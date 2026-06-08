#!/usr/bin/env python3
"""Stage2 runner for Alpha158 practice with explicit time-decay reweighting.

This script mirrors the Qlib qrun workflow but instantiates a custom
TimeDecayReweighter so we can tune the half-life without modifying the
installed qlib package.
"""
from __future__ import annotations

import argparse
import inspect
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from qlib.config import C
from qlib.constant import REG_CN
from qlib.data.dataset import Dataset, TSDataSampler
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.utils import auto_filter_kwargs, fill_placeholder, flatten_dict, init_instance_by_config
from qlib.workflow import R

from time_decay_reweighter import TimeDecayReweighter

logger = get_module_logger("stage2_practice")


def _materialize_label_data(label_obj):
    if isinstance(label_obj, (pd.Series, pd.DataFrame)):
        return label_obj
    if isinstance(label_obj, TSDataSampler):
        values = []
        for idx in range(len(label_obj)):
            arr = np.asarray(label_obj[idx])
            if arr.ndim == 0:
                values.append(float(arr))
            elif arr.ndim == 1:
                values.append(float(arr[-1]))
            else:
                values.append(float(arr[-1, -1]))
        return pd.DataFrame({"label": values}, index=label_obj.get_index())
    raise TypeError(f"Unsupported label object type: {type(label_obj)!r}")


def _predict_for_segment(model: Model, dataset: Dataset, segment: str):
    try:
        signature = inspect.signature(model.predict)
        if "segment" in signature.parameters:
            return model.predict(dataset, segment=segment)
    except (TypeError, ValueError):
        pass

    original_segments = getattr(dataset, "segments", {}).copy()
    if segment not in original_segments:
        raise KeyError(f"Dataset segment not found: {segment}")

    patched_segments = original_segments.copy()
    patched_segments["test"] = original_segments[segment]
    try:
        dataset.config(segments=patched_segments)
        return model.predict(dataset)
    finally:
        dataset.config(segments=original_segments)


# ── Factor IC helpers ──────────────────────────────────────────────────

def _get_feature_names(dataset: Dataset) -> list[str] | None:
    """Try to get feature names from dataset handler's data_loader config."""
    try:
        handler = getattr(dataset, "handler", None)
        if handler is None:
            return None
        dl = getattr(handler, "_data_loader", None) or getattr(handler, "data_loader", None)
        if dl is None:
            return None
        # 从 dict 配置中提取
        if isinstance(dl, dict):
            feat_cfg = dl.get("kwargs", {}).get("config", {}).get("feature", [])
            if isinstance(feat_cfg, (list, tuple)) and len(feat_cfg) == 2:
                return list(feat_cfg[1])
        # 从 QlibDataLoader 实例的 fields 中提取
        fields = getattr(dl, "fields", None)
        if fields is not None:
            if isinstance(fields, dict):
                # 优先从 "feature" 分组提取
                if "feature" in fields:
                    return list(fields["feature"][1])
                # 回退: 收集所有分组名称 (但排除 label 分组)
                names = []
                for grp_key, (exprs, grp_names) in fields.items():
                    if grp_key != "label":
                        names.extend(grp_names)
                return names if names else None
            elif isinstance(fields, (list, tuple)) and len(fields) == 2:
                return list(fields[1])
    except Exception:
        pass
    return None


def _materialize_features(dataset: Dataset, segment: str) -> pd.DataFrame:
    """Convert feature TSDataSampler to DataFrame [datetime, instrument] × features."""
    sampler = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)

    # Already a DataFrame
    if isinstance(sampler, pd.DataFrame):
        return sampler

    # Fast path: direct numpy access
    data_arr = getattr(sampler, "data_arr", None)
    idx = getattr(sampler, "data_idx", None)
    if data_arr is not None and idx is not None:
        return pd.DataFrame(data_arr, index=idx)

    # Fallback: iterate via sampler indexing
    idx = getattr(sampler, "get_index", lambda: None)()
    n = len(sampler)
    if n == 0:
        return pd.DataFrame()

    first = np.asarray(sampler[0]).ravel()
    n_feats = len(first)
    data = np.empty((n, n_feats), dtype=np.float64)
    data[0] = first
    for i in range(1, n):
        data[i] = np.asarray(sampler[i]).ravel()[:n_feats]

    if idx is not None:
        return pd.DataFrame(data, index=idx)
    return pd.DataFrame(data)


def _compute_factor_ic(dataset: Dataset, segment: str) -> pd.DataFrame:
    """Compute per-factor Rank IC (Spearman) and Pearson IC for a segment.

    Returns DataFrame with columns:
        factor, mean_RankIC, std_RankIC, IR_RankIC, pos_RankIC,
        mean_PearsonIC, std_PearsonIC, IR_PearsonIC, pos_PearsonIC, n_dates
    """
    feat_df = _materialize_features(dataset, segment)
    if feat_df.empty:
        logger.warning("_compute_factor_ic: empty feature data for segment=%s", segment)
        return pd.DataFrame()

    label_obj = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)
    label_df = _materialize_label_data(label_obj)
    if label_df.empty:
        logger.warning("_compute_factor_ic: empty label data for segment=%s", segment)
        return pd.DataFrame()

    feat_names = _get_feature_names(dataset)
    if feat_names is None:
        # 尝试从 feat_df 自带列名恢复
        existing_cols = [c for c in feat_df.columns if isinstance(c, str) and c != ""]
        if existing_cols:
            feat_names = existing_cols
        else:
            feat_names = [f"F{i}" for i in range(feat_df.shape[1])]
    feat_names = feat_names[: feat_df.shape[1]]
    feat_df.columns = feat_names

    # Align features and labels on MultiIndex
    label_col = "label" if "label" in label_df.columns else label_df.columns[0]
    joined = feat_df.join(label_df[[label_col]], how="inner")
    if joined.empty:
        logger.warning("_compute_factor_ic: no overlap between features and labels")
        return pd.DataFrame()

    factor_cols = [c for c in feat_names if c in joined.columns]
    if not factor_cols:
        return pd.DataFrame()

    results = []
    for factor in factor_cols:
        daily = joined.groupby(level="datetime").apply(
            lambda g: pd.Series({
                "rank_ic": (
                    g[factor].corr(g[label_col], method="spearman")
                    if g[factor].notna().sum() >= 5 and g[label_col].notna().sum() >= 5
                    else np.nan
                ),
                "pearson_ic": (
                    g[factor].corr(g[label_col])
                    if g[factor].notna().sum() >= 5 and g[label_col].notna().sum() >= 5
                    else np.nan
                ),
            })
        )
        rank_ic = daily["rank_ic"].dropna()
        pearson_ic = daily["pearson_ic"].dropna()
        n = len(rank_ic)

        results.append({
            "factor": factor,
            "mean_RankIC": float(rank_ic.mean()) if n else np.nan,
            "std_RankIC": float(rank_ic.std(ddof=1)) if n > 1 else np.nan,
            "IR_RankIC": float(rank_ic.mean() / rank_ic.std(ddof=1)) if n > 1 and rank_ic.std() > 1e-12 else np.nan,
            "pos_RankIC": float((rank_ic > 0).mean()) if n else np.nan,
            "mean_PearsonIC": float(pearson_ic.mean()) if n else np.nan,
            "std_PearsonIC": float(pearson_ic.std(ddof=1)) if n > 1 else np.nan,
            "IR_PearsonIC": float(pearson_ic.mean() / pearson_ic.std(ddof=1)) if n > 1 and pearson_ic.std() > 1e-12 else np.nan,
            "pos_PearsonIC": float((pearson_ic > 0).mean()) if n else np.nan,
            "n_dates": n,
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df
    return df.sort_values("mean_RankIC", key=abs, ascending=False, na_position="last")


def _get_model_feature_importance(model: Model, feature_names: list[str]) -> pd.DataFrame:
    """Extract LightGBM feature importance (gain + split)."""
    try:
        booster = getattr(model, "booster_", None)
        if booster is None:
            return pd.DataFrame()
        gain = booster.feature_importance(importance_type="gain")
        split = booster.feature_importance(importance_type="split")
    except Exception:
        return pd.DataFrame()

    model_names = booster.feature_name()
    names = model_names if model_names else feature_names[: len(gain)]

    df = pd.DataFrame({
        "factor": list(names)[: len(gain)],
        "gain": gain,
        "split": split,
    })
    df["gain_pct"] = df["gain"] / df["gain"].sum() * 100 if df["gain"].sum() > 0 else 0.0
    df["split_pct"] = df["split"] / df["split"].sum() * 100 if df["split"].sum() > 0 else 0.0
    return df.sort_values("gain", ascending=False)


def _print_factor_ic_report(
    fic_valid: pd.DataFrame,
    fic_test: pd.DataFrame,
    fi: pd.DataFrame,
    top_n: int = 30,
) -> None:
    """Print formatted factor IC report to stdout."""

    def _fmt(v, width=8):
        if pd.isna(v):
            return f"{'NaN':>{width}}"
        return f"{v:>{width}.4f}"

    def _print_table(df: pd.DataFrame, title: str, date_info: str = ""):
        if df.empty:
            print(f"  (empty)\n")
            return
        n = min(top_n, len(df))
        print(f"  {title} {date_info}")
        header = f"  {'Factor':<30s} {'RankIC':>8s} {'RankIR':>8s} {'Pos%':>7s} {'PearIC':>8s} {'PearIR':>8s} {'Pos%':>7s}  n"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, row in df.head(n).iterrows():
            print(
                f"  {row['factor']:<30s} "
                f"{_fmt(row.get('mean_RankIC'))} "
                f"{_fmt(row.get('IR_RankIC'))} "
                f"{_fmt(row.get('pos_RankIC') * 100 if pd.notna(row.get('pos_RankIC')) else np.nan, 6):>7} "
                f"{_fmt(row.get('mean_PearsonIC'))} "
                f"{_fmt(row.get('IR_PearsonIC'))} "
                f"{_fmt(row.get('pos_PearsonIC') * 100 if pd.notna(row.get('pos_PearsonIC')) else np.nan, 6):>7} "
                f"{int(row.get('n_dates', 0)):>4d}"
            )
        print()

    def _print_fi_table(df: pd.DataFrame):
        if df.empty:
            print("  (empty)\n")
            return
        n = min(top_n, len(df))
        print(f"  LightGBM Feature Importance (top {n})")
        header = f"  {'Factor':<30s} {'gain%':>8s} {'split%':>8s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, row in df.head(n).iterrows():
            print(
                f"  {row['factor']:<30s} "
                f"{_fmt(row.get('gain_pct'), 7):>8} "
                f"{_fmt(row.get('split_pct'), 7):>8}"
            )
        print()

    print()
    print("=" * 75)
    print("  Factor IC Analysis")
    print("=" * 75)

    # Valid dates info
    v_info = ""
    if not fic_valid.empty and "n_dates" in fic_valid.columns:
        v_info = f"({int(fic_valid['n_dates'].iloc[0])} dates)"
    _print_table(fic_valid, "Single-Factor IC — Valid", v_info)

    t_info = ""
    if not fic_test.empty and "n_dates" in fic_test.columns:
        t_info = f"({int(fic_test['n_dates'].iloc[0])} dates)"
    _print_table(fic_test, "Single-Factor IC — Test", t_info)

    _print_fi_table(fi)
    print("=" * 75)


def _save_split_predictions(model: Model, dataset: Dataset) -> None:
    valid_pred = _predict_for_segment(model, dataset, "valid")
    valid_label = _materialize_label_data(dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L))
    test_pred = _predict_for_segment(model, dataset, "test")
    test_label = _materialize_label_data(dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L))

    R.save_objects(
        **{
            "valid_pred.pkl": valid_pred,
            "valid_label.pkl": valid_label,
            "test_pred_snapshot.pkl": test_pred,
            "test_label_snapshot.pkl": test_label,
        }
    )

def _coerce_scalar(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        if any(ch in text.lower() for ch in [".", "e"]):
            return float(text)
        return int(text)
    except ValueError:
        return value


def _normalize_config_types(obj):
    if isinstance(obj, dict):
        return {k: _normalize_config_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_config_types(v) for v in obj]
    return _coerce_scalar(obj)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return _normalize_config_types(yaml.safe_load(f))


def _init_qlib(qlib_init: dict[str, Any], uri_folder: str) -> None:
    import qlib

    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / uri_folder)
    qlib.init(**qlib_init, exp_manager=exp_manager)


def _log_task_info(task_config: dict[str, Any]) -> None:
    R.log_params(**flatten_dict(task_config))
    R.save_objects(**{"task": task_config})
    R.set_tags(**{"hostname": os.uname().nodename})


def _load_warm_start_checkpoint(path: str | None):
    if not path:
        return None
    ckpt = Path(path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {ckpt}")
    with open(ckpt, "rb") as f:
        return pickle.load(f)


def _build_fit_kwargs(model: Model, warm_start_model, reweighter: TimeDecayReweighter) -> dict[str, Any]:
    fit_kwargs: dict[str, Any] = {"reweighter": reweighter}
    if warm_start_model is None:
        return fit_kwargs

    warm_inner_model = getattr(warm_start_model, "model", None)
    if warm_inner_model is None:
        return fit_kwargs

    model_name = type(model).__name__
    if model_name == "XGBModel":
        fit_kwargs["xgb_model"] = warm_inner_model
    elif model_name == "LGBModel":
        fit_kwargs["init_model"] = warm_inner_model
    else:
        logger.warning("Warm start is not implemented for model type: %s", model_name)
    return fit_kwargs


def _exe_task(task_config: dict[str, Any], reweighter: TimeDecayReweighter, warm_start_path: str | None = None) -> None:
    rec = R.get_recorder()
    model: Model = init_instance_by_config(task_config["model"], accept_types=Model)
    dataset: Dataset = init_instance_by_config(task_config["dataset"], accept_types=Dataset)
    warm_start_model = _load_warm_start_checkpoint(warm_start_path)
    if warm_start_path:
        logger.info("Warm start checkpoint: %s", warm_start_path)
    else:
        logger.info("Warm start checkpoint: <none>")
    fit_kwargs = _build_fit_kwargs(model, warm_start_model, reweighter)

    auto_filter_kwargs(model.fit)(dataset, **fit_kwargs)
    R.save_objects(**{"params.pkl": model})
    dataset.config(dump_all=False, recursive=True)
    R.save_objects(**{"dataset": dataset})
    _save_split_predictions(model, dataset)

    # ── Per-factor IC + feature importance ──
    try:
        feat_names = _get_feature_names(dataset)
        fic_valid = _compute_factor_ic(dataset, "valid")
        fic_test = _compute_factor_ic(dataset, "test")
        fi = _get_model_feature_importance(model, feat_names or [])
        R.save_objects(**{
            "factor_ic_valid.pkl": fic_valid,
            "factor_ic_test.pkl": fic_test,
            "feature_importance.pkl": fi,
        })
        _print_factor_ic_report(fic_valid, fic_test, fi)
    except Exception:
        logger.exception("Factor IC computation failed, continuing")

    placehorder_value = {"<MODEL>": model, "<DATASET>": dataset}
    task_config = fill_placeholder(task_config, placehorder_value)
    records = task_config.get("record", [])
    if isinstance(records, dict):
        records = [records]
    for record in records:
        r = init_instance_by_config(
            record,
            recorder=rec,
            default_module="qlib.workflow.record_temp",
            try_kwargs={"model": model, "dataset": dataset},
        )
        r.generate()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Generated practice YAML")
    ap.add_argument("--experiment-name", required=True)
    ap.add_argument("--uri-folder", default="mlruns")
    ap.add_argument("--half-life", type=int, default=252)
    ap.add_argument("--floor", type=float, default=0.2)
    ap.add_argument("--warm-start", default=None, help="Previous fold params.pkl checkpoint for warm start")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    qlib_init = cfg.get("qlib_init", {})
    task = cfg.get("task", {})

    if args.half_life <= 0:
        raise ValueError("half-life must be positive")
    if not (0 < args.floor <= 1):
        raise ValueError("floor must be in (0, 1]")

    _init_qlib(qlib_init, args.uri_folder)

    reweighter = TimeDecayReweighter(half_life=args.half_life, floor=args.floor)

    with R.start(experiment_name=args.experiment_name):
        _log_task_info(task)
        _exe_task(task, reweighter=reweighter, warm_start_path=args.warm_start)


if __name__ == "__main__":
    main()
