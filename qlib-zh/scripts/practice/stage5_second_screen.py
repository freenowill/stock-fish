#!/usr/bin/env python3
"""
stage5_second_screen.py
二筛：
    1. 从 risk_eval 读取财务 / 风险 / 估值标签
    2. 先做硬约束：高风险、ST、停牌剔除
    3. 再用财务 / 估值 / 风险 / 负面公告做软惩罚
    4. 从 model_predict/scores.csv 补充 score / rank / percentile /
         annualized_return / max_drawdown / sharpe_ratio / ICIR / monthly_win_rate
    5. 以模型分数为主，叠加 IC、稳定性、近期表现与基本面惩罚构建综合分
    6. 取 Top-K 进入目标组合
输出: <output>/second_screen.csv
"""
import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

warnings.filterwarnings("ignore")


def _normalize_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"^[A-Za-z]+", "", regex=True).str.zfill(6)


def _risk_order(series: pd.Series) -> pd.Series:
    mapping = {"低": 0, "中": 1, "高": 2}
    return series.map(mapping).fillna(3)


def _valuation_order(series: pd.Series) -> pd.Series:
    mapping = {"低": 0, "中": 1, "高": 2}
    return series.map(mapping).fillna(3)


def _bool_series(series: pd.Series | None, index: pd.Index, default: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index)
    if isinstance(series, pd.Series):
        return series.fillna(default).astype(bool)
    return pd.Series(default, index=index)


def _normalize_industry(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    return out.replace({"nan": "未知", "None": "未知", "": "未知"}).fillna("未知")


def _minmax(series: pd.Series, ascending: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() <= 1:
        return pd.Series(0.0, index=series.index)
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=series.index)
    scaled = (s - lo) / (hi - lo)
    if not ascending:
        scaled = 1 - scaled
    return scaled.fillna(0.0)


def _resolve_scores_csv(pred_dir: str) -> Path:
    pred_path = Path(pred_dir)
    scores_csv = pred_path / "scores.csv"
    if scores_csv.exists():
        return scores_csv
    candidates = sorted(pred_path.glob("walk_forward/*/model_predict/scores.csv"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"scores.csv 不存在: {scores_csv}")


def _read_csv_or_empty(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except EmptyDataError:
        return pd.DataFrame()


_QLIB_INITED = False


def _qlib_root() -> str:
    return os.environ.get("QLIB_DATA_DIR", "/root/.qlib/qlib_data/cn_data")


def _ensure_qlib():
    global _QLIB_INITED
    if _QLIB_INITED:
        return
    import qlib
    from qlib.constant import REG_CN

    qlib.init(provider_uri=_qlib_root(), region=REG_CN)
    _QLIB_INITED = True


def _to_qlib_code(code: str) -> str:
    code = str(code).zfill(6)
    prefix = "SH" if code.startswith(("5", "6", "9")) else "SZ"
    return f"{prefix}{code}"


def _history_volatility(codes: list[str], pred_date: str, lookback_days: int = 60) -> pd.DataFrame:
    _ensure_qlib()
    from qlib.data import D

    if not codes:
        return pd.DataFrame(columns=["code", "volatility_60d"])

    end_ts = pd.Timestamp(pred_date)
    start_ts = end_ts - pd.Timedelta(days=lookback_days * 3)
    q_codes = [_to_qlib_code(code) for code in codes]
    close_df = D.features(q_codes, ["$close"], start_time=start_ts.strftime("%Y-%m-%d"), end_time=pred_date)
    if close_df is None or len(close_df) == 0:
        return pd.DataFrame({"code": codes, "volatility_60d": np.nan})

    close_df = close_df.reset_index()
    close_df.columns = ["instrument", "datetime", "close"]
    close_df["code"] = close_df["instrument"].astype(str).str.replace(r"^[A-Za-z]+", "", regex=True).str.zfill(6)
    close_df["datetime"] = pd.to_datetime(close_df["datetime"], errors="coerce")
    close_df["close"] = pd.to_numeric(close_df["close"], errors="coerce")
    close_df = close_df.sort_values(["code", "datetime"])
    close_df["ret"] = close_df.groupby("code")["close"].pct_change()
    vol = close_df.groupby("code")["ret"].apply(lambda s: float(s.dropna().tail(lookback_days).std(ddof=1)) if s.dropna().shape[0] >= 5 else np.nan)
    return vol.rename("volatility_60d").reset_index()


def _apply_risk_parity_weights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    score = pd.to_numeric(out.get("composite_score"), errors="coerce")
    if score.notna().sum() > 1:
        score_weight = _minmax(score)
    else:
        score_weight = pd.Series(1.0 / max(len(out), 1), index=out.index)

    vol = pd.to_numeric(out.get("volatility_60d"), errors="coerce")
    inv_vol = 1.0 / vol.replace(0, np.nan)
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)
    if inv_vol.notna().sum() == 0:
        combined = score_weight
    else:
        inv_vol = inv_vol.fillna(inv_vol.dropna().median() if inv_vol.notna().any() else 1.0)
        vol_weight = inv_vol / inv_vol.sum()
        combined = 0.7 * score_weight + 0.3 * vol_weight

    if combined.isna().all() or combined.sum() <= 0:
        out["risk_parity_weight"] = 1.0 / max(len(out), 1)
    else:
        combined = combined.fillna(combined.dropna().median() if combined.notna().any() else 1.0)
        out["risk_parity_weight"] = combined / combined.sum()
    out["weight"] = out["risk_parity_weight"]
    return out


def _apply_industry_cap(df: pd.DataFrame, hold_num: int, cap_ratio: float = 0.40) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    if out.empty:
        return out

    if "industry" not in out.columns:
        out["industry"] = "未知"
    out["industry"] = _normalize_industry(out["industry"])

    max_per_industry = max(1, int(np.floor(hold_num * cap_ratio)))
    selected_rows: list[int] = []
    industry_counts: dict[str, int] = {}

    for idx, row in out.iterrows():
        industry = str(row["industry"])
        if industry_counts.get(industry, 0) >= max_per_industry:
            continue
        selected_rows.append(idx)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if len(selected_rows) >= hold_num:
            break

    if len(selected_rows) < hold_num:
        print(
            f"  ⚠ 行业约束后可选股票仅 {len(selected_rows)} 只，低于目标 {hold_num} 只；"
            f" 当前行业上限={max_per_industry}"
        )

    return out.loc[selected_rows].reset_index(drop=True)


def second_screen(risk_eval_csv: str, pred_dir: str,
                  output_dir: str, hold_num: int = 5):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── 1. 加载风险评估结果 ───────────────────────
    risk_df = _read_csv_or_empty(risk_eval_csv)
    if risk_df.empty:
        print(f"⚠ 风险评估文件为空或不存在: {risk_eval_csv}，回退使用 scores.csv 作为候选池")
        scores_csv = _resolve_scores_csv(pred_dir)
        risk_df = _read_csv_or_empty(scores_csv)
        if risk_df.empty:
            print("⚠ scores.csv 也为空，输出空的二筛结果")
            (out_path / "second_screen.csv").write_text("", encoding="utf-8")
            (out_path / "second_screen_candidates.csv").write_text("", encoding="utf-8")
            return
    if "code" not in risk_df.columns and "instrument" in risk_df.columns:
        risk_df["code"] = risk_df["instrument"]
    if "code" in risk_df.columns:
        risk_df["code"] = _normalize_code(risk_df["code"])
    for col, default in {
        "risk_label": "中",
        "valuation_label": "中",
        "financial_label": "中",
        "is_st": False,
        "is_suspended": False,
    }.items():
        if col not in risk_df.columns:
            risk_df[col] = default
    print(f"✓ 加载 risk_eval: {len(risk_df)} 只股票")

    # ── 2. 先做硬约束：高风险 / ST / 停牌剔除 ──────
    before = len(risk_df)
    is_st = _bool_series(risk_df["is_st"] if "is_st" in risk_df.columns else None, risk_df.index, default=False)
    is_suspended = _bool_series(risk_df["is_suspended"] if "is_suspended" in risk_df.columns else None, risk_df.index, default=False)
    risk_label = risk_df["risk_label"] if "risk_label" in risk_df.columns else pd.Series("中", index=risk_df.index)
    mask_ok = (
        (risk_label != "高")
        & (~is_st)
        & (~is_suspended)
    )
    filtered = risk_df[mask_ok].copy()
    print(f"✓ 过滤高风险/ST/停牌: {before} → {len(filtered)} 只")

    if filtered.empty:
        print("  ⚠ 严格约束后无剩余股票，将放宽为仅排除停牌/ST")
        filtered = risk_df[(~is_st) & (~is_suspended)].copy()

    # ── 3. 从 model_predict/scores.csv 补充得分信息 ─
    scores_csv = _resolve_scores_csv(pred_dir)
    scores_df = pd.read_csv(scores_csv)

    # 统一 code 字段
    if "code" not in scores_df.columns and "instrument" in scores_df.columns:
        scores_df["code"] = _normalize_code(scores_df["instrument"])
    if "code" in scores_df.columns:
        scores_df["code"] = _normalize_code(scores_df["code"])

    merge_cols = ["code", "stock", "date", "score", "score_final", "rank", "percentile", "rank_pct"]
    perf_cols  = ["annualized_return", "max_drawdown", "sharpe_ratio", "ICIR", "monthly_win_rate"]
    for c in perf_cols:
        if c in scores_df.columns:
            merge_cols.append(c)

    # 如果 filtered 里已有 score 列就先丢掉，避免重复
    drop_cols = [c for c in merge_cols if c in filtered.columns and c != "code"]
    filtered  = filtered.drop(columns=drop_cols, errors="ignore")

    result = filtered.merge(
        scores_df[merge_cols],
        on="code",
        how="left"
    )
    pred_date = None
    for col in ["date", "pred_date"]:
        if col in result.columns and result[col].notna().any():
            pred_date = str(result[col].dropna().astype(str).iloc[0])
            break

    # ── 4. 对候选池排序，并同时保存完整候选列表 ──────
    result["_risk_order"] = _risk_order(result.get("risk_label", pd.Series(index=result.index, dtype=object)))
    result["_valuation_order"] = _valuation_order(result.get("valuation_label", pd.Series(index=result.index, dtype=object)))
    result["_financial_score"] = pd.to_numeric(result.get("financial_score"), errors="coerce")
    primary_score = result.get("score_final", result.get("score"))
    result["_score"] = pd.to_numeric(primary_score, errors="coerce")
    result["_ann_norm"] = _minmax(result.get("annualized_return", pd.Series(index=result.index, dtype=float)))
    result["_mdd_norm"] = _minmax(result.get("max_drawdown", pd.Series(index=result.index, dtype=float)), ascending=True)
    result["_sharpe_norm"] = _minmax(result.get("sharpe_ratio", pd.Series(index=result.index, dtype=float)))
    result["_icir_norm"] = _minmax(result.get("ICIR", pd.Series(index=result.index, dtype=float)))
    result["_win_norm"] = _minmax(result.get("monthly_win_rate", pd.Series(index=result.index, dtype=float)))
    result["_score_norm"] = _minmax(result["_score"])
    result["_fin_quality"] = _minmax(result.get("financial_score", pd.Series(index=result.index, dtype=float)))
    result["_val_quality"] = _minmax(result.get("valuation_score", pd.Series(index=result.index, dtype=float)))
    risk_score_series = pd.to_numeric(result.get("risk_score", pd.Series(50.0, index=result.index)), errors="coerce")
    news_series = pd.to_numeric(result.get("negative_news_count", pd.Series(0.0, index=result.index)), errors="coerce").fillna(0.0)
    result["_risk_quality"] = _minmax(100.0 - risk_score_series)
    result["_news_quality"] = _minmax(3.0 - news_series)
    result["_vol_quality"] = 1.0 - _minmax(result.get("volatility_60d", pd.Series(index=result.index, dtype=float)), ascending=True)
    result["composite_score"] = (
        0.42 * result["_score_norm"]
        + 0.12 * result["_icir_norm"]
        + 0.08 * result["_sharpe_norm"]
        + 0.07 * result["_ann_norm"]
        + 0.05 * result["_win_norm"]
        + 0.04 * result["_mdd_norm"]
        + 0.10 * result["_fin_quality"]
        + 0.06 * result["_val_quality"]
        + 0.04 * result["_risk_quality"]
        + 0.02 * result["_news_quality"]
        + 0.02 * result["_vol_quality"]
    )
    if "rank" in result.columns:
        result["rank"] = pd.to_numeric(result["rank"], errors="coerce")
    result = result.sort_values(
        ["_risk_order", "_valuation_order", "composite_score", "rank", "_financial_score", "_score"],
        ascending=[True, True, False, True, False, False],
    )

    candidates = result.drop(
        columns=[
            "_risk_order", "_valuation_order", "_financial_score", "_score",
            "_ann_norm", "_mdd_norm", "_sharpe_norm", "_icir_norm", "_win_norm", "_score_norm",
        ]
    ).reset_index(drop=True)

    if pred_date:
        vol_df = _history_volatility(candidates["code"].astype(str).tolist(), pred_date, lookback_days=60)
        candidates = candidates.merge(vol_df, on="code", how="left")
    else:
        candidates["volatility_60d"] = np.nan

    candidates_csv = out_path / "second_screen_candidates.csv"
    result = _apply_industry_cap(candidates, hold_num=hold_num, cap_ratio=0.40)
    result = _apply_risk_parity_weights(result)

    candidate_weights = _apply_risk_parity_weights(result.copy())[["code", "risk_parity_weight", "weight"]]
    candidates = candidates.merge(candidate_weights, on="code", how="left")
    candidates.to_csv(candidates_csv, index=False, encoding="utf-8-sig")

    print(f"\n◆ 二筛完成，选出 {len(result)} 只股票:")
    display_cols = ["code", "score", "rank", "rank_pct",
                    "percentile", "composite_score", "volatility_60d", "risk_parity_weight", "financial_label", "valuation_label", "risk_label",
                    "financial_score", "valuation_score", "risk_score",
                    "annualized_return", "max_drawdown", "sharpe_ratio", "ICIR", "monthly_win_rate"]
    display_cols = [c for c in display_cols if c in result.columns]
    print(result[display_cols].to_string(index=False))

    out_csv = out_path / "second_screen.csv"
    result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✓ 二筛结果保存: {out_csv}")
    print(f"✓ 二筛完整候选池保存: {candidates_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk-eval", required=True, dest="risk_eval")
    ap.add_argument("--pred-dir",  required=True, dest="pred_dir")
    ap.add_argument("--output",    required=True)
    ap.add_argument("--hold-num",  type=int, default=5, dest="hold_num")
    args = ap.parse_args()
    second_screen(args.risk_eval, args.pred_dir, args.output, args.hold_num)


if __name__ == "__main__":
    main()
