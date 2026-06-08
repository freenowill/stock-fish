#!/usr/bin/env python3
"""
export_model_predict.py
从 MLflow run artifacts 读取 pred.pkl / label.pkl / portfolio_analysis，
导出股票预测得分、绩效指标到 CSV，并生成 overview.html。
输出:
  <output>/scores.csv      — 每只股票的预测得分（按 pred_date 日截面）
  <output>/metrics.csv     — IC / ICIR / 月胜率 / 最大回撤等汇总指标
    <output>/overview.html   — 图表化分析总览
"""
import argparse
import importlib
import math
import pickle
import os
import sys
import subprocess
from pathlib import Path
import warnings

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
analysis_position = None
analysis_model = None
_ANALYSIS_IMPORT_ERROR = None

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────

def _load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _max_drawdown(series: pd.Series) -> float:
    """计算最大回撤（负数）"""
    cummax = (1 + series).cumprod().cummax()
    dd = (1 + series).cumprod() / cummax - 1
    return float(dd.min())


def _monthly_win_rate(series: pd.Series) -> float:
    """月度胜率 = 月收益率 > 0 的比例"""
    monthly = (1 + series).resample("ME").prod() - 1
    if monthly.empty:
        return float("nan")
    return float((monthly > 0).mean())


def _safe_load_df(path: Path):
    obj = _load_pkl(path)
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame("value")
    return None


def _normalize_index_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.MultiIndex):
        # 统一成按 datetime 索引的单层时间轴，便于后续统计
        if "datetime" in out.index.names:
            out.index = pd.to_datetime(out.index.get_level_values("datetime"))
        else:
            out.index = pd.to_datetime(out.index.get_level_values(-1))
    else:
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out


def _load_qlib_report_modules():
    global analysis_position, analysis_model, _ANALYSIS_IMPORT_ERROR
    if analysis_position is not None or _ANALYSIS_IMPORT_ERROR is not None:
        return analysis_position, analysis_model

    original_sys_path = sys.path[:]
    try:
        blocked = {ROOT.resolve(), Path(os.getcwd()).resolve()}
        cleaned = []
        for entry in original_sys_path:
            entry_path = Path(entry or ".").resolve()
            if entry_path in blocked:
                continue
            cleaned.append(entry)
        sys.path = cleaned
        analysis_position = importlib.import_module("qlib.contrib.report.analysis_position")
        analysis_model = importlib.import_module("qlib.contrib.report.analysis_model")
    except Exception as exc:  # pragma: no cover - environment dependent
        missing_name = getattr(exc, "name", "") or str(exc)
        if "statsmodels" in missing_name and "statsmodels" not in sys.modules:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "statsmodels"],
                    check=True,
                )
                analysis_position = importlib.import_module("qlib.contrib.report.analysis_position")
                analysis_model = importlib.import_module("qlib.contrib.report.analysis_model")
                _ANALYSIS_IMPORT_ERROR = None
                return analysis_position, analysis_model
            except Exception as retry_exc:  # pragma: no cover - environment dependent
                exc = retry_exc
        _ANALYSIS_IMPORT_ERROR = exc
        analysis_position = None
        analysis_model = None
    finally:
        sys.path = original_sys_path

    return analysis_position, analysis_model


def _robust_scale(series: pd.Series) -> pd.Series:
    """Robust z-score based on median and MAD, clipped to [-3, 3]."""
    s = pd.to_numeric(series, errors="coerce")
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(0.0, index=s.index)
    z = (s - med) / (1.4826 * mad)
    return z.clip(-3, 3)


def _compute_ic_metrics(pred_df: pd.DataFrame, label_df: pd.DataFrame) -> dict:
    if pred_df is None or label_df is None or pred_df.empty or label_df.empty:
        return {"IC_mean": float("nan"), "IC_std": float("nan"), "ICIR": float("nan"), "rank_IC_mean": float("nan")}

    pred = pred_df.copy()
    label = label_df.copy()
    if label.shape[1] != 1:
        label = label.iloc[:, :1]
    label.columns = ["label"]

    joined = pred.join(label, how="inner")
    if joined.empty:
        return {"IC_mean": float("nan"), "IC_std": float("nan"), "ICIR": float("nan"), "rank_IC_mean": float("nan")}

    ic_df = joined.groupby(level="datetime", group_keys=False).apply(lambda x: pd.Series({
        "ic": x["label"].corr(x["score"]),
        "rank_ic": x["label"].corr(x["score"], method="spearman"),
    }))
    ic_series = ic_df["ic"].dropna() if "ic" in ic_df.columns else pd.Series(dtype=float)
    rank_ic_series = ic_df["rank_ic"].dropna() if "rank_ic" in ic_df.columns else pd.Series(dtype=float)

    ic_mean = float(ic_series.mean()) if len(ic_series) else float("nan")
    ic_std = float(ic_series.std()) if len(ic_series) else float("nan")
    icir = ic_mean / ic_std if pd.notna(ic_mean) and pd.notna(ic_std) and ic_std not in (0, 0.0) else float("nan")
    rank_ic_mean = float(rank_ic_series.mean()) if len(rank_ic_series) else float("nan")
    return {"IC_mean": ic_mean, "IC_std": ic_std, "ICIR": icir, "rank_IC_mean": rank_ic_mean}


def _fallback_report_figures(report_df: pd.DataFrame) -> list:
    try:
        go = importlib.import_module("plotly.graph_objects")
        make_subplots = importlib.import_module("plotly.subplots").make_subplots
    except Exception:
        return []

    if not isinstance(report_df, pd.DataFrame) or report_df.empty:
        return []

    rpt = _normalize_index_datetime(report_df)
    required_cols = {"return", "cost", "bench", "turnover"}
    if not required_cols.issubset(rpt.columns):
        return []

    cum_strategy = pd.to_numeric(rpt["return"], errors="coerce").fillna(0).cumsum()
    cum_strategy_net = (pd.to_numeric(rpt["return"], errors="coerce") - pd.to_numeric(rpt["cost"], errors="coerce")).fillna(0).cumsum()
    cum_bench = pd.to_numeric(rpt["bench"], errors="coerce").fillna(0).cumsum()
    cum_excess = (pd.to_numeric(rpt["return"], errors="coerce") - pd.to_numeric(rpt["bench"], errors="coerce")).fillna(0).cumsum()
    turnover = pd.to_numeric(rpt["turnover"], errors="coerce")
    drawdown = cum_strategy_net - cum_strategy_net.cummax()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Cumulative Return", "Drawdown", "Turnover"),
    )
    fig.add_trace(go.Scatter(x=rpt.index, y=cum_bench, name="Benchmark", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=rpt.index, y=cum_strategy, name="Strategy Gross", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=rpt.index, y=cum_strategy_net, name="Strategy Net", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=rpt.index, y=cum_excess, name="Excess Gross", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=rpt.index, y=drawdown, name="Net Drawdown", mode="lines", fill="tozeroy"), row=2, col=1)
    fig.add_trace(go.Bar(x=rpt.index, y=turnover.fillna(0), name="Turnover"), row=3, col=1)
    fig.update_layout(height=980, title="Qlib-style Backtest Report", legend=dict(orientation="h"))
    return [fig]


def _fallback_ic_figures(pred_label: pd.DataFrame) -> list:
    try:
        go = importlib.import_module("plotly.graph_objects")
    except Exception:
        return []

    if pred_label is None or pred_label.empty:
        return []

    joined = pred_label[["score", "label"]].dropna().copy()
    if joined.empty:
        return []

    ic_df = joined.groupby(level="datetime", group_keys=False).apply(
        lambda x: pd.Series({
            "ic": x["label"].corr(x["score"]),
            "rank_ic": x["label"].corr(x["score"], method="spearman"),
        })
    ).dropna(how="all")
    if ic_df.empty:
        return []

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ic_df.index, y=ic_df["ic"], name="IC", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=ic_df.index, y=ic_df["rank_ic"], name="Rank IC", mode="lines+markers"))
    fig.update_layout(title="Qlib-style Score IC", height=420)
    return [fig]


def _fallback_model_figures(pred_label: pd.DataFrame, groups: int = 5) -> list:
    try:
        go = importlib.import_module("plotly.graph_objects")
    except Exception:
        return []

    if pred_label is None or pred_label.empty:
        return []

    joined = pred_label[["score", "label"]].dropna().copy()
    if joined.empty:
        return []

    def _daily_group_return(x: pd.DataFrame) -> pd.Series:
        x = x.sort_values("score", ascending=False)
        n = len(x)
        bucket = max(n // groups, 1)
        top = x.iloc[:bucket]["label"].mean()
        bottom = x.iloc[-bucket:]["label"].mean()
        avg = x["label"].mean()
        return pd.Series({"long_short": top - bottom, "long_avg": top - avg})

    group_ret = joined.groupby(level="datetime", group_keys=False).apply(_daily_group_return).dropna(how="all")
    auto_corr = joined.assign(prev_score=joined.groupby(level="instrument")["score"].shift(1)).groupby(level="datetime", group_keys=False).apply(
        lambda x: x["score"].rank(pct=True).corr(x["prev_score"].rank(pct=True))
    ).dropna()

    figs = []
    if not group_ret.empty:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=group_ret.index, y=group_ret["long_short"].cumsum(), name="Long-Short", mode="lines"))
        fig1.add_trace(go.Scatter(x=group_ret.index, y=group_ret["long_avg"].cumsum(), name="Long-Average", mode="lines"))
        fig1.update_layout(title="Qlib-style Group Return", height=420)
        figs.append(fig1)
    if not auto_corr.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=auto_corr.index, y=auto_corr.values, name="Pred AutoCorr", mode="lines+markers"))
        fig2.update_layout(title="Qlib-style Prediction Auto Correlation", height=420)
        figs.append(fig2)
    return figs


def _extract_portfolio_metrics(port_analysis_df: pd.DataFrame, report_df: pd.DataFrame) -> dict:
    metrics = {
        "annualized_return": float("nan"),
        "max_drawdown": float("nan"),
        "monthly_win_rate": float("nan"),
        "sharpe_ratio": float("nan"),
        "information_ratio": float("nan"),
        "excess_return_with_cost_mean": float("nan"),
        "excess_return_without_cost_mean": float("nan"),
    }

    if isinstance(port_analysis_df, pd.DataFrame) and not port_analysis_df.empty:
        if isinstance(port_analysis_df.index, pd.MultiIndex):
            for grp, prefix in [("excess_return_with_cost", "with_cost"), ("excess_return_without_cost", "without_cost")]:
                if grp in port_analysis_df.index.get_level_values(0):
                    sub = port_analysis_df.loc[grp]
                    metrics[f"excess_return_{prefix}_mean"] = float(sub.loc["mean", "risk"]) if "mean" in sub.index else float("nan")
                    if prefix == "with_cost":
                        metrics["annualized_return"] = float(sub.loc["annualized_return", "risk"]) if "annualized_return" in sub.index else float("nan")
                        metrics["information_ratio"] = float(sub.loc["information_ratio", "risk"]) if "information_ratio" in sub.index else float("nan")
                        metrics["max_drawdown"] = float(sub.loc["max_drawdown", "risk"]) if "max_drawdown" in sub.index else float("nan")
                        daily_std = float(sub.loc["std", "risk"]) if "std" in sub.index else float("nan")
                        daily_mean = float(sub.loc["mean", "risk"]) if "mean" in sub.index else float("nan")
                        metrics["sharpe_ratio"] = daily_mean / daily_std * math.sqrt(252) if pd.notna(daily_std) and daily_std > 0 and pd.notna(daily_mean) else float("nan")
        elif "risk" in port_analysis_df.columns:
            # 兼容非 MultiIndex 的极简结构
            for col in ["annualized_return", "max_drawdown", "information_ratio"]:
                if col in port_analysis_df.index:
                    metrics[col] = float(port_analysis_df.loc[col, "risk"])

    if isinstance(report_df, pd.DataFrame) and not report_df.empty and "return" in report_df.columns:
        rpt = _normalize_index_datetime(report_df)
        ret = pd.to_numeric(rpt["return"], errors="coerce").dropna()
        if not ret.empty:
            monthly = (1 + ret).resample("ME").prod() - 1
            metrics["monthly_win_rate"] = float((monthly > 0).mean()) if not monthly.empty else float("nan")

    return metrics


def _build_overview_html(out_path: Path, title: str, summary_metrics: dict, top_df: pd.DataFrame, figs: list, ic_figs: list, model_figs: list):
    def fmt(v):
        if v is None:
            return "N/A"
        try:
            if pd.isna(v):
                return "N/A"
        except Exception:
            pass
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    metrics_rows = "".join(
        f"<tr><th>{k}</th><td>{fmt(v)}</td></tr>" for k, v in summary_metrics.items()
    )
    top_table = top_df.head(20).to_html(index=False, escape=False, classes="table table-sm table-striped")

    fig_html_parts = []
    for i, fig in enumerate([*figs, *ic_figs, *model_figs], start=1):
        fig_html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 1 else False))

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2 {{ margin: 0.2em 0 0.6em; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; align-items: start; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); background: #fff; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e5e7eb; }}
    th {{ background: #f9fafb; width: 240px; }}
    .muted {{ color: #6b7280; font-size: 0.95em; }}
    .section {{ margin-top: 20px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="muted">基于 Alpha158 + LightGBM 的 stage2 结果，包含预测得分、回测统计与图形化分析。</div>

  <div class="grid section">
    <div class="card">
      <h2>核心指标</h2>
      <table>{metrics_rows}</table>
    </div>
    <div class="card">
      <h2>Top 20 预测得分</h2>
      {top_table}
    </div>
  </div>

  <div class="section card">
    <h2>图表分析</h2>
    {''.join(f'<div style="margin-bottom:24px;">{part}</div>' for part in fig_html_parts)}
  </div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


# ──────────────────────────────────────────────────
# 主逻辑
# ──────────────────────────────────────────────────

def export_predict(run_dir: str, output_dir: str, pred_date: str):
    run_path   = Path(run_dir)
    out_path   = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    artifacts  = run_path / "artifacts"

    # ── 1. 预测得分 (pred.pkl) ───────────────────
    pred_pkl   = artifacts / "pred.pkl"
    if not pred_pkl.exists():
        raise FileNotFoundError(f"pred.pkl not found: {pred_pkl}")

    label_pkl = artifacts / "label.pkl"
    if not label_pkl.exists():
        raise FileNotFoundError(f"label.pkl not found: {label_pkl}")

    pred_df = _safe_load_df(pred_pkl)
    label_df = _safe_load_df(label_pkl)
    print(f"pred.pkl loaded, shape={pred_df.shape}, index levels={pred_df.index.nlevels}")

    # pred_df 通常是 MultiIndex (datetime, instrument) → score 列
    if pred_df.index.nlevels == 2:
        pred_df.index.names = ["datetime", "instrument"]
        pred_df.columns      = ["score"]
    elif pred_df.index.nlevels == 1:
        # 尝试从列推断
        if "instrument" in pred_df.columns:
            pred_df = pred_df.set_index("instrument")
        pred_df.columns = pred_df.columns[:1].tolist()
        pred_df.columns = ["score"]

    # 取 pred_date 截面（如果没有精确匹配则取最后一个可用交易日）
    selected_pred_date = pred_date
    if pred_df.index.nlevels == 2:
        dates_available = pred_df.index.get_level_values("datetime").unique()
        dates_str = dates_available.astype(str)
        if pred_date in dates_str:
            slice_df = pred_df.xs(pred_date, level="datetime")
        else:
            # 取最近日期
            latest = dates_available.max()
            print(f"⚠ pred_date={pred_date} 在数据中不存在，使用最近日期: {latest}")
            selected_pred_date = pd.Timestamp(latest).strftime("%Y-%m-%d")
            slice_df = pred_df.xs(latest, level="datetime")
    else:
        slice_df = pred_df

    slice_df = slice_df.reset_index()
    if "instrument" not in slice_df.columns and slice_df.columns[0] != "score":
        slice_df.columns = ["instrument", "score"]
    elif "instrument" not in slice_df.columns:
        slice_df.index.name = "instrument"
        slice_df = slice_df.reset_index()

    # 全截面排名（仅输出实盘预测可用字段，不附带未来收益类信息）
    slice_df = slice_df.sort_values("score", ascending=False).reset_index(drop=True)
    n = len(slice_df)
    slice_df["rank"]       = range(1, n + 1)
    slice_df["rank_pct"]   = (slice_df["rank"] / n * 100).round(2).astype(str) + "%"
    slice_df["pred_date"]  = selected_pred_date
    slice_df["score_quantile"] = (1.0 - (slice_df["rank"] - 1) / max(n, 1)).round(6)
    slice_df["percentile"] = slice_df["score_quantile"]
    slice_df["quantile_bucket"] = np.select(
        [
            slice_df["rank"] <= max(int(np.ceil(n * 0.01)), 1),
            slice_df["rank"] <= max(int(np.ceil(n * 0.05)), 1),
            slice_df["rank"] <= max(int(np.ceil(n * 0.10)), 1),
        ],
        ["top_1pct", "top_5pct", "top_10pct"],
        default="others",
    )
    slice_df["code"] = slice_df["instrument"].astype(str).str.replace(r'^[A-Za-z]+', '', regex=True).str.zfill(6)
    slice_df["stock"] = slice_df["code"]
    slice_df["date"] = slice_df["pred_date"]

    export_scores_df = slice_df[[
        "stock",
        "code",
        "instrument",
        "date",
        "pred_date",
        "score",
        "rank",
        "percentile",
        "rank_pct",
        "score_quantile",
        "quantile_bucket",
    ]].copy()

    # ── 2. 绩效指标 ─────────────────────────────
    portfolio_dir = artifacts / "portfolio_analysis"
    port_analysis_pkl = portfolio_dir / "port_analysis_1day.pkl"
    report_normal_pkl = portfolio_dir / "report_normal_1day.pkl"
    indicator_analysis_pkl = portfolio_dir / "indicator_analysis_1day.pkl"

    port_analysis_df = _safe_load_df(port_analysis_pkl) if port_analysis_pkl.exists() else pd.DataFrame()
    report_normal_df = _safe_load_df(report_normal_pkl) if report_normal_pkl.exists() else pd.DataFrame()
    indicator_df = _safe_load_df(indicator_analysis_pkl) if indicator_analysis_pkl.exists() else pd.DataFrame()

    metrics = _compute_ic_metrics(pred_df, label_df)
    portfolio_metrics = _extract_portfolio_metrics(port_analysis_df, report_normal_df)
    metrics.update(portfolio_metrics)

    # 进一步补充：如果缺失，则尝试从 report_normal 计算
    if pd.isna(metrics.get("sharpe_ratio", float("nan"))) and isinstance(report_normal_df, pd.DataFrame) and not report_normal_df.empty:
        rpt = _normalize_index_datetime(report_normal_df)
        if "return" in rpt.columns:
            ret_series = pd.to_numeric(rpt["return"], errors="coerce").dropna()
            daily_std = ret_series.std()
            if len(ret_series) > 0 and daily_std > 0:
                metrics["sharpe_ratio"] = float(ret_series.mean() / daily_std * np.sqrt(252))

    # 额外指标：命中率/胜率，若 indicator 文件存在则写入
    if isinstance(indicator_df, pd.DataFrame) and not indicator_df.empty:
        for col in ["ffr", "pa", "pos"]:
            if col in indicator_df.index:
                metrics[col] = float(indicator_df.loc[col].iloc[0])

    # 额外统计：用于判断信号是否过于集中
    if len(slice_df) > 1:
        top_k = min(10, len(slice_df))
        bot_k = min(10, len(slice_df))
        metrics["score_mean"] = float(pd.to_numeric(slice_df["score"], errors="coerce").mean())
        metrics["score_std"] = float(pd.to_numeric(slice_df["score"], errors="coerce").std())
        metrics["score_spread_top_bottom"] = float(
            slice_df["score"].head(top_k).mean() - slice_df["score"].tail(bot_k).mean()
        )

    # ── 4. 写出 CSV ───────────────────────────
    scores_csv  = out_path / "scores.csv"
    metrics_csv = out_path / "metrics.csv"
    overview_html = out_path / "overview.html"

    export_scores_df.to_csv(scores_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    # ④ Export ALL test-period prediction dates for full-cycle backtest aggregation.
    # all_scores.csv has every (datetime, instrument, score) in pred.pkl —
    # used by _load_fold_scores in run_stage2_walk_forward.py.  scores.csv
    # (single-date snapshot) is kept unchanged for stage3/4/5/6 downstream.
    if pred_df is not None and not pred_df.empty and pred_df.index.nlevels == 2:
        all_df = pred_df.reset_index().copy()
        all_df.columns = [pred_df.index.names[0], pred_df.index.names[1], "score"][:len(all_df.columns)]
        if len(all_df.columns) == 3:
            all_df.columns = ["datetime", "instrument", "score"]
        all_df["datetime"] = pd.to_datetime(all_df["datetime"]).dt.strftime("%Y-%m-%d")
        all_df["score"] = pd.to_numeric(all_df["score"], errors="coerce")
        all_df = all_df.dropna(subset=["score"])
        all_df.to_csv(out_path / "all_scores.csv", index=False, encoding="utf-8-sig")
        print(f"✓ 全周期得分已保存: {out_path / 'all_scores.csv'}  ({len(all_df)} 行, {all_df['datetime'].nunique()} 日期)")

    # ── 5. 生成图形化分析 HTML ─────────────────
    try:
        report_figs = []
        ic_figs = []
        model_figs = []
        analysis_position_mod, analysis_model_mod = _load_qlib_report_modules()
        if analysis_position_mod is not None:
            if isinstance(report_normal_df, pd.DataFrame) and not report_normal_df.empty:
                rpt = _normalize_index_datetime(report_normal_df)
                report_figs = list(analysis_position_mod.report_graph(rpt, show_notebook=False))

            pred_label = pred_df.join(label_df.rename(columns={label_df.columns[0]: "label"}), how="inner")
            if not pred_label.empty:
                pred_label = pred_label.dropna()
                if not pred_label.empty:
                    ic_figs = list(analysis_position_mod.score_ic_graph(pred_label, show_notebook=False))
                    if analysis_model_mod is not None:
                        model_figs = list(
                            analysis_model_mod.model_performance_graph(
                                pred_label,
                                show_notebook=False,
                                graph_names=["group_return", "pred_ic", "pred_autocorr"],
                            )
                        )
        else:
            print(f"⚠ analysis_position 不可用，跳过图表生成: {_ANALYSIS_IMPORT_ERROR}")

        if not report_figs:
            report_figs = _fallback_report_figures(report_normal_df)
        if not ic_figs:
            pred_label = pred_df.join(label_df.rename(columns={label_df.columns[0]: "label"}), how="inner")
            ic_figs = _fallback_ic_figures(pred_label)
        if not model_figs:
            pred_label = pred_df.join(label_df.rename(columns={label_df.columns[0]: "label"}), how="inner")
            model_figs = _fallback_model_figures(pred_label)

        _build_overview_html(
            overview_html,
            title="Stage2 Model Predict Overview",
            summary_metrics={k: metrics.get(k, float("nan")) for k in [
                "IC_mean",
                "IC_std",
                "ICIR",
                "rank_IC_mean",
                "annualized_return",
                "max_drawdown",
                "monthly_win_rate",
                "sharpe_ratio",
                "information_ratio",
            ]},
            top_df=export_scores_df,
            figs=report_figs,
            ic_figs=ic_figs,
            model_figs=model_figs,
        )
        print(f"✓ 详细图表已保存: {overview_html}")
    except Exception as exc:
        print(f"⚠ overview.html 生成失败: {exc}")

    print(f"✓ 得分表已保存: {scores_csv}  ({len(slice_df)} 行)")
    print(f"✓ 指标表已保存: {metrics_csv}")
    print("\n═══ 绩效摘要 ═══")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else f"  {k:25s}: N/A")
    print(f"\n◆ Top10 预测得分:")
    cols = [c for c in ["instrument", "score", "rank", "rank_pct", "quantile_bucket"] if c in export_scores_df.columns]
    print(export_scores_df[cols].head(10).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",   required=True, dest="run_dir")
    ap.add_argument("--output",    required=True)
    ap.add_argument("--pred-date", required=True, dest="pred_date")
    args = ap.parse_args()
    export_predict(args.run_dir, args.output, args.pred_date)


if __name__ == "__main__":
    main()
