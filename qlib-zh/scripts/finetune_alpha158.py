"""
Qlib 模型微调脚本 — 在已有模型基础上，使用近 6 年数据进行单折微调。

运行方式（Docker 内部）:
    python3 scripts/finetune_alpha158.py \\
        --base-model-dir /work/DATA/analysis_outputs/2026-06-07-csi300-alpha158 \\
        --output-name 2026-06-07-csi300-alpha158-fintune \\
        --template /work/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml

输出:
    {output_root}/
        model_predict/
            scores.csv, metrics.csv, all_scores.csv, pred.pkl
            report_of_backtest.txt
            walk_forward/{valid_start}/
                model_runs/lightgbm/
                    workflow_config_practice.yaml
                    mlflow_run/artifacts/params.pkl, pred.pkl, label.pkl, ...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# qlib
import qlib
from qlib.config import REG_CN
from qlib.contrib.evaluate import backtest_daily, risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.data import D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.practice.gen_practice_yaml import patch_yaml


def _find_latest_checkpoint(base_model_dir: Path) -> str | None:
    """在基础模型目录中找到最新 fold 的 params.pkl 路径."""
    walk_dir = base_model_dir / "model_predict" / "walk_forward"
    if not walk_dir.exists():
        return None

    fold_dirs = sorted(
        [d for d in walk_dir.iterdir() if d.is_dir() and d.name[0].isdigit()],
        reverse=True,
    )
    for fold_dir in fold_dirs:
        for sub in ("lightgbm", "xgboost"):
            pkl = fold_dir / "model_runs" / sub / "mlflow_run" / "artifacts" / "params.pkl"
            if pkl.exists():
                return str(pkl)
    return None


def _get_last_trade_date() -> str:
    """从 qlib 日历获取最近交易日."""
    try:
        calendar = D.calendar()
        if len(calendar) > 0:
            return pd.Timestamp(calendar[-1]).strftime("%Y-%m-%d")
    except Exception:
        pass
    return datetime.now().strftime("%Y-%m-%d")


def _get_prev_trade_date(date_str: str) -> str:
    """获取指定日期的前一个交易日."""
    dt = pd.Timestamp(date_str)
    try:
        calendar = D.calendar()
        cal = [pd.Timestamp(c) for c in calendar]
        past = [c for c in cal if c < dt]
        if past:
            return past[-1].strftime("%Y-%m-%d")
    except Exception:
        pass
    return (dt - timedelta(days=1)).strftime("%Y-%m-%d")


def _get_trade_date_offset(date_str: str, offset_years: int) -> str:
    """获取指定日期 N 年前后的第一个交易日."""
    dt = pd.Timestamp(date_str)
    target = dt - pd.DateOffset(years=offset_years)
    try:
        calendar = D.calendar()
        cal = [pd.Timestamp(c) for c in calendar]
        future = [c for c in cal if c >= target]
        if future:
            return future[0].strftime("%Y-%m-%d")
    except Exception:
        pass
    return target.strftime("%Y-%m-%d")


def _export_scores(artifacts_dir: Path, output_dir: Path, pred_date: str):
    """从 MLflow artifacts 导出 scores.csv 和 metrics.csv."""
    # 尝试多个可能的预测文件
    candidates = [
        artifacts_dir / "test_pred_snapshot.pkl",
        artifacts_dir / "pred.pkl",
        artifacts_dir / "valid_pred_snapshot.pkl",
        artifacts_dir / "valid_pred.pkl",
    ]
    pred_path = None
    for p in candidates:
        if p.exists():
            pred_path = p
            break

    if pred_path is None:
        print(f"[finetune] 未找到预测文件，检查: {[str(p) for p in candidates]}")
        return {}

    import pickle
    with open(pred_path, "rb") as f:
        pred_data = pickle.load(f)

    if isinstance(pred_data, pd.Series):
        scores_df = pred_data.to_frame("score").reset_index()
    elif isinstance(pred_data, pd.DataFrame):
        scores_df = pred_data.reset_index()
    else:
        print(f"[finetune] 未知预测数据类型: {type(pred_data)}")
        return {}

    # 识别 instrument 列和 score 列
    score_col = None
    inst_col = None
    for col in scores_df.columns:
        if "score" in col.lower():
            score_col = col
        sample = str(scores_df[col].iloc[0]) if len(scores_df) > 0 else ""
        if sample.startswith(("SZ", "SH", "BJ")) or (sample.isdigit() and len(sample) == 6):
            inst_col = col

    if score_col is None:
        score_col = scores_df.columns[-1]
    if inst_col is not None:
        scores_df["instrument"] = scores_df[inst_col]
    elif "instrument" not in scores_df.columns:
        scores_df["instrument"] = scores_df.index.astype(str)

    result_df = pd.DataFrame()
    result_df["instrument"] = scores_df["instrument"]
    result_df["code"] = scores_df["instrument"].apply(
        lambda x: x[2:] if isinstance(x, str) and x[:2] in ("SZ", "SH", "BJ") else x
    )
    result_df["date"] = pred_date
    result_df["pred_date"] = pred_date
    result_df["score"] = scores_df[score_col]
    result_df = result_df.sort_values("score", ascending=False)
    result_df["rank"] = range(1, len(result_df) + 1)
    n = len(result_df)
    result_df["percentile"] = result_df["rank"].apply(lambda r: (n - r) / n * 100)
    result_df["rank_pct"] = result_df["rank"] / n
    try:
        result_df["score_quantile"] = pd.qcut(result_df["score"], q=10, labels=False, duplicates="drop") + 1
    except Exception:
        result_df["score_quantile"] = 1
    result_df["quantile_bucket"] = result_df["score_quantile"].apply(lambda x: f"q{x}")
    result_df["score_final"] = result_df["score"]

    output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_dir / "scores.csv", index=False, encoding="utf-8-sig")
    print(f"[finetune] scores.csv → {len(result_df)} 行")

    # all_scores.csv
    result_df["datetime"] = pd.Timestamp(pred_date)
    all_df = result_df[["datetime", "instrument", "score"]].copy()
    all_df.to_csv(output_dir / "all_scores.csv", index=False, encoding="utf-8-sig")

    # metrics.csv
    metrics = {
        "pred_date": pred_date,
        "total_stocks": len(result_df),
        "score_mean": round(float(result_df["score"].mean()), 6),
        "score_std": round(float(result_df["score"].std()), 6),
        "score_min": round(float(result_df["score"].min()), 6),
        "score_max": round(float(result_df["score"].max()), 6),
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    # copy pred.pkl
    shutil.copy2(pred_path, output_dir / "pred.pkl")

    print(f"[finetune] score: mean={metrics['score_mean']}, max={metrics['score_max']}")
    return metrics


def _run_backtest(scores_csv: Path, output_dir: Path, pred_date: str,
                  benchmark: str = "SH000300", topk: int = 20) -> dict:
    """运行简化回测."""
    print(f"[finetune] 运行回测 benchmark={benchmark} topk={topk}...")
    try:
        pred_df = pd.read_csv(scores_csv)
        if "instrument" not in pred_df.columns:
            print("[finetune] scores.csv 缺少 instrument 列")
            return {}

        pred_df["datetime"] = pd.to_datetime(pred_df["date"])
        score_series = pred_df.set_index(["datetime", "instrument"])["score"]

        strategy = TopkDropoutStrategy(signal=score_series, topk=topk, n_drop=5)
        pred_dt = pd.Timestamp(pred_date)
        cfg = {
            "start_time": str(pred_dt - timedelta(days=30)),
            "end_time": pred_date,
            "account": 100000,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "limit_threshold": 0.095, "deal_price": "close",
                "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5,
            },
        }

        portfolio_metric_dict, indicator_dict = backtest_daily(strategy=strategy, **cfg)
        analysis = risk_analysis(portfolio_metric_dict["portfolio"])
        report = dict(indicator_dict)
        report.update({k: round(float(v), 6) for k, v in analysis.items()})

        lines = [
            "=" * 60,
            f"Finetune Backtest Report",
            f"Model: {output_dir.parent.name}",
            f"Pred Date: {pred_date}",
            f"Benchmark: {benchmark}",
            f"TopK: {topk}",
            "=" * 60,
        ]
        for k, v in report.items():
            lines.append(f"{k:>30s}: {v}")
        (output_dir / "report_of_backtest.txt").write_text("\n".join(lines), encoding="utf-8")

        print(f"[finetune] 夏普: {report.get('sharpe_ratio', 'N/A')}, "
              f"年化: {report.get('annualized_return', 'N/A')}")
        return report

    except Exception as e:
        print(f"[finetune] 回测失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


def main():
    ap = argparse.ArgumentParser(description="Qlib 模型微调")
    ap.add_argument("--base-model-dir", required=True)
    ap.add_argument("--output-name", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--experiment-name", default=None)
    ap.add_argument("--train-years", type=int, default=5)
    ap.add_argument("--valid-years", type=int, default=1)
    ap.add_argument("--hold-num", type=int, default=20)
    ap.add_argument("--model-mode", default="robust", choices=["default", "robust"])
    args = ap.parse_args()

    base_model_dir = Path(args.base_model_dir)
    output_root = Path(args.output_root)
    output_predict = output_root / "model_predict"
    output_predict.mkdir(parents=True, exist_ok=True)
    experiment_name = args.experiment_name or args.output_name

    print(f"[finetune] ===== Qlib 模型微调 =====")
    print(f"[finetune] 基础模型: {base_model_dir}")
    print(f"[finetune] 输出目录: {output_root}")

    # ---- Step 1: 初始化 qlib ----
    provider_uri = os.environ.get("QLIB_DATA_DIR", "/root/.qlib/qlib_data/cn_data")
    print(f"[finetune] 初始化 qlib, 数据: {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # ---- Step 2: 计算日期 ----
    last_trade = _get_last_trade_date()
    print(f"[finetune] 最近交易日: {last_trade}")

    valid_start = _get_trade_date_offset(last_trade, args.valid_years)
    valid_end = last_trade

    train_end = _get_prev_trade_date(valid_start)
    train_start = _get_trade_date_offset(train_end, args.train_years)

    print(f"[finetune] 训练: {train_start} ~ {train_end} ({args.train_years}年)")
    print(f"[finetune] 验证: {valid_start} ~ {valid_end} ({args.valid_years}年)")

    # ---- Step 3: 生成 YAML 配置（使用 patch_yaml 写出文件） ----
    config_path = output_predict / "walk_forward" / valid_start / "model_runs" / "lightgbm" / "workflow_config_practice.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    dates = {
        "train_start": train_start,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "test_start": valid_start,
        "test_end": valid_end,
    }

    os.environ["HOLD_NUM"] = str(args.hold_num)
    os.environ["CASH_TOTAL"] = os.environ.get("CASH_TOTAL", "100000")

    patch_yaml(
        template_path=str(Path(args.template).resolve()),
        output_path=str(config_path),
        dates=dates,
        model_mode=args.model_mode,
        practice_mode=True,  # finetune 模式：valid 和 test 可用同一窗口
    )
    print(f"[finetune] 配置已生成: {config_path}")

    # ---- Step 4: 查找检查点 ----
    checkpoint = _find_latest_checkpoint(base_model_dir)
    if checkpoint:
        print(f"[finetune] 找到检查点: {checkpoint}")
    else:
        print(f"[finetune] 警告: 未找到检查点，从头训练")

    # ---- Step 5: 执行训练（通过 run_stage2_practice.py 子进程） ----
    print(f"[finetune] 开始微调训练...")

    stage2_script = str(ROOT / "scripts" / "practice" / "run_stage2_practice.py")
    cmd = [
        sys.executable, stage2_script,
        "--config", str(config_path),
        "--experiment-name", experiment_name,
        "--uri-folder", "mlruns",
    ]
    if checkpoint:
        cmd += ["--warm-start", checkpoint]

    print(f"[finetune] 运行: {' '.join(cmd[-6:])}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.splitlines():
            if line.strip():
                print(f"  {line}")
    if result.returncode != 0:
        print(f"[finetune] 训练失败 (code={result.returncode})")
        if result.stderr:
            print(f"  stderr: {result.stderr[:1000]}")
        # 输出 JSON 失败结果
        print(json.dumps({"success": False, "error": f"训练失败, 退出码 {result.returncode}"}))
        return

    print(f"[finetune] 训练完成")

    # ---- Step 6: 查找 MLflow artifacts ----
    print(f"[finetune] 导出预测结果...")
    mlruns_dir = ROOT / "mlruns"
    run_dirs = sorted(mlruns_dir.rglob("*/artifacts"), reverse=True)
    artifacts_dir = None
    for ad in run_dirs:
        if (ad / "params.pkl").exists():
            artifacts_dir = ad
            break

    if not artifacts_dir and run_dirs:
        artifacts_dir = run_dirs[0]

    if artifacts_dir:
        metrics = _export_scores(artifacts_dir, output_predict, valid_end)
        # 复制 artifacts 到 fold 结构
        fold_artifacts = output_predict / "walk_forward" / valid_start / "model_runs" / "lightgbm" / "mlflow_run" / "artifacts"
        fold_artifacts.mkdir(parents=True, exist_ok=True)
        for f in artifacts_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, fold_artifacts / f.name)
        print(f"[finetune] 模型参数已复制到 fold 结构")
    else:
        print(f"[finetune] 未找到 MLflow artifacts")
        metrics = {}

    # ---- Step 6b: 创建 segments 目录（推理时 _find_latest_fold_checkpoint 需要） ----
    seg_key = f"{valid_start[:4]}-{valid_end[:4]}"
    seg_dir = output_predict / "walk_forward" / "segments" / seg_key
    seg_dir.mkdir(parents=True, exist_ok=True)
    seg_csv = seg_dir / "segment_folds.csv"
    import pandas as pd
    seg_df = pd.DataFrame([{
        "fold_tag": valid_start,
        "signal_date": valid_start,
        "train_start": train_start,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "test_start": valid_start,
        "test_end": valid_end,
    }])
    seg_df.to_csv(seg_csv, index=False, encoding="utf-8-sig")
    print(f"[finetune] segments 已创建: segments/{seg_key}/segment_folds.csv")

    # workflow_config_practice.yaml 已由 gen_practice_yaml.py 写入 fold 目录，无需复制

    # ---- Step 7: 回测 ----
    scores_csv = output_predict / "scores.csv"
    backtest_report = {}
    if scores_csv.exists():
        backtest_report = _run_backtest(
            scores_csv, output_predict, valid_end,
            benchmark=os.environ.get("TARGET_BENCHMARK", "SH000300"),
            topk=args.hold_num,
        )

    # ---- Step 8: 摘要 ----
    summary = {
        "finetune_completed": True,
        "base_model": str(base_model_dir),
        "output_name": args.output_name,
        "train_period": f"{train_start} ~ {train_end}",
        "valid_period": f"{valid_start} ~ {valid_end}",
        "pred_date": valid_end,
        "metrics": metrics,
        "backtest": backtest_report,
    }
    (output_root / "finetune_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print(f"[finetune] ===== 微调完成 =====")
    print(f"[finetune] 输出: {output_root}")
    print(f"[finetune] 模型: {args.output_name}")

    # stdout JSON（被 app.py 捕获用于获取回测指标）
    print(json.dumps({
        "success": True,
        "model_name": args.output_name,
        "output_dir": str(output_root),
        "backtest_metrics": backtest_report,
    }))


if __name__ == "__main__":
    main()
