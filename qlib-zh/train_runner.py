"""
Qlib 训练执行器 — 在 Docker 中运行 stage2 walk-forward 全量训练（含回测）。

用法:
    from train_runner import run_training
    result = run_training(market="csi300", progress_callback=cb)
    print(result["model_name"])
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

# ---- paths (同 infer_runner.py) ----
_HOST_PROJECT_ROOT = Path(os.environ.get("QLIB_HOST_PROJECT_ROOT", Path(__file__).resolve().parent))
_HOST_QLIB_DATA = Path(os.environ.get("QLIB_HOST_DATA_DIR", Path.home() / ".qlib"))
_HOST_MLRUNS = Path(os.environ.get("QLIB_HOST_MLRUNS_DIR", Path.home() / "github" / "qlib-zh" / "mlruns"))

_LOCAL_PROJECT_ROOT = Path(__file__).resolve().parent

DOCKER_IMAGE = os.environ.get("DOCKER_IMAGE", "zhuhai123/qlib-rdagent:v1")
WORKDIR = "/work"

# 训练超时 2 小时
TRAIN_TIMEOUT = 7200


def _resolve_config(market: str) -> dict:
    """根据市场名称解析脚本 / YAML 模板."""
    if market == "csi1000":
        return {
            "market": "csi1000",
            "benchmark": "SH000852",
            "script": "scripts/small/run_stage2_walk_forward_small.py",
            "template": (
                f"{WORKDIR}/scripts/small/templates/"
                "workflow_config_lightgbm_Alpha158_csi1000.yaml"
            ),
            "qlib_data_dir": "/root/.qlib/qlib_data/cn_data",
            "use_models_dir": True,
        }
    else:
        # 默认 CSI300
        return {
            "market": "csi300",
            "benchmark": "SH000300",
            "script": "scripts/practice/run_stage2_walk_forward.py",
            "template": (
                f"{WORKDIR}/examples/benchmarks/LightGBM/"
                "workflow_config_lightgbm_Alpha158.yaml"
            ),
            "qlib_data_dir": "/root/.qlib/qlib_data/cn_data",
            "use_models_dir": False,
        }


def _get_last_trade_date() -> str:
    """从 qlib 日历获取最近交易日."""
    calendar_file = Path.home() / ".qlib" / "qlib_data" / "cn_data" / "calendars" / "day.txt"
    if calendar_file.exists():
        dates = calendar_file.read_text().strip().splitlines()
        if dates:
            return dates[-1].strip()
    # fallback: 今天
    return datetime.now().strftime("%Y-%m-%d")


def _ensure_mlruns_dir():
    """确保 mlruns 目录存在."""
    _HOST_MLRUNS.mkdir(parents=True, exist_ok=True)


def _parse_backtest_metrics(analysis_root: Path, model_name: str) -> dict:
    """从训练输出中提取回测指标."""
    metrics = {}

    # 尝试读取 report_of_backtest.txt
    report_txt = analysis_root / "model_predict" / "walk_forward" / "full_backtest" / "report_of_backtest.txt"
    if not report_txt.exists():
        # 尝试其他路径
        alt_paths = [
            analysis_root / "model_predict" / "report_of_backtest.txt",
            analysis_root / "full_backtest" / "report_of_backtest.txt",
        ]
        for p in alt_paths:
            if p.exists():
                report_txt = p
                break

    if report_txt.exists():
        content = report_txt.read_text(encoding="utf-8")
        patterns = {
            "annualized_return": r"annualized_return[:\s]+([-.\d]+)",
            "max_drawdown": r"max_drawdown[:\s]+([-.\d]+)",
            "sharpe_ratio": r"sharpe_ratio[:\s]+([-.\d]+)",
            "information_ratio": r"information_ratio[:\s]+([-.\d]+)",
            "ic_mean": r"ic_mean[:\s]+([-.\d]+)",
            "icir": r"icir[:\s]+([-.\d]+)",
            "annualized_excess_return": r"annualized_excess_return[:\s]+([-.\d]+)",
            "turnover_mean": r"turnover_mean[:\s]+([-.\d]+)",
            "monthly_win_rate": r"monthly_win_rate[:\s]+([-.\d]+)",
        }
        for key, pattern in patterns.items():
            m = re.search(pattern, content, re.IGNORECASE)
            if m:
                try:
                    metrics[key] = round(float(m.group(1)), 4)
                except ValueError:
                    pass
        metrics["raw_report"] = content[:2000]  # 截取前 2000 字符
    else:
        # 尝试从 metrics.csv 读取
        metrics_csv = analysis_root / "model_predict" / "metrics.csv"
        if metrics_csv.exists():
            with open(metrics_csv, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for k, v in row.items():
                        try:
                            metrics[k] = round(float(v), 4)
                        except (ValueError, TypeError):
                            metrics[k] = v

    # 检查 scores.csv 是否存在（训练成功标志）
    scores_csv = analysis_root / "model_predict" / "scores.csv"
    metrics["has_scores"] = scores_csv.exists()
    if scores_csv.exists():
        with open(scores_csv, newline="", encoding="utf-8-sig") as f:
            metrics["score_count"] = sum(1 for _ in csv.DictReader(f))

    return metrics


def _copy_to_models_dir(model_name: str, analysis_root: Path):
    """将训练好的模型复制到 models/ 目录，使其对推理端点可见."""
    models_dir = _LOCAL_PROJECT_ROOT / "models"
    target_dir = models_dir / model_name

    # 复制 model_predict 子目录（关键是 scores.csv 和模型检查点）
    src_predict = analysis_root / "model_predict"
    if src_predict.exists():
        target_predict = target_dir / "model_predict"
        target_predict.mkdir(parents=True, exist_ok=True)
        for item in src_predict.iterdir():
            if item.is_dir():
                shutil.copytree(item, target_predict / item.name, dirs_exist_ok=True)
            elif item.suffix in (".csv", ".pkl", ".json", ".html"):
                shutil.copy2(item, target_predict / item.name)
        return True
    return False


def run_training(
    market: str = "csi300",
    model_mode: str = "robust",
    hold_num: int = 20,
    lightgbm_only: bool = True,
    progress_callback: Callable | None = None,
) -> dict:
    """
    在 Docker 中运行完整的 qlib walk-forward 训练（含回测）。

    Args:
        market: "csi300" 或 "csi1000"
        model_mode: "default" 或 "robust"
        hold_num: 持仓股票数量
        lightgbm_only: 仅训练 LightGBM（跳过 XGBoost）
        progress_callback: 可选回调，接收 dict(message, progress, ...)

    Returns:
        {"success": True/False, "model_name": "...", "message": "...",
         "backtest_metrics": {...}}
    """

    def _log(msg: str, **extra):
        if progress_callback:
            progress_callback({"message": msg, **extra})

    cfg = _resolve_config(market)
    last_trade_date = _get_last_trade_date()
    model_name = f"{last_trade_date}-{market}-alpha158"

    # 确保 mlruns 目录存在
    _ensure_mlruns_dir()

    # 检查 qlib 数据是否就绪
    qlib_data = Path.home() / ".qlib" / "qlib_data" / "cn_data"
    calendar_file = qlib_data / "calendars" / "day.txt"
    features_dir = qlib_data / "features"
    if not calendar_file.exists() and not features_dir.exists():
        _log("❌ qlib 数据未下载！请先在网页勾选「获取最新qlib数据」→ 点击「执行」下载数据", status="failed")
        return {
            "success": False,
            "model_name": model_name,
            "message": "qlib 数据未下载，请先使用「获取最新qlib数据」功能下载数据",
            "backtest_metrics": {},
        }

    _log(f"市场: {cfg['market']}")
    _log(f"基准: {cfg['benchmark']}")
    _log(f"模型名称: {model_name}")
    _log(f"模型模式: {model_mode}")
    _log(f"持仓数量: {hold_num}")
    _log(f"LightGBM only: {lightgbm_only}")
    _log(f"镜像: {DOCKER_IMAGE}")

    base_dir = f"{WORKDIR}/models" if cfg.get("use_models_dir") else f"{WORKDIR}/DATA/analysis_outputs"
    output_root_ctr = f"{base_dir}/{model_name}/model_predict"
    analysis_root_ctr = f"{base_dir}/{model_name}"

    cmd = [
        "docker", "run", "--rm",
        "-e", f"QLIB_DATA_DIR={cfg['qlib_data_dir']}",
        "-e", f"TARGET_MARKET={cfg['market']}",
        "-e", f"TARGET_BENCHMARK={cfg['benchmark']}",
        "-e", "HOLD_NUM={}".format(hold_num),
        "-e", "CASH_TOTAL=100000",
        "-e", "TX_FEE_RATE=0.0001",
        "-e", "STAMP_DUTY_RATE=0.0005",
    ]

    if lightgbm_only:
        cmd += ["-e", "STAGE2_LIGHTGBM_ONLY=1"]

    cmd += [
        "-v", f"{_HOST_PROJECT_ROOT}:{WORKDIR}",
        "-v", f"{_HOST_QLIB_DATA}:/root/.qlib",
        "-v", f"{_HOST_MLRUNS}:{WORKDIR}/mlruns",
        "-w", WORKDIR,
        DOCKER_IMAGE,
        "python3", cfg["script"],
        "--template", cfg["template"],
        "--output-root", output_root_ctr,
        "--analysis-root", analysis_root_ctr,
        "--experiment-name", model_name,
        "--uri-folder", "mlruns",
        "--walk-forward-end", last_trade_date,
        "--model-mode", model_mode,
        "--hold-num", str(hold_num),
    ]

    _log(f"启动 Docker 训练 (超时 {TRAIN_TIMEOUT}s)...")
    _log(f"脚本: {cfg['script']}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        import time as _time
        line_buffer: list[str] = []

        for line in process.stdout:
            line = line.rstrip()
            if line:
                line_buffer.append(line)
                # 每行都推送到日志（截断过长行）
                _log(f"[Docker] {line[:200]}")

        process.wait(timeout=TRAIN_TIMEOUT)

        if process.returncode != 0:
            # 输出最后 30 行用于诊断
            for err_line in line_buffer[-30:]:
                _log(f"[Docker ERR] {err_line[:300]}")
            raise RuntimeError(
                f"Docker 退出码: {process.returncode}，"
                f"请检查上方 Docker 日志定位具体错误"
            )

        _log("Docker 训练完成，正在提取结果...")

    except subprocess.TimeoutExpired:
        process.kill()
        raise RuntimeError(f"训练超时 ({TRAIN_TIMEOUT}s)")

    # 提取结果
    _base_local = _LOCAL_PROJECT_ROOT / "models" if cfg.get("use_models_dir") else _LOCAL_PROJECT_ROOT / "DATA" / "analysis_outputs"
    analysis_root = _base_local / model_name

    backtest_metrics = _parse_backtest_metrics(analysis_root, model_name)

    # 复制到 models/ 目录（使推理端点可见）
    if not cfg.get("use_models_dir"):
        copied = _copy_to_models_dir(model_name, analysis_root)
        if copied:
            _log(f"模型已复制到: models/{model_name}")
        else:
            _log("警告: 模型复制到 models/ 失败（但训练结果仍在 DATA/analysis_outputs/ 中）")

    # 构建结果消息
    msg_parts = [f"训练完成: {model_name}"]
    if backtest_metrics.get("sharpe_ratio") is not None:
        msg_parts.append(f"夏普比: {backtest_metrics['sharpe_ratio']}")
    if backtest_metrics.get("annualized_return") is not None:
        msg_parts.append(f"年化收益: {backtest_metrics['annualized_return']*100:.2f}%")
    if backtest_metrics.get("max_drawdown") is not None:
        msg_parts.append(f"最大回撤: {backtest_metrics['max_drawdown']*100:.2f}%")
    if backtest_metrics.get("ic_mean") is not None:
        msg_parts.append(f"IC均值: {backtest_metrics['ic_mean']}")
    msg = " | ".join(msg_parts)

    _log(msg, progress=1.0, status="completed", model_name=model_name)

    return {
        "success": True,
        "model_name": model_name,
        "message": msg,
        "backtest_metrics": backtest_metrics,
    }


def _print_progress(data):
    """CLI 进度回调."""
    msg = data.get("message", "")
    status = data.get("status", "")
    pct = data.get("progress", 0)
    if pct:
        print(f"[{pct*100:.0f}%] {msg}", file=sys.stderr)
    else:
        print(f"[train] {msg}", file=sys.stderr)


# ---- CLI 入口 ----
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Qlib 训练执行器")
    ap.add_argument("--market", default="csi300", choices=["csi300", "csi1000"])
    ap.add_argument("--model-mode", default="robust", choices=["default", "robust"])
    ap.add_argument("--hold-num", type=int, default=20)
    ap.add_argument("--lightgbm-only", action="store_true", default=True)
    args = ap.parse_args()

    result = run_training(
        market=args.market,
        model_mode=args.model_mode,
        hold_num=args.hold_num,
        lightgbm_only=args.lightgbm_only,
        progress_callback=_print_progress,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
