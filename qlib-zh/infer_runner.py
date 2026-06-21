"""
Qlib 推理执行器 — 在 Docker 中运行 stage2 predict-only 推理，提取 Top N 股票。

用法:
    from qlib_zh.infer_runner import run_inference
    result = run_inference("2026-05-27-csi300-alpha158", top_n=20, progress_callback=cb)
    print(result["stocks"])  # "300394/601899/..."
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# ---- paths ----
# 宿主机路径（Docker-in-Docker 时通过环境变量覆盖）
# 当 StockFish 本身运行在 Docker 中时，内部 docker run -v 需要宿主机路径
_HOST_PROJECT_ROOT = Path(os.environ.get("QLIB_HOST_PROJECT_ROOT", Path(__file__).resolve().parent))
_HOST_QLIB_DATA = Path(os.environ.get("QLIB_HOST_DATA_DIR", Path.home() / ".qlib"))
_HOST_MLRUNS = Path(os.environ.get("QLIB_HOST_MLRUNS_DIR", Path.home() / "github" / "qlib-zh" / "mlruns"))

# 容器本地路径（用于读取文件等本地操作）
_LOCAL_PROJECT_ROOT = Path(__file__).resolve().parent  # 始终使用实际所在目录
_LOCAL_QLIB_DATA = Path.home() / ".qlib"

DOCKER_IMAGE = os.environ.get("DOCKER_IMAGE", "zhuhai123/qlib-rdagent:v1")
WORKDIR = "/work"


def _resolve_model(model_name: str) -> dict:
    """根据模型名称解析 market / 脚本 / YAML 模板."""
    name_lower = model_name.lower()

    if "csi1000" in name_lower:
        return {
            "market": "csi1000",
            "benchmark": "SH000852",
            "script": "scripts/small/run_stage2_walk_forward_small.py",
            "template": (
                f"{WORKDIR}/scripts/small/templates/"
                "workflow_config_lightgbm_Alpha158_csi1000.yaml"
            ),
            "qlib_data_dir": "/root/.qlib/qlib_data/cn_data",
            # CSI1000 walk_forward 在 models/ 目录下，不在 DATA/analysis_outputs/
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
        }


def _get_pred_date() -> str:
    """获取最近一个交易日作为预测日期（从 qlib 日历文件读取）."""
    calendar_file = _LOCAL_QLIB_DATA / "qlib_data" / "cn_data" / "calendars" / "day.txt"
    if calendar_file.exists():
        dates = calendar_file.read_text().strip().splitlines()
        if dates:
            return dates[-1].strip()
    # fallback: 最近周五
    today = datetime.now()
    friday = today - timedelta(days=(today.weekday() - 4) % 7)
    return friday.strftime("%Y-%m-%d")


def _instrument_to_code(instrument: str) -> str:
    """将 qlib instrument 格式转为纯数字代码. SZ300394 → 300394, SH601899 → 601899."""
    for prefix in ("SZ", "SH", "BJ"):
        if instrument.startswith(prefix):
            return instrument[len(prefix):]
    return instrument


def _code_to_instrument(code: str) -> str:
    """将纯数字代码转为 qlib instrument 格式. 600519 → SH600519, 000001 → SZ000001."""
    code = code.strip()
    if not code or len(code) != 6:
        return code
    if code.startswith(("60", "68")):
        return f"SH{code}"
    else:
        return f"SZ{code}"


def run_inference(
    model_name: str,
    top_n: int = 20,
    progress_callback=None,
    holdings: str = "",
) -> dict:
    """
    在 Docker 中运行 qlib predict_only 推理，返回 Top N 股票。

    Args:
        model_name: 模型名称 (如 "2026-05-27-csi300-alpha158")
        top_n: 取前 N 只股票
        progress_callback: 可选回调函数，接收 dict(status, message, ...)
        holdings: 当前持仓，/ 分隔的纯数字代码 (如 "600018/600066/600309")

    Returns:
        {"stocks": "300394/601899/...", "count": 20, "pred_date": "...",
         "scores": [{"code": "300394", "score": 0.96, "rank": 1}, ...]}
    """

    def _log(msg: str, **extra):
        if progress_callback:
            progress_callback({"message": msg, **extra})

    cfg = _resolve_model(model_name)
    pred_date = os.environ.get("PRED_DATE_OVERRIDE") or _get_pred_date()

    # 如果模型自带 predict_only YAML（如 finetune 模型），优先使用
    base_dir = f"{WORKDIR}/models" if cfg.get("use_models_dir") else f"{WORKDIR}/DATA/analysis_outputs"
    model_dir = _LOCAL_PROJECT_ROOT / "DATA" / "analysis_outputs" / model_name
    _find_predict_yaml = lambda root: next(
        (p for p in sorted(Path(root).rglob("workflow_config_practice.yaml"), reverse=True)
         if "predict_only" in str(p)),
        None,
    )
    local_template = _find_predict_yaml(model_dir)
    if local_template:
        rel = local_template.relative_to(_LOCAL_PROJECT_ROOT)
        cfg["template"] = f"{WORKDIR}/{rel}"
        _log(f"使用模型自带 YAML: {rel}")

    _log(f"模型: {model_name}")
    _log(f"市场: {cfg['market']} | 预测日期: {pred_date}")
    _log(f"镜像: {DOCKER_IMAGE}")

    # 构建 Docker 命令
    output_root_ctr = f"{base_dir}/{model_name}/model_predict"
    analysis_root_ctr = f"{base_dir}/{model_name}"

    cmd = [
        "docker", "run", "--rm",
        # env
        "-e", f"QLIB_DATA_DIR={cfg['qlib_data_dir']}",
        "-e", f"TARGET_MARKET={cfg['market']}",
        "-e", f"TARGET_BENCHMARK={cfg['benchmark']}",
        "-e", f"PRED_DATE={pred_date}",
        "-e", "STAGE2_LIGHTGBM_ONLY=1",
        "-e", "HOLD_NUM=20",
        "-e", "CASH_TOTAL=100000",
        "-e", "TX_FEE_RATE=0.0001",
        "-e", "STAMP_DUTY_RATE=0.0005",
        "-e", "PYTHONUNBUFFERED=1",
        # mounts
        "-v", f"{_HOST_PROJECT_ROOT}:{WORKDIR}",
        "-v", f"{_HOST_QLIB_DATA}:/root/.qlib",
        "-v", f"{_HOST_MLRUNS}:{WORKDIR}/mlruns",
        # workdir
        "-w", WORKDIR,
        # image
        DOCKER_IMAGE,
        # python script + args
        "python3", cfg["script"],
        "--predict-only",
        "--template", cfg["template"],
        "--output-root", output_root_ctr,
        "--analysis-root", analysis_root_ctr,
        "--experiment-name", model_name,
        "--uri-folder", "mlruns",
        "--walk-forward-end", pred_date,
        "--model-mode", "robust",
        "--hold-num", "20",
    ]

    _log(f"启动 Docker 推理...")
    _log(f"脚本: {cfg['script']}")

    try:
        # 使用 Popen 流式读取输出，便于实时推送进度
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        line_buffer: list[str] = []
        for line in process.stdout:
            line = line.rstrip()
            if line:
                line_buffer.append(line)
                _log(f"[Docker] {line[:200]}")

        process.wait(timeout=900)  # 15 分钟超时

        if process.returncode != 0:
            for err_line in line_buffer[-30:]:
                _log(f"[Docker ERR] {err_line[:300]}")
            raise RuntimeError(
                f"Docker 退出码: {process.returncode}，"
                f"请检查上方 Docker 日志定位具体错误"
            )

        _log("Docker 推理完成，读取结果...")

    except subprocess.TimeoutExpired:
        process.kill()
        raise RuntimeError("推理超时 (900s)")

    # 读取 scores.csv
    # 读取 scores.csv（路径需与 Docker --output-root 一致）
    _base_local = _LOCAL_PROJECT_ROOT / "models" if cfg.get("use_models_dir") else _LOCAL_PROJECT_ROOT / "DATA" / "analysis_outputs"
    scores_csv = _base_local / model_name / "model_predict" / "scores.csv"

    if not scores_csv.exists():
        # 尝试 predict_only 子目录
        alt_dir = _base_local / model_name / "model_predict" / "walk_forward"
        found_csvs = list(alt_dir.glob(f"predict_only_{pred_date.replace('-', '')}/**/scores.csv"))
        if found_csvs:
            scores_csv = found_csvs[0]

    if not scores_csv.exists():
        # 列出所有 csv 文件帮助调试
        out_dir = _base_local / model_name
        all_csvs = list(out_dir.glob("**/*.csv"))
        raise FileNotFoundError(
            f"未找到 scores.csv，尝试的路径:\n"
            f"  {scores_csv}\n"
            f"  {alt_dir if 'alt_dir' in dir() else 'N/A'}\n"
            f"输出目录下 CSV 文件: {[str(p.relative_to(out_dir)) for p in all_csvs[:20]]}"
        )

    _log(f"读取: {scores_csv}")

    # ── 读取全部股票（用于 dropout buffer） ──────────────────────
    all_rows = []
    with open(scores_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            instrument = row.get("instrument", "")
            code = _instrument_to_code(instrument)
            score = float(row.get("score", 0))
            rank = int(row.get("rank", "999999"))
            all_rows.append({"code": code, "instrument": instrument, "score": score, "rank": rank})

    all_rows.sort(key=lambda r: r["rank"])
    total_universe = len(all_rows)
    _log(f"全市场 {total_universe} 只股票，准备 dropout buffer 分析...")

    # 全市场排名映射
    rank_map: dict[str, int] = {r["code"]: r["rank"] for r in all_rows}
    code_to_inst: dict[str, str] = {r["code"]: r["instrument"] for r in all_rows}

    # Top-N 用于展示
    top_n_rows = [r for r in all_rows if r["rank"] <= top_n]
    stocks = [r["code"] for r in top_n_rows]
    scores_detail = [{"code": r["code"], "score": round(r["score"], 4), "rank": r["rank"]} for r in top_n_rows]

    # ── Strategy B Master Veto Analysis ────────────────────────────
    # 传入 top-50 给 Strategy B 做否决分析（扩大覆盖，避免漏判）
    strategy_b_result: dict = {}
    try:
        _log("运行 Strategy B Master Veto 分析...")
        _sb_n = min(50, total_universe)
        _sb_rows = [r for r in all_rows if r["rank"] <= _sb_n]
        _sb_stocks = [r["instrument"] for r in _sb_rows]
        _sb_scores = [str(r["score"]) for r in _sb_rows]

        # 转换用户持仓为 Qlib instrument 格式
        _sb_holdings = ""
        if holdings:
            codes = [c.strip() for c in holdings.split("/") if c.strip()]
            _sb_holdings = ",".join(_code_to_instrument(c) for c in codes)

        sb_cmd = [
            "docker", "run", "--rm",
            "-e", f"QLIB_DATA_DIR={cfg['qlib_data_dir']}",
            "-v", f"{_HOST_PROJECT_ROOT}:{WORKDIR}",
            "-v", f"{_HOST_QLIB_DATA}:/root/.qlib",
            "-w", WORKDIR,
            DOCKER_IMAGE,
            "python3", "strategy_b_analyze.py",
            "--stocks", ",".join(_sb_stocks),
            "--scores", ",".join(_sb_scores),
            "--date", pred_date,
            "--top-n", "5",
        ]
        if _sb_holdings:
            sb_cmd.extend(["--holdings", _sb_holdings])

        sb_process = subprocess.run(sb_cmd, capture_output=True, text=True, timeout=120)
        if sb_process.returncode == 0 and sb_process.stdout.strip():
            strategy_b_result = json.loads(sb_process.stdout.strip())
            _log(f"Strategy B: vetoed={len(strategy_b_result.get('vetoed',[]))}, "
                 f"buy={len(strategy_b_result.get('buy',[]))}")
        else:
            _log(f"Strategy B analysis failed: {sb_process.stderr[:200]}")
    except Exception as e:
        _log(f"Strategy B analysis error: {e}")

    # ── Dropout Buffer 调仓（对齐回测逻辑） ────────────────────────
    if strategy_b_result and holdings:
        dropout_buffer_pct = float(os.environ.get("DROPOUT_BUFFER_PCT", "0.5"))
        hold_num = int(os.environ.get("HOLD_NUM", "5"))
        keep_rank_cutoff = max(hold_num, int(math.ceil(total_universe * dropout_buffer_pct)))
        _log(f"Dropout buffer: keep if rank <= {keep_rank_cutoff}/{total_universe} (Top {dropout_buffer_pct*100:.0f}%)")

        # 解析被否决的股票代码
        vetoed_codes: set[str] = set()
        for v in strategy_b_result.get("vetoed", []):
            v_code = _instrument_to_code(v if isinstance(v, str) else v.get("stock", ""))
            vetoed_codes.add(v_code)

        holding_codes = [c.strip() for c in holdings.split("/") if c.strip()]

        keep_list = []
        sell_list = []
        for h in holding_codes:
            rank = rank_map.get(h, 999999)
            if h in vetoed_codes:
                sell_list.append({"stock": h, "reason": "Strategy B 否决"})
            elif rank >= 999999:
                sell_list.append({"stock": h,
                    "reason": f"不在模型覆盖范围 (数据缺失或未纳入成分股)"})
            elif rank > keep_rank_cutoff:
                sell_list.append({"stock": h,
                    "reason": f"排名 {rank}/{total_universe} > 阈值 {keep_rank_cutoff}"})
            elif rank <= hold_num:
                keep_list.append({"stock": h,
                    "reason": f"Top-{hold_num} 强推荐 (排名 {rank}/{total_universe})"})
            else:
                keep_list.append({"stock": h,
                    "reason": f"在 buffer 内 (排名 {rank}/{total_universe})"})

        # 建议买入：全市场 Top-5 非否决、非已持有
        held_set = set(holding_codes)
        buy_candidates = [r for r in all_rows
                          if r["code"] not in held_set
                          and r["code"] not in vetoed_codes]
        buy_list = [{"stock": r["code"], "score": round(r["score"], 4)}
                    for r in buy_candidates[:5]]

        strategy_b_result["keep"] = keep_list
        strategy_b_result["sell"] = sell_list
        strategy_b_result["buy"] = buy_list
        _log(f"调仓: keep={len(keep_list)}, sell={len(sell_list)}, buy={len(buy_list)}")

    result = {
        "stocks": "/".join(stocks),
        "count": len(stocks),
        "pred_date": pred_date,
        "scores": scores_detail,
        "strategy_b": strategy_b_result,
    }

    _log(
        f"完成 — Top {len(stocks)} 股票: {result['stocks'][:80]}...",
        status="completed",
        stocks=result["stocks"],
        count=result["count"],
        pred_date=result["pred_date"],
        scores=result["scores"],
        strategy_b=result["strategy_b"],
    )

    return result


# ---- CLI 入口 ----
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Qlib 推理执行器")
    ap.add_argument("model", help="模型名称")
    ap.add_argument("--top-n", type=int, default=20, help="返回 Top N 股票 (默认 20)")
    ap.add_argument("--json", action="store_true", help="以 JSON 格式输出结果")
    args = ap.parse_args()

    def _print_progress(data):
        msg = data.get("message", "")
        print(f"[qlib] {msg}", file=sys.stderr)

    result = run_inference(args.model, top_n=args.top_n, progress_callback=_print_progress)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["stocks"])
