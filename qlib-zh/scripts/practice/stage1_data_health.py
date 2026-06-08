#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


EXCLUDED_PREFIXES = ("300", "301", "688", "689")


def _normalize_code(code: str) -> str:
    value = str(code).strip()
    if value.count(".") == 1:
        left, right = value.split(".")
        value = right if right.isdigit() else left
    if value.upper().startswith(("SH", "SZ")):
        value = value[2:]
    return value.zfill(6)


def _is_excluded_board(code: str) -> bool:
    return _normalize_code(code).startswith(EXCLUDED_PREFIXES)


def _validate_required_paths(qlib_dir: Path, market: str) -> dict[str, object]:
    required = {
        "qlib_dir": qlib_dir,
        "calendar_day": qlib_dir / "calendars" / "day.txt",
        "all_instruments": qlib_dir / "instruments" / "all.txt",
        f"{market}_instruments": qlib_dir / "instruments" / f"{market}.txt",
        "features_dir": qlib_dir / "features",
    }
    return {name: path.exists() for name, path in required.items()}


def _latest_calendar_date(qlib_dir: Path) -> str | None:
    calendar_path = qlib_dir / "calendars" / "day.txt"
    if not calendar_path.exists():
        return None
    rows = [line.strip() for line in calendar_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[-1] if rows else None


def _read_instruments_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["code", "listed_date", "delisted_date"])
    df = pd.read_csv(path, sep="\t", header=None, names=["code", "listed_date", "delisted_date"])
    df["code"] = df["code"].astype(str).map(_normalize_code)
    df["listed_date"] = pd.to_datetime(df["listed_date"], errors="coerce")
    df["delisted_date"] = pd.to_datetime(df["delisted_date"], errors="coerce")
    return df


def _write_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src, target_is_directory=src.is_dir())


def _build_filtered_qlib_dir(source_dir: Path, target_dir: Path, market: str) -> dict[str, object]:
    if not source_dir.exists():
        raise FileNotFoundError(f"源 qlib 数据目录不存在: {source_dir}")

    target_dir = target_dir.resolve()
    source_dir = source_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    # Keep calendars/features as lightweight symlinks so health checks still
    # operate on the source data while instruments are filtered locally.
    _write_symlink(source_dir / "calendars", target_dir / "calendars")
    _write_symlink(source_dir / "features", target_dir / "features")

    instruments_src = source_dir / "instruments"
    instruments_dst = target_dir / "instruments"
    instruments_dst.mkdir(parents=True, exist_ok=True)

    all_df = _read_instruments_file(instruments_src / "all.txt")
    market_df = _read_instruments_file(instruments_src / f"{market}.txt")

    if all_df.empty:
        raise RuntimeError(f"源数据缺少 instruments/all.txt: {source_dir}")
    if market_df.empty:
        raise RuntimeError(f"源数据缺少 instruments/{market}.txt: {source_dir}")

    all_filtered = all_df[~all_df["code"].map(_is_excluded_board)].copy()
    market_filtered = market_df[~market_df["code"].map(_is_excluded_board)].copy()

    def _dump(df: pd.DataFrame, path: Path) -> None:
        out = df.copy()
        out["listed_date"] = out["listed_date"].dt.strftime("%Y-%m-%d")
        out["delisted_date"] = out["delisted_date"].dt.strftime("%Y-%m-%d")
        out = out.fillna("")
        out.to_csv(path, sep="\t", header=False, index=False, encoding="utf-8")

    _dump(all_filtered, instruments_dst / "all.txt")
    _dump(market_filtered, instruments_dst / f"{market}.txt")

    meta = {
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "market": market,
        "all_before": int(len(all_df)),
        "all_after": int(len(all_filtered)),
        f"{market}_before": int(len(market_df)),
        f"{market}_after": int(len(market_filtered)),
        "excluded_prefixes": list(EXCLUDED_PREFIXES),
    }
    (target_dir / "filter_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _run_health_check(qlib_dir: Path, market: str = "all") -> None:
    script_path = Path(__file__).resolve().parents[1] / "check_data_health.py"
    cmd = [
        sys.executable,
        str(script_path),
        "check_data",
        "--qlib_dir",
        str(qlib_dir),
        "--market",
        market,
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage1 Qlib data health check and filtered-data builder")
    parser.add_argument("--source-qlib-dir", default="/root/.qlib/qlib_data/cn_data")
    parser.add_argument("--qlib-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pred-date", default=None)
    parser.add_argument("--market", default=os.environ.get("TARGET_MARKET", "csi300"))
    args = parser.parse_args()

    source_dir = Path(args.source_qlib_dir).expanduser().resolve()
    qlib_dir = Path(args.qlib_dir).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = _build_filtered_qlib_dir(source_dir, qlib_dir, args.market)
    path_status = _validate_required_paths(qlib_dir, args.market)
    missing_paths = [name for name, exists in path_status.items() if not exists]
    if missing_paths:
        raise SystemExit(f"缺少必要数据路径: {missing_paths}")

    latest_date = _latest_calendar_date(qlib_dir)
    _run_health_check(qlib_dir, args.market)

    summary = {
        "source_qlib_dir": str(source_dir),
        "qlib_dir": str(qlib_dir),
        "latest_calendar_date": latest_date,
        "path_status": path_status,
        "filter_meta": meta,
        "status": "ok",
    }

    summary_path = output_dir / "stage1_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"✓ stage1 summary saved: {summary_path}")


if __name__ == "__main__":
    main()