#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


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


def _write_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src, target_is_directory=src.is_dir())


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _build_smallcap_qlib_dir(source_dir: Path, target_dir: Path, market: str) -> dict[str, object]:
    if not source_dir.exists():
        raise FileNotFoundError(f"源 qlib 数据目录不存在: {source_dir}")

    source_dir = source_dir.resolve()
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    _write_symlink(source_dir / "calendars", target_dir / "calendars")
    _write_symlink(source_dir / "features", target_dir / "features")

    instruments_src = source_dir / "instruments"
    instruments_dst = target_dir / "instruments"
    if instruments_dst.exists():
        shutil.rmtree(instruments_dst)
    instruments_dst.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    for src in sorted(instruments_src.glob("*.txt")):
        shutil.copy2(src, instruments_dst / src.name)
        copied_files.append(src.name)

    market_path = instruments_dst / f"{market}.txt"
    if not market_path.exists():
        raise RuntimeError(f"源数据缺少 instruments/{market}.txt: {source_dir}")

    meta = {
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "market": market,
        "copied_instruments": copied_files,
        "all_rows": _line_count(instruments_dst / "all.txt"),
        f"{market}_rows": _line_count(market_path),
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
    parser = argparse.ArgumentParser(description="Stage1 CSI1000 data health check and isolated qlib builder")
    parser.add_argument("--source-qlib-dir", default="/root/.qlib/qlib_data/cn_data")
    parser.add_argument("--qlib-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pred-date", default=None)
    parser.add_argument("--market", default="csi1000")
    args = parser.parse_args()

    source_dir = Path(args.source_qlib_dir).expanduser().resolve()
    qlib_dir = Path(args.qlib_dir).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = _build_smallcap_qlib_dir(source_dir, qlib_dir, args.market)
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