#!/usr/bin/env python3
"""Stage1 data health check for AlphaExtra (cn_extra_data).

Checks feature completeness by reading binary .day.bin files directly,
filters stocks with too many missing values, and creates a filtered
data directory (symlink-based, only instrument lists differ).

Usage (inside container):
    python3 scripts/practice/stage1_data_health_extra.py \
        --source-qlib-dir /root/.qlib/qlib_data/cn_extra_data \
        --qlib-dir /root/.qlib/qlib_data/cn_extra_data_filtered \
        --output /work/DATA/analysis_outputs/<exp>/data_health \
        --market all \
        --missing-threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd


def _discover_expected_features(source_features_dir: Path) -> frozenset:
    """Auto-detect expected features from the first stock's bin files."""
    stock_dirs = sorted(d for d in source_features_dir.iterdir() if d.is_dir())
    if not stock_dirs:
        return frozenset()
    return frozenset(
        f.name.replace(".day.bin", "")
        for f in stock_dirs[0].glob("*.day.bin")
    )


EXPECTED_FEATURES = None  # None = auto-detect at runtime


def _normalize_code(code: str) -> str:
    value = str(code).strip()
    if value.count(".") == 1:
        left, right = value.split(".")
        value = right.lower() + left
    return value.lower()


def _read_instruments_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        df = pd.DataFrame(columns=["code", "listed_date", "delisted_date"])
        df["listed_date"] = pd.to_datetime(df["listed_date"], errors="coerce")
        df["delisted_date"] = pd.to_datetime(df["delisted_date"], errors="coerce")
        return df
    df = pd.read_csv(path, sep="\t", header=None, names=["code", "listed_date", "delisted_date"])
    df["code"] = df["code"].astype(str).map(_normalize_code)
    df["listed_date"] = pd.to_datetime(df["listed_date"], errors="coerce")
    df["delisted_date"] = pd.to_datetime(df["delisted_date"], errors="coerce")
    return df


def _write_symlink(src: Path, dst: Path, absolute: bool = False) -> None:
    """Create a symlink from dst -> src.

    Uses relative paths by default. Set absolute=True to use the src path directly
    (needed when the filtered dir will be mounted at a different path than src).
    """
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if absolute:
        dst.symlink_to(str(src), target_is_directory=src.is_dir())
    else:
        rel = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel, target_is_directory=src.is_dir())


def _read_feature_file(filepath: Path) -> tuple[int, np.ndarray, int, int]:
    """Read a qlib .day.bin file.

    Returns (start_index, data_array, nan_count, total_count).
    Format: [start_index: float32][value1: float32][value2: float32]...
    """
    if not filepath.exists():
        return 0, np.array([]), 0, 0
    raw = np.fromfile(filepath, dtype="<f")
    if len(raw) < 2:
        return int(raw[0]) if len(raw) == 1 else 0, np.array([]), 0, 0
    start_index = int(raw[0])
    data = raw[1:]
    nan_count = int(np.isnan(data).sum())
    return start_index, data, nan_count, len(data)


def _check_stock(source_features_dir: Path, stock_name: str) -> dict:
    """Check one stock's feature completeness.

    Returns dict with:
        - present: number of expected features found
        - total_nan: total NaN values across all present features
        - total_data: total data points across all present features
        - missing_ratio: overall missing ratio
        - feature_details: per-feature stats
    """
    stock_dir = source_features_dir / stock_name
    if not stock_dir.is_dir():
        return {
            "present": 0, "total_nan": 0, "total_data": 0,
            "missing_ratio": 1.0, "feature_details": {},
            "has_dir": False,
        }

    feature_details = {}
    total_nan = 0
    total_data = 0
    present = 0

    for feat in EXPECTED_FEATURES:
        fpath = stock_dir / f"{feat}.day.bin"
        _, _, nan_cnt, cnt = _read_feature_file(fpath)
        if cnt > 0:
            present += 1
            total_nan += nan_cnt
            total_data += cnt
            feature_details[feat] = {"nan": int(nan_cnt), "total": int(cnt)}
        else:
            feature_details[feat] = {"nan": 0, "total": 0, "missing_file": True}

    # Expected data points: if all features present, total = 25 * per_feature_len
    # Missing features count as fully missing
    max_expected = len(EXPECTED_FEATURES)
    effective_nan = total_nan + (max_expected - present) * (total_data // max(present, 1)) if present > 0 else max_expected * 100
    effective_total = total_data + (max_expected - present) * (total_data // max(present, 1)) if present > 0 else max_expected * 100
    missing_ratio = effective_nan / effective_total if effective_total > 0 else 1.0

    return {
        "present": present,
        "total_nan": int(total_nan),
        "total_data": int(total_data),
        "missing_ratio": round(float(missing_ratio), 6),
        "feature_details": feature_details,
        "has_dir": True,
    }


def _check_stock_worker(args: tuple[str, Path, frozenset]) -> tuple[str, dict]:
    """Multiprocessing worker: check one stock."""
    code, source_features_dir, expected_features = args
    # Restore global for this worker process
    global EXPECTED_FEATURES
    EXPECTED_FEATURES = expected_features
    return code, _check_stock(source_features_dir, code)


def _check_all_stocks(
    source_features_dir: Path,
    stock_codes: list[str],
    workers: int = 4,
) -> dict[str, dict]:
    """Check all stocks for missing data, using multiprocessing.

    Args:
        workers: Number of parallel worker processes (0 = auto-detect).
    """
    n_workers = workers if workers > 0 else min(multiprocessing.cpu_count(), 8)
    total = len(stock_codes)

    # Package arguments for each worker: (code, features_dir, expected_features)
    work_items = [(code, source_features_dir, EXPECTED_FEATURES) for code in stock_codes]

    results: dict[str, dict] = {}
    print(f"  Parallel check: {total} stocks, {n_workers} workers")

    with multiprocessing.Pool(processes=n_workers) as pool:
        chunk_size = max(1, total // (n_workers * 50))
        for i, (code, result) in enumerate(
            pool.imap_unordered(_check_stock_worker, work_items, chunksize=chunk_size), start=1
        ):
            results[code] = result
            if i % 500 == 0 or i == total:
                print(f"  ... checked {i}/{total} stocks", flush=True)
            elif i % 50 == 0:
                print(f"  ... checked {i}/{total} stocks", end="\r", flush=True)
        # Ensure a clean newline after \r progress updates
        if total % 500 != 0:
            print(flush=True)

    return results


def _compute_feature_coverage(results: dict[str, dict], expected_features: frozenset) -> dict:
    """Compute per-feature coverage across all stocks."""
    coverage = {}
    for feat in expected_features:
        stocks_with = sum(1 for r in results.values() if r.get("feature_details", {}).get(feat, {}).get("total", 0) > 0)
        coverage[feat] = {
            "stocks_with_feature": stocks_with,
            "total_stocks": len(results),
            "coverage": round(stocks_with / len(results), 6) if results else 0,
        }
    return coverage


def _build_filtered_qlib_dir(
    source_dir: Path, target_dir: Path, market: str, results: dict[str, dict], threshold: float,
    features_source: Path | None = None,
) -> dict:
    """Create filtered data directory with symlinks and filtered instrument lists.

    Args:
        features_source: If given, calendars/features symlinks point to this path.
                         Use when the filtered dir will be mounted at a different path
                         than the source data (e.g., filtered mounts over original path).
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source qlib data dir not found: {source_dir}")

    source_dir = source_dir.resolve()
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    # Symlinks for calendars/features: use absolute symlinks if features_source
    # is provided (for cross-mount resolution in Docker), otherwise relative.
    _link_src = features_source.resolve() if features_source else source_dir
    _use_absolute = features_source is not None
    _write_symlink(_link_src / "calendars", target_dir / "calendars", absolute=_use_absolute)
    _write_symlink(_link_src / "features", target_dir / "features", absolute=_use_absolute)

    instruments_src = source_dir / "instruments"
    instruments_dst = target_dir / "instruments"
    instruments_dst.mkdir(parents=True, exist_ok=True)

    all_df = _read_instruments_file(instruments_src / "all.txt")

    market_file = instruments_src / f"{market}.txt"
    market_df = _read_instruments_file(market_file)
    if market_df.empty and not all_df.empty and not market_file.exists():
        print(f"  {market}.txt 不存在, 使用 all.txt 代替 (股票数: {len(all_df)})")
        market_df = all_df.copy()

    # Filter by missing ratio
    excluded_codes = {code for code, r in results.items() if r["missing_ratio"] > threshold}

    all_filtered = all_df[~all_df["code"].isin(excluded_codes)].copy()
    market_filtered = market_df[~market_df["code"].isin(excluded_codes)].copy()

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
        "missing_threshold": threshold,
        "all_before": int(len(all_df)),
        "all_after": int(len(all_filtered)),
        f"{market}_before": int(len(market_df)),
        f"{market}_after": int(len(market_filtered)),
        "excluded_stocks": sorted(excluded_codes),
        "excluded_count": len(excluded_codes),
    }
    (target_dir / "filter_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage1 data health check and filtered-data builder for AlphaExtra"
    )
    parser.add_argument("--source-qlib-dir", default="/root/.qlib/qlib_data/cn_extra_data_h5")
    parser.add_argument("--qlib-dir", required=True, help="Output filtered qlib data directory")
    parser.add_argument("--output", required=True, help="Output report directory")
    parser.add_argument("--market", default=os.environ.get("TARGET_MARKET", "all"))
    parser.add_argument("--missing-threshold", type=float, default=0.5,
                        help="Stocks with missing ratio > this are excluded (default: 0.5)")
    parser.add_argument("--pred-date", default=None)
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers for stock check (0=auto, default: auto)")
    parser.add_argument("--features-source", default=None,
                        help="Path calendars/features symlinks point to (default: source-qlib-dir)")
    args = parser.parse_args()

    source_dir = Path(args.source_qlib_dir).expanduser().resolve()
    qlib_dir = Path(args.qlib_dir).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        raise SystemExit(f"Source qlib data directory not found: {source_dir}")

    source_features = source_dir / "features"
    instruments_src = source_dir / "instruments"

    # Auto-detect expected features from the first stock's bin files
    global EXPECTED_FEATURES
    if EXPECTED_FEATURES is None:
        EXPECTED_FEATURES = _discover_expected_features(source_features)

    # Load instrument codes from the market definition
    all_df = _read_instruments_file(instruments_src / "all.txt")
    stock_codes = sorted(all_df["code"].tolist())

    print(f"Checking {len(stock_codes)} stocks for missing data...")
    print(f"Missing threshold: {args.missing_threshold}")
    print(f"Expected features per stock: {len(EXPECTED_FEATURES)}")

    results = _check_all_stocks(source_features, stock_codes, workers=args.workers)

    # Summary statistics
    ratios = [r["missing_ratio"] for r in results.values()]
    excluded = [code for code, r in results.items() if r["missing_ratio"] > args.missing_threshold]

    print(f"\nMissing ratio stats across {len(ratios)} stocks:")
    print(f"  min:    {min(ratios):.4f}")
    print(f"  max:    {max(ratios):.4f}")
    print(f"  mean:   {np.mean(ratios):.4f}")
    print(f"  median: {np.median(ratios):.4f}")
    print(f"  p90:    {np.percentile(ratios, 90):.4f}")
    print(f"  p95:    {np.percentile(ratios, 95):.4f}")
    print(f"  p99:    {np.percentile(ratios, 99):.4f}")
    print(f"\n  Excluded (> {args.missing_threshold}): {len(excluded)} stocks")
    if excluded:
        print(f"  Examples: {excluded[:10]}")

    # Per-feature coverage
    feature_coverage = _compute_feature_coverage(results, EXPECTED_FEATURES)
    low_coverage_features = [
        (feat, c["coverage"]) for feat, c in feature_coverage.items() if c["coverage"] < 0.9
    ]
    if low_coverage_features:
        print(f"\nFeatures with < 90% stock coverage: {len(low_coverage_features)}")
        for feat, cov in sorted(low_coverage_features, key=lambda x: x[1]):
            print(f"  {feat}: {cov:.2%}")

    # Build filtered data directory
    print(f"\nBuilding filtered data directory: {qlib_dir}")
    features_source = Path(args.features_source).expanduser() if args.features_source else None
    meta = _build_filtered_qlib_dir(source_dir, qlib_dir, args.market, results, args.missing_threshold, features_source)
    print(f"  {meta['all_before']} → {meta['all_after']} instruments (all)")
    print(f"  {meta[f'{args.market}_before']} → {meta[f'{args.market}_after']} instruments ({args.market})")

    # Write summary JSON
    ratio_distribution = {}
    for r in ratios:
        bucket = int(r * 10) / 10  # 0.0, 0.1, 0.2, ...
        bucket_key = f"{bucket:.1f}-{bucket + 0.1:.1f}"
        ratio_distribution[bucket_key] = ratio_distribution.get(bucket_key, 0) + 1

    summary = {
        "source_qlib_dir": str(source_dir),
        "qlib_dir": str(qlib_dir),
        "market": args.market,
        "missing_threshold": args.missing_threshold,
        "total_stocks_checked": len(stock_codes),
        "excluded_stocks": sorted(excluded),
        "excluded_count": len(excluded),
        "missing_ratio_stats": {
            "min": float(min(ratios)),
            "max": float(max(ratios)),
            "mean": float(np.mean(ratios)),
            "median": float(np.median(ratios)),
            "p90": float(np.percentile(ratios, 90)),
            "p95": float(np.percentile(ratios, 95)),
            "p99": float(np.percentile(ratios, 99)),
        },
        "missing_ratio_distribution": ratio_distribution,
        "feature_coverage": {k: v for k, v in feature_coverage.items()},
        "filter_meta": meta,
        "status": "ok",
    }

    summary_path = output_dir / "stage1_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nStage1 summary saved: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
