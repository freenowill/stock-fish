#!/usr/bin/env python3
"""Re-run only the full_backtest using existing walk-forward fold outputs.

Usage (inside Docker):
    python3 scripts/practice/rerun_full_backtest.py <experiment_dir> [--hold-num 5]

The experiment_dir should contain model_predict/ with walk_forward/ subdirectories
from a completed run_new_factor_practice run.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("WORKDIR", Path(__file__).resolve().parents[2]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Re-run full_backtest from existing fold outputs")
    parser.add_argument("experiment_dir", help="Path to experiment output directory (e.g. DATA/analysis_outputs/xxx)")
    parser.add_argument("--hold-num", type=int, default=None,
                        help="Number of stocks to hold (default: from env HOLD_NUM or 5)")
    parser.add_argument("--template",
                        default="examples/benchmarks/LightGBM/workflow_config_lightgbm_AlphaExtra.yaml",
                        help="YAML template path")
    parser.add_argument("--uri-folder", default="mlruns", help="MLflow URI folder")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        print(f"ERROR: experiment directory not found: {exp_dir}")
        sys.exit(1)

    model_predict_dir = exp_dir / "model_predict"
    if not model_predict_dir.exists():
        print(f"ERROR: model_predict not found in {exp_dir}")
        sys.exit(1)

    hold_num = args.hold_num or int(os.environ.get("HOLD_NUM", 5))
    template_path = ROOT / args.template

    # Load template config
    import yaml
    with open(template_path) as f:
        template_cfg = yaml.safe_load(f)

    # Collect fold outputs (walk_forward/*/model_predict/ directories)
    wf_dir = model_predict_dir / "walk_forward"
    fold_dirs = sorted(
        [d for d in wf_dir.iterdir() if d.is_dir() and (d / "model_predict" / "scores.csv").exists()],
        key=lambda d: d.name,
    )
    if not fold_dirs:
        print(f"ERROR: no fold outputs found under {wf_dir}")
        sys.exit(1)

    print(f"Found {len(fold_dirs)} fold outputs:")
    for fd in fold_dirs:
        print(f"  {fd.name} -> {fd / 'model_predict'}")

    # Per-fold metadata
    folds_csv = model_predict_dir / "walk_forward_folds.csv"
    if folds_csv.exists():
        fold_df = pd.read_csv(folds_csv)
        fold_rows = fold_df.to_dict("records")
    else:
        fold_rows = [{"fold": i + 1, "fold_tag": d.name} for i, d in enumerate(fold_dirs)]

    # IMPORTANT: fold_outputs are the model_predict subdirectories, not the fold roots
    fold_outputs = [d / "model_predict" for d in fold_dirs]
    # Only use the last fold for full backtest
    fold_outputs = fold_outputs[-1:]

    # Remove old full_backtest output
    old_full = model_predict_dir / "full_backtest"
    if old_full.exists():
        import shutil
        print(f"\nRemoving old full_backtest: {old_full}")
        shutil.rmtree(old_full)

    old_overview = model_predict_dir / "full_backtest_overview.html"
    if old_overview.exists():
        old_overview.unlink()

    # Import and call the report function
    from scripts.practice.run_stage2_walk_forward import _write_full_backtest_report

    print(f"\nRunning full_backtest with {len(fold_outputs)} folds, hold_num={hold_num}...")
    _write_full_backtest_report(
        output_root=model_predict_dir,
        analysis_root=exp_dir,
        fold_rows=fold_rows,
        fold_outputs=fold_outputs,
        template_cfg=template_cfg,
        uri_folder=args.uri_folder,
        hold_num=hold_num,
    )

    # Check output
    full_dir = model_predict_dir / "full_backtest"
    if full_dir.exists():
        equity = full_dir / "equity_curve.csv"
        signal = full_dir / "signal.csv"
        overview = full_dir / "overview.html"
        report_txt = exp_dir / "report_of_backtest.txt"
        print(f"\nFull backtest completed:")
        print(f"  equity_curve: {equity} ({equity.stat().st_size:,} bytes)" if equity.exists() else "  equity_curve: MISSING")
        print(f"  signal: {signal} ({signal.stat().st_size:,} bytes)" if signal.exists() else "  signal: MISSING")
        print(f"  overview: {overview} ({overview.stat().st_size:,} bytes)" if overview.exists() else "  overview: MISSING")
        print(f"  report: {report_txt} ({report_txt.stat().st_size:,} bytes)" if report_txt.exists() else "  report: MISSING")

        if equity.exists():
            eq = pd.read_csv(equity)
            print(f"\n  Equity curve: {len(eq)} days, columns: {list(eq.columns)}")
            if "excess_return" in eq.columns:
                print(f"  Excess return (last 5 days):")
                print(eq[["datetime", "excess_return"]].tail().to_string(index=False))

        if report_txt.exists():
            print(f"\n  Report content:")
            print(report_txt.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
