#!/usr/bin/env python3
"""Simplified live trading stage for run_alpha158_small.

Replaces stages 3-6 with a straightforward top-K buffered equal-weight approach.

Logic per weekly run:
1. Load stage2 scores.csv
2. Apply price-cap filter using qlib close price data
3. Rank by score descending
4. Load previous week's holdings from holdings_tracker.json
5. Keep holdings that remain above buffer_rank_cutoff in the full universe
6. Fill remaining slots (up to hold_num) with top-scoring affordable stocks
7. Assign equal weights = 1/hold_num
8. Save holdings for next week
9. Output final_result.csv with code, rank, weight, action (keep/buy)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_qlib_instrument(code: str) -> str:
    code = str(code).strip().zfill(6)
    return f"SH{code}" if code.startswith(("6", "9")) else f"SZ{code}"


def _extract_6digit(code: str) -> str:
    return str(code).strip().replace("SH", "").replace("SZ", "").replace("sh", "").replace("sz", "").zfill(6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simplified stage3: buffered top-K live trading")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pred-date", required=True)
    parser.add_argument("--hold-num", type=int, default=int(os.environ.get("HOLD_NUM", "5")))
    parser.add_argument("--buffer-pct", type=float, default=float(os.environ.get("FULL_BACKTEST_BUFFER_PCT", "0.5")))
    parser.add_argument("--price-cap", type=float, default=float(os.environ.get("MAX_STOCK_PRICE", "50.0")))
    parser.add_argument("--market", default=os.environ.get("TARGET_MARKET", "csi1000"))
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load scores.csv
    scores_csv = pred_dir / "scores.csv"
    if not scores_csv.exists():
        scores_csv = pred_dir / "walk_forward" / "scores.csv"
    if not scores_csv.exists():
        raise FileNotFoundError(f"scores.csv not found under {pred_dir}")

    df = pd.read_csv(scores_csv)
    if df.empty:
        raise RuntimeError("scores.csv is empty")

    score_col = "score_final" if "score_final" in df.columns else "score"
    instrument_col = "instrument" if "instrument" in df.columns else "code"
    df["code"] = df[instrument_col].astype(str).apply(_extract_6digit)
    df["score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["code", "score"])

    # 2. Apply price-cap filter using qlib close price data
    try:
        import qlib
        from qlib.data import D

        qlib.init(
            provider_uri=str(os.environ.get("QLIB_DATA_DIR", "~/.qlib/qlib_data/cn_data_filter_small")),
            region="cn",
        )

        instruments = df["code"].apply(_resolve_qlib_instrument).unique().tolist()
        end_time = pd.Timestamp(args.pred_date)
        start_time = end_time - pd.Timedelta(days=30)

        close_df = D.features(
            instruments, ["$close"],
            start_time=start_time.strftime("%Y-%m-%d"),
            end_time=end_time.strftime("%Y-%m-%d"),
        )
        if close_df is not None and len(close_df) > 0:
            if isinstance(close_df, pd.Series):
                close_df = close_df.to_frame("close")
            if isinstance(close_df.index, pd.MultiIndex):
                close_df = close_df.reset_index()
            close_col = "close" if "close" in close_df.columns else close_df.columns[-1]
            close_df["instrument"] = close_df["instrument"].astype(str)
            close_df[close_col] = pd.to_numeric(close_df[close_col], errors="coerce")
            close_df = close_df.dropna(subset=[close_col])
            latest_close = close_df.groupby("instrument")[close_col].last().reset_index()
            latest_close["code"] = latest_close["instrument"].apply(_extract_6digit)
            df = df.merge(latest_close[["code", close_col]], on="code", how="left")
            df = df.rename(columns={close_col: "close"})
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            before = len(df)
            df = df[df["close"].notna() & (df["close"] <= args.price_cap)].copy()
            print(f"  Price cap filter ({args.price_cap}): {before} -> {len(df)} stocks")
    except Exception as e:
        print(f"  Qlib price filter unavailable: {e}, skipping price cap")

    # 3. Rank by score descending
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    total = len(df)
    df["rank"] = range(1, total + 1)
    df["rank_pct"] = (df["rank"] / max(total, 1) * 100).round(2)

    # 4. Load previous holdings
    tracker_path = output_dir / "holdings_tracker.json"
    previous_holdings: set = set()
    if tracker_path.exists():
        try:
            prev = json.loads(tracker_path.read_text(encoding="utf-8"))
            previous_holdings = set(prev.get("holdings", []))
            print(f"  Previous holdings ({len(previous_holdings)} stocks): {previous_holdings}")
        except Exception:
            pass

    # 5. Buffer logic: keep holdings still in top buffer_pct of rank
    keep_rank_cutoff = max(int(args.hold_num), int(np.ceil(total * max(args.buffer_pct, 0.0))))
    code_to_rank = dict(zip(df["code"].astype(str), df["rank"]))

    kept = [
        code
        for code in previous_holdings
        if code in code_to_rank and int(code_to_rank[code]) <= keep_rank_cutoff
    ]

    all_codes = df["code"].astype(str).tolist()
    additions = [code for code in all_codes if code not in kept]
    selected_codes = kept + additions[:max(int(args.hold_num) - len(kept), 0)]

    if not selected_codes:
        print("  No stocks selected, falling back to top-N by score")
        selected_codes = all_codes[: args.hold_num]

    # 6. Assign equal weights
    selected = df[df["code"].astype(str).isin(selected_codes)].copy()
    code_order = {c: i for i, c in enumerate(selected_codes)}
    selected["_order"] = selected["code"].map(code_order)
    selected = selected.sort_values("_order").drop(columns=["_order"])
    selected["weight"] = 1.0 / max(len(selected), 1)
    selected["instrument"] = selected["code"].apply(_resolve_qlib_instrument)
    selected["action"] = selected["code"].apply(lambda c: "keep" if str(c) in kept else "buy")

    # 7. Save holdings for next week
    tracker_path.write_text(
        json.dumps(
            {"holdings": selected_codes, "pred_date": args.pred_date, "market": args.market},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 8. Output files
    out_cols = ["code", "instrument", "score", "rank", "rank_pct", "weight", "action"]
    if "close" in selected.columns:
        out_cols.insert(2, "close")

    final_df = selected[out_cols].reset_index(drop=True)
    final_csv = output_dir / "final_result.csv"
    final_df.to_csv(final_csv, index=False, encoding="utf-8-sig")
    print(f"  Final result: {final_csv} ({len(final_df)} stocks)")
    kept_count = int((selected["action"] == "keep").sum())
    buy_count = int((selected["action"] == "buy").sum())
    print(f"  Keep: {kept_count}  Buy: {buy_count}")
    print(f"  Holdings: {selected_codes}")

    # Compat output files for downstream consumers
    compat = final_df.copy()
    compat["name"] = compat["code"]
    compat["pred_date"] = str(args.pred_date)
    compat["date"] = str(args.pred_date)
    compat["score_final"] = compat["score"]
    compat["stock"] = compat["code"]
    compat.to_csv(output_dir / "result.csv", index=False, encoding="utf-8-sig")
    compat.to_csv(output_dir / "result_update.csv", index=False, encoding="utf-8-sig")

    print(f"\n  Trade Summary ({args.pred_date}):")
    print(f"  {'=' * 40}")
    print(f"  Holdings: {len(final_df)} / {args.hold_num}  (keep {kept_count} + buy {buy_count})")


if __name__ == "__main__":
    main()
