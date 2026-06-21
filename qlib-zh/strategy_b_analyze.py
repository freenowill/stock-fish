#!/usr/bin/env python3
"""Lightweight Strategy B (Master Veto) analysis for inference results.

Takes model's top-20 stock predictions and applies the 4 veto rules,
outputting a filtered buy list and rebalancing advice.

Usage (inside Docker):
    python3 strategy_b_analyze.py --stocks SH600000,SH600009,... \
        --scores 0.14,0.12,... --date 2026-06-20 \
        [--holdings SH600123,SH600456,...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure qlib-zh scripts are importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Strategy B Master Veto analysis")
    parser.add_argument("--stocks", required=True, help="Comma-separated stock instruments (SH600000,SZ000001,...)")
    parser.add_argument("--scores", required=True, help="Comma-separated model scores")
    parser.add_argument("--date", required=True, help="Analysis date (YYYY-MM-DD)")
    parser.add_argument("--holdings", default="", help="Current holdings (comma-separated)")
    parser.add_argument("--top-n", type=int, default=5, help="Number of stocks to recommend")
    args = parser.parse_args()

    stocks = [s.strip() for s in args.stocks.split(",") if s.strip()]
    scores = [float(s.strip()) for s in args.scores.split(",") if s.strip()]
    holdings = [h.strip() for h in args.holdings.split(",") if h.strip()] if args.holdings else []
    as_of_date = pd.Timestamp(args.date)
    top_n = args.top_n

    if len(stocks) < 3:
        print(json.dumps({"error": "Need at least 3 stocks", "buy": [], "sell": [], "keep": []}))
        return

    # ── Initialize Qlib ────────────────────────────────────────────
    import qlib
    from qlib.data import D

    provider_uri = os.environ.get("QLIB_DATA_DIR", "/root/.qlib/qlib_data/cn_data")
    qlib.init(provider_uri=provider_uri, region="cn")

    # ── Query 6-dimension factor expressions ───────────────────────
    from scripts.practice.stage2_master_strategy import (
        FACTOR_EXPRESSIONS,
        _ALL_EXPRESSIONS,
        _extract_no_leakage_factors,
        _cross_sectional_percentile,
        _assess_stock_risk_opportunity,
    )

    start_time = (as_of_date - pd.Timedelta(days=800)).strftime("%Y-%m-%d")
    end_time = as_of_date.strftime("%Y-%m-%d")

    factor_df = D.features(stocks, _ALL_EXPRESSIONS, start_time=start_time, end_time=end_time)

    if factor_df is None or factor_df.empty:
        print(json.dumps({"error": "Factor query returned empty", "buy": stocks[:top_n], "sell": [], "keep": []}))
        return

    # Normalize index
    if not isinstance(factor_df.index, pd.MultiIndex):
        factor_df = factor_df.reset_index()
        renamed = {}
        for col in factor_df.columns:
            cl = col.lower()
            if cl in ("datetime", "date", "trade_date"):
                renamed[col] = "datetime"
            elif cl in ("instrument", "code", "symbol", "stock"):
                renamed[col] = "instrument"
        factor_df = factor_df.rename(columns=renamed)
        factor_df["datetime"] = pd.to_datetime(factor_df["datetime"])
        factor_df["instrument"] = factor_df["instrument"].astype(str)
        factor_df = factor_df.set_index(["datetime", "instrument"])

    # Extract no-leakage factor snapshot
    fs = _extract_no_leakage_factors(factor_df, stocks, as_of_date)
    if fs.empty or len(fs) < 3:
        print(json.dumps({"error": "Insufficient factor data", "buy": stocks[:top_n], "sell": [], "keep": []}))
        return

    # Compute dimension scores
    dim_factors = {}
    for dim, factors in FACTOR_EXPRESSIONS.items():
        avail = [(e, d, w) for e, d, w in factors if e in fs.columns]
        if avail:
            dim_factors[dim] = avail

    dim_scores = {}
    for dim, factors in dim_factors.items():
        vals = np.zeros(len(fs))
        tw = 0.0
        for expr, direction, weight in factors:
            if expr not in fs.columns:
                continue
            raw = fs[expr].astype(float).values
            pct = _cross_sectional_percentile(raw)
            if direction == -1:
                pct = 1.0 - pct
            vals += weight * pct
            tw += weight
        if tw > 0:
            vals /= tw
        dim_scores[dim] = vals

    # ── Run Strategy B veto ────────────────────────────────────────
    vetoed, overweight, reasons = _assess_stock_risk_opportunity(
        dim_scores, fs.index.tolist(), fs
    )

    # ── Build ranked buy list (model score order, skip vetoed) ─────
    scored = list(zip(stocks, scores))
    scored.sort(key=lambda x: -x[1])

    buy_list = []
    veto_list = []
    for s, sc in scored:
        if s in vetoed:
            veto_list.append({"stock": s, "score": sc, "reason": "vetoed"})
        else:
            buy_list.append({"stock": s, "score": sc})

    # ── Rebalancing advice ─────────────────────────────────────────
    buy_recs = [b["stock"] for b in buy_list[:top_n]]
    sell_recs = []
    keep_recs = []

    if holdings:
        holdings_set = set(holdings)
        all_candidates = set(stocks)  # 模型 Top-20
        for h in holdings:
            if h in vetoed:
                sell_recs.append({"stock": h, "reason": "Strategy B 否决"})
            elif h not in all_candidates:
                # 不在模型 Top-20 内 → 建议卖出
                sell_recs.append({"stock": h, "reason": "不在模型 Top-20"})
            elif h in [b["stock"] for b in buy_list[:top_n]]:
                keep_recs.append({"stock": h, "reason": f"在 Top-{top_n} 推荐中"})
            else:
                # 在 Top-20 但不在 Top-5 → 持有但关注
                keep_recs.append({"stock": h, "reason": f"在模型 Top-20 (非 Top-{top_n})"})

        # Also suggest buying stocks not currently held
        buy_recs = [b for b in buy_recs if b not in holdings_set]

    # ── Output ─────────────────────────────────────────────────────
    result = {
        "date": args.date,
        "buy": buy_list[:top_n],  # 保留完整 dict（stock + score）
        "sell": sell_recs,        # 保留完整 dict（stock + reason）
        "keep": keep_recs,        # 保留完整 dict（stock + reason）
        "vetoed": veto_list,      # 保留完整 dict（stock + score + reason）
        "veto_reasons": reasons[:5],
        "overweight_signals": list(overweight.keys()),
        "top20_ranked": [{"stock": s, "score": round(sc, 4), "vetoed": s in vetoed}
                         for s, sc in scored],
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
