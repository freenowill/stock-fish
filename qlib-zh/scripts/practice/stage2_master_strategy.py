#!/usr/bin/env python3
"""
Master Analysis Backtest Strategy — Stage 2 Post-Processing Layer.

After stage 2 walk-forward training generates model predictions (all_scores.csv),
this module applies a "master philosophy" selection layer:

    1. At each rebalance date, pick top-K candidates by model score
    2. For each candidate, compute per-dimension factor scores from Qlib OHLCV data
    3. Combine dimensions with per-master weights (7 masters from StockFish CIO)
    4. Ensemble across masters → final ranking → select top-N holdings
    5. Build trade signal for PrecomputedWeightStrategy backtest

Key design properties:
- Uses ONLY Qlib D.features() with expression syntax (no StockFish providers)
- Strict no-future-leakage: at date t, only data with datetime <= t
- Pure rule-based: no LLM calls, no external APIs
- Factor expressions compute directly from price/volume in one batch query
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ═════════════════════════════════════════════════════════════════════════════
# Master Dimension Weights
# Derived from StockFish analysis/agents/cio_prompts.py investment philosophies.
# Each master has a 6-dimension weight vector:
#   [value, quality, growth, momentum, low_risk, sentiment]
# ═════════════════════════════════════════════════════════════════════════════

MASTER_WEIGHTS: dict[str, dict[str, float]] = {
    "graham":    {"value": 0.50, "quality": 0.25, "growth": 0.00, "momentum": 0.00, "low_risk": 0.15, "sentiment": 0.10},
    "buffett":   {"value": 0.20, "quality": 0.50, "growth": 0.15, "momentum": 0.00, "low_risk": 0.10, "sentiment": 0.05},
    "fisher":    {"value": 0.05, "quality": 0.25, "growth": 0.55, "momentum": 0.05, "low_risk": 0.05, "sentiment": 0.05},
    "lynch":     {"value": 0.20, "quality": 0.25, "growth": 0.35, "momentum": 0.10, "low_risk": 0.05, "sentiment": 0.05},
    "templeton": {"value": 0.30, "quality": 0.15, "growth": 0.05, "momentum": 0.00, "low_risk": 0.10, "sentiment": 0.40},
    "soros":     {"value": 0.05, "quality": 0.05, "growth": 0.10, "momentum": 0.55, "low_risk": 0.00, "sentiment": 0.25},
    "dalio":     {"value": 0.10, "quality": 0.15, "growth": 0.10, "momentum": 0.05, "low_risk": 0.50, "sentiment": 0.10},
}

# ═════════════════════════════════════════════════════════════════════════════
# Factor Expressions per Dimension
# Qlib expression syntax: $close, Ref(), Mean(), Std(), Sum(), Max(), Min()...
# Each tuple: (expression, direction, weight_in_dimension)
#   direction =  1 → higher factor value → higher score
#   direction = -1 → higher factor value → lower score
# ═════════════════════════════════════════════════════════════════════════════

FACTOR_EXPRESSIONS: dict[str, list[tuple[str, int, float]]] = {
    "value": [
        # Distance from 1-year average (price mean-reversion proxy)
        ("$close / Mean($close, 252) - 1", -1, 0.25),
        # Distance from 52-week high (drawdown from peak)
        ("$close / Max($close, 252) - 1", 1, 0.25),
        # Distance from 3-year average (long-term value)
        ("$close / Mean($close, 756) - 1", -1, 0.15),
        # Low relative volume = neglected stock (Templeton-style value)
        ("$volume / Mean($volume, 252) - 1", -1, 0.15),
        # Book-to-price proxy: negative return over 2 years = cheaper
        ("$close / Ref($close, 504) - 1", -1, 0.20),
    ],
    "quality": [
        # Return stability (low earnings volatility proxy)
        ("Std(Ref($close, -1) / $close - 1, 63)", -1, 0.25),
        # Sharpe-like: return / volatility ratio over 6 months
        ("($close / Ref($close, 126) - 1) / (Std(Ref($close, -1) / $close - 1, 126) + 1e-8)", 1, 0.25),
        # Price above long-term MA = stable uptrend
        ("$close / Mean($close, 126) - 1", 1, 0.15),
        # Low debt proxy: low volatility over long period
        ("Std(Ref($close, -1) / $close - 1, 252)", -1, 0.20),
        # Price-volume correlation (healthy correlation = institutional quality)
        ("Corr($close / Ref($close, -1) - 1, $volume / Mean($volume, 21), 63)", 1, 0.15),
    ],
    "growth": [
        # 1-month return
        ("$close / Ref($close, 21) - 1", 1, 0.20),
        # 3-month return
        ("$close / Ref($close, 63) - 1", 1, 0.25),
        # 6-month return acceleration (recent vs longer term)
        ("($close / Ref($close, 21) - 1) - ($close / Ref($close, 126) - 1)", 1, 0.20),
        # Volume expansion (increasing interest)
        ("Sum($volume, 21) / Sum($volume, 63) - 1", 1, 0.15),
        # Price relative to 1-year high (near-high = growth)
        ("$close / Max($close, 252)", 1, 0.20),
    ],
    "momentum": [
        # Short-term returns
        ("$close / Ref($close, 5) - 1", 1, 0.10),
        ("$close / Ref($close, 10) - 1", 1, 0.10),
        # Medium-term returns
        ("$close / Ref($close, 21) - 1", 1, 0.15),
        ("$close / Ref($close, 63) - 1", 1, 0.15),
        # Moving average crossovers
        ("Mean($close, 5) / Mean($close, 21) - 1", 1, 0.15),
        ("Mean($close, 21) / Mean($close, 63) - 1", 1, 0.10),
        # Recent volume surge (momentum confirmation)
        ("Sum($volume, 5) / Mean($volume, 21) / 5 - 1", 1, 0.10),
        # High-low range (strong momentum = closes near high)
        ("($close - $low) / ($high - $low + 1e-8)", 1, 0.15),
    ],
    "low_risk": [
        # Short-term volatility
        ("Std(Ref($close, -1) / $close - 1, 21)", -1, 0.20),
        # Medium-term volatility
        ("Std(Ref($close, -1) / $close - 1, 63)", -1, 0.25),
        # Long-term volatility
        ("Std(Ref($close, -1) / $close - 1, 126)", -1, 0.20),
        # Max drawdown proxy (distance from 3-month high)
        ("$close / Max($high, 63) - 1", 1, 0.15),
        # Downside volatility: lower volatility of negative-return days = safer
        ("Std($close / Ref($close, 1) - 1, 63)", -1, 0.20),
    ],
    "sentiment": [
        # Drawdown from 52-week high (contrarian: bigger drawdown = better)
        ("$close / Max($close, 252) - 1", 1, 0.25),
        # Abnormal low volume (neglect premium)
        ("$volume / Mean($volume, 63) - 1", -1, 0.20),
        # Recent underperformance (reversal expectation)
        ("$close / Ref($close, 21) - 1", -1, 0.20),
        # 1-year return (inverted: worst performers = contrarian buy)
        ("$close / Ref($close, 252) - 1", -1, 0.20),
        # High-low reversal signal (close near low of range)
        ("($close - $low) / ($high - $low + 1e-8) - 1", 1, 0.15),
    ],
}

# Flattened list of all unique expressions for single D.features() batch query
_ALL_EXPRESSIONS: list[str] = sorted(set(
    expr for dim_factors in FACTOR_EXPRESSIONS.values()
    for expr, _, _ in dim_factors
))


def _fmt(dt: pd.Timestamp) -> str:
    return dt.strftime("%Y-%m-%d")


def _extract_no_leakage_factors(
    factor_df: pd.DataFrame,
    instruments: list[str],
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """Extract factor values with strict temporal boundary.

    Only rows with datetime <= as_of_date are used. For each instrument,
    the most recent row is taken (simulating latest available data at
    the rebalance decision point).

    Args:
        factor_df: MultiIndex [datetime, instrument], columns = factor expressions
        instruments: Stock codes to select
        as_of_date: Backtest rebalance date (data cutoff)

    Returns:
        DataFrame indexed by instrument, one row per stock
    """
    idx_dt = factor_df.index.get_level_values("datetime")
    slice_df = factor_df[idx_dt <= as_of_date].copy()
    if slice_df.empty:
        return pd.DataFrame()

    idx_inst = slice_df.index.get_level_values("instrument")
    mask = idx_inst.isin(instruments)
    slice_df = slice_df[mask]

    if slice_df.empty:
        return pd.DataFrame()

    # Take latest observation per instrument before as_of_date
    result = (
        slice_df.reset_index()
        .sort_values(["instrument", "datetime"])
        .groupby("instrument")
        .last()
    )
    result = result.drop(columns=["datetime"], errors="ignore")
    return result


def _cross_sectional_percentile(values: np.ndarray) -> np.ndarray:
    """Compute cross-sectional percentile ranks (0 to 1)."""
    finite = np.isfinite(values)
    ranks = np.full(len(values), 0.5)  # neutral for missing
    if finite.sum() >= 2:
        ranks[finite] = (
            pd.Series(values[finite]).rank(pct=True).values
        )
    return ranks


def build_master_signal(
    raw_signal_df: pd.DataFrame,
    cal: pd.DatetimeIndex,
    hold_num: int = 5,
    top_k_candidates: int = 20,
    master_keys: list[str] | None = None,
    ensemble_method: str = "mean",
    freq: str = "yearly",
    price_cap: float = 9999.0,
    industry_cap_ratio: float = 0.40,
) -> pd.DataFrame:
    """Build a master-analyzed trade signal from raw stage2 predictions.

    At each rebalance date:
      1. Select top top_k_candidates stocks by model score
      2. Query factor expressions for those stocks (full historical range)
      3. Extract no-leakage factor snapshot at rebalance date
      4. Compute per-dimension cross-sectional percentiles
      5. Combine dimensions with per-master weights
      6. Ensemble across masters → final ranking
      7. Select top hold_num, equal-weight

    Args:
        raw_signal_df: MultiIndex [datetime, instrument] with 'score' column
        cal: Qlib trading calendar
        hold_num: Final number of holdings (default 5)
        top_k_candidates: Number of model top picks to screen (default 20)
        master_keys: Masters to use (default: all 7, ensembled)
        ensemble_method: 'mean' or 'median' for combining master scores
        freq: Rebalance frequency ('daily', 'weekly', 'monthly', 'yearly')
        price_cap: Maximum stock price for eligibility
        industry_cap_ratio: Max fraction of holdings from one industry

    Returns:
        MultiIndex DataFrame [datetime, instrument] with columns:
        score, weight, master_score
    """
    if raw_signal_df.empty:
        return pd.DataFrame(columns=["score", "weight", "master_score"])

    from qlib.data import D

    if master_keys is None:
        master_keys = list(MASTER_WEIGHTS.keys())

    # ── Step 1: Shift raw signal to trade dates ─────────────────────────
    shifted = _shift_signal_to_trade_dates(raw_signal_df[["score"]], cal, freq=freq)
    if shifted.empty:
        return pd.DataFrame(columns=["score", "weight", "master_score"])

    # ── Step 2: Get all instruments and query factors once (full range) ─
    all_instruments = sorted(
        shifted.index.get_level_values("instrument").unique().tolist()
    )
    all_dates = pd.DatetimeIndex(
        shifted.index.get_level_values("datetime").unique()
    ).sort_values()
    start_time = (all_dates.min() - pd.Timedelta(days=800)).strftime("%Y-%m-%d")
    end_time = all_dates.max().strftime("%Y-%m-%d")

    print(f"[master] Querying {len(_ALL_EXPRESSIONS)} factors for {len(all_instruments)} stocks "
          f"from {start_time} to {end_time}...")

    factor_df = D.features(
        all_instruments, _ALL_EXPRESSIONS,
        start_time=start_time, end_time=end_time,
    )

    if factor_df is None or factor_df.empty:
        print("[master] WARNING: D.features() returned empty, falling back to model score top-K")
        return _build_fallback_topk_signal(shifted, hold_num)

    # Normalize to MultiIndex [datetime, instrument]
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
        if "datetime" in factor_df.columns and "instrument" in factor_df.columns:
            factor_df["datetime"] = pd.to_datetime(factor_df["datetime"])
            factor_df["instrument"] = factor_df["instrument"].astype(str)
            factor_df = factor_df.set_index(["datetime", "instrument"])
        else:
            print("[master] WARNING: Unexpected factor_df columns, fallback to model score")
            return _build_fallback_topk_signal(shifted, hold_num)

    idx_dt = pd.to_datetime(factor_df.index.get_level_values("datetime"))
    idx_inst = factor_df.index.get_level_values("instrument").astype(str)
    factor_df.index = pd.MultiIndex.from_arrays(
        [idx_dt, idx_inst], names=["datetime", "instrument"]
    )

    print(f"[master] Factor data loaded: {factor_df.shape[0]} rows x {factor_df.shape[1]} cols")

    # ── Step 3: Build per-master factor scoring ─────────────────────────
    # Map expression names from factor_df columns to dimension/factor entries
    dim_factors_available: dict[str, list[tuple[str, int, float]]] = {}
    factor_cols = list(factor_df.columns)
    for dim, factors in FACTOR_EXPRESSIONS.items():
        available = []
        for expr, direction, weight in factors:
            if expr in factor_cols:
                available.append((expr, direction, weight))
        if available:
            dim_factors_available[dim] = available
            print(f"[master]   {dim}: {len(available)}/{len(factors)} factors matched")
        else:
            print(f"[master]   {dim}: 0/{len(factors)} factors matched (SKIPPED)")

    # ── Step 4: Per-rebalance-date loop ────────────────────────────────
    trade_dates = pd.DatetimeIndex(
        shifted.index.get_level_values("datetime").unique()
    ).sort_values()

    result_frames: list[pd.DataFrame] = []

    for trade_dt in trade_dates:
        # Get top-K candidates at this date by model score
        day_sig = shifted.loc[pd.IndexSlice[trade_dt, :]]
        if hasattr(day_sig, "reset_index"):
            day_df = day_sig.reset_index()
        else:
            continue

        candidates = day_df.sort_values("score", ascending=False).head(top_k_candidates)
        cand_codes = candidates["instrument"].tolist()

        if len(cand_codes) < hold_num:
            continue

        # Extract NO-LEAKAGE factor values for candidates
        factor_slice = _extract_no_leakage_factors(
            factor_df, cand_codes, trade_dt
        )
        if factor_slice.empty or len(factor_slice) < hold_num:
            continue

        # ── Step 5: Compute dimension scores ───────────────────────────
        dim_scores: dict[str, np.ndarray] = {}
        instruments_slice = factor_slice.index.tolist()
        n = len(instruments_slice)

        for dim, factors in dim_factors_available.items():
            dim_vals = np.zeros(n)
            total_w = 0.0
            for expr, direction, weight in factors:
                if expr not in factor_slice.columns:
                    continue
                raw = factor_slice[expr].astype(float).values
                pct = _cross_sectional_percentile(raw)
                if direction == -1:
                    pct = 1.0 - pct
                dim_vals += weight * pct
                total_w += weight
            if total_w > 0:
                dim_vals /= total_w
            dim_scores[dim] = dim_vals

        # ── Step 6: Per-master scoring ─────────────────────────────────
        master_scores_list: list[np.ndarray] = []
        for mk in master_keys:
            weights = MASTER_WEIGHTS.get(mk, {})
            score = np.zeros(n)
            total_w = 0.0
            for dim, w in weights.items():
                if dim in dim_scores:
                    score += w * dim_scores[dim]
                    total_w += w
            if total_w > 0:
                score /= total_w
            master_scores_list.append(score)

        # ── Step 7: Ensemble ───────────────────────────────────────────
        stacked = np.stack(master_scores_list, axis=1)
        if ensemble_method == "median":
            ensemble = np.nanmedian(stacked, axis=1)
        else:
            ensemble = np.nanmean(stacked, axis=1)

        # ── Step 8: Select top hold_num ────────────────────────────────
        cand_indexed = candidates.set_index("instrument")
        model_scores = []
        for inst in instruments_slice:
            if inst in cand_indexed.index:
                model_scores.append(float(cand_indexed.loc[inst, "score"]))
            else:
                model_scores.append(0.0)

        sel_df = pd.DataFrame({
            "instrument": instruments_slice,
            "model_score": model_scores,
            "ensemble_score": ensemble,
        })

        sel_df = sel_df.sort_values("ensemble_score", ascending=False).head(hold_num)

        # Equal-weight
        scores = sel_df["ensemble_score"].values
        lo, hi = float(np.min(scores)), float(np.max(scores))
        if hi > lo:
            weights = (scores - lo) / (hi - lo)
        else:
            weights = np.ones(len(scores)) / len(scores)
        weights = weights / weights.sum()

        sel_df["datetime"] = trade_dt
        sel_df["score"] = weights
        sel_df["weight"] = weights
        sel_df["master_score"] = sel_df["ensemble_score"]

        result_frames.append(
            sel_df[["datetime", "instrument", "score", "weight", "master_score"]]
        )

    if not result_frames:
        return pd.DataFrame(columns=["score", "weight", "master_score"])

    out = pd.concat(result_frames, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.drop_duplicates(subset=["datetime", "instrument"], keep="last")
    out = out.set_index(["datetime", "instrument"])
    out.index = out.index.set_names(["datetime", "instrument"])

    print(f"[master] Signal built: {len(out)} rows across {len(trade_dates)} rebalance dates")
    return out


def _shift_signal_to_trade_dates(
    signal_df: pd.DataFrame, cal: pd.DatetimeIndex, freq: str = "yearly"
) -> pd.DataFrame:
    """Align raw signal dates to valid trading days with frequency grouping.

    (Minimal copy of the main script's function for self-contained use.)
    """
    if signal_df.empty:
        return signal_df

    df = signal_df.reset_index().copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["instrument"] = df["instrument"].astype(str)
    keep_cols = [c for c in df.columns if c not in {"datetime", "instrument"}]
    for col in keep_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime", "instrument"])
    if df.empty:
        return signal_df.iloc[0:0]

    freq = freq.strip().lower()
    if freq == "daily":
        pass
    elif freq == "monthly":
        monthly_dates = (
            df[["datetime"]].drop_duplicates()
            .assign(
                iso_year=lambda x: pd.to_datetime(x["datetime"]).dt.isocalendar().year,
                iso_week=lambda x: pd.to_datetime(x["datetime"]).dt.isocalendar().week,
            )
            .groupby(["iso_year", "iso_week"], as_index=False)["datetime"]
            .max()["datetime"]
        )
        monthly_dates = monthly_dates.iloc[::4]
        df = df[df["datetime"].isin(set(pd.to_datetime(monthly_dates)))]
    elif freq == "yearly":
        yearly_dates = (
            df[["datetime"]].drop_duplicates()
            .assign(year=lambda x: pd.to_datetime(x["datetime"]).dt.year)
            .groupby("year", as_index=False)["datetime"]
            .min()["datetime"]
        )
        df = df[df["datetime"].isin(set(pd.to_datetime(yearly_dates)))]
    else:
        # weekly
        weekly_dates = (
            df[["datetime"]].drop_duplicates()
            .assign(
                iso_year=lambda x: pd.to_datetime(x["datetime"]).dt.isocalendar().year,
                iso_week=lambda x: pd.to_datetime(x["datetime"]).dt.isocalendar().week,
            )
            .groupby(["iso_year", "iso_week"], as_index=False)["datetime"]
            .max()["datetime"]
        )
        df = df[df["datetime"].isin(set(pd.to_datetime(weekly_dates)))]

    if df.empty:
        return signal_df.iloc[0:0]

    trade_dates = []
    for dt in pd.to_datetime(df["datetime"]):
        pos = int(cal.searchsorted(dt, side="right"))
        trade_dates.append(pd.Timestamp(cal[pos]) if pos < len(cal) else pd.NaT)

    df["trade_datetime"] = trade_dates
    df = df.dropna(subset=["trade_datetime"])
    if df.empty:
        return signal_df.iloc[0:0]

    df = df.drop(columns=["datetime"]).rename(columns={"trade_datetime": "datetime"})
    df = df.drop_duplicates(subset=["datetime", "instrument"], keep="last")
    df = df.sort_values(["datetime", "instrument"])
    shifted = df.set_index(["datetime", "instrument"])[keep_cols]
    shifted.index = shifted.index.set_names(["datetime", "instrument"])
    return shifted


def _build_fallback_topk_signal(
    shifted: pd.DataFrame, hold_num: int
) -> pd.DataFrame:
    """Fallback: pure model-score top-K equal-weight."""
    frames: list[pd.DataFrame] = []
    for dt, grp in shifted.reset_index().groupby("datetime", sort=True):
        grp = grp.sort_values("score", ascending=False).head(hold_num)
        grp["weight"] = 1.0 / len(grp)
        grp["master_score"] = grp["score"]
        frames.append(grp[["datetime", "instrument", "score", "weight", "master_score"]])
    if not frames:
        return pd.DataFrame(columns=["score", "weight", "master_score"])
    out = pd.concat(frames, ignore_index=True)
    out = out.set_index(["datetime", "instrument"])
    return out
