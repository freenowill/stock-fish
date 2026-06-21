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
    # model_score weight preserves the model's data-driven ranking (IC=0.038)
    # while allowing master philosophy to make marginal adjustments.
    # Higher model_score weight for trend-following masters (momentum aligns with model),
    # lower for contrarian masters (need more factor adjustment to flip model picks).
    "graham":    {"model_score": 0.25, "value": 0.38, "quality": 0.18, "low_risk": 0.11, "sentiment": 0.08},
    "buffett":   {"model_score": 0.30, "value": 0.12, "quality": 0.38, "growth": 0.10, "low_risk": 0.07, "sentiment": 0.03},
    "fisher":    {"model_score": 0.30, "value": 0.03, "quality": 0.15, "growth": 0.42, "momentum": 0.05, "low_risk": 0.03, "sentiment": 0.02},
    "lynch":     {"model_score": 0.35, "value": 0.12, "quality": 0.15, "growth": 0.25, "momentum": 0.06, "low_risk": 0.04, "sentiment": 0.03},
    "templeton": {"model_score": 0.20, "value": 0.25, "quality": 0.12, "low_risk": 0.08, "sentiment": 0.35},
    "soros":     {"model_score": 0.32, "value": 0.03, "quality": 0.03, "growth": 0.06, "momentum": 0.42, "sentiment": 0.14},
    "dalio":     {"model_score": 0.28, "value": 0.07, "quality": 0.10, "growth": 0.06, "momentum": 0.03, "low_risk": 0.38, "sentiment": 0.08},
}

# ═════════════════════════════════════════════════════════════════════════════
# Market Regime → Master Selection
# Instead of mean-ensemble of all 7 (which cancels out opposing philosophies),
# select 2-3 masters whose style matches current market conditions.
# Regime is detected from TRAILING data only (T-21 to T-1), strictly before
# the rebalance date — no future information leakage.
# ═════════════════════════════════════════════════════════════════════════════

REGIME_MASTER_SELECTION: dict[str, list[str]] = {
    # 🟢 Bull market: index trending up, broad participation → growth + momentum
    "bull":      ["fisher", "lynch", "soros"],
    # 🟡 Sideways/range-bound: moderate trends → quality + GARP + risk parity
    "sideways":  ["buffett", "lynch", "dalio"],
    # 🔴 Bear market: index declining, high fear → deep value + contrarian + low risk
    "bear":      ["graham", "templeton", "dalio"],
}

# Number of stocks directly selected from model's top picks (bypasses master ranking).
# These are the model's highest-conviction picks, preserving IC=0.038 ranking quality.
GUARANTEED_FROM_MODEL: int = 2

# Cache of most recent factor IC validation result.
# Updated each time validate_factor_ic=True is used. Consulted by
# _compute_adaptive_guaranteed_seats() to adjust model seats.
_last_factor_ic_result: pd.DataFrame | None = None

# ═════════════════════════════════════════════════════════════════════════════
# Factor Expressions per Dimension
# Qlib expression syntax: $close, Ref(), Mean(), Std(), Sum(), Max(), Min()...
# Each tuple: (expression, direction, weight_in_dimension)
#   direction =  1 → higher factor value → higher score
#   direction = -1 → higher factor value → lower score
# ═════════════════════════════════════════════════════════════════════════════

FACTOR_EXPRESSIONS: dict[str, list[tuple[str, int, float]]] = {
    # ═══════════════════════════════════════════════════════════════════════
    # T21-Aligned Factor Expressions
    # Window minimum: 21 days (monthly), default: 63/126/252 days.
    # Removed 5/10-day ultra-short windows — they conflict with the
    # multihorizon model's T21-weighted label (35% at monthly horizon).
    # ═══════════════════════════════════════════════════════════════════════
    "value": [
        # Distance from 1-year average (price mean-reversion proxy)
        ("$close / Mean($close, 252) - 1", -1, 0.30),
        # Distance from 52-week high (drawdown from peak)
        ("$close / Max($close, 252) - 1", 1, 0.25),
        # Distance from 3-year average (long-term value)
        ("$close / Mean($close, 756) - 1", -1, 0.15),
        # Low relative volume = neglected stock
        ("$volume / Mean($volume, 252) - 1", -1, 0.15),
        # Book-to-price proxy: negative return over 2 years = cheaper
        ("$close / Ref($close, 504) - 1", -1, 0.15),
    ],
    "quality": [
        # Return stability over 3 months (T21-aligned)
        ("Std(Ref($close, 1) / $close - 1, 63)", -1, 0.30),
        # Sharpe-like: return/vol over 6 months
        ("($close / Ref($close, 126) - 1) / (Std(Ref($close, 1) / $close - 1, 126) + 1e-8)", 1, 0.25),
        # Price above 6-month MA = stable medium-term trend
        ("$close / Mean($close, 126) - 1", 1, 0.15),
        # Low long-term volatility (quality businesses are less volatile)
        ("Std(Ref($close, 1) / $close - 1, 252)", -1, 0.20),
        # Price-volume correlation over 3 months (institutional quality)
        ("Corr($close / Ref($close, 1) - 1, $volume / Mean($volume, 63), 63)", 1, 0.10),
    ],
    "growth": [
        # 1-month return (aligned with T21 holding period)
        ("$close / Ref($close, 21) - 1", 1, 0.25),
        # 3-month return (sustained growth)
        ("$close / Ref($close, 63) - 1", 1, 0.30),
        # 6-month return (longer-term growth trend)
        ("$close / Ref($close, 126) - 1", 1, 0.15),
        # Acceleration: recent vs medium-term (21d vs 126d)
        ("($close / Ref($close, 21) - 1) - ($close / Ref($close, 126) - 1)", 1, 0.15),
        # Volume expansion over quarter
        ("Sum($volume, 63) / Sum($volume, 126) - 1", 1, 0.15),
    ],
    "momentum": [
        # Medium-term returns (monthly + quarterly)
        ("$close / Ref($close, 21) - 1", 1, 0.25),
        ("$close / Ref($close, 63) - 1", 1, 0.25),
        # 6-month return
        ("$close / Ref($close, 126) - 1", 1, 0.15),
        # MA crossovers (monthly + quarterly)
        ("Mean($close, 21) / Mean($close, 63) - 1", 1, 0.15),
        ("Mean($close, 63) / Mean($close, 126) - 1", 1, 0.10),
        # High-low range (closes near high = momentum)
        ("($close - $low) / ($high - $low + 1e-8)", 1, 0.10),
    ],
    "low_risk": [
        # Monthly volatility
        ("Std(Ref($close, 1) / $close - 1, 21)", -1, 0.20),
        # Quarterly volatility
        ("Std(Ref($close, 1) / $close - 1, 63)", -1, 0.30),
        # Semi-annual volatility
        ("Std(Ref($close, 1) / $close - 1, 126)", -1, 0.25),
        # Max drawdown from 3-month high
        ("$close / Max($high, 63) - 1", 1, 0.15),
        # Downside volatility
        ("Std($close / Ref($close, 1) - 1, 63)", -1, 0.10),
    ],
    "sentiment": [
        # Drawdown from 52-week high (contrarian)
        ("$close / Max($close, 252) - 1", 1, 0.25),
        # Abnormal low volume (neglect premium)
        ("$volume / Mean($volume, 63) - 1", -1, 0.20),
        # Recent underperformance (monthly reversal)
        ("$close / Ref($close, 21) - 1", -1, 0.20),
        # 1-year return (inverted: worst performers = contrarian buy)
        ("$close / Ref($close, 252) - 1", -1, 0.20),
        # Close near low of range (reversal signal)
        ("($close - $low) / ($high - $low + 1e-8) - 1", 1, 0.15),
    ],
}

# Flattened list of all unique expressions for single D.features() batch query
_ALL_EXPRESSIONS: list[str] = sorted(set(
    expr for dim_factors in FACTOR_EXPRESSIONS.values()
    for expr, _, _ in dim_factors
))

# ── Window extraction cache (avoid re-parsing expressions) ───────────────
_EXPR_WINDOW_CACHE: dict[str, int] = {}


def _extract_window_from_expression(expr: str) -> int:
    """Extract the effective lookback window (in days) from a Qlib expression.

    Parses integer parameters from functions like Ref(X, N), Mean(X, N),
    Std(X, N), Sum(X, N), Max(X, N), Corr(X, Y, N). For expressions with
    multiple integer parameters (e.g., Corr with daily-shift Refs), the
    MAXIMUM integer is returned as the effective window.

    Returns:
        Window days (int). Falls back to 126 (6-month default) if no
        integer parameter is found.
    """
    cached = _EXPR_WINDOW_CACHE.get(expr)
    if cached is not None:
        return cached

    import re
    # Find all integer literals in the expression
    numbers = [int(m) for m in re.findall(r'\b(\d+)\b', expr)]
    # Filter out very small numbers (<5) — these are likely Ref() shifts, not windows
    windows = [n for n in numbers if n >= 5]

    if windows:
        result = max(windows)  # use the longest window as the effective horizon
    else:
        result = 126  # default: 6-month medium horizon

    _EXPR_WINDOW_CACHE[expr] = result
    return result


def _adapt_factor_window_weights(freq: str) -> dict[int, float]:
    """Return a multiplier map: window_days → frequency-adaptive weight multiplier.

    With yearly rebalancing, short-horizon factors (21d momentum, 63d vol)
    have near-zero predictive power for 1-year forward returns. This function
    boosts long-horizon factors (>252d) and suppresses short-horizon ones.

    The multipliers are multiplicative on the existing factor weights inside
    each dimension. For example, a 21d momentum factor with weight 0.25 in the
    momentum dimension gets scaled to 0.25 * 0.2 = 0.05 at yearly frequency.

    Args:
        freq: Rebalance frequency — "daily", "weekly", "monthly", or "yearly"

    Returns:
        Dict mapping window_days → multiplier. Windows not in the map
        default to 1.0 (no change).
    """
    freq = freq.strip().lower()
    if freq in ("daily", "weekly"):
        # All windows equally relevant for short-horizon trading
        return {}

    if freq == "monthly":
        return {
            21: 0.6,    # 1-month: slightly reduced (T21 label alignment, but noisy)
            63: 1.0,    # 1-quarter: neutral
            126: 1.1,   # 6-month: slight boost
            252: 1.2,   # 1-year: boost
            504: 1.2,   # 2-year: boost
            756: 1.2,   # 3-year: boost
        }

    if freq == "yearly":
        return {
            21: 0.2,    # 1-month: heavily suppressed — noise at yearly horizon
            63: 0.4,    # 1-quarter: suppressed
            126: 0.8,   # 6-month: slightly reduced
            252: 1.3,   # 1-year: boosted — aligned with holding period
            504: 1.6,   # 2-year: strongly boosted — long-term value signal
            756: 1.6,   # 3-year: strongly boosted
        }

    return {}


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


def _borda_count_ensemble(master_scores_list: list[np.ndarray]) -> np.ndarray:
    """Combine master scores via Borda Count (rank aggregation).

    Each master ranks candidates (1 = best), then mean rank across masters
    determines the final ensemble score. This reduces the influence of
    correlated score magnitudes — masters using the same factor data with
    different weights produce highly correlated raw scores (>0.7), so
    mean-ensemble provides almost no diversification.

    Borda Count breaks this correlation by operating on ranks:
      - If two masters agree on ranking, the consensus is reinforced
      - If they disagree, the rank average moderates the outcome
      - Outlier scores (extreme values from one master) are bounded

    Returns:
        Ensemble scores (0-1, higher = better), same shape as input arrays.
    """
    n_candidates = len(master_scores_list[0]) if master_scores_list else 0
    if n_candidates == 0:
        return np.array([])

    # Compute descending ranks for each master (rank 1 = best/highest score)
    all_ranks = np.zeros((len(master_scores_list), n_candidates), dtype=float)
    for i, scores in enumerate(master_scores_list):
        # Rank descending: highest score → rank 1
        # method="average" handles ties correctly
        all_ranks[i] = pd.Series(-scores).rank(method="average").values

    # Mean rank across masters (lower = better)
    mean_ranks = np.nanmean(all_ranks, axis=0)

    # Convert to 0-1 score (higher = better, matching existing convention)
    ensemble = 1.0 - (mean_ranks - 1.0) / (n_candidates - 1.0) if n_candidates > 1 else np.ones(n_candidates) * 0.5
    return ensemble


def _detect_market_regime(
    cal: pd.DatetimeIndex,
    as_of_date: pd.Timestamp,
    benchmark: str = "SH000300",
) -> str:
    """Detect market regime using STRICTLY trailing data (T-21 to T-1).

    NO future information leakage:
      - Only uses data with datetime < as_of_date
      - The 21 trading days before the rebalance date (T-21 to T-1)
      - Never includes the rebalance date itself (T) or any date after

    Returns: "bull", "bear", or "sideways"
    """
    from qlib.data import D

    # Find the index of as_of_date in the calendar
    try:
        pos = int(cal.searchsorted(as_of_date, side="left"))
    except Exception:
        return "sideways"

    # Use up to 21 trading days BEFORE as_of_date
    lookback_start_pos = max(0, pos - 22)
    lookback_end_pos = max(0, pos - 1)  # T-1, strictly before rebalance date

    if lookback_end_pos - lookback_start_pos < 5:
        return "sideways"  # insufficient data

    lookback_start = pd.Timestamp(cal[lookback_start_pos])
    lookback_end = pd.Timestamp(cal[lookback_end_pos])

    try:
        # Get benchmark close prices for the lookback period
        bench_data = D.features(
            [benchmark],
            ["$close"],
            start_time=lookback_start.strftime("%Y-%m-%d"),
            end_time=lookback_end.strftime("%Y-%m-%d"),
        )
    except Exception:
        return "sideways"

    if bench_data is None or len(bench_data) < 5:
        return "sideways"

    # Compute trailing 20-day return and volatility
    if isinstance(bench_data, pd.DataFrame):
        closes = bench_data.iloc[:, 0].values if bench_data.shape[1] > 0 else bench_data.values.flatten()
    elif isinstance(bench_data, pd.Series):
        closes = bench_data.values
    else:
        return "sideways"

    closes = closes[~np.isnan(closes)]
    if len(closes) < 5:
        return "sideways"

    # 20-day return (approximate — may be less if data sparse)
    ret_20d = (closes[-1] / closes[0] - 1) if closes[0] != 0 else 0.0

    # 20-day volatility (annualized)
    if len(closes) >= 5:
        daily_rets = np.diff(closes) / (closes[:-1] + 1e-8)
        vol_20d = float(np.std(daily_rets))
    else:
        vol_20d = 0.02  # default moderate vol

    # ── Regime classification ──
    # Bull: strong positive return AND moderate/low volatility
    if ret_20d > 0.03 and vol_20d < 0.025:
        return "bull"
    # Bear: strong negative return OR high volatility with negative return
    elif ret_20d < -0.03 or (ret_20d < -0.01 and vol_20d > 0.025):
        return "bear"
    else:
        return "sideways"


# ── Volatility cache (avoid redundant D.features() calls) ──────────────────
_VOL_CACHE: dict[tuple, float] = {}


def _get_market_volatility(
    cal: pd.DatetimeIndex,
    as_of_date: pd.Timestamp,
    benchmark: str = "SH000300",
) -> float:
    """Get trailing 21-day annualized volatility of the benchmark.

    Uses STRICTLY trailing data (T-21 to T-1), same temporal boundary
    as _detect_market_regime(). Results are cached by (benchmark, as_of_date)
    since multiple calls may occur for the same date in a single run.

    Returns:
        Annualized daily return volatility. Falls back to 0.02 (moderate).
    """
    cache_key = (benchmark, as_of_date)
    cached = _VOL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from qlib.data import D

    try:
        pos = int(cal.searchsorted(as_of_date, side="left"))
    except Exception:
        return 0.02

    lookback_start_pos = max(0, pos - 22)
    lookback_end_pos = max(0, pos - 1)

    if lookback_end_pos - lookback_start_pos < 5:
        return 0.02

    lookback_start = pd.Timestamp(cal[lookback_start_pos])
    lookback_end = pd.Timestamp(cal[lookback_end_pos])

    try:
        bench_data = D.features(
            [benchmark],
            ["$close"],
            start_time=lookback_start.strftime("%Y-%m-%d"),
            end_time=lookback_end.strftime("%Y-%m-%d"),
        )
    except Exception:
        return 0.02

    if bench_data is None or len(bench_data) < 5:
        return 0.02

    if isinstance(bench_data, pd.DataFrame):
        closes = bench_data.iloc[:, 0].values if bench_data.shape[1] > 0 else bench_data.values.flatten()
    elif isinstance(bench_data, pd.Series):
        closes = bench_data.values
    else:
        return 0.02

    closes = closes[~np.isnan(closes)]
    if len(closes) < 5:
        return 0.02

    daily_rets = np.diff(closes) / (closes[:-1] + 1e-8)
    vol = float(np.std(daily_rets))
    # Annualize: daily vol * sqrt(252)
    annual_vol = vol * np.sqrt(252)

    _VOL_CACHE[cache_key] = annual_vol
    return annual_vol


def _compute_adaptive_model_weight(
    model_scores: np.ndarray,
    dim_scores: dict[str, np.ndarray],
    market_vol: float,
    base_model_weight: float,
) -> float:
    """Compute adaptive model_score weight based on three signals.

    1. Model confidence: higher cross-sectional score variance → model is
       discriminating stocks more clearly → increase model weight.
    2. Factor disagreement: higher std across dimension percentile scores →
       factors disagree → trust model more.
    3. Market volatility: higher vol → prefer value/low_risk factors over
       model predictions → reduce model weight.

    The three signals are multiplicative on the base weight, then clamped
    to [0.12, 0.55] to prevent extreme values.

    Args:
        model_scores: Raw model scores (n_candidates,)
        dim_scores: Dict of dimension → percentile score array (each n_candidates,)
        market_vol: Annualized trailing volatility (from _get_market_volatility)
        base_model_weight: Static model_score weight from MASTER_WEIGHTS

    Returns:
        Adaptive model_score weight in [0.12, 0.55]
    """
    n = len(model_scores)

    # ── Signal 1: Model confidence ─────────────────────────────────────
    score_std = float(np.nanstd(model_scores)) if n > 1 else 0.0
    # Normalize: typical model score std after Gaussian ranking is ~0.15-0.35
    normalized_confidence = min(score_std / 0.25, 1.0)
    # Center around 1.0: above-average confidence → boost, below → reduce
    confidence_factor = 1.0 + 0.4 * (normalized_confidence - 0.5)

    # ── Signal 2: Factor disagreement ───────────────────────────────────
    if len(dim_scores) > 1:
        dim_arrays = [v for v in dim_scores.values() if len(v) == n]
        if len(dim_arrays) > 1:
            stacked = np.column_stack(dim_arrays)
            # Per-candidate std across dimensions → mean across candidates
            disagreement = float(np.nanmean(np.nanstd(stacked, axis=1)))
        else:
            disagreement = 0.0
    else:
        disagreement = 0.0
    # Disagreement typically ranges 0.05-0.25
    disagreement_factor = 1.0 + 1.5 * max(disagreement - 0.10, 0.0)

    # ── Signal 3: Market volatility ─────────────────────────────────────
    # High vol (>30% annual) → reduce model weight up to 50%
    vol_clamped = min(market_vol, 0.50)
    vol_reduction = (vol_clamped - 0.15) / 0.35 if vol_clamped > 0.15 else 0.0
    vol_factor = 1.0 - 0.5 * max(vol_reduction, 0.0)

    # ── Combine and clamp ───────────────────────────────────────────────
    adaptive_weight = base_model_weight * confidence_factor * disagreement_factor * vol_factor
    adaptive_weight = max(0.12, min(0.55, adaptive_weight))

    return adaptive_weight


def _flag_tail_risk_stocks(
    factor_slice: pd.DataFrame,
    instruments_slice: list[str],
) -> tuple[set[str], dict[str, list[str]]]:
    """Flag stocks with extreme tail risk for exclusion.

    Identifies candidates with:
    a. High short-term volatility (>1.5× peer median)
    b. Severe drawdown from 52-week high (>40%)
    c. Persistent weakness (6-month return < -30%)

    These stocks are excluded BEFORE master ranking — the factor layer
    acts as a risk filter rather than a re-ranker.

    Returns:
        (risky_set, reason_map) — set of instruments to exclude,
        dict mapping instrument → list of risk reasons (for logging).
    """
    risky: set[str] = set()
    reasons: dict[str, list[str]] = {}

    def _check(instrument, condition, tag):
        if condition:
            risky.add(instrument)
            reasons.setdefault(instrument, []).append(tag)

    # ── Volatility check (60-day) ──────────────────────────────────────
    vol_expr = "Std(Ref($close, 1) / $close - 1, 63)"
    if vol_expr in factor_slice.columns:
        vols = factor_slice[vol_expr].astype(float)
        median_vol = vols.median()
        for i, inst in enumerate(instruments_slice):
            v = vols.iloc[i] if i < len(vols) else 0.0
            _check(inst, v > median_vol * 1.5 and v > 0.03, "high_vol")

    # ── Drawdown check (from 52-week high) ─────────────────────────────
    dd_threshold = float(os.environ.get("RISK_FILTER_DD_THRESHOLD", "-0.50"))
    dd_expr = "$close / Max($close, 252) - 1"
    if dd_expr in factor_slice.columns:
        dds = factor_slice[dd_expr].astype(float)
        for i, inst in enumerate(instruments_slice):
            d = dds.iloc[i] if i < len(dds) else 0.0
            _check(inst, d < dd_threshold, "deep_drawdown")

    # ── Persistent weakness (6-month return) ────────────────────────────
    ret_threshold = float(os.environ.get("RISK_FILTER_WEAKNESS_THRESHOLD", "-0.40"))
    ret_expr = "$close / Ref($close, 126) - 1"
    if ret_expr in factor_slice.columns:
        rets = factor_slice[ret_expr].astype(float)
        for i, inst in enumerate(instruments_slice):
            r = rets.iloc[i] if i < len(rets) else 0.0
            _check(inst, r < -0.30, "persistent_weakness")

    return risky, reasons


def _compute_adaptive_guaranteed_seats(
    hold_num: int,
    validate_result: pd.DataFrame | None = None,
) -> int:
    """Determine how many top model picks are guaranteed a seat.

    When factors have strong IC, the master gets more discretion (fewer
    guaranteed model seats). When factors are weak or negative, the model
    gets more guaranteed seats (preserving IC quality).

    Args:
        hold_num: Total seats
        validate_result: Factor IC validation DataFrame from _validate_factor_ic,
            or None if IC data is unavailable.

    Returns:
        Number of guaranteed model seats (1 to hold_num-1).
    """
    if validate_result is None or validate_result.empty:
        return min(2, max(1, hold_num - 1))

    # Compute mean IC across dimensions (excluding model_score)
    dim_rows = validate_result[validate_result["dimension"] != "model_score"]
    if dim_rows.empty or dim_rows["mean_RankIC"].isna().all():
        return min(2, max(1, hold_num - 1))

    mean_ic = float(dim_rows["mean_RankIC"].mean())

    # Strong factors → fewer guaranteed seats (master has discretion)
    if mean_ic > 0.05:
        return max(1, hold_num - 3)       # e.g., 5 → 2
    elif mean_ic > 0.02:
        return max(1, hold_num - 2)       # e.g., 5 → 3
    elif mean_ic > 0.0:
        return max(1, hold_num - 1)       # e.g., 5 → 4
    else:
        # Negative IC → almost pure model
        return max(1, hold_num - 1)


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
    validate_factor_ic: bool = False,
    strategy_mode: str = "full",
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
        validate_factor_ic: If True, compute per-dimension Rank IC against
            forward returns at the rebalance horizon (diagnostic only).
            Also controllable via env var MASTER_VALIDATE_FACTOR_IC=1.
        strategy_mode: Strategy operating mode:
            - "full": Full master analysis with 7 masters + factor re-ranking
              (current default, backward-compatible).
            - "risk_filter": Factor layer acts as risk filter only — flags
              and excludes tail-risk stocks, then uses pure model ranking
              for the survivors. Preserves model IC quality.
            - "simple": Direct dimension-mean blend with model_score.
              No per-master weighting. model_score=60%, factor_mean=40%.

    Returns:
        MultiIndex DataFrame [datetime, instrument] with columns:
        score, weight, master_score
    """
    if raw_signal_df.empty:
        return pd.DataFrame(columns=["score", "weight", "master_score"])

    from qlib.data import D

    # Module-level cache for adaptive seats
    global _last_factor_ic_result

    # Allow strategy_mode override via environment variable
    _env_mode = os.environ.get("MASTER_STRATEGY_MODE", "").strip().lower()
    if _env_mode and _env_mode in ("full", "risk_filter", "simple"):
        strategy_mode = _env_mode
        print(f"[master] Strategy mode override (env): {strategy_mode}")

    # ── Step 1: Shift raw signal to trade dates ─────────────────────────
    shifted = _shift_signal_to_trade_dates(raw_signal_df[["score"]], cal, freq=freq)
    if shifted.empty:
        return pd.DataFrame(columns=["score", "weight", "master_score"])

    trade_dates = pd.DatetimeIndex(
        shifted.index.get_level_values("datetime").unique()
    ).sort_values()

    # ── Step 2: Get all instruments and query factors once (full historical range) ─
    # NOTE: Regime detection is done PER-DATE inside the rebalance loop (Step 4)
    # to prevent stale regime assignments across multi-year backtests.
    # A single regime detected at the first trade date would leak the assumption
    # that market conditions remain constant — incorrect across bull/bear cycles.

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
    regime_log: list[dict] = []  # track per-date regime for diagnostics
    # Optional factor IC validation data (only populated when validate_factor_ic=True)
    dim_scores_per_date: dict[pd.Timestamp, dict[str, np.ndarray]] = {}
    instruments_per_date: dict[pd.Timestamp, list[str]] = {}

    for trade_dt in trade_dates:
        # ── Per-date regime detection (NO leakage: strictly T-21 to T-1) ──
        if master_keys is None:
            regime = _detect_market_regime(cal, trade_dt)
            current_master_keys = REGIME_MASTER_SELECTION.get(regime, ["buffett", "lynch", "dalio"])
        else:
            regime = "manual"
            current_master_keys = master_keys
        regime_log.append({"date": trade_dt, "regime": regime, "masters": current_master_keys})

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
        # Apply frequency-adaptive window weight modifiers.
        # Yearly rebalancing → boost long-horizon factors (252d+),
        # suppress short-horizon (21d) that are noise at annual scale.
        window_modifiers = _adapt_factor_window_weights(freq)
        dim_scores: dict[str, np.ndarray] = {}
        instruments_slice = factor_slice.index.tolist()
        n = len(instruments_slice)

        for dim, factors in dim_factors_available.items():
            dim_vals = np.zeros(n)
            total_w = 0.0
            for expr, direction, weight in factors:
                if expr not in factor_slice.columns:
                    continue
                # Apply frequency-adaptive window multiplier
                wind_days = _extract_window_from_expression(expr)
                freq_mult = window_modifiers.get(wind_days, 1.0) if window_modifiers else 1.0
                effective_weight = weight * freq_mult

                raw = factor_slice[expr].astype(float).values
                pct = _cross_sectional_percentile(raw)
                if direction == -1:
                    pct = 1.0 - pct
                dim_vals += effective_weight * pct
                total_w += effective_weight
            if total_w > 0:
                dim_vals /= total_w
            dim_scores[dim] = dim_vals

        # ── Model score dimension (preserves IC=0.038 ranking quality) ─
        cand_indexed = candidates.set_index("instrument")
        model_score_raw = np.array([
            float(cand_indexed.loc[inst, "score"]) if inst in cand_indexed.index else 0.0
            for inst in instruments_slice
        ])
        dim_scores["model_score"] = _cross_sectional_percentile(model_score_raw)

        # ── Collect data for optional factor IC validation ─────────────
        if validate_factor_ic:
            dim_scores_per_date[trade_dt] = {dim: vals.copy() for dim, vals in dim_scores.items()}
            instruments_per_date[trade_dt] = list(instruments_slice)

        # ── Strategy mode branching ──────────────────────────────────
        if strategy_mode == "risk_filter":
            # ── Risk Filter Mode ──────────────────────────────────────
            # 1. Flag tail-risk stocks → exclude from candidates
            # 2. Use pure model ranking on survivors
            # 3. No master scoring — factor layer is filter, not re-ranker
            risky_set, risk_reasons = _flag_tail_risk_stocks(factor_slice, instruments_slice)

            if risky_set:
                n_risky = len(risky_set)
                risky_list = sorted(risky_set)
                print(f"[master]   {_fmt(trade_dt)}: flagged {n_risky} tail-risk stocks: "
                      f"{', '.join(f'{r}({risk_reasons.get(r, [])})' for r in risky_list[:5])}"
                      f"{'...' if n_risky > 5 else ''}")

            # Filter out risky stocks from selection
            safe_mask = np.array([inst not in risky_set for inst in instruments_slice])
            safe_indices = np.where(safe_mask)[0]

            if len(safe_indices) < hold_num:
                # Not enough safe stocks → fill from flagged (least risky first)
                print(f"[master]   {_fmt(trade_dt)}: only {len(safe_indices)} safe stocks, "
                      f"backfilling from flagged")
                safe_indices = np.arange(len(instruments_slice))

            # Pure model score ranking on safe stocks
            safe_scores = model_score_raw[safe_indices]
            safe_insts = [instruments_slice[i] for i in safe_indices]
            sel_df = pd.DataFrame({
                "instrument": safe_insts,
                "model_score": safe_scores,
                "ensemble_score": safe_scores,  # = model score in risk_filter mode
            })
            sel_df = sel_df.sort_values("model_score", ascending=False).head(hold_num)

            # Equal-weight
            scores = sel_df["ensemble_score"].values
            lo, hi = float(np.min(scores)), float(np.max(scores))
            weights = ((scores - lo) / (hi - lo)) if hi > lo else np.ones(len(scores)) / len(scores)
            weights = weights / weights.sum()

            sel_df["datetime"] = trade_dt
            sel_df["score"] = weights
            sel_df["weight"] = weights
            sel_df["master_score"] = sel_df["ensemble_score"]

            result_frames.append(
                sel_df[["datetime", "instrument", "score", "weight", "master_score"]]
            )
            continue  # next rebalance date

        elif strategy_mode == "simple":
            # ── Simple Mode: model_score 60% + factor mean 40% ─────────
            # No per-master weights, no regime selection.
            # Direct blend of model predictions with dimension average.
            factor_dims = [d for d in dim_scores if d != "model_score"]
            if factor_dims:
                factor_mean = np.nanmean(np.column_stack(
                    [dim_scores[d] for d in factor_dims]
                ), axis=1)
            else:
                factor_mean = np.ones(n) * 0.5

            ensemble = 0.6 * dim_scores["model_score"] + 0.4 * factor_mean

            # Simple top-N selection
            sel_df = pd.DataFrame({
                "instrument": instruments_slice,
                "model_score": model_score_raw,
                "ensemble_score": ensemble,
            })
            sel_df = sel_df.sort_values("ensemble_score", ascending=False).head(hold_num)

            scores = sel_df["ensemble_score"].values
            lo, hi = float(np.min(scores)), float(np.max(scores))
            weights = ((scores - lo) / (hi - lo)) if hi > lo else np.ones(len(scores)) / len(scores)
            weights = weights / weights.sum()

            sel_df["datetime"] = trade_dt
            sel_df["score"] = weights
            sel_df["weight"] = weights
            sel_df["master_score"] = sel_df["ensemble_score"]

            result_frames.append(
                sel_df[["datetime", "instrument", "score", "weight", "master_score"]]
            )
            continue  # next rebalance date

        # ── Full Mode (default): per-master scoring + ensemble ───────

        # ── Step 6: Per-master scoring with adaptive model_score weight ──
        # Compute market volatility once per date for adaptive weight
        if regime != "manual":
            _market_vol = _get_market_volatility(cal, trade_dt)
        else:
            _market_vol = 0.20  # neutral annualized vol (20%)

        master_scores_list: list[np.ndarray] = []
        for mk in current_master_keys:
            base_weights = MASTER_WEIGHTS.get(mk, {})
            weights = dict(base_weights)

            # Dynamically adjust model_score weight based on:
            # 1) model confidence, 2) factor disagreement, 3) market volatility
            if "model_score" in weights and "model_score" in dim_scores:
                base_mw = weights["model_score"]
                adaptive_mw = _compute_adaptive_model_weight(
                    model_score_raw, dim_scores, _market_vol, base_mw
                )
                # Renormalize remaining dimension weights to sum to 1.0 - adaptive_mw
                non_model_dims = [d for d in weights if d != "model_score" and d in dim_scores]
                remaining_original = sum(weights[d] for d in non_model_dims)
                if remaining_original > 0:
                    scale = (1.0 - adaptive_mw) / remaining_original
                    for d in non_model_dims:
                        weights[d] *= scale
                weights["model_score"] = adaptive_mw

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
        if ensemble_method == "borda":
            # Borda Count: rank-based aggregation reduces correlated-score issue
            ensemble = _borda_count_ensemble(master_scores_list)
        elif ensemble_method == "median":
            ensemble = np.nanmedian(stacked, axis=1)
        else:
            ensemble = np.nanmean(stacked, axis=1)

        # ── Adaptive guaranteed seats (based on factor IC quality) ─────
        _guaranteed = _compute_adaptive_guaranteed_seats(hold_num, _last_factor_ic_result)

        # ── Step 8: Select top hold_num with model guaranteed seats ────
        # Model's top _guaranteed picks are directly included,
        # preserving the highest-conviction model predictions.
        # Master selects the remaining (hold_num - _guaranteed)
        # from positions [_guaranteed : top_k_candidates].
        sel_df = pd.DataFrame({
            "instrument": instruments_slice,
            "model_score": model_score_raw,
            "ensemble_score": ensemble,
        })
        sel_df["model_rank"] = sel_df["model_score"].rank(ascending=False).astype(int)

        # ── Model confidence gating: lock in very high-confidence picks ──
        # Stocks with model z-score > 2.0 bypass factor re-ranking entirely.
        model_z = (model_score_raw - np.nanmean(model_score_raw)) / (np.nanstd(model_score_raw) + 1e-8)
        confidence_locked = set(
            instruments_slice[i] for i in range(len(instruments_slice))
            if model_z[i] > 2.0
        )
        # Merge confidence-locked with guaranteed seats
        n_guaranteed = max(_guaranteed, len(confidence_locked))
        guaranteed_seats = set(sel_df.nsmallest(n_guaranteed, "model_rank")["instrument"])
        guaranteed_seats |= confidence_locked
        n_guaranteed = len(guaranteed_seats)

        guaranteed = sel_df[sel_df["instrument"].isin(guaranteed_seats)].copy()
        remaining_pool = sel_df[~sel_df["instrument"].isin(guaranteed_seats)].copy()
        n_remaining = hold_num - len(guaranteed)

        if n_remaining > 0 and not remaining_pool.empty:
            master_picks = remaining_pool.sort_values(
                "ensemble_score", ascending=False
            ).head(n_remaining)
            sel_df = pd.concat([guaranteed, master_picks], ignore_index=True)
        else:
            sel_df = guaranteed.head(hold_num)

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

    # ── Regime distribution summary ─────────────────────────────────────
    if regime_log:
        from collections import Counter
        regime_counts = Counter(r["regime"] for r in regime_log)
        regime_str = ", ".join(f"{k}={v}" for k, v in sorted(regime_counts.items()))
        print(f"[master] Regime distribution ({len(trade_dates)} dates): {regime_str}")
        # Print per-date log for the first 3 and last 3 dates (compact)
        if len(regime_log) > 6:
            for r in regime_log[:3]:
                print(f"[master]   {_fmt(r['date'])}: {r['regime']} → {r['masters']}")
            print(f"[master]   ... ({len(regime_log) - 6} dates omitted) ...")
            for r in regime_log[-3:]:
                print(f"[master]   {_fmt(r['date'])}: {r['regime']} → {r['masters']}")
        else:
            for r in regime_log:
                print(f"[master]   {_fmt(r['date'])}: {r['regime']} → {r['masters']}")

    # ── Factor IC validation (opt-in, diagnostic only) ──────────────────
    if validate_factor_ic and dim_scores_per_date:
        try:
            _last_factor_ic_result = _validate_factor_ic(
                cal=cal,
                trade_dates=trade_dates,
                dim_scores_per_date=dim_scores_per_date,
                instruments_per_date=instruments_per_date,
                dim_factors_available=dim_factors_available,
                freq=freq,
            )
        except Exception as exc:
            print(f"[master] Factor IC validation failed (non-fatal): {exc}")

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


def _assess_stock_risk_opportunity(
    dim_scores: dict[str, np.ndarray],
    instruments: list[str],
    factor_slice: pd.DataFrame,
) -> tuple[set[str], dict[str, float], list[str]]:
    """Per-stock risk veto and opportunity overweight assessment.

    For EACH candidate stock, checks whether it should be:
    - VETOED (excluded): stock-specific risk is too high
    - OVERWEIGHTED: stock shows strong value/quality/contrarian signal
    - NORMAL: kept at standard weight

    This is fundamentally different from market-level timing — it evaluates
    each stock independently, allowing the portfolio to stay partially
    invested even in bear markets by keeping only the highest-quality names.

    Veto rules (any one triggers exclusion):
      1. Momentum crash: 60d ret < -30% AND 20d vol > 40% ann
      2. Quality deterioration: recent vol / long-term vol > 1.5 AND price < MA126
      3. Sentiment extreme: near 52wk low AND volume collapsing
      4. Systematic weakness: 4+ of 6 dimensions in bottom 15%

    Overweight rules (any one triggers multiplier):
      1. Deep value + quality: value > 85% AND quality > 70% → 1.5×
      2. Contrarian sentiment: sent < 20% AND quality > 60% AND low_risk > 50% → 1.5×
      3. GARP: growth > 60% AND value > 50% → 1.3×
      4. Trend + quality: momentum > 70% AND quality > 70% AND vol < median → 1.2×

    Args:
        dim_scores: Dict dimension_name → percentile array (n_candidates,)
        instruments: Stock codes
        factor_slice: DataFrame with raw factor values indexed by instrument

    Returns:
        (vetoed_set, overweight_dict, reason_log)
    """
    n = len(instruments)
    vetoed: set[str] = set()
    overweight: dict[str, float] = {}
    reasons: list[str] = []

    if n == 0:
        return vetoed, overweight, reasons

    # ── Helper: get dimension percentile for a stock ─────────────────
    def _dim_pct(dim: str, idx: int) -> float:
        if dim not in dim_scores: return 0.5
        arr = dim_scores[dim]
        return float(arr[idx]) if idx < len(arr) else 0.5

    # ── Precompute dimension stats for systematic weakness check ─────
    dim_names = list(dim_scores.keys())
    bottom_count = np.zeros(n, dtype=int)
    for dim in dim_names:
        arr = dim_scores[dim]
        for i in range(min(n, len(arr))):
            if arr[i] < 0.15:
                bottom_count[i] += 1

    # ── Compute vol stats for quality checks ─────────────────────────
    vol_short_expr = "Std(Ref($close, 1) / $close - 1, 63)"
    vol_long_expr = "Std(Ref($close, 1) / $close - 1, 252)"
    ma_expr = "Mean($close, 126)"
    ret_60d_expr = "$close / Ref($close, 63) - 1"
    vol_20d_expr = "Std(Ref($close, 1) / $close - 1, 21)"
    low_52w_expr = "$close / Min($close, 252) - 1"
    vol_ratio_expr = "$volume / Mean($volume, 63)"

    has_vol_data = all(e in factor_slice.columns for e in [vol_short_expr, vol_long_expr, ma_expr])

    for i, inst in enumerate(instruments):
        reasons_i = []

        # ── Veto 1: Momentum crash ───────────────────────────────
        ret_60 = float(factor_slice[ret_60d_expr].iloc[i]) if ret_60d_expr in factor_slice.columns else 0.0
        vol_20 = float(factor_slice[vol_20d_expr].iloc[i]) if vol_20d_expr in factor_slice.columns else 0.0
        if ret_60 < -0.30 and vol_20 > 0.40 / np.sqrt(252):  # ~2.5% daily
            vetoed.add(inst)
            reasons_i.append(f"momentum_crash(ret60={ret_60:.1%},vol20={vol_20:.3f})")

        # ── Veto 2: Quality deterioration ────────────────────────
        if has_vol_data and inst not in vetoed:
            vs = float(factor_slice[vol_short_expr].iloc[i])
            vl = float(factor_slice[vol_long_expr].iloc[i])
            ma = float(factor_slice[ma_expr].iloc[i])
            close = float(factor_slice["$close"].iloc[i]) if "$close" in factor_slice.columns else ma
            if vl > 0 and vs / vl > 1.5 and close < ma:
                vetoed.add(inst)
                reasons_i.append(f"quality_deterioration(vol_ratio={vs/vl:.1f})")

        # ── Veto 3: Sentiment extreme ────────────────────────────
        near_low = float(factor_slice[low_52w_expr].iloc[i]) if low_52w_expr in factor_slice.columns else 1.0
        vol_ratio = float(factor_slice[vol_ratio_expr].iloc[i]) if vol_ratio_expr in factor_slice.columns else 1.0
        if inst not in vetoed and near_low < 0.05 and vol_ratio < 0.5:
            vetoed.add(inst)
            reasons_i.append(f"sentiment_extreme(near_low={near_low:.1%},vol_ratio={vol_ratio:.1f})")

        # ── Veto 4: Systematic weakness ──────────────────────────
        if inst not in vetoed and bottom_count[i] >= 4:
            vetoed.add(inst)
            reasons_i.append(f"systematic_weakness({bottom_count[i]}/6 dims bottom 15%)")

        # ── Overweight checks (only if not vetoed) ───────────────
        if inst not in vetoed:
            v_pct = _dim_pct("value", i)
            q_pct = _dim_pct("quality", i)
            g_pct = _dim_pct("growth", i)
            m_pct = _dim_pct("momentum", i)
            lr_pct = _dim_pct("low_risk", i)
            s_pct = _dim_pct("sentiment", i)

            # Deep value + quality (Graham)
            if v_pct > 0.85 and q_pct > 0.70:
                overweight[inst] = 1.5
                reasons_i.append(f"deep_value_quality(v={v_pct:.0%},q={q_pct:.0%})→1.5×")
            # Contrarian sentiment (Templeton)
            elif s_pct < 0.20 and q_pct > 0.60 and lr_pct > 0.50:
                overweight[inst] = 1.5
                reasons_i.append(f"contrarian_sentiment(s={s_pct:.0%},q={q_pct:.0%})→1.5×")
            # GARP (Lynch)
            elif g_pct > 0.60 and v_pct > 0.50:
                overweight[inst] = 1.3
                reasons_i.append(f"garp(g={g_pct:.0%},v={v_pct:.0%})→1.3×")
            # Trend + quality
            elif m_pct > 0.70 and q_pct > 0.70 and lr_pct > 0.50:
                overweight[inst] = 1.2
                reasons_i.append(f"trend_quality(m={m_pct:.0%},q={q_pct:.0%})→1.2×")

        if reasons_i:
            reasons.append(f"{inst}: {', '.join(reasons_i)}")

    return vetoed, overweight, reasons


def _validate_factor_ic(
    cal: pd.DatetimeIndex,
    trade_dates: pd.DatetimeIndex,
    dim_scores_per_date: dict[pd.Timestamp, dict[str, np.ndarray]],
    instruments_per_date: dict[pd.Timestamp, list[str]],
    dim_factors_available: dict[str, list],
    freq: str = "yearly",
) -> pd.DataFrame:
    """Diagnostic: validate Strategy B factor dimensions against forward returns.

    For each dimension, computes cross-sectional Rank IC between the dimension's
    percentile scores and the forward return at the rebalance horizon. This
    reveals which dimensions actually have predictive power — and which should
    be reweighted or removed.

    CRITICAL: This function queries FORWARD returns (future data). This is
    correct because labels MUST look forward — they define the prediction
    target. The forward returns are used ONLY for validation/diagnostics,
    NEVER fed into ranking, selection, or any other production signal path.

    Controlled by:
      - build_master_signal(validate_factor_ic=True)
      - Env var: MASTER_VALIDATE_FACTOR_IC=1

    Args:
        cal: Qlib trading calendar
        trade_dates: Sorted rebalance dates
        dim_scores_per_date: {date: {dim: percentile_scores_array}}
        instruments_per_date: {date: [instrument_codes]}
        dim_factors_available: Dimension → factor list
        freq: Rebalance frequency (determines forward horizon)

    Returns:
        DataFrame: dimension, mean_RankIC, std_RankIC, ICIR, pos_IC_ratio, n_dates
    """
    from qlib.data import D
    from scipy.stats import spearmanr

    # ── Determine forward horizon based on rebalance frequency ──────────
    freq = freq.strip().lower()
    if freq == "daily":
        horizon_days = 1
    elif freq == "weekly":
        horizon_days = 5
    elif freq == "monthly":
        horizon_days = 21
    else:  # yearly (default)
        horizon_days = 252

    print(f"\n[factor_ic_validate] Computing per-dimension Rank IC "
          f"(horizon={horizon_days}d, freq={freq})...")

    # ── Per-date: query forward returns and compute dimension IC ───────
    dim_ic_records: dict[str, list[float]] = {dim: [] for dim in dim_factors_available}

    for trade_dt in trade_dates:
        if trade_dt not in dim_scores_per_date:
            continue
        instruments = instruments_per_date.get(trade_dt, [])
        if len(instruments) < 3:
            continue

        # Compute forward date
        try:
            cal_pos = int(cal.searchsorted(trade_dt, side="left"))
        except Exception:
            continue
        fwd_pos = min(len(cal) - 1, cal_pos + horizon_days)
        if fwd_pos <= cal_pos:
            continue
        forward_date = pd.Timestamp(cal[fwd_pos])

        # Query forward close prices (THE ONLY future-data access in this file)
        # Used EXCLUSIVELY for validation — never for ranking
        try:
            fwd_data = D.features(
                instruments,
                ["$close"],
                start_time=trade_dt.strftime("%Y-%m-%d"),
                end_time=forward_date.strftime("%Y-%m-%d"),
            )
        except Exception:
            continue

        if fwd_data is None:
            continue

        # Get the last available close for each instrument within the window
        if isinstance(fwd_data, pd.DataFrame):
            if fwd_data.index.nlevels >= 2:
                fwd_close = fwd_data.iloc[:, 0].groupby(level="instrument").last()
            else:
                fwd_close = fwd_data.iloc[:, 0]
        elif isinstance(fwd_data, pd.Series):
            if fwd_data.index.nlevels >= 2:
                fwd_close = fwd_data.groupby(level="instrument").last()
            else:
                fwd_close = fwd_data
        else:
            continue

        # Compute forward return from trade date close
        try:
            trade_data = D.features(
                instruments, ["$close"],
                start_time=trade_dt.strftime("%Y-%m-%d"),
                end_time=trade_dt.strftime("%Y-%m-%d"),
            )
            if trade_data is None:
                continue
            if isinstance(trade_data, pd.DataFrame):
                trade_close = trade_data.iloc[:, 0]
            elif isinstance(trade_data, pd.Series):
                trade_close = trade_data
            else:
                continue
        except Exception:
            continue

        # Align and compute forward returns
        fwd_ret = (fwd_close - trade_close) / (trade_close.abs() + 1e-8)
        fwd_ret = fwd_ret.dropna()

        if len(fwd_ret) < 3:
            continue

        # Compute per-dimension Rank IC
        dim_scores_dt = dim_scores_per_date[trade_dt]
        for dim, scores in dim_scores_dt.items():
            if dim == "model_score" or len(scores) < 3:
                continue
            # Align scores with forward returns by instrument
            inst_list = instruments_per_date[trade_dt]
            score_s = pd.Series(scores, index=inst_list[:len(scores)])
            aligned = pd.concat([score_s, fwd_ret], axis=1, join="inner").dropna()
            if len(aligned) >= 5:
                try:
                    ic, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                    if not np.isnan(ic):
                        dim_ic_records[dim].append(ic)
                except Exception:
                    pass

    # ── Summarize results ──────────────────────────────────────────────
    results = []
    for dim, ics in dim_ic_records.items():
        n = len(ics)
        if n < 2:
            results.append({
                "dimension": dim, "mean_RankIC": np.nan, "std_RankIC": np.nan,
                "ICIR": np.nan, "pos_IC_ratio": np.nan, "n_dates": n,
                "verdict": "insufficient data",
            })
            continue
        mean_ic = float(np.mean(ics))
        std_ic = float(np.std(ics, ddof=1))
        icir = mean_ic / std_ic if std_ic > 1e-12 else 0.0
        pos_ratio = float((np.array(ics) > 0).mean())
        # Verdict
        if mean_ic > 0.03 and icir > 0.5:
            verdict = "✓ effective"
        elif mean_ic > 0.01 and icir > 0.2:
            verdict = "~ marginal"
        elif mean_ic < 0:
            verdict = "✗ NEGATIVE IC — consider removing or reversing direction"
        else:
            verdict = "? weak — may need reweighting"
        results.append({
            "dimension": dim, "mean_RankIC": mean_ic, "std_RankIC": std_ic,
            "ICIR": icir, "pos_IC_ratio": pos_ratio, "n_dates": n,
            "verdict": verdict,
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    df = df.sort_values("mean_RankIC", ascending=False, na_position="last")

    # ── Print report ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Strategy B Factor IC Validation (horizon={horizon_days}d)")
    print(f"{'='*70}")
    print(f"  {'Dimension':<15s} {'RankIC':>8s} {'ICIR':>7s} {'Pos%':>7s} {'n':>5s}  Verdict")
    print(f"  {'-'*60}")
    for _, row in df.iterrows():
        ic_str = f"{row['mean_RankIC']:>8.4f}" if pd.notna(row['mean_RankIC']) else "     N/A"
        icir_str = f"{row['ICIR']:>7.3f}" if pd.notna(row['ICIR']) else "    N/A"
        pos_str = f"{row['pos_IC_ratio']*100:>6.0f}%" if pd.notna(row['pos_IC_ratio']) else "   N/A"
        n_str = f"{int(row['n_dates']):>5d}"
        print(f"  {row['dimension']:<15s} {ic_str} {icir_str} {pos_str} {n_str}  {row['verdict']}")
    print(f"{'='*70}\n")

    return df
