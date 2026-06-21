#!/usr/bin/env python3
"""
Turnover-aware training utilities for monthly-rebalanced A-share portfolios.

Provides two mechanisms to reduce portfolio turnover through model-level changes:

1. **Prediction Smoothing (EMA)**: Post-training exponential moving average of
   predictions across consecutive trading days per stock. This directly reduces
   day-to-day prediction volatility, which translates to lower portfolio turnover.
   Simple, fast, no training changes needed.

2. **Custom eval metric: Rank IC (Spearman)**: Replaces the default MSE-based
   validation metric with cross-sectional Rank IC. Since the actual trading
   objective is ranking quality (not regression accuracy), early stopping
   on Rank IC directly optimizes for the downstream metric.

Usage:
    from turnover_loss import prediction_ema_smooth, rank_ic_eval_metric

    # After model.predict(), smooth predictions across time:
    smoothed_pred = prediction_ema_smooth(predictions_df, alpha=0.3)

    # During training, pass feval for Rank IC early stopping:
    model_kwargs["feval"] = rank_ic_eval_metric
"""

import os

import numpy as np
import pandas as pd


# ── Prediction Smoothing (EMA across time) ────────────────────────────────

def prediction_ema_smooth(
    pred_df: pd.DataFrame,
    alpha: float = 0.3,
    date_col: str = "datetime",
    inst_col: str = "instrument",
    score_col: str = "score",
) -> pd.DataFrame:
    """Apply exponential moving average smoothing to predictions across time.

    For each stock, smooth the prediction series:
        smoothed[t] = alpha * raw[t] + (1-alpha) * smoothed[t-1]

    This reduces day-to-day prediction "jitter" that causes unnecessary
    turnover. A higher alpha means more smoothing (less responsive).

    Args:
        pred_df: DataFrame with columns [datetime, instrument, score]
        alpha: Smoothing factor (0 = no change, 1 = no smoothing).
               Default 0.3 means 30% weight on current, 70% on history.
        date_col: Name of the date column
        inst_col: Name of the instrument column
        score_col: Name of the score column

    Returns:
        DataFrame with smoothed score column
    """
    if pred_df.empty:
        return pred_df

    alpha = float(os.environ.get("TURNOVER_EMA_ALPHA", str(alpha)))

    df = pred_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values([inst_col, date_col]).reset_index(drop=True)

    smoothed = []
    for _, grp in df.groupby(inst_col, sort=False):
        grp = grp.sort_values(date_col)
        raw_scores = grp[score_col].values.astype(float)
        ema = np.empty_like(raw_scores)
        if len(raw_scores) > 0:
            ema[0] = raw_scores[0]
            for i in range(1, len(raw_scores)):
                ema[i] = alpha * raw_scores[i] + (1 - alpha) * ema[i - 1]
        grp = grp.copy()
        grp[score_col] = ema
        smoothed.append(grp)

    result = pd.concat(smoothed, ignore_index=True)
    return result.sort_values([date_col, inst_col]).reset_index(drop=True)


# ── Rank IC Eval Metric (for LightGBM early stopping) ─────────────────────

def rank_ic_eval_metric(preds, train_data):
    """LightGBM custom eval metric: cross-sectional Rank IC (Spearman).

    Computes the Spearman rank correlation between predictions and labels
    at each boosting round. Higher is better (maximizing ranking quality).

    This replaces MSE-based early stopping, which is poorly correlated
    with the actual downstream objective of portfolio ranking.

    Args:
        preds: Model predictions for the validation set
        train_data: LightGBM Dataset object with .get_label()

    Returns:
        ('rank_ic', value, higher_is_better=True)
    """
    labels = train_data.get_label()
    if len(preds) < 5:
        return ('rank_ic', 0.0, True)

    try:
        from scipy.stats import spearmanr
        # Sample down if very large (spearmanr is O(n log n))
        if len(preds) > 5000:
            idx = np.random.RandomState(42).choice(len(preds), size=5000, replace=False)
            ic, _ = spearmanr(preds[idx], labels[idx])
        else:
            ic, _ = spearmanr(preds, labels)
        return ('rank_ic', ic if not np.isnan(ic) else 0.0, True)
    except Exception:
        # Fallback: use simple Pearson correlation
        try:
            ic = np.corrcoef(preds, labels)[0, 1]
            return ('rank_ic', ic if not np.isnan(ic) else 0.0, True)
        except Exception:
            return ('rank_ic', 0.0, True)


# ── Turnover-Aware Custom Objective (experimental) ─────────────────────────

def turnover_aware_mse_objective(preds, train_data):
    """Custom LightGBM objective with turnover/stability regularization.

    Adds a penalty to the gradient when predictions differ from the
    previous round's predictions for the same stock. This encourages
    the model to produce temporally consistent rankings.

    NOTE: This requires maintaining per-stock prediction history across
    boosting rounds, which is complex in LightGBM's training loop.
    Currently EXPERIMENTAL — prefer prediction_ema_smooth() for
    practical turnover reduction.

    grad = (pred - label) + lambda_turnover * (pred - prev_pred)
    hess = 1 + lambda_turnover
    """
    labels = train_data.get_label()
    lambda_turnover = float(os.environ.get("TURNOVER_REG_STRENGTH", "0.01"))

    grad = (preds - labels).astype(np.float64)
    hess = np.ones_like(grad)

    # Stability penalty: if we have previous round predictions,
    # penalize large changes from round to round
    prev_preds = getattr(train_data, '_prev_preds', None)
    if prev_preds is not None and len(prev_preds) == len(preds):
        delta = preds - prev_preds
        grad += lambda_turnover * delta
        hess += lambda_turnover

    # Store current predictions for next round
    train_data._prev_preds = preds.copy()

    return grad, hess


# ── Utility: compute cross-sectional Rank IC ───────────────────────────────

def compute_oos_rank_ic(
    pred_df: pd.DataFrame,
    label_df: pd.DataFrame,
    date_col: str = "datetime",
) -> float:
    """Compute out-of-sample cross-sectional Rank IC.

    Args:
        pred_df: Predictions with [datetime, instrument, score]
        label_df: Labels with [datetime, instrument, label]
        date_col: Name of the date column

    Returns:
        Mean Rank IC across all dates
    """
    from scipy.stats import spearmanr

    if pred_df.empty or label_df.empty:
        return 0.0

    merged = pred_df.merge(label_df, on=[date_col, "instrument"], how="inner")
    if merged.empty:
        return 0.0

    ics = []
    for _, grp in merged.groupby(date_col):
        if len(grp) >= 5:
            ic, _ = spearmanr(grp["score"], grp["label"])
            if not np.isnan(ic):
                ics.append(ic)

    return float(np.mean(ics)) if ics else 0.0


if __name__ == "__main__":
    # Quick smoke test
    print("Turnover loss module loaded.")
    print(f"  TURNOVER_EMA_ALPHA: {os.environ.get('TURNOVER_EMA_ALPHA', '0.3 (default)')}")
    print(f"  TURNOVER_REG_STRENGTH: {os.environ.get('TURNOVER_REG_STRENGTH', '0.01 (default)')}")
