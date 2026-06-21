#!/usr/bin/env python3
"""
Multi-horizon label handler for monthly-rebalanced A-share trading.

Extends Qlib's Alpha158 to produce labels at multiple forward horizons
(T1, T5, T10, T21) aligned with the actual holding period, and replaces
CSZScoreNorm with robust time-series-based normalization that preserves
cross-day magnitude information.

Key improvements over default Alpha158:
  1. Multi-horizon labels: T1 (daily), T5 (weekly), T10 (biweekly), T21 (monthly)
  2. Combined label = weighted sum across horizons (T21 gets highest weight)
  3. Robust label normalization: rolling-median centering + MAD winsorization
     instead of CSZScoreNorm (which destroys inter-day magnitude info)
  4. Net-return label option: subtract estimated transaction costs for T21

Usage:
  Set environment variable MODEL_MODE=multihorizon to activate.
  Optional env vars:
    LABEL_HORIZONS: comma-separated horizon days (default: "1,5,10,21")
    LABEL_WEIGHTS:  comma-separated weights matching horizons (default: "0.15,0.25,0.25,0.35")
    TRANSACTION_COST_RATE: round-trip cost rate for net-return label (default: 0.001)
    LABEL_NORM: label normalization method: "robust" (default), "cs_rank", "cs_zscore"
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure qlib is importable
_QLIB_ROOT = Path(__file__).resolve().parents[2]
if str(_QLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(_QLIB_ROOT))


class RobustLabelProcessor:
    """Replace CSZScoreNorm with time-series robust normalization.

    Instead of cross-sectional z-scoring labels per day (which destroys
    the magnitude of returns and makes bull/bear days indistinguishable),
    this processor:

    1. Computes a rolling 252-day median return per stock (removes stock bias)
    2. Subtracts the rolling median from raw returns
    3. Winsorizes at 5x the rolling median absolute deviation (handles outliers)
    4. Preserves the cross-day structure: a 5% return day still looks different
       from a -3% return day across ALL stocks

    This preserves the model's ability to distinguish high-return from
    low-return market environments, enabling better position sizing and
    risk allocation decisions.
    """

    def __init__(self, fields_group="label", rolling_window=252, winsor_sigma=5.0):
        self.fields_group = fields_group
        self.rolling_window = rolling_window
        self.winsor_sigma = winsor_sigma
        self._centers = {}   # per-instrument rolling median
        self._scales = {}    # per-instrument rolling MAD

    def fit(self, df: pd.DataFrame):
        """Compute per-instrument rolling median and MAD on training data."""
        if self.fields_group not in df.columns.get_level_values("feature").unique():
            return self

        label_cols = [c for c in df.columns if c[0] == self.fields_group]
        if not label_cols:
            return self

        instruments = df.index.get_level_values("instrument").unique()
        window = self.rolling_window

        for inst in instruments:
            inst_data = df.xs(inst, level="instrument", drop_level=False)
            if inst_data.empty:
                continue
            series = inst_data[label_cols[0]]
            if len(series) < window:
                self._centers[inst] = float(series.mean())
                self._scales[inst] = max(float(series.std()), 1e-8)
            else:
                rolling = series.rolling(window=window, min_periods=min(63, len(series)))
                self._centers[inst] = float(rolling.median().iloc[-1])
                mad = (series - rolling.median()).abs().rolling(
                    window=window, min_periods=min(63, len(series))
                ).median().iloc[-1]
                self._scales[inst] = max(float(mad) * 1.4826, 1e-8)

        return self

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or self.fields_group not in df.columns.get_level_values("feature").unique():
            return df

        label_cols = [c for c in df.columns if c[0] == self.fields_group]
        if not label_cols:
            return df

        out = df.copy()
        for col in label_cols:
            for inst in df.index.get_level_values("instrument").unique():
                idx = df.index.get_level_values("instrument") == inst
                if inst in self._centers:
                    center = self._centers[inst]
                    scale = self._scales.get(inst, 1e-8)
                else:
                    vals = df.loc[idx, col]
                    center = float(vals.median()) if len(vals) > 0 else 0.0
                    scale = max(float(vals.abs().median()) * 1.4826, 1e-8) if len(vals) > 0 else 1e-8

                vals = df.loc[idx, col].values.astype(float)
                centered = vals - center
                threshold = self.winsor_sigma * scale
                winsorized = np.clip(centered, -threshold, threshold)
                out.loc[idx, col] = winsorized

        return out


def _parse_horizons_and_weights():
    """Read horizon/weight configuration from environment variables."""
    horizons_str = os.environ.get("LABEL_HORIZONS", "1,5,10,21")
    weights_str = os.environ.get("LABEL_WEIGHTS", "0.15,0.25,0.25,0.35")

    horizons = [int(h.strip()) for h in horizons_str.split(",")]
    weights = [float(w.strip()) for w in weights_str.split(",")]

    if len(horizons) != len(weights):
        raise ValueError(
            f"LABEL_HORIZONS ({len(horizons)} items) and LABEL_WEIGHTS "
            f"({len(weights)} items) must have the same length"
        )

    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    return horizons, weights


def _build_multi_horizon_label_config():
    """Build Qlib label expressions for multiple forward horizons.

    Qlib expression convention: Ref($close, -N) = close at t+N (future).
    The return from T+1 to T+N+1 is:
        Ref($close, -(N+1)) / Ref($close, -1) - 1

    This accounts for the T+1 settlement rule: signal generated at T close,
    executed at T+1 open, return measured from T+1 close to T+N+1 close.

    Returns:
        (label_expressions, label_names, weights)
    """
    horizons, weights = _parse_horizons_and_weights()

    cost_rate = float(os.environ.get("TRANSACTION_COST_RATE", "0.001"))
    use_net_return = os.environ.get("LABEL_USE_NET_RETURN", "0") == "1"

    expressions = []
    names = []

    for h in horizons:
        # Return from T+1 close to T+h+1 close
        expr = f"Ref($close, -{h + 1}) / Ref($close, -1) - 1"
        expressions.append(expr)
        names.append(f"LABEL_T{h}")

    # Add combined weighted label
    combined_parts = []
    for i, (expr, w) in enumerate(zip(expressions, weights)):
        combined_parts.append(f"{w} * ({expr})")
    combined_expr = " + ".join(combined_parts)
    expressions.append(combined_expr)
    names.append("LABEL_COMBINED")

    # Add net-return label for the longest horizon (monthly)
    if use_net_return:
        longest_h = horizons[-1]
        net_expr = f"Ref($close, -{longest_h + 1}) / Ref($close, -1) - 1 - {cost_rate}"
        expressions.append(net_expr)
        names.append("LABEL_NET")

    return expressions, names, weights


# Cache the label config so it's consistent within a process
_MULTI_HORIZON_LABEL_EXPRS, _MULTI_HORIZON_LABEL_NAMES, _MULTI_HORIZON_WEIGHTS = (
    _build_multi_horizon_label_config()
)


class MultiHorizonAlpha158:
    """Mixin that replaces Alpha158's label configuration with multi-horizon labels.

    Usage:
        from qlib.contrib.data.handler import Alpha158

        class MultiHorizonAlpha158Handler(MultiHorizonAlpha158, Alpha158):
            pass

    Then use MultiHorizonAlpha158Handler as the handler class in YAML config.
    """

    def get_label_config(self):
        # LightGBM only supports single-label regression.
        # Return only the weighted combined label (LABEL_COMBINED) as the training target.
        # Individual horizon labels (T1/T5/T10/T21) are available for post-hoc
        # IC analysis via the same Qlib expressions if needed.
        combined_idx = _MULTI_HORIZON_LABEL_NAMES.index("LABEL_COMBINED")
        return (
            [_MULTI_HORIZON_LABEL_EXPRS[combined_idx]],
            [_MULTI_HORIZON_LABEL_NAMES[combined_idx]],
        )


# ── Concrete handler class for Qlib config system ──
# Qlib loads the handler class from module_path and instantiates it.
# This concrete class combines MultiHorizonAlpha158 mixin + Alpha158 base.
try:
    from qlib.contrib.data.handler import Alpha158 as _Alpha158

    class MultiHorizonAlpha158Handler(MultiHorizonAlpha158, _Alpha158):
        """Concrete handler: Alpha158 features + multi-horizon labels.

        Replaces the default 1-day label with T1/T5/T10/T21 forward returns
        and uses robust time-series label normalization instead of CSZScoreNorm.
        Use with model_mode="multihorizon" in gen_practice_yaml.py.
        """
        pass

except ImportError:
    # Fallback: when qlib is not importable (e.g., outside Docker)
    class MultiHorizonAlpha158Handler:
        pass


def get_learn_processors():
    """Get label processors based on LABEL_NORM env var.

    Options:
      - "robust" (default): RobustLabelProcessor — time-series median centering + MAD winsorize
      - "cs_rank": CSRankNorm — cross-sectional rank (preserves ordinal info)
      - "cs_zscore": CSZScoreNorm — cross-sectional z-score (original qlib default)
    """
    norm_method = os.environ.get("LABEL_NORM", "robust").strip().lower()

    if norm_method == "cs_zscore":
        # Original qlib default
        return [
            {"class": "DropnaLabel"},
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
        ]

    if norm_method == "cs_rank":
        return [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ]

    # "robust" — default
    return [
        {"class": "DropnaLabel"},
        {
            "class": "RobustLabelProcessor",
            "module_path": "scripts.practice.multi_horizon_label_handler",
            "kwargs": {
                "fields_group": "label",
                "rolling_window": 252,
                "winsor_sigma": 5.0,
            },
        },
    ]


def get_combined_label_name():
    """Return the name of the combined label column for use as primary target."""
    return "LABEL_COMBINED"


def get_label_weights():
    """Return horizon weights dict for downstream analysis."""
    return dict(zip(_MULTI_HORIZON_LABEL_NAMES, _MULTI_HORIZON_WEIGHTS))


# ── Standalone test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Multi-horizon label configuration:")
    print(f"  Expressions: {_MULTI_HORIZON_LABEL_EXPRS}")
    print(f"  Names:       {_MULTI_HORIZON_LABEL_NAMES}")
    print(f"  Weights:     {_MULTI_HORIZON_WEIGHTS}")
    print()
    print(f"  Learn processors: {get_learn_processors()}")
    print(f"  Combined label name: {get_combined_label_name()}")
