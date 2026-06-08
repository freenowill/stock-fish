#!/usr/bin/env python3
"""
build_features_from_h5.py — 从 H5 文件构建 qlib 二进制特征

读取 factor-mining 生成的 daily_pv_all.h5 (58 个原始字段),
计算全部因子 (Alpha158 + new_factor.md 独立因子), 应用截面 z-score 归一化,
输出 qlib 二进制格式。

因子集:
  - Alpha158: 9 Kbar + 4 Price + 145 Rolling = 158 个
  - 独立因子: 32 个 (来自 new_factor.md, 无 Alpha158 等效)
  - 重叠因子: 9 个 (与 Alpha158 重叠, 计算但不作为独立特征)

用法 (Docker 内):
  python3 scripts/practice/build_features_from_h5.py \
      --h5 /data/git_ignore_folder/daily_pv_all.h5 \
      --output /root/.qlib/qlib_data/cn_extra_data_h5 \
      --batch-size 400
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────
BIN_SUFFIX = ".day.bin"

# Alpha158 因子名集合
ALPHA158_KBAR = {"KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"}
ALPHA158_PRICE = {"OPEN0", "HIGH0", "LOW0", "VWAP0"}
_ALPHA158_ROLLING_OPS = [
    "ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN",
    "QTLU", "QTLD", "RANK", "RSV", "IMAX", "IMIN", "IMXD",
    "CORR", "CORD", "CNTP", "CNTN", "CNTD", "SUMP", "SUMN", "SUMD",
    "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD",
]
ALPHA158_NAMES = ALPHA158_KBAR | ALPHA158_PRICE | {
    f"{op}{d}" for d in [5, 10, 20, 30, 60] for op in _ALPHA158_ROLLING_OPS
}

H5_COLUMN_MAP = {
    "close": "$close", "open": "$open", "high": "$high", "low": "$low",
    "vwap": "$vwap", "volume": "$volume",
    "turnover": "$turnover", "eps": "$eps", "pb": "$pb",
    "pe_ttm": "$pe_ttm", "roe_yearly": "$roe_yearly",
    "netprofit_margin": "$netprofit_margin",
    "total_mv": "$total_mv", "free_sh": "$free_sh",
    "ps_ttm": "$ps_ttm", "dv_ratio": "$dv_ratio",
    "ocfps": "$ocfps", "revenue": "$revenue",
    "total_assets": "$total_assets", "total_liab": "$total_liab",
    "total_equity": "$total_equity",
    # v2.0 新增字段 (SUE, AssetGrowth, AccrualsRatio, RevenueGrowth)
    "eps_yoy": "$eps_yoy",
    "assets_yoy": "$assets_yoy",
    "revenue_yoy": "$revenue_yoy",
    "n_income": "$n_income",
    "ocf": "$ocf",
    # v3.0 新增字段 (Amihud, FCF_Yield, ROA, NetProfitGrowth, EPS_Quality)
    "amount": "$amount",
    "fcf": "$fcf",
    "circ_mv": "$circ_mv",
    "roa_yearly": "$roa_yearly",
    "netprofit_yoy": "$netprofit_yoy",
    "npta": "$npta",
    # v4.0 新增字段 (CSI300 优化因子集: BP, Leverage)
    "bps": "$bps",
    "debt_to_assets": "$debt_to_assets",
}

# 42 个独立因子 (v3.0: v2.0 34 + 新增8)
INDEPENDENT_FACTORS = [
    "RealizedVolatility_20d", "volume_change_5d", "obv_slope_10d",
    "sharpe_10d", "reversal_1d", "volume_weighted_momentum_5d",
    "earnings_yield", "vwap_deviation_10d", "avg_normalized_range_5d",
    "turnover_trend", "vwap_deviation_5d", "reversal_2d",
    "momentum_vol_adjusted_20", "risk_adjusted_momentum_5d_20d",
    "trailing_PE_ratio", "PB_Ratio", "PriceToSales", "DividendYield", "Size",
    "Delta_roe", "Delta_net_profit_margin",
    "Delta_DebtToEquity", "Delta_AssetTurnover", "Delta_OperatingCashFlowYield",
    "Turnover",
    "Sector_Relative_PB", "Sector_Relative_PE", "Sector_Relative_DividendYield",
    # v2.0 新增
    "SUE", "AssetGrowth", "AccrualsRatio", "DebtToEquity", "MAX_20d", "RevenueGrowth",
    # v3.0 新增
    "Amihud_20d", "FCF_Yield", "ROA", "Skewness_20d",
    "NetProfitGrowth", "EPS_Quality", "TurnoverVol_20d", "Reversal_3d",
]

# 9 个与 Alpha158 重叠的因子 (也计算, 供参考)
OVERLAP_FACTORS = [
    "momentum_5d", "momentum_10d", "momentum_20d",
    "reversal_5d", "reversal_20d",
    "rsi_14d", "intraday_volatility", "volume_ratio_5d",
    "avg_volume_ratio_20d",
]

# 不参与归一化的原始字段 (行情数据本身不作为因子特征)
EXCLUDE_FROM_NORM = {
    "close", "adjclose", "open", "high", "low", "vwap", "volume", "amount",
    "change", "factor",
}

# ── Rolling helpers (pandas-based, same formulas as process_extra_data.py) ──

def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(arr)
    return s.rolling(window, min_periods=max(3, window // 2)).mean().values


def _rolling_std(arr: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    s = pd.Series(arr)
    return s.rolling(window, min_periods=max(3, window // 2)).std(ddof=ddof).values


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(arr)
    return s.rolling(window, min_periods=max(3, window // 2)).max().values


def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(arr)
    return s.rolling(window, min_periods=max(3, window // 2)).min().values


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(arr)
    return s.rolling(window, min_periods=1).sum().values


def _rolling_quantile(arr: np.ndarray, window: int, q: float) -> np.ndarray:
    s = pd.Series(arr)
    return s.rolling(window, min_periods=max(3, window // 2)).quantile(q).values


def _rolling_skew(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动偏度 (rolling skewness)"""
    s = pd.Series(arr)
    return s.rolling(window, min_periods=max(5, window // 2)).skew().values


def _rolling_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    sx = pd.Series(x)
    sy = pd.Series(y)
    return sx.rolling(window, min_periods=max(5, window // 2)).corr(sy).values


def _rolling_slope(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    xs = np.arange(window, dtype=np.float64)
    x_mean = xs.mean()
    denom = ((xs - x_mean) ** 2).sum()
    if denom < 1e-12:
        return out
    min_p = max(5, window // 2)
    for i in range(window - 1, n):
        ys = arr[i - window + 1 : i + 1]
        valid = ~np.isnan(ys)
        if valid.sum() >= min_p:
            out[i] = ((xs[valid] - x_mean) * (ys[valid] - ys[valid].mean())).sum() / denom
    return out


def _rolling_rsquare(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    xs = np.arange(window, dtype=np.float64)
    x_mean = xs.mean()
    denom = ((xs - x_mean) ** 2).sum()
    if denom < 1e-12:
        return out
    min_p = max(5, window // 2)
    for i in range(window - 1, n):
        ys = arr[i - window + 1 : i + 1]
        valid = ~np.isnan(ys)
        if valid.sum() >= min_p:
            slope = ((xs[valid] - x_mean) * (ys[valid] - ys[valid].mean())).sum() / denom
            intercept = ys[valid].mean() - slope * x_mean
            pred = slope * xs[valid] + intercept
            ss_res = ((ys[valid] - pred) ** 2).sum()
            ss_tot = ((ys[valid] - ys[valid].mean()) ** 2).sum()
            if ss_tot > 1e-12:
                out[i] = 1.0 - ss_res / ss_tot
    return out


def _rolling_resi(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    xs = np.arange(window, dtype=np.float64)
    x_mean = xs.mean()
    denom = ((xs - x_mean) ** 2).sum()
    if denom < 1e-12:
        return out
    min_p = max(5, window // 2)
    for i in range(window - 1, n):
        ys = arr[i - window + 1 : i + 1]
        valid = ~np.isnan(ys)
        if valid.sum() >= min_p:
            slope = ((xs[valid] - x_mean) * (ys[valid] - ys[valid].mean())).sum() / denom
            intercept = ys[valid].mean() - slope * x_mean
            last_pred = slope * (window - 1) + intercept
            out[i] = ys[valid][-1] - last_pred
    return out


def _rolling_idxmax(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        seg = arr[i - window + 1 : i + 1]
        valid_idx = np.where(~np.isnan(seg))[0]
        if len(valid_idx) > 0:
            out[i] = float(valid_idx[np.argmax(seg[valid_idx])])
    return out


def _rolling_idxmin(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        seg = arr[i - window + 1 : i + 1]
        valid_idx = np.where(~np.isnan(seg))[0]
        if len(valid_idx) > 0:
            out[i] = float(valid_idx[np.argmin(seg[valid_idx])])
    return out


def _rolling_rank(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    min_p = max(3, window // 2)
    for i in range(window - 1, n):
        seg = arr[i - window + 1 : i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) >= min_p:
            out[i] = (valid < seg[-1]).sum() / (len(valid) - 1) if len(valid) > 1 else 0.5
    return out


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b != 0, a / b, np.nan)


# ── H5 数据加载 ───────────────────────────────────────────────────────

def load_h5(h5_path: str) -> pd.DataFrame:
    """加载 H5 文件, 返回 MultiIndex DataFrame [datetime, instrument]."""
    print(f"Loading {h5_path} ...", end=" ", flush=True)
    df = pd.read_hdf(h5_path, key="data")
    print(f"shape={df.shape}, instruments={df.index.get_level_values('instrument').nunique()}")
    return df


# ── practice_factor.md 解析 ───────────────────────────────────────────

def parse_practice_factors(md_path: str) -> set | None:
    """从 practice_factor.md 提取因子名列表.

    解析 ## 因子总览 表格中的 因子名称 列。
    返回 None 如果文件不存在或解析失败。
    """
    path = Path(md_path)
    if not path.exists():
        print(f"WARNING: {md_path} 不存在, 无法解析因子列表")
        return None

    factors = set()
    in_table = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "## 因子总览" in line:
                in_table = True
                continue
            if in_table:
                if line.startswith("##") or line.startswith("---"):
                    break
                # 解析表格行: | 1 | FactorName | ... |
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3 and parts[1].isdigit():
                    name = parts[2]
                    if name and name not in ("因子名称", ""):
                        factors.add(name)

    if not factors:
        print(f"WARNING: 从 {md_path} 未解析到任何因子")
        return None

    print(f"从 {md_path} 解析到 {len(factors)} 个实践因子")
    return factors


def parse_alpha158_exclusions(md_path: str) -> set:
    """从 practice_factor.md 的 ## Alpha158 排除列表 节解析排除的 Alpha158 因子名.

    解析两种表格:
      1. 因子级排除表 (KBar/Price) — 列名 | 因子 | 排除 | ...
      2. Rolling 算子级排除表 — 列名 | 算子 | 排除窗口 | ...

    Returns:
        set of excluded factor names (e.g. {"KMID", "KLEN", "MA30", "BETA5", ...}).
    """
    path = Path(md_path)
    if not path.exists():
        return set()

    excluded = set()
    in_section = False
    in_factor_table = False
    in_rolling_table = False
    _WINDOWS = [5, 10, 20, 30, 60]

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            if "## Alpha158 排除列表" in stripped:
                in_section = True
                continue

            if not in_section:
                continue

            # Exit section at next ## heading
            if stripped.startswith("## ") and "Alpha158 排除列表" not in stripped:
                break

            # Detect table headers
            if stripped.startswith("| 因子 |"):
                in_factor_table = True
                in_rolling_table = False
                continue
            if stripped.startswith("| 算子 |"):
                in_rolling_table = True
                in_factor_table = False
                continue

            # Skip separator lines (|----|----|)
            if set(stripped) <= {"|", "-", " "}:
                continue

            if in_factor_table:
                parts = [p.strip() for p in stripped.split("|")]
                if len(parts) >= 4:
                    factor_name = parts[1]
                    if factor_name and factor_name != "因子":
                        decision = parts[2]
                        if "❌" in decision or decision == "排除":
                            excluded.add(factor_name)
                continue

            if in_rolling_table:
                parts = [p.strip() for p in stripped.split("|")]
                if len(parts) >= 4:
                    op = parts[1]
                    if not op or op == "算子":
                        continue
                    exclude_windows = parts[2]
                    if exclude_windows in ("所有", "全部", "所有窗口", "全部窗口"):
                        for d in _WINDOWS:
                            excluded.add(f"{op}{d}")
                    elif exclude_windows and exclude_windows not in ("—", "-", ""):
                        for w in exclude_windows.split(","):
                            w = w.strip()
                            if w.isdigit():
                                excluded.add(f"{op}{w}")

    if excluded:
        print(f"从 {md_path} 解析到 {len(excluded)} 个排除的 Alpha158 因子")
    return excluded


# ── 单只股票因子计算 ──────────────────────────────────────────────────

def compute_factors_for_stock(raw: dict) -> dict:
    """对一只股票计算全部因子 (Alpha158 + 独立 + 重叠).

    Args:
        raw: dict of {field_name: np.ndarray}, 所有数组等长.
             field_name 不带 $ 前缀 (如 'close', 'pe_ttm').

    Returns:
        dict of {factor_name: np.ndarray}, 全部因子.
    """
    close = raw.get("close")
    open_ = raw.get("open")
    high = raw.get("high")
    low = raw.get("low")
    vwap = raw.get("vwap")
    volume = raw.get("volume")
    turnover = raw.get("turnover")

    if close is None:
        return {}

    n = len(close)
    factors = {}

    # 日收益率 (多个因子共用)
    ret = np.full(n, np.nan, dtype=np.float64)
    ret[1:] = _safe_div(close[1:] - close[:-1], close[:-1])

    # ═══════════════════════════════════════════════════════════
    # Alpha158 Kbar (9)
    # ═══════════════════════════════════════════════════════════
    if open_ is not None and high is not None and low is not None:
        eps = 1e-12
        h_l = high - low
        greater_oc = np.maximum(open_, close)
        less_oc = np.minimum(open_, close)

        factors["KMID"] = _safe_div(close - open_, open_)
        factors["KLEN"] = _safe_div(h_l, open_)
        factors["KMID2"] = _safe_div(close - open_, h_l + eps)
        factors["KUP"] = _safe_div(high - greater_oc, open_)
        factors["KUP2"] = _safe_div(high - greater_oc, h_l + eps)
        factors["KLOW"] = _safe_div(less_oc - low, open_)
        factors["KLOW2"] = _safe_div(less_oc - low, h_l + eps)
        factors["KSFT"] = _safe_div(2 * close - high - low, open_)
        factors["KSFT2"] = _safe_div(2 * close - high - low, h_l + eps)

    # ═══════════════════════════════════════════════════════════
    # Alpha158 Price (4)
    # ═══════════════════════════════════════════════════════════
    if open_ is not None:
        factors["OPEN0"] = _safe_div(open_, close)
    if high is not None:
        factors["HIGH0"] = _safe_div(high, close)
    if low is not None:
        factors["LOW0"] = _safe_div(low, close)
    if vwap is not None:
        factors["VWAP0"] = _safe_div(vwap, close)

    # ═══════════════════════════════════════════════════════════
    # Alpha158 Rolling (145 = 5 windows × 29 operators)
    # ═══════════════════════════════════════════════════════════
    WINDOWS = [5, 10, 20, 30, 60]
    abs_ret = np.abs(ret)

    # 预计算 (不依赖窗口)
    up = np.full(n, np.nan, dtype=np.float64)
    dn = np.full(n, np.nan, dtype=np.float64)
    diff_arr = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        up[i] = 1.0 if close[i] > close[i - 1] else 0.0
        dn[i] = 1.0 if close[i] < close[i - 1] else 0.0
        diff_arr[i] = close[i] - close[i - 1]
    gain = np.maximum(diff_arr, 0)
    loss_arr = np.maximum(-diff_arr, 0)
    abs_diff = np.abs(diff_arr)

    close_ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if close[i - 1] > 0:
            close_ratio[i] = close[i] / close[i - 1]

    if volume is not None:
        vol_ratio_arr = np.full(n, np.nan, dtype=np.float64)
        for i in range(1, n):
            if volume[i - 1] > 0:
                vol_ratio_arr[i] = volume[i] / volume[i - 1]
        log_vol = np.log(np.maximum(volume, 0) + 1)
        log_vol_ratio = np.log(np.maximum(vol_ratio_arr, 0) + 1)

        v_diff = np.full(n, np.nan, dtype=np.float64)
        for i in range(1, n):
            v_diff[i] = volume[i] - volume[i - 1]
        v_gain = np.maximum(v_diff, 0)
        v_loss = np.maximum(-v_diff, 0)
        v_abs = np.abs(v_diff)
        wv = abs_ret * volume

    for d in WINDOWS:
        ds = str(d)

        # ROC
        roc = np.full(n, np.nan, dtype=np.float64)
        for i in range(d, n):
            if close[i - d] > 0 and close[i] > 0:
                roc[i] = close[i] / close[i - d] - 1.0
        factors[f"ROC{ds}"] = roc

        ma_raw = _rolling_mean(close, d)
        std_raw = _rolling_std(close, d)
        factors[f"MA{ds}"] = _safe_div(ma_raw, close)
        factors[f"STD{ds}"] = _safe_div(std_raw, close)
        factors[f"BETA{ds}"] = _safe_div(_rolling_slope(close, d), close)
        factors[f"RSQR{ds}"] = _rolling_rsquare(close, d)
        factors[f"RESI{ds}"] = _safe_div(_rolling_resi(close, d), close)

        if high is not None:
            factors[f"MAX{ds}"] = _safe_div(_rolling_max(high, d), close)
        if low is not None:
            factors[f"MIN{ds}"] = _safe_div(_rolling_min(low, d), close)

        factors[f"QTLU{ds}"] = _safe_div(_rolling_quantile(close, d, 0.8), close)
        factors[f"QTLD{ds}"] = _safe_div(_rolling_quantile(close, d, 0.2), close)
        factors[f"RANK{ds}"] = _rolling_rank(close, d)

        if high is not None and low is not None:
            max_high = _rolling_max(high, d)
            min_low = _rolling_min(low, d)
            rsv = np.full(n, np.nan, dtype=np.float64)
            for i in range(d - 1, n):
                lo = min_low[i]
                hi = max_high[i]
                if not np.isnan(lo) and not np.isnan(hi) and hi - lo > 1e-12:
                    rsv[i] = (close[i] - lo) / (hi - lo + 1e-12)
            factors[f"RSV{ds}"] = rsv

        if high is not None:
            factors[f"IMAX{ds}"] = _rolling_idxmax(high, d) / d
        if low is not None:
            factors[f"IMIN{ds}"] = _rolling_idxmin(low, d) / d
        if high is not None and low is not None:
            imax_raw = _rolling_idxmax(high, d)
            imin_raw = _rolling_idxmin(low, d)
            imxd = np.full(n, np.nan, dtype=np.float64)
            for i in range(d - 1, n):
                if not np.isnan(imax_raw[i]) and not np.isnan(imin_raw[i]):
                    imxd[i] = (imax_raw[i] - imin_raw[i]) / d
            factors[f"IMXD{ds}"] = imxd

        if volume is not None:
            factors[f"CORR{ds}"] = _rolling_corr(close, log_vol, d)
            factors[f"CORD{ds}"] = _rolling_corr(close_ratio, log_vol_ratio, d)

        cntp = _rolling_mean(up, d)
        cntn = _rolling_mean(dn, d)
        factors[f"CNTP{ds}"] = cntp
        factors[f"CNTN{ds}"] = cntn
        cntd = np.full(n, np.nan, dtype=np.float64)
        for i in range(d - 1, n):
            if not np.isnan(cntp[i]) and not np.isnan(cntn[i]):
                cntd[i] = cntp[i] - cntn[i]
        factors[f"CNTD{ds}"] = cntd

        sum_gain = _rolling_sum(gain, d)
        sum_loss = _rolling_sum(loss_arr, d)
        sum_abs = _rolling_sum(abs_diff, d)
        factors[f"SUMP{ds}"] = _safe_div(sum_gain, sum_abs + 1e-12)
        factors[f"SUMN{ds}"] = _safe_div(sum_loss, sum_abs + 1e-12)
        sumd = np.full(n, np.nan, dtype=np.float64)
        for i in range(d - 1, n):
            sf_p = factors[f"SUMP{ds}"][i]
            sf_n = factors[f"SUMN{ds}"][i]
            if not np.isnan(sf_p) and not np.isnan(sf_n):
                sumd[i] = sf_p - sf_n
        factors[f"SUMD{ds}"] = sumd

        if volume is not None:
            factors[f"VMA{ds}"] = _safe_div(_rolling_mean(volume, d), volume + 1e-12)
            factors[f"VSTD{ds}"] = _safe_div(_rolling_std(volume, d), volume + 1e-12)
            wv_std = _rolling_std(wv, d)
            wv_mean = _rolling_mean(wv, d)
            factors[f"WVMA{ds}"] = _safe_div(wv_std, wv_mean + 1e-12)

            sum_v_gain = _rolling_sum(v_gain, d)
            sum_v_loss = _rolling_sum(v_loss, d)
            sum_v_abs = _rolling_sum(v_abs, d)
            v_sump = _safe_div(sum_v_gain, sum_v_abs + 1e-12)
            v_sumn = _safe_div(sum_v_loss, sum_v_abs + 1e-12)
            factors[f"VSUMP{ds}"] = v_sump
            factors[f"VSUMN{ds}"] = v_sumn
            v_sumd = np.full(n, np.nan, dtype=np.float64)
            for i in range(d - 1, n):
                if not np.isnan(v_sump[i]) and not np.isnan(v_sumn[i]):
                    v_sumd[i] = v_sump[i] - v_sumn[i]
            factors[f"VSUMD{ds}"] = v_sumd

    # ROC120 (扩展窗口, 不在标准 Alpha158 中)
    roc120 = np.full(n, np.nan, dtype=np.float64)
    for i in range(120, n):
        if close[i - 120] > 0 and close[i] > 0:
            roc120[i] = close[i] / close[i - 120] - 1.0
    factors["ROC120"] = roc120

    # ═══════════════════════════════════════════════════════════
    # 重叠因子 (与 Alpha158 等价, 但仍计算用于参考)
    # ═══════════════════════════════════════════════════════════
    # momentum_5d
    mom5 = np.full(n, np.nan, dtype=np.float64)
    mask5 = (close[5:] > 0) & (close[:-5] > 0)
    mom5[5:] = np.where(mask5, close[5:] / close[:-5] - 1.0, np.nan)
    factors["momentum_5d"] = mom5

    # momentum_10d
    mom10 = np.full(n, np.nan, dtype=np.float64)
    mask10 = (close[10:] > 0) & (close[:-10] > 0)
    mom10[10:] = np.where(mask10, close[10:] / close[:-10] - 1.0, np.nan)
    factors["momentum_10d"] = mom10

    # momentum_20d
    mom20 = np.full(n, np.nan, dtype=np.float64)
    mask20 = (close[20:] > 0) & (close[:-20] > 0)
    mom20[20:] = np.where(mask20, close[20:] / close[:-20] - 1.0, np.nan)
    factors["momentum_20d"] = mom20

    # reversal_5d = -momentum_5d
    factors["reversal_5d"] = -mom5

    # reversal_20d = -momentum_20d
    factors["reversal_20d"] = -mom20

    # rsi_14d (≈ SUMP14)
    rsi_gain = _rolling_sum(gain, 14)
    rsi_loss = _rolling_sum(loss_arr, 14)
    rs = _safe_div(rsi_gain, rsi_loss + 1e-12)
    factors["rsi_14d"] = 100.0 - 100.0 / (1.0 + rs)

    # intraday_volatility = (high - low) / close
    if high is not None and low is not None:
        factors["intraday_volatility"] = _safe_div(high - low, close)

    # volume_ratio_5d
    if volume is not None:
        vol_ma5 = _rolling_mean(volume, 5)
        factors["volume_ratio_5d"] = _safe_div(volume, vol_ma5)

    # avg_volume_ratio_20d
    if volume is not None:
        vol_ma20 = _rolling_mean(volume, 20)
        factors["avg_volume_ratio_20d"] = _safe_div(volume, vol_ma20)

    # ═══════════════════════════════════════════════════════════
    # 独立因子 (32 个, new_factor.md [独立])
    # ═══════════════════════════════════════════════════════════

    # RealizedVolatility_20d: std(ret, 20) * sqrt(252)
    rvol20 = _rolling_std(ret, 20)
    if rvol20 is not None:
        factors["RealizedVolatility_20d"] = rvol20 * np.sqrt(252)

    # volume_change_5d
    if volume is not None:
        vol_chg5 = np.full(n, np.nan, dtype=np.float64)
        mask = volume[5:] > 0
        vol_chg5[5:] = np.where(mask, volume[5:] / volume[:-5] - 1.0, np.nan)
        factors["volume_change_5d"] = vol_chg5

    # obv_slope_10d
    if volume is not None:
        obv = np.full(n, np.nan, dtype=np.float64)
        obv[0] = 0.0
        for i in range(1, n):
            prev = obv[i - 1] if not np.isnan(obv[i - 1]) else 0.0
            if np.isnan(volume[i]) or np.isnan(close[i]) or np.isnan(close[i - 1]):
                obv[i] = prev
            elif close[i] > close[i - 1]:
                obv[i] = prev + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = prev - volume[i]
            else:
                obv[i] = prev
        factors["obv_slope_10d"] = _rolling_slope(obv, 10)

    # sharpe_10d
    ret_mean10 = _rolling_mean(ret, 10)
    ret_std10 = _rolling_std(ret, 10)
    if ret_mean10 is not None and ret_std10 is not None:
        factors["sharpe_10d"] = _safe_div(ret_mean10, ret_std10)

    # reversal_1d = -ret
    factors["reversal_1d"] = -ret

    # volume_weighted_momentum_5d
    if volume is not None:
        vwmom5 = np.full(n, np.nan, dtype=np.float64)
        for i in range(5, n):
            vols = volume[i - 5 : i]
            rets = ret[i - 5 : i]
            mask = ~np.isnan(vols) & ~np.isnan(rets) & (vols > 0)
            if mask.sum() >= 3:
                vwmom5[i] = np.sum(vols[mask] * rets[mask]) / np.sum(vols[mask])
        factors["volume_weighted_momentum_5d"] = vwmom5

    # earnings_yield = eps / close
    eps_arr = raw.get("eps")
    if eps_arr is not None:
        factors["earnings_yield"] = _safe_div(eps_arr, close)

    # vwap_deviation_10d
    if vwap is not None:
        vwap_ma10 = _rolling_mean(vwap, 10)
        if vwap_ma10 is not None:
            factors["vwap_deviation_10d"] = _safe_div(close - vwap_ma10, vwap_ma10)

    # avg_normalized_range_5d
    if high is not None and low is not None:
        daily_range = _safe_div(high - low, close)
        factors["avg_normalized_range_5d"] = _rolling_mean(daily_range, 5)

    # turnover_trend
    if turnover is not None:
        to_ma5 = _rolling_mean(turnover, 5)
        to_ma20 = _rolling_mean(turnover, 20)
        if to_ma5 is not None and to_ma20 is not None:
            factors["turnover_trend"] = _safe_div(to_ma5 - to_ma20, to_ma20)

    # vwap_deviation_5d
    if vwap is not None:
        vwap_ma5 = _rolling_mean(vwap, 5)
        if vwap_ma5 is not None:
            factors["vwap_deviation_5d"] = _safe_div(close - vwap_ma5, vwap_ma5)

    # reversal_2d
    rev2 = np.full(n, np.nan, dtype=np.float64)
    rev2[2:] = np.where(close[:-2] > 0, -(close[2:] / close[:-2] - 1.0), np.nan)
    factors["reversal_2d"] = rev2

    # volume_ratio_5d_20d
    if volume is not None:
        vma5 = _rolling_mean(volume, 5)
        vma20 = _rolling_mean(volume, 20)
        if vma5 is not None and vma20 is not None:
            factors["volume_ratio_5d_20d"] = _safe_div(vma5, vma20)

    # momentum_vol_adjusted_20
    ret_std20 = _rolling_std(ret, 20)
    if ret_std20 is not None:
        factors["momentum_vol_adjusted_20"] = _safe_div(mom20, ret_std20)

    # risk_adjusted_momentum_5d_20d
    if ret_std20 is not None:
        vol_ann20 = ret_std20 * np.sqrt(252)
        factors["risk_adjusted_momentum_5d_20d"] = _safe_div(mom5, vol_ann20)

    # ── 直接映射因子 (原始字段即因子, 后续截面 z-score 归一化) ──
    if raw.get("pe_ttm") is not None:
        factors["trailing_PE_ratio"] = raw["pe_ttm"].astype(np.float64)
    pb_arr = raw.get("pb")
    if pb_arr is not None:
        factors["PB_Ratio"] = pb_arr.astype(np.float64)
    if raw.get("ps_ttm") is not None:
        factors["PriceToSales"] = raw["ps_ttm"].astype(np.float64)
    dv_ratio_arr = raw.get("dv_ratio")
    if dv_ratio_arr is not None:
        factors["DividendYield"] = dv_ratio_arr.astype(np.float64)

    # BP: 账面市值比 (Book-to-Price)
    bps_arr = raw.get("bps")
    if bps_arr is not None:
        factors["BP"] = _safe_div(bps_arr.astype(np.float64), close)

    # ROE: 净资产收益率
    roe_arr = raw.get("roe_yearly")
    if roe_arr is not None:
        factors["ROE"] = roe_arr.astype(np.float64) / 100.0

    # EPS_YoY: EPS 同比增速
    eps_yoy_arr_new = raw.get("eps_yoy")
    if eps_yoy_arr_new is not None:
        factors["EPS_YoY"] = eps_yoy_arr_new.astype(np.float64) / 100.0

    # Leverage: 资产负债率
    dta_arr = raw.get("debt_to_assets")
    if dta_arr is not None:
        factors["Leverage"] = dta_arr.astype(np.float64)

    # ── 财务比率季度差分 (Delta = 当期 − 60交易日前) ──
    _DELTA_WINDOW = 60  # 约一个季度
    _delta = lambda arr: np.concatenate([
        np.full(_DELTA_WINDOW, np.nan, dtype=np.float64),
        arr[_DELTA_WINDOW:] - arr[:-_DELTA_WINDOW]
    ]) if arr is not None and len(arr) > _DELTA_WINDOW else np.full(len(close), np.nan, dtype=np.float64)

    if raw.get("roe_yearly") is not None:
        factors["Delta_roe"] = _delta(raw["roe_yearly"].astype(np.float64))
    if raw.get("netprofit_margin") is not None:
        factors["Delta_net_profit_margin"] = _delta(raw["netprofit_margin"].astype(np.float64))

    # OperatingCashFlowYield: ocfps / close → Delta
    ocfps_arr = raw.get("ocfps")
    if ocfps_arr is not None:
        ocf_yield = _safe_div(ocfps_arr, close)
        factors["Delta_OperatingCashFlowYield"] = _delta(ocf_yield)

    # AssetTurnover: revenue / total_assets → Delta
    revenue_arr = raw.get("revenue")
    total_assets_arr = raw.get("total_assets")
    if revenue_arr is not None and total_assets_arr is not None:
        at_val = _safe_div(revenue_arr, total_assets_arr)
        factors["Delta_AssetTurnover"] = _delta(at_val)

    # DebtToEquity: total_liab / total_equity → Delta
    total_liab_arr = raw.get("total_liab")
    total_equity_arr = raw.get("total_equity")
    if total_liab_arr is not None and total_equity_arr is not None:
        dte_val = _safe_div(total_liab_arr, total_equity_arr)
        factors["Delta_DebtToEquity"] = _delta(dte_val)

    # ── 流动性 / 波动率 ──
    # Size: ln(total_mv)
    total_mv_arr = raw.get("total_mv")
    if total_mv_arr is not None:
        factors["Size"] = np.log(np.maximum(total_mv_arr, 1))

    # Turnover: volume / free_sh
    free_sh_arr = raw.get("free_sh")
    if volume is not None and free_sh_arr is not None:
        factors["Turnover"] = _safe_div(volume, free_sh_arr)

    # Liquidity_Turnover_5d: 5日平均换手率
    if turnover is not None:
        factors["Liquidity_Turnover_5d"] = _rolling_mean(turnover, 5)

    # Volatility_5d: std(ret, 5)
    factors["Volatility_5d"] = _rolling_std(ret, 5)

    # Volatility_10d: std(ret, 10)
    factors["Volatility_10d"] = _rolling_std(ret, 10)

    # Volatility_60d: 60日年化已实现波动率
    rvol60 = _rolling_std(ret, 60)
    if rvol60 is not None:
        factors["Volatility_60d"] = rvol60 * np.sqrt(252)

    # avg_turnover_10d: 10日平均换手率
    if turnover is not None:
        factors["avg_turnover_10d"] = _rolling_mean(turnover, 10)

    # ═══════════════════════════════════════════════════════════
    # v2.0 新增因子 (6 个)
    # ═══════════════════════════════════════════════════════════

    # SUE: 标准化意外盈余
    eps_yoy_arr = raw.get("eps_yoy")
    if eps_yoy_arr is not None:
        eps_yoy_std = _rolling_std(eps_yoy_arr, 20)
        sue = _safe_div(eps_yoy_arr.astype(np.float64), eps_yoy_std)
        sue = np.where(np.isinf(sue) | np.isnan(sue), 0.0, sue)
        factors["SUE"] = sue

    # AssetGrowth: 总资产增长率 (assets_yoy 已是 YoY %)
    assets_yoy_arr = raw.get("assets_yoy")
    if assets_yoy_arr is not None:
        factors["AssetGrowth"] = assets_yoy_arr.astype(np.float64) / 100.0

    # AccrualsRatio: 应计比率 = (净利润 - 经营现金流) / 总资产
    n_income_arr = raw.get("n_income")
    ocf_arr = raw.get("ocf")
    total_assets_arr = raw.get("total_assets")
    if n_income_arr is not None and ocf_arr is not None and total_assets_arr is not None:
        accruals = _safe_div(
            n_income_arr.astype(np.float64) - ocf_arr.astype(np.float64),
            total_assets_arr.astype(np.float64),
        )
        factors["AccrualsRatio"] = accruals

    # DebtToEquity: 杠杆水平 (水平值, 非差分)
    total_liab_arr = raw.get("total_liab")
    total_equity_arr = raw.get("total_equity")
    if total_liab_arr is not None and total_equity_arr is not None:
        factors["DebtToEquity"] = _safe_div(
            total_liab_arr.astype(np.float64),
            total_equity_arr.astype(np.float64),
        )

    # MAX_20d: 过去20日最大日收益率
    max20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        rets_20 = ret[i - 19 : i + 1]
        valid = rets_20[~np.isnan(rets_20)]
        if len(valid) > 0:
            max20[i] = np.max(valid)
    factors["MAX_20d"] = max20

    # RevenueGrowth: 营业收入增长率
    revenue_yoy_arr = raw.get("revenue_yoy")
    if revenue_yoy_arr is not None:
        factors["RevenueGrowth"] = revenue_yoy_arr.astype(np.float64) / 100.0

    # ═══════════════════════════════════════════════════════════
    # v3.0 新增因子 (8 个)
    # ═══════════════════════════════════════════════════════════

    # Amihud_20d: Amihud (2002) 非流动性度量
    amount_arr = raw.get("amount")
    if amount_arr is not None:
        illiq = _safe_div(np.abs(ret), amount_arr)
        factors["Amihud_20d"] = _rolling_mean(illiq, 20) * 1e8

    # FCF_Yield: 自由现金流收益率 = fcf / circ_mv
    fcf_arr = raw.get("fcf")
    circ_mv_arr = raw.get("circ_mv")
    if fcf_arr is not None and circ_mv_arr is not None:
        factors["FCF_Yield"] = _safe_div(fcf_arr.astype(np.float64), circ_mv_arr.astype(np.float64))

    # ROA: 资产收益率 (百分比→小数)
    roa_yearly_arr = raw.get("roa_yearly")
    if roa_yearly_arr is not None:
        factors["ROA"] = roa_yearly_arr.astype(np.float64) / 100.0

    # Skewness_20d: 收益率偏度
    factors["Skewness_20d"] = _rolling_skew(ret, 20)

    # NetProfitGrowth: 净利润同比增长率 (百分比→小数)
    netprofit_yoy_arr = raw.get("netprofit_yoy")
    if netprofit_yoy_arr is not None:
        factors["NetProfitGrowth"] = netprofit_yoy_arr.astype(np.float64) / 100.0

    # EPS_Quality: 经常性损益占比 = 1 - |npta|/max(|eps|, 0.01)
    npta_arr = raw.get("npta")
    eps_arr = raw.get("eps")
    if npta_arr is not None and eps_arr is not None:
        epsq_denom = np.maximum(np.abs(eps_arr.astype(np.float64)), 0.01)
        eps_quality = 1.0 - np.abs(npta_arr.astype(np.float64)) / epsq_denom
        eps_quality = np.clip(eps_quality, 0.0, 1.0)
        factors["EPS_Quality"] = eps_quality

    # TurnoverVol_20d: 换手率变异系数 (std/mean)
    if turnover is not None:
        to_mean20 = _rolling_mean(turnover, 20)
        to_std20 = _rolling_std(turnover, 20)
        factors["TurnoverVol_20d"] = _safe_div(to_std20, to_mean20)

    # Reversal_3d: 3日反转
    rev3 = np.full(n, np.nan, dtype=np.float64)
    rev3[3:] = np.where(close[:-3] > 0, -(close[3:] / close[:-3] - 1.0), np.nan)
    factors["Reversal_3d"] = rev3

    # Sector_Relative 截面因子由 compute_sector_relative_factors() 后处理填充

    return factors


# ── 截面因子: 行业中性化 ──────────────────────────────────────────────

def _load_sw_industry_mapping(csv_path: str) -> dict[str, str]:
    """加载申万行业映射: {symbol_lower: industry_name}.

    从 tushare/fetch_sw_industry.py 生成的 CSV 读取。
    文件不存在或为空时返回空 dict，下游回退到板块分类。
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"  申万行业文件不存在: {csv_path}, 回退到板块分类")
        return {}
    try:
        df = pd.read_csv(path, dtype=str)
        if "symbol" not in df.columns or "industry_name" not in df.columns:
            print(f"  WARNING: {csv_path} 缺少 symbol/industry_name 列, 回退到板块分类")
            return {}
        mapping = {}
        for _, row in df.iterrows():
            sym = str(row["symbol"]).strip().lower()
            ind = str(row["industry_name"]).strip()
            if sym and ind:
                mapping[sym] = ind
        print(f"  加载申万行业映射: {len(mapping)} 只股票, {len(set(mapping.values()))} 个行业")
        return mapping
    except Exception as e:
        print(f"  WARNING: 加载 {csv_path} 失败: {e}, 回退到板块分类")
        return {}


def _stock_industry(symbol: str, sw_map: dict[str, str]) -> str:
    """返回申万一级行业名称，找不到时回退到板块分类。

    Args:
        symbol: 股票代码 (如 SH600000)
        sw_map: {symbol_lower: industry_name} 申万行业映射

    Returns:
        行业名称 (如 "银行", "医药生物") 或板块名 (如 "sh_main")
    """
    name = sw_map.get(symbol.lower(), "")
    if name:
        return name
    # 回退到板块分类 (兼容北交所、新上市等申万未覆盖的股票)
    code = symbol[2:]
    if symbol.startswith("SH"):
        return "star" if code.startswith("688") else "sh_main"
    elif symbol.startswith("SZ"):
        if code.startswith("30"):
            return "chinext"
        if code.startswith("002") or code.startswith("003"):
            return "sme"
        return "sz_main"
    return "other"


# 行业中性化映射: {截面输出因子 → 源因子名}
SECTOR_RELATIVE_MAP = {
    "Sector_Relative_PB": "PB_Ratio",
    "Sector_Relative_PE": "trailing_PE_ratio",
    "Sector_Relative_DividendYield": "DividendYield",
    "Sector_Rel_BP": "BP",
}


def compute_sector_relative_factors(
    feat_root: Path, stock_list: list[str], start_idx: int,
    sw_map: dict[str, str] | None = None,
) -> list[str]:
    """计算行业中性化截面因子 (PB/PE/DividendYield) 并写入 bin.

    对每个源因子: sector_relative = stock_value − sector_median(peers)

    Args:
        sw_map: 申万一级行业映射 {symbol_lower: industry_name}. None 则回退板块分类.
    """
    computed = []
    for sector_name, source_factor in SECTOR_RELATIVE_MAP.items():
        # 读取所有股票的源因子数据
        src_data = {}
        for sym in stock_list:
            fpath = feat_root / sym.lower() / f"{source_factor}{BIN_SUFFIX}"
            if not fpath.exists():
                continue
            raw = np.fromfile(str(fpath), dtype="<f4")
            if len(raw) < 2:
                continue
            src_data[sym.lower()] = raw[1:].astype(np.float64)

        if len(src_data) < 2:
            print(f"  {sector_name}: 不足 2 只有效股票, 跳过")
            continue

        n = len(next(iter(src_data.values())))
        _map = sw_map or {}
        sectors = {sym: _stock_industry(sym.upper(), _map) for sym in src_data}
        sector_stocks = {}
        for sym, sec in sectors.items():
            sector_stocks.setdefault(sec, []).append(sym)

        for sec, syms in sector_stocks.items():
            if len(syms) < 2:
                continue
            matrix = np.full((len(syms), n), np.nan, dtype=np.float64)
            for r, sym in enumerate(syms):
                arr = src_data.get(sym)
                if arr is not None and len(arr) == n:
                    matrix[r] = arr

            with np.errstate(invalid="ignore"):
                sector_median = np.nanmedian(matrix, axis=0)

            for r, sym in enumerate(syms):
                rel_val = np.full(n, np.nan, dtype=np.float64)
                valid = ~np.isnan(matrix[r]) & ~np.isnan(sector_median)
                if valid.any():
                    n_peers = np.sum(~np.isnan(matrix), axis=0) - ~np.isnan(matrix[r])
                    peer_mask = n_peers >= 2
                    idx = valid & peer_mask
                    if idx.any():
                        if len(syms) <= 10:
                            for i in range(n):
                                if not idx[i]:
                                    continue
                                col = matrix[:, i]
                                peers_vals = np.delete(col, r)
                                peers_valid = peers_vals[~np.isnan(peers_vals)]
                                if len(peers_valid) >= 2:
                                    rel_val[i] = matrix[r, i] - np.median(peers_valid)
                        else:
                            rel_val[idx] = matrix[r, idx] - sector_median[idx]

                fpath = feat_root / sym / f"{sector_name}{BIN_SUFFIX}"
                write_bin(fpath, start_idx, rel_val.astype(np.float32))

        computed.append(sector_name)
        print(f"  {sector_name}: {len(src_data)} 只股票, {len(sector_stocks)} 个行业")

    return computed


# ── Qlib 二进制读写 ───────────────────────────────────────────────────

def write_bin(filepath: Path, start_idx: int, data: np.ndarray) -> None:
    header = np.array([float(start_idx)], dtype="<f4")
    np.concatenate([header, data.astype("<f4")]).tofile(str(filepath))


def read_bin(filepath: Path):
    if not filepath.exists():
        return 0, np.array([])
    raw = np.fromfile(str(filepath), dtype="<f4")
    if len(raw) < 2:
        return int(raw[0]) if len(raw) == 1 else 0, np.array([])
    return int(raw[0]), raw[1:].astype(np.float64)


# ── 基准数据 ──────────────────────────────────────────────────────────

def ensure_benchmark_data(output_dir: Path, calendar: list[str], start_idx: int = 0) -> bool:
    """确保输出目录中有 SH000300 基准数据 (close.day.bin).

    从已有 qlib 数据源 (cn_data, cn_extra_data) 中提取 SH000300 收盘价,
    对齐到当前日历, 写入 output_dir/features/sh000300/close.day.bin。

    返回 True 如果成功写入基准数据。
    """
    benchmark = "SH000300"
    bench_feat_dir = output_dir / "features" / benchmark.lower()
    bench_close = bench_feat_dir / f"close{BIN_SUFFIX}"

    if bench_close.exists():
        print(f"基准 {benchmark} 已存在, 跳过")
        return True

    # 候选数据源 (按优先级)
    candidates = [
        Path(os.path.expanduser("~/.qlib/qlib_data/cn_data")),
        Path(os.path.expanduser("~/.qlib/qlib_data/cn_extra_data")),
        Path("/root/.qlib/qlib_data/cn_data"),
    ]
    if os.environ.get("QLIB_DATA_DIR"):
        candidates.insert(0, Path(os.environ["QLIB_DATA_DIR"]))

    src_close = None
    src_calendar_path = None
    for cand in candidates:
        cf = cand / "features" / benchmark.lower() / "close.day.bin"
        cal = cand / "calendars" / "day.txt"
        if cf.exists() and cal.exists():
            src_close = cf
            src_calendar_path = cal
            print(f"  找到基准数据源: {cand}")
            break

    if src_close is None:
        print(f"  WARNING: 找不到 {benchmark} 基准数据, 将跳过回测中的基准比较")
        return False

    # 读取源日历
    with open(src_calendar_path) as f:
        src_calendar = [line.strip() for line in f if line.strip()]
    src_compact = [d.replace("-", "") for d in src_calendar]

    # 读取源二进制数据
    _si, src_data = read_bin(src_close)
    if len(src_data) == 0:
        print(f"  WARNING: {benchmark} 源数据为空")
        return False

    # 对齐到目标日历
    dst_compact = [d.replace("-", "") for d in calendar]
    dst_data = np.full(len(calendar), np.nan, dtype=np.float32)

    # 构建源日期→值映射 (考虑 start_idx 偏移)
    # qlib 二进制格式: raw[0]=start_idx, raw[1:]=从 start_idx 开始的连续数据
    # 所以 src_data[i] 对应日历位置 (_si + i) 的日期
    src_date_to_val = {}
    for i in range(len(src_data)):
        cal_pos = _si + i
        if cal_pos < len(src_compact) and not np.isnan(src_data[i]):
            src_date_to_val[src_compact[cal_pos]] = src_data[i]

    for i, d in enumerate(dst_compact):
        if d in src_date_to_val:
            dst_data[i] = np.float32(src_date_to_val[d])
        elif d in src_date_to_idx:
            idx = src_date_to_idx[d]
            if idx < len(src_data) and not np.isnan(src_data[idx]):
                dst_data[i] = np.float32(src_data[idx])

    n_valid = np.sum(~np.isnan(dst_data))
    if n_valid < 10:
        print(f"  WARNING: {benchmark} 对齐后仅有 {n_valid} 个有效值, 跳过")
        return False

    # 写入
    bench_feat_dir.mkdir(parents=True, exist_ok=True)
    write_bin(bench_close, start_idx, dst_data)
    print(f"  基准 {benchmark}: {n_valid}/{len(calendar)} 个有效日期")

    # 添加到 instruments
    inst_dir = output_dir / "instruments"
    inst_path = inst_dir / "all.txt"
    if inst_path.exists():
        first_date = calendar[0]
        last_date = calendar[-1]
        line = f"{benchmark}\t{first_date}\t{last_date}\n"
        with open(inst_path, "a") as f:
            f.write(line)
        # 同步到市场专属 instrument 文件 (csi300.txt 等)
        for market_file in inst_dir.glob("*.txt"):
            if market_file.name == "all.txt":
                continue
            with open(market_file, "a") as f:
                f.write(line)
        print(f"  benchmark 已注册到 all.txt 及 {len(list(inst_dir.glob('*.txt'))) - 1} 个市场文件")

    return True


# ── 截面归一化 ────────────────────────────────────────────────────────

def normalize_cross_sectional(
    feat_root: Path,
    feature_list: list[str],
    stock_list: list[str],
    batch_size: int = 400,
    min_stocks_per_date: int = 5,
    min_valid_stocks: int = 10,
    winsorize: tuple[float, float] | None = (0.01, 0.99),
) -> dict:
    """对每个特征做跨股票截面 winsorize + z-score 归一化并写回 bin 文件.

    先对每日期截面做 winsorize 缩尾 (移除极端值污染),
    再计算截面 z-score (mean=0, std=1).

    Args:
        min_stocks_per_date: 每个日期至少需要多少只有效股票才做归一化
        min_valid_stocks: 特征层面至少需要多少只有效股票才处理该特征
        winsorize: (lo_pct, hi_pct) 缩尾分位阈值, None 表示不缩尾.
                   默认 (0.01, 0.99) 即 1%/99% 缩尾.
    """
    stock_dirs = [feat_root / s.lower() for s in stock_list if (feat_root / s.lower()).is_dir()]
    if not stock_dirs:
        return {}

    first_feat = feature_list[0]
    first_path = stock_dirs[0] / f"{first_feat}{BIN_SUFFIX}"
    if not first_path.exists():
        print(f"  WARNING: first feature {first_feat} not found")
        return {}

    sample = np.fromfile(str(first_path), dtype="<f4")
    start_idx = int(sample[0]) if len(sample) > 0 else 0
    n_dates = len(sample) - 1
    n_stocks = len(stock_list)

    results = {}

    for feat in feature_list:
        # Phase 1: 加载所有股票数据
        all_data = np.full((n_stocks, n_dates), np.nan, dtype=np.float64)
        valid_mask = np.zeros(n_stocks, dtype=bool)

        for batch_start in range(0, n_stocks, batch_size):
            batch_end = min(batch_start + batch_size, n_stocks)
            for i in range(batch_start, batch_end):
                fpath = stock_dirs[i] / f"{feat}{BIN_SUFFIX}"
                if not fpath.exists():
                    continue
                raw = np.fromfile(str(fpath), dtype="<f4")
                if len(raw) < 2:
                    continue
                si = int(raw[0])
                data = raw[1:].astype(np.float64)
                if si == start_idx and len(data) >= n_dates:
                    all_data[i, :n_dates] = data[:n_dates]
                    valid_mask[i] = True
                elif si != start_idx:
                    offset = si - start_idx
                    src_start = max(0, -offset)
                    dst_start = max(0, offset)
                    length = min(n_dates - dst_start, len(data) - src_start)
                    if length > 0:
                        all_data[i, dst_start:dst_start + length] = data[src_start:src_start + length]
                        valid_mask[i] = True

        valid_stocks = int(valid_mask.sum())
        if valid_stocks < min_valid_stocks:
            results[feat] = {"stocks_processed": 0, "dates_normalized": 0}
            continue

        # Phase 2: 截面 winsorize (去极端值污染)
        if winsorize is not None:
            lo_pct, hi_pct = winsorize
            with np.errstate(invalid="ignore"):
                for t in range(n_dates):
                    col = all_data[:n_stocks, t]
                    valid = ~np.isnan(col)
                    n_valid = valid.sum()
                    if n_valid < min_stocks_per_date:
                        continue
                    lo = np.nanquantile(col, lo_pct)
                    hi = np.nanquantile(col, hi_pct)
                    if lo < hi:
                        col[valid] = np.clip(col[valid], lo, hi)

        # Phase 3: 每日期横截面 z-score
        dates_normalized = 0
        with np.errstate(invalid="ignore"):
            for t in range(n_dates):
                col = all_data[:n_stocks, t]
                valid = ~np.isnan(col)
                n_valid = valid.sum()
                if n_valid < min_stocks_per_date:
                    continue
                mean_t = np.mean(col[valid])
                std_t = np.std(col[valid])
                if std_t < 1e-8:
                    continue
                all_data[valid, t] = (col[valid] - mean_t) / std_t
                dates_normalized += 1

        # Phase 4: 写回
        for i in range(n_stocks):
            if not valid_mask[i]:
                continue
            fpath = stock_dirs[i] / f"{feat}{BIN_SUFFIX}"
            write_bin(fpath, start_idx, all_data[i].astype(np.float32))

        results[feat] = {"stocks_processed": valid_stocks, "dates_normalized": dates_normalized}

    print(f"  归一化完成: {len(results)} 个特征")
    return results


# ── 日历 & instruments ────────────────────────────────────────────────

def build_calendars_and_instruments(
    h5_df: pd.DataFrame, output_dir: Path
) -> list[str]:
    """从 H5 数据提取交易日历并写入 qlib 格式."""
    dates = sorted(h5_df.index.get_level_values("datetime").unique())
    calendar = [d.strftime("%Y-%m-%d") for d in dates]

    cal_dir = output_dir / "calendars"
    cal_dir.mkdir(parents=True, exist_ok=True)
    with open(cal_dir / "day.txt", "w") as f:
        for d in calendar:
            f.write(d + "\n")
    print(f"日历: {len(calendar)} 天 ({calendar[0]} ~ {calendar[-1]})")

    # Instruments: 从 H5 获取每只股票的日期范围
    inst_dir = output_dir / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)

    instruments = h5_df.index.get_level_values("instrument").unique()
    records = []
    for inst in sorted(instruments):
        inst_dates = h5_df.loc[pd.IndexSlice[:, inst], :].index.get_level_values("datetime")
        if len(inst_dates) > 0:
            start = inst_dates.min().strftime("%Y-%m-%d")
            end = inst_dates.max().strftime("%Y-%m-%d")
            records.append((inst, start, end))

    with open(inst_dir / "all.txt", "w") as f:
        for sym, sd, ed in records:
            f.write(f"{sym}\t{sd}\t{ed}\n")
    print(f"instruments: {len(records)} 只")

    return calendar


# ── 主流程 ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="从 H5 文件构建 qlib 二进制特征 (Alpha158 + new_factor.md 因子)"
    )
    parser.add_argument("--h5", required=True, help="H5 文件路径")
    parser.add_argument("--output", required=True, help="输出 qlib 数据目录")
    parser.add_argument("--batch-size", type=int, default=400, help="每批股票数 (控制内存)")
    parser.add_argument("--stock-codes", nargs="*", default=None, help="指定股票代码 (默认全部)")
    parser.add_argument("--skip-normalize", action="store_true", help="跳过截面归一化")
    parser.add_argument("--new-factor-only", action="store_true",
                        help="仅提取 practice_factor.md 中列出的因子, 跳过全部 Alpha158")
    parser.add_argument("--practice-factor-file", default="tushare/practice_factor.md",
                        help="practice_factor.md 路径 (默认: tushare/practice_factor.md)")
    parser.add_argument("--workers", type=int, default=0,
                        help="每批并行 worker 数 (0=auto, 默认: auto)")
    parser.add_argument("--sw-industry-file", default="tushare/cn_data/sw_industry.csv",
                        help="申万行业映射 CSV 路径 (默认: tushare/cn_data/sw_industry.csv)")
    parser.add_argument("--include-alpha158", action="store_true",
                        help="与 --new-factor-only 联合: 在实践因子基础上, 额外包含所有 Alpha158 因子")
    parser.add_argument("--market", default=None,
                        help="市场名称 (如 csi300/csi1000), 用于生成市场专属 instruments 文件")
    args = parser.parse_args()

    h5_path = args.h5
    output_dir = Path(args.output)
    batch_size = args.batch_size

    if not Path(h5_path).exists():
        print(f"ERROR: H5 文件不存在: {h5_path}")
        sys.exit(1)

    t_start = time.time()

    # ── 1. 加载 H5 ──
    df = load_h5(h5_path)

    # ── 2. 获取股票列表 ──
    all_instruments = sorted(df.index.get_level_values("instrument").unique())
    if args.stock_codes:
        stock_list = [s.upper() for s in args.stock_codes if s.upper() in all_instruments]
    else:
        stock_list = all_instruments
    print(f"股票数量: {len(stock_list)}")

    # ── 3. 构建全局日历 ──
    calendar = build_calendars_and_instruments(df, output_dir)
    # 如果指定了市场, 复制 all.txt 作为市场专属 instruments 文件
    if args.market and args.market != "all":
        inst_dir = output_dir / "instruments"
        market_file = inst_dir / f"{args.market}.txt"
        import shutil
        shutil.copy(inst_dir / "all.txt", market_file)
        print(f"  市场 instruments 文件: {market_file.name}")
    cal_compact = [d.replace("-", "") for d in calendar]
    global_start_idx = 0  # H5 数据从 index 0 开始

    # ── 4. 确定需要的原始字段 ──
    needed_raw = list(H5_COLUMN_MAP.keys())
    # 映射: 内部名 → H5 列名
    h5_col_to_internal = {v: k for k, v in H5_COLUMN_MAP.items()}

    # ── 4b. --new-factor-only: 解析 practice_factor.md 白名单 ──
    practice_whitelist = None
    _include_alpha158 = args.include_alpha158
    if args.new_factor_only:
        practice_whitelist = parse_practice_factors(args.practice_factor_file)
        if practice_whitelist is None:
            print("ERROR: --new-factor-only 需要有效的 practice_factor.md")
            sys.exit(1)
        if _include_alpha158:
            print(f"  Alpha158+模式: Alpha158(158) + 白名单 {len(practice_whitelist)} 个独立因子")
        else:
            print(f"  白名单模式: 仅输出 {len(practice_whitelist)} 个实践因子")

    # ── 5. 逐批计算因子 & 写入 bin ──
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有计算出的因子名
    all_factor_names = None  # 延迟确定

    batches = [stock_list[i:i + batch_size] for i in range(0, len(stock_list), batch_size)]
    print(f"批次: {len(batches)} (batch_size={batch_size})")

    # 确保 close 被保留 (用于 label 计算)
    KEEP_RAW = {"close"}

    # 并行 worker 数
    n_workers = args.workers if args.workers > 0 else min(multiprocessing.cpu_count(), 8)

    for bi, batch in enumerate(batches):
        print(f"  Batch {bi+1}/{len(batches)}: {len(batch)} stocks (workers={n_workers}) ...",
              end=" ", flush=True)
        t_batch = time.time()

        # Phase 1: 串行从 H5 提取原始数据 (HDF5 不支持高并发读)
        batch_data = []  # list of (symbol, raw_dict)
        for symbol in batch:
            sym_lower = symbol.lower()
            try:
                stock_df = df.loc[pd.IndexSlice[:, symbol], :]
            except KeyError:
                continue
            if stock_df.empty:
                continue
            stock_df = stock_df.sort_index(level="datetime")
            raw = {}
            for internal_name, h5_col in H5_COLUMN_MAP.items():
                if h5_col in stock_df.columns:
                    raw[internal_name] = stock_df[h5_col].values.astype(np.float64)
            if "close" in raw:
                batch_data.append((symbol, raw))

        # Phase 2: 并行计算因子
        if n_workers > 1 and len(batch_data) > 1:
            with multiprocessing.Pool(processes=n_workers) as pool:
                raw_list = [d[1] for d in batch_data]
                factor_results = pool.map(compute_factors_for_stock, raw_list)
        else:
            factor_results = [compute_factors_for_stock(d[1]) for d in batch_data]

        # Phase 3: 串行写 bin 文件
        for (symbol, raw), factors in zip(batch_data, factor_results):
            sym_lower = symbol.lower()

            if all_factor_names is None:
                if practice_whitelist is not None:
                    if _include_alpha158:
                        whitelisted_independent = practice_whitelist & set(factors.keys())
                        excluded_alpha158 = parse_alpha158_exclusions(args.practice_factor_file)
                        if excluded_alpha158:
                            effective_alpha158 = ALPHA158_NAMES - excluded_alpha158
                            n_excluded = len(excluded_alpha158 & ALPHA158_NAMES)
                        else:
                            effective_alpha158 = ALPHA158_NAMES
                            n_excluded = 0
                        all_factor_names = sorted(effective_alpha158 | whitelisted_independent)
                        skipped = sorted(set(factors.keys()) - ALPHA158_NAMES - whitelisted_independent)
                        print(f"  Alpha158({len(effective_alpha158)}/{len(ALPHA158_NAMES)})"
                              f" + {len(whitelisted_independent)} 独立"
                              f" (排除 {n_excluded} Alpha158, 跳过 {len(skipped)} 个)",
                              end=" ", flush=True)
                    else:
                        all_factor_names = sorted(practice_whitelist & set(factors.keys()))
                        skipped = sorted(set(factors.keys()) - practice_whitelist)
                        print(f"  白名单模式: {len(all_factor_names)} 个因子 (跳过 {len(skipped)} 个)",
                              end=" ", flush=True)
                else:
                    all_factor_names = sorted(factors.keys())
                    print(f"  检测到 {len(all_factor_names)} 个因子", end=" ", flush=True)

            feat_dir = features_dir / sym_lower
            feat_dir.mkdir(parents=True, exist_ok=True)

            for fname in all_factor_names:
                arr = factors.get(fname)
                if arr is None:
                    arr = np.full(len(raw["close"]), np.nan, dtype=np.float64)
                write_bin(feat_dir / f"{fname}{BIN_SUFFIX}", global_start_idx, arr.astype(np.float32))

            # 保留 close (用于 label 计算, 不参与归一化)
            close_path = feat_dir / f"close{BIN_SUFFIX}"
            if not close_path.exists():
                write_bin(close_path, global_start_idx, raw["close"].astype(np.float32))

        print(f"({time.time() - t_batch:.1f}s)")

    # ── 6. 截面因子 (行业中性化, 需要所有股票数据) ──
    if practice_whitelist is None or any(
        s in practice_whitelist for s in SECTOR_RELATIVE_MAP
    ):
        # 加载申万行业映射 (不存在则回退板块分类)
        sw_map = _load_sw_industry_mapping(args.sw_industry_file)
        print("计算截面因子 (Sector_Relative_PB/PE/DividendYield) ...")
        cs_computed = compute_sector_relative_factors(
            features_dir, stock_list, global_start_idx, sw_map=sw_map,
        )
        for cs_name in cs_computed:
            if practice_whitelist is None or cs_name in practice_whitelist:
                if cs_name not in (all_factor_names or []):
                    all_factor_names = (all_factor_names or []) + [cs_name]
    else:
        print("  跳过截面因子 (不在白名单中)")

    # ── 7. 基准数据 ──
    print("检查基准数据 (SH000300) ...")
    ensure_benchmark_data(output_dir, calendar, global_start_idx)

    # ── 8. 截面归一化 ──
    if not args.skip_normalize and all_factor_names:
        # 排除不需要归一化的字段 (行情原始数据 + close)
        norms = [f for f in all_factor_names if f not in EXCLUDE_FROM_NORM]
        # 根据股票数量自适应调整最小阈值
        _min_valid = max(3, min(10, len(stock_list) // 2))
        _min_per_date = max(3, min(5, len(stock_list) // 2))
        print(f"截面归一化: {len(norms)} 个特征, {len(stock_list)} 只股票 "
              f"(min_valid={_min_valid}, min_per_date={_min_per_date}) ...")
        t_norm = time.time()
        norm_results = normalize_cross_sectional(
            features_dir, norms, stock_list, batch_size=batch_size,
            min_valid_stocks=_min_valid, min_stocks_per_date=_min_per_date,
        )
        n_ok = sum(1 for v in norm_results.values() if v["stocks_processed"] > 0)
        print(f"  归一化耗时: {time.time() - t_norm:.1f}s ({n_ok}/{len(norms)} 个特征已归一化)")

    # Winsorize 已集成到 normalize_cross_sectional() 中,
    # 对所有归一化特征统一做截面缩尾 (1%/99%), 不再单独处理特选因子.

    # ── 9. 因子清单 ──
    manifest = {
        "description": "从 H5 文件构建的因子特征 (Alpha158 + practice_factor.md v3.0)",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_h5": str(h5_path),
        "stocks_processed": len(stock_list),
        "total_factors": len(all_factor_names or []),
        "alpha158_kbar": sorted(ALPHA158_KBAR),
        "alpha158_price": sorted(ALPHA158_PRICE),
        "independent_factors": sorted(INDEPENDENT_FACTORS),
        "overlap_factors": sorted(OVERLAP_FACTORS),
        "all_factors": sorted(all_factor_names or []),
    }
    manifest_path = output_dir / "factor_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"因子清单: {manifest_path}")

    # ── 10. 统计 ──
    stock_dirs = list(features_dir.glob("*"))
    if stock_dirs:
        sample = stock_dirs[0]
        n_bins = len(list(sample.glob(f"*{BIN_SUFFIX}")))
        print(f"每只股票: {n_bins} 个文件")

    print(f"\n总耗时: {time.time() - t_start:.1f}s")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
