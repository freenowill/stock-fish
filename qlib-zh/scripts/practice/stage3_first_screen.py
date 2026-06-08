#!/usr/bin/env python3
"""
stage3_first_screen.py
初筛选股：
    1. 从目标指数（默认 CSI300，可通过 --market 改为 CSI500 等）成分股中
  2. 排除 ST、涨停/跌停、停牌、上市不足60天、日均成交额排名后20%
    3. 从初筛池中取模型得分最高的 top_n（默认20）支
数据源：掘金量化 gm 免费数据
输出: <output>/first_screen.csv
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from datetime import timedelta
from pathlib import Path

import pandas as pd

try:
    from gm_client import GmClient
except Exception:  # pragma: no cover - environment dependent
    GmClient = None  # type: ignore[assignment]

warnings.filterwarnings("ignore")

_QLIB_INITED = False


def _affordable_price_cap() -> float:
    cash_total = float(os.environ.get("CASH_TOTAL", "20000"))
    hold_num = int(os.environ.get("HOLD_NUM", "5"))
    fee = float(os.environ.get("TX_FEE_RATE", "0.0001"))
    per_stock_budget = cash_total / max(hold_num, 1)
    return max(per_stock_budget / (100 * (1 + fee)), 0.0)


def _qlib_root() -> Path:
    return Path(os.environ.get("QLIB_DATA_DIR", "/root/.qlib/qlib_data/cn_data"))


def _ensure_qlib():
    global _QLIB_INITED
    if _QLIB_INITED:
        return
    try:
        import qlib
        from qlib.constant import REG_CN
        qlib.init(provider_uri=str(_qlib_root()), region=REG_CN)
        _QLIB_INITED = True
    except Exception as exc:
        raise RuntimeError(f"无法初始化本地 Qlib 数据：{exc}") from exc


def _normalize_qlib_code(code: str) -> str:
    code = _to_6digit(code)
    prefix = "SH" if code.startswith(("5", "6", "9")) else "SZ"
    return f"{prefix}{code}"


def _parse_instruments_file(path: Path, pred_date: str | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["code", "listed_date", "delisted_date"])
    df = pd.read_csv(path, sep="\t", header=None, names=["code", "listed_date", "delisted_date"])
    df["listed_date"] = pd.to_datetime(df["listed_date"], errors="coerce")
    df["delisted_date"] = pd.to_datetime(df["delisted_date"], errors="coerce")
    if pred_date is not None:
        dt = pd.to_datetime(pred_date)
        df = df[(df["listed_date"].isna() | (df["listed_date"] <= dt)) & (df["delisted_date"].isna() | (df["delisted_date"] >= dt))]
    return df


def _local_basic_info(codes: list[str]) -> pd.DataFrame:
    inst = _parse_instruments_file(_qlib_root() / "instruments" / "all.txt")
    inst["code"] = inst["code"].astype(str).map(_normalize_symbol)
    out = inst[inst["code"].isin({_normalize_symbol(c) for c in codes})].copy()
    if out.empty:
        out = pd.DataFrame({"code": [_normalize_symbol(c) for c in codes]})
        out["listed_date"] = pd.NaT
    out["sec_name"] = out["code"]
    out["symbol"] = out["code"].map(lambda s: f"SHSE.{s}" if s.startswith(("5", "6", "9")) else f"SZSE.{s}")
    return out[["code", "sec_name", "listed_date", "symbol"]].drop_duplicates(subset=["code"])


def _local_market_components(market: str, pred_date: str) -> set[str]:
    path = _qlib_root() / "instruments" / f"{market}.txt"
    if not path.exists():
        return set()
    df = pd.read_csv(path, sep="\t", header=None, names=["code", "listed_date", "delisted_date"])
    df["listed_date"] = pd.to_datetime(df["listed_date"], errors="coerce")
    df["delisted_date"] = pd.to_datetime(df["delisted_date"], errors="coerce")
    dt = pd.to_datetime(pred_date)
    df = df[(df["listed_date"].isna() | (df["listed_date"] <= dt)) & (df["delisted_date"].isna() | (df["delisted_date"] >= dt))]
    return set(df["code"].astype(str).map(_normalize_symbol))


def _local_history(symbols: list[str], start_time: str, end_time: str, fields: list[str]) -> pd.DataFrame:
    _ensure_qlib()
    from qlib.data import D

    q_symbols = [_normalize_qlib_code(s) for s in symbols]
    q_fields = [f if f.startswith("$") else f"${f}" for f in fields]
    df = D.features(q_symbols, q_fields, start_time=start_time, end_time=end_time)
    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    if "instrument" in df.columns:
        df["symbol"] = df["instrument"].map(lambda s: str(s).replace("SH", "SHSE.").replace("SZ", "SZSE."))
    elif "code" not in df.columns:
        df["symbol"] = q_symbols[0] if len(q_symbols) == 1 else None
    return df


def _to_6digit(code: str) -> str:
    code = str(code).strip()
    if code.count(".") == 1:
        left, right = code.split(".")
        code = right if right.isdigit() else left
    if code.upper().startswith(("SH", "SZ")):
        code = code[2:]
    return code.zfill(6)


def _to_gm_symbol(code: str) -> str:
    code = _to_6digit(code)
    prefix = "SHSE" if code.startswith(("5", "6", "9")) else "SZSE"
    return f"{prefix}.{code}"


def _normalize_symbol(value: str) -> str:
    return _to_6digit(value)


def _resolve_scores_csv(pred_path: Path) -> Path:
    scores_csv = pred_path / "scores.csv"
    if scores_csv.exists():
        return scores_csv

    candidates = sorted(pred_path.glob("walk_forward/*/model_predict/scores.csv"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"scores.csv 不存在: {scores_csv}")


def _to_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def _get_basic_info(gm: GmClient | None, codes: list[str]) -> pd.DataFrame:
    symbols = [_to_gm_symbol(code) for code in codes]
    if gm is None:
        return _local_basic_info(codes)
    try:
        df = gm.get_instruments(symbols=symbols, fields="symbol,sec_name,listed_date,delisted_date")
        if df is not None and not df.empty:
            out = df.copy()
            out["code"] = out["symbol"].astype(str).map(_normalize_symbol)
            out["listed_date"] = pd.to_datetime(out.get("listed_date"), errors="coerce")
            out["sec_name"] = out.get("sec_name", "")
            return out[["code", "sec_name", "listed_date", "symbol"]].drop_duplicates(subset=["code"])
        print("  ⚠ gm get_instruments 为空，切换到本地 Qlib instruments/all.txt")
    except Exception as exc:
        print(f"  ⚠ gm get_instruments 失败，切换到本地 Qlib 数据: {exc}")
    return _local_basic_info(codes)


def _get_market_components(gm: GmClient | None, market: str, pred_date: str) -> set[str]:
    print(f"  [gm] 获取 {market.upper()} 成分股...")
    if gm is None:
        print(f"  ⚠ 未提供 gm 客户端，切换到本地 Qlib {market}.txt")
        return _local_market_components(market, pred_date)
    start_date = (pd.to_datetime(pred_date) - timedelta(days=90)).strftime("%Y-%m-%d")
    index_code = {
        "csi300": "SHSE.000300",
        "csi500": "SHSE.000905",
        "csi800": "SHSE.000906",
        "csi1000": "SHSE.000852",
    }.get(market.lower())
    if index_code is None:
        print(f"  ⚠ 未知 market={market}，回退到本地 Qlib {market}.txt")
        return _local_market_components(market, pred_date)
    try:
        df = gm.get_history_constituents(index=index_code, start_date=start_date, end_date=pred_date)
    except Exception as exc:
        print(f"  ⚠ get_history_constituents 失败: {exc}")
        return _local_market_components(market, pred_date)
    if df is None or df.empty:
        print(f"  ⚠ gm {market.upper()} 成分为空，切换到本地 Qlib {market}.txt")
        return _local_market_components(market, pred_date)
    code_col = next((c for c in df.columns if c.lower() in {"symbol", "code", "con_code"}), None)
    if code_col is None:
        return _local_market_components(market, pred_date)
    return set(df[code_col].astype(str).map(_normalize_symbol))


def _get_st_list(gm: GmClient | None, basic_df: pd.DataFrame, pred_date: str) -> set[str]:
    print("  [gm] 获取 ST 股票列表...")
    st_set = set(basic_df.loc[basic_df["sec_name"].astype(str).str.contains("ST", case=False, na=False), "code"])
    if gm is None:
        return st_set
    try:
        df = gm.get_history_instruments(
            symbols=list(basic_df["symbol"].dropna().astype(str).head(3000)),
            start_date=pred_date,
            end_date=pred_date,
            fields="symbol,trade_date,is_st,is_suspended,sec_name",
        )
        if df is not None and not df.empty:
            if "is_st" in df.columns:
                st_set.update(df.loc[_to_bool_series(df["is_st"]), "symbol"].astype(str).map(_normalize_symbol))
            if "sec_name" in df.columns:
                st_set.update(df.loc[df["sec_name"].astype(str).str.contains("ST", case=False, na=False), "symbol"].astype(str).map(_normalize_symbol))
    except Exception as exc:
        print(f"  ⚠ get_history_instruments 不可用，当前回退方案无法稳定识别 ST: {exc}")
    return st_set


def _history_quotes(gm: GmClient | None, symbols: list[str], start_time: str, end_time: str, fields: str) -> pd.DataFrame:
    frames = []
    if gm is None:
        return _local_history(symbols, start_time, end_time, fields.split(","))
    for i in range(0, len(symbols), 50):
        chunk = symbols[i : i + 50]
        df = gm.history(chunk, start_time=start_time, end_time=end_time, fields=fields, frequency="1d")
        if df is not None and not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _latest_quotes(gm: GmClient | None, codes: set[str], pred_date: str, basic_df: pd.DataFrame) -> pd.DataFrame:
    print(f"  [gm] 获取 {len(codes)} 只股票最近行情...")
    symbols = [_to_gm_symbol(code) for code in sorted(codes)]
    start_time = (pd.to_datetime(pred_date) - timedelta(days=15)).strftime("%Y-%m-%d")
    end_time = pd.to_datetime(pred_date).strftime("%Y-%m-%d")
    try:
        df = _history_quotes(gm, symbols, start_time, end_time, fields="close,amount,volume")
    except Exception as exc:
        print(f"  ⚠ gm history 失败，切换到本地 Qlib 数据: {exc}")
        df = _local_history(list(codes), start_time, end_time, fields=["close", "amount", "volume"])
    if df.empty:
        base = basic_df[basic_df["code"].isin(codes)][["code", "sec_name"]].copy()
        base["price"] = pd.NA
        base["change_pct"] = pd.NA
        base["turnover_amount"] = pd.NA
        base["is_suspended"] = True
        return base

    date_col = next((c for c in ["eob", "trade_date", "datetime"] if c in df.columns), None)
    if date_col is None:
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if "symbol" in df.columns:
        df["code"] = df["symbol"].astype(str).map(_normalize_symbol)
    elif "instrument" in df.columns:
        df["code"] = df["instrument"].astype(str).map(_normalize_symbol)
    df["close"] = pd.to_numeric(df.get("close", df.get("$close")), errors="coerce")
    df["amount"] = pd.to_numeric(df.get("amount", df.get("$amount")), errors="coerce")
    df = df.sort_values(["code", date_col])

    latest = []
    for code, grp in df.groupby("code"):
        grp = grp.dropna(subset=["close"]).tail(2)
        if grp.empty:
            continue
        last = grp.iloc[-1]
        prev = grp.iloc[-2]["close"] if len(grp) >= 2 else pd.NA
        latest.append(
            {
                "code": code,
                "price": last["close"],
                "change_pct": ((last["close"] / prev - 1) * 100) if pd.notna(prev) and prev not in (0, 0.0) else pd.NA,
                "turnover_amount": last.get("amount", pd.NA),
            }
        )
    latest_df = pd.DataFrame(latest)

    suspend_set = set()
    if gm is None:
        base = basic_df[basic_df["code"].isin(codes)][["code", "sec_name"]].copy()
        out = base.merge(latest_df, on="code", how="left")
        out["is_suspended"] = out["price"].isna() | (out["price"].fillna(0) <= 0)
        return out
    try:
        status = gm.get_history_instruments(
            symbols=symbols,
            start_date=pred_date,
            end_date=pred_date,
            fields="symbol,trade_date,is_suspended",
        )
        if status is not None and not status.empty and "is_suspended" in status.columns:
            suspend_set = set(status.loc[_to_bool_series(status["is_suspended"]), "symbol"].astype(str).map(_normalize_symbol))
    except Exception as exc:
        print(f"  ⚠ 停牌状态获取失败，回退为不识别停牌: {exc}")

    base = basic_df[basic_df["code"].isin(codes)][["code", "sec_name"]].copy()
    out = base.merge(latest_df, on="code", how="left")
    out["is_suspended"] = out["code"].isin(suspend_set) | out["price"].isna() | (out["price"].fillna(0) <= 0)
    return out


def _avg_turnover(gm: GmClient | None, codes: set[str], pred_date: str, days: int = 20) -> dict[str, float]:
    print(f"  [gm] 获取 {len(codes)} 只股票近{days}日成交额...")
    symbols = [_to_gm_symbol(code) for code in sorted(codes)]
    start_time = (pd.to_datetime(pred_date) - timedelta(days=max(days * 3, 45))).strftime("%Y-%m-%d")
    end_time = pd.to_datetime(pred_date).strftime("%Y-%m-%d")
    try:
        df = _history_quotes(gm, symbols, start_time, end_time, fields="amount")
    except Exception as exc:
        print(f"  ⚠ gm history 失败，切换到本地 Qlib 数据: {exc}")
        df = _local_history(list(codes), start_time, end_time, fields=["amount"])
    if df.empty:
        return {code: 0.0 for code in codes}
    date_col = next((c for c in ["eob", "trade_date", "datetime"] if c in df.columns), None)
    if date_col is None:
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if "symbol" in df.columns:
        df["code"] = df["symbol"].astype(str).map(_normalize_symbol)
    elif "instrument" in df.columns:
        df["code"] = df["instrument"].astype(str).map(_normalize_symbol)
    df["amount"] = pd.to_numeric(df.get("amount", df.get("$amount")), errors="coerce")
    df = df.sort_values(["code", date_col])
    return df.groupby("code")["amount"].apply(lambda s: float(s.dropna().tail(days).mean()) if not s.dropna().empty else 0.0).to_dict()


def first_screen(pred_dir: str, output_dir: str, pred_date: str, top_n: int = 20, max_price: float = 50.0, market: str = "csi300"):
    gm = None
    gm_token = os.getenv("GM_TOKEN", "").strip()
    if gm_token and GmClient is not None:
        try:
            gm = GmClient(gm_token)
        except Exception as exc:
            print(f"  ⚠ gm 初始化失败，切换到本地 Qlib 数据: {exc}")
    elif gm_token and GmClient is None:
        print("  ⚠ gm_client 不可用，切换到本地 Qlib 数据")
    pred_path = Path(pred_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    scores_csv = _resolve_scores_csv(pred_path)
    scores_df = pd.read_csv(scores_csv)
    if "pred_date" in scores_df.columns and not scores_df["pred_date"].dropna().empty:
        file_pred_date = str(scores_df["pred_date"].dropna().astype(str).iloc[0])
        if file_pred_date != pred_date:
            print(f"  ⚠ 输入 pred_date={pred_date} 与 scores.csv 中的 pred_date={file_pred_date} 不一致，已自动对齐为后者")
            pred_date = file_pred_date
    scores_df["code"] = scores_df["instrument"].astype(str).map(_normalize_symbol)
    scores_df["pred_date"] = pred_date
    print(f"✓ 模型得分加载完成，共 {len(scores_df)} 只股票")

    basic_df = _get_basic_info(gm, scores_df["code"].dropna().astype(str).tolist())
    components = _get_market_components(gm, market, pred_date)
    if components:
        pool = scores_df[scores_df["code"].isin(components)].copy()
        print(f"✓ {market.upper()} 筛选后剩余: {len(pool)} 只")
        if pool.empty:
            print(f"  ⚠ {market.upper()} 成分与预测池无交集，回退为全部得分股票作为候选池")
            pool = scores_df.copy()
    else:
        print(f"  ⚠ {market.upper()} 成分获取失败，使用全部得分股票作为候选池")
        pool = scores_df.copy()

    pool = pool.merge(
        basic_df[["code", "sec_name", "symbol", "listed_date"]].rename(columns={"sec_name": "name"}),
        on="code",
        how="left",
    )

    st_set = _get_st_list(gm, basic_df, pred_date)
    before = len(pool)
    pool = pool[~pool["code"].isin(st_set)]
    print(f"✓ 排除 ST 后剩余: {len(pool)} (移除 {before - len(pool)} 只)")

    base_pool = pool.copy()
    codes_now = set(base_pool["code"].tolist())
    realtime = _latest_quotes(gm, codes_now, pred_date, basic_df)

    def _screen(apply_limit: bool, apply_turnover: bool, apply_price_cap: bool) -> pd.DataFrame:
        candidate = base_pool.copy()

        if not realtime.empty and apply_limit:
            realtime_map = realtime.set_index("code").to_dict("index")

            def _is_limit(code):
                info = realtime_map.get(code, {})
                pct = info.get("change_pct", 0) or 0
                return abs(float(pct)) >= 9.5

            def _is_suspended(code):
                return bool(realtime_map.get(code, {}).get("is_suspended", False))

            before_local = len(candidate)
            candidate = candidate[~candidate["code"].apply(_is_limit)]
            candidate = candidate[~candidate["code"].apply(_is_suspended)]
            print(f"✓ 排除涨跌停/停牌后剩余: {len(candidate)} (移除 {before_local - len(candidate)} 只)")

        if apply_turnover and not realtime.empty:
            before_local = len(candidate)
            turnover_map = _avg_turnover(gm, set(candidate["code"].tolist()), pred_date)
            candidate["avg_turnover"] = candidate["code"].map(turnover_map)
            valid_turnover = pd.to_numeric(candidate["avg_turnover"], errors="coerce").dropna()
            if not valid_turnover.empty and (valid_turnover > 0).any():
                q20 = valid_turnover.quantile(0.2)
                candidate = candidate[pd.to_numeric(candidate["avg_turnover"], errors="coerce").fillna(0) > q20]
                print(f"✓ 排除成交额后20%后剩余: {len(candidate)} (移除 {before_local - len(candidate)} 只，阈值={q20:,.0f})")
            else:
                print("  ⚠ 成交额数据不足，跳过后20%流动性过滤")
        else:
            candidate["avg_turnover"] = float("nan")

        if apply_price_cap and max_price > 0 and "price" in candidate.columns:
            before_local = len(candidate)
            price_num = pd.to_numeric(candidate["price"], errors="coerce")
            candidate = candidate[price_num.notna() & (price_num <= max_price)].copy()
            print(f"✓ 价格上限过滤(≤{max_price:g})后剩余: {len(candidate)} (移除 {before_local - len(candidate)} 只)")

        pred_dt = pd.to_datetime(pred_date)
        before_local = len(candidate)
        candidate = candidate[(pred_dt - pd.to_datetime(candidate["listed_date"], errors="coerce")).dt.days.fillna(9999) >= 60]
        print(f"✓ 排除上市不足60天后剩余: {len(candidate)} (移除 {before_local - len(candidate)} 只)")

        return candidate

    # 先走严格版；如果被过度过滤清空，则逐级放宽，但不会影响原本已经可用的结果。
    attempts = [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, False, True),
        (False, False, False),
    ]
    pool = pd.DataFrame()
    for apply_limit, apply_turnover, apply_price_cap in attempts:
        candidate = _screen(apply_limit, apply_turnover, apply_price_cap)
        if not candidate.empty:
            pool = candidate
            break

    if pool.empty:
        print("  ⚠ 所有筛选尝试后仍为空，输出空表")
        out_csv = out_path / "first_screen.csv"
        pool.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n✓ 初筛结果保存: {out_csv}")
        return

    if "score_final" in pool.columns:
        score_col = "score_final"
    elif "robust_score" in pool.columns:
        score_col = "robust_score"
    else:
        score_col = "score"
    pool = pool.sort_values(score_col, ascending=False).head(top_n).reset_index(drop=True)
    print(f"\n◆ 初筛完成，选出 {len(pool)} 只股票:")
    display_cols = [c for c in ["code", score_col, "score", "rank_pct", "price", "avg_turnover"] if c in pool.columns]
    print(pool[display_cols].to_string(index=False))

    out_csv = out_path / "first_screen.csv"
    pool.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✓ 初筛结果保存: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, dest="pred_dir")
    ap.add_argument("--output", required=True)
    ap.add_argument("--pred-date", required=True, dest="pred_date")
    ap.add_argument("--top-n", type=int, default=20, dest="top_n")
    ap.add_argument("--max-price", type=float, default=0.0, dest="max_price")
    ap.add_argument("--market", default=os.environ.get("TARGET_MARKET", "csi300"))
    args = ap.parse_args()
    first_screen(args.pred_dir, args.output, args.pred_date, top_n=args.top_n, max_price=args.max_price, market=args.market)


if __name__ == "__main__":
    main()