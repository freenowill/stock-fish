from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("/Users/apple/github/qlib/DATA/myquant")
DEFAULT_INDEX = "SHSE.000300"
DEFAULT_START_DATE = "2018-01-01"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_df(df: pd.DataFrame, path: Path) -> None:
    _ensure_parent(path)
    df.to_csv(path, index=False)


def _unique_symbols(df: pd.DataFrame) -> list[str]:
    for col in ("symbol", "stock_symbol", "constituent_symbol", "secu_code"):
        if col in df.columns:
            symbols = (
                df[col]
                .dropna()
                .astype(str)
                .map(str.strip)
            )
            return sorted({s for s in symbols if s})

    if "constituents" in df.columns:
        symbols: set[str] = set()
        for item in df["constituents"].dropna():
            if isinstance(item, dict):
                symbols.update(str(k).strip() for k in item.keys() if str(k).strip())
        if symbols:
            return sorted(symbols)

    raise RuntimeError(
        "无法从指数成分结果中识别股票代码，请先检查 gm 返回字段结构。"
    )


def _require_gm():
    try:
        from gm.api import (  # type: ignore
            get_instruments,
            get_trading_dates,
            history_n,
            set_token,
            stk_get_daily_basic_pt,
            stk_get_daily_valuation_pt,
            stk_get_index_constituents,
            stk_get_index_history_constituents,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "当前 Python 环境无法导入 gm SDK。请在 Linux 容器内运行，或先安装掘金量化 gm SDK。"
        ) from exc

    return {
        "get_instruments": get_instruments,
        "get_trading_dates": get_trading_dates,
        "history_n": history_n,
        "set_token": set_token,
        "stk_get_daily_basic_pt": stk_get_daily_basic_pt,
        "stk_get_daily_valuation_pt": stk_get_daily_valuation_pt,
        "stk_get_index_constituents": stk_get_index_constituents,
        "stk_get_index_history_constituents": stk_get_index_history_constituents,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the minimum gm datasets used by the Alpha158 practice pipeline."
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--index", default=DEFAULT_INDEX, help="Index code, e.g. SHSE.000300")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Download start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"), help="Download end date (YYYY-MM-DD)")
    parser.add_argument("--history-days", type=int, default=0, help="Optional historical daily bars per symbol")
    parser.add_argument("--token", default="", help="GM token; overrides GM_TOKEN env")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    token = args.token.strip() or os.environ.get("GM_TOKEN", "").strip()
    if not token:
        print("GM_TOKEN 未设置，无法访问掘金量化数据。", file=sys.stderr)
        return 2

    gm = _require_gm()
    gm["set_token"](token)

    out_dir = Path(args.output).expanduser().resolve()
    raw_dir = out_dir / "raw"
    history_dir = out_dir / "history_n"
    raw_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "index": args.index,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "history_days": args.history_days,
    }

    # 交易日
    trading_dates = gm["get_trading_dates"]("SHSE", args.start_date, args.end_date)
    trading_dates_df = pd.DataFrame({"trade_date": trading_dates})
    _save_df(trading_dates_df, raw_dir / "trading_dates.csv")
    manifest["trading_dates_count"] = int(len(trading_dates_df))

    # 指数历史成分股
    hist_df = gm["stk_get_index_history_constituents"](args.index, args.start_date, args.end_date)
    if not isinstance(hist_df, pd.DataFrame):
        hist_df = pd.DataFrame(hist_df)
    _save_df(hist_df, raw_dir / "index_history_constituents.csv")
    manifest["index_history_constituents_rows"] = int(len(hist_df))

    # 指数最新成分股
    latest_df = gm["stk_get_index_constituents"](args.index, args.end_date)
    if not isinstance(latest_df, pd.DataFrame):
        latest_df = pd.DataFrame(latest_df)
    _save_df(latest_df, raw_dir / "index_latest_constituents.csv")
    manifest["index_latest_constituents_rows"] = int(len(latest_df))

    symbols = _unique_symbols(latest_df if len(latest_df) else hist_df)
    pd.DataFrame({"symbol": symbols}).to_csv(raw_dir / "index_latest_symbols.csv", index=False)
    manifest["symbol_count"] = int(len(symbols))

    # 最新标的快照
    instruments_df = gm["get_instruments"](symbols=symbols, df=True)
    if not isinstance(instruments_df, pd.DataFrame):
        instruments_df = pd.DataFrame(instruments_df)
    _save_df(instruments_df, raw_dir / "instruments_snapshot.csv")
    manifest["instruments_rows"] = int(len(instruments_df))

    # 最新截面基础数据
    basic_df = gm["stk_get_daily_basic_pt"](symbols, fields="", trade_date=args.end_date, df=True)
    if not isinstance(basic_df, pd.DataFrame):
        basic_df = pd.DataFrame(basic_df)
    _save_df(basic_df, raw_dir / "daily_basic_snapshot.csv")
    manifest["daily_basic_rows"] = int(len(basic_df))

    # 最新截面估值数据
    valuation_df = gm["stk_get_daily_valuation_pt"](symbols, fields="", trade_date=args.end_date, df=True)
    if not isinstance(valuation_df, pd.DataFrame):
        valuation_df = pd.DataFrame(valuation_df)
    _save_df(valuation_df, raw_dir / "daily_valuation_snapshot.csv")
    manifest["daily_valuation_rows"] = int(len(valuation_df))

    # 可选：回填最近若干日的日线数据，便于后续二次筛选/风控。
    if args.history_days > 0:
        history_errors: list[dict[str, str]] = []
        for idx, symbol in enumerate(symbols, start=1):
            try:
                bars_df = gm["history_n"](symbol, "1d", args.history_days, end_time=args.end_date, df=True)
                if not isinstance(bars_df, pd.DataFrame):
                    bars_df = pd.DataFrame(bars_df)
                _save_df(bars_df, history_dir / f"{symbol.replace('.', '_')}.csv")
            except Exception as exc:  # pragma: no cover - network dependent
                history_errors.append({"symbol": symbol, "error": str(exc)})
            if idx % 25 == 0:
                print(f"[gm] history_n progress: {idx}/{len(symbols)}")
        if history_errors:
            pd.DataFrame(history_errors).to_csv(raw_dir / "history_errors.csv", index=False)
        manifest["history_files"] = int(len(list(history_dir.glob('*.csv'))))
        manifest["history_error_count"] = int(len(history_errors))

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())