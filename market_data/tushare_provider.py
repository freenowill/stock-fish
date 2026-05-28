"""
Tushare Pro A 股数据提供者

使用 sxsc_tushare 包（参考 qlib-zh/tushare 的封装方式）。
提供行情/财务/基本面数据。

用法:
    provider = TushareBackend(token="your_token")
    quote = provider.get_quote("600519")
"""
import os
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import pandas as pd
from loguru import logger

from market_data.a_stock_provider import (
    BaseStockBackend, Quote, FinancialSummary, NewsItem, GubaPost
)


# 简单滑动窗口限速器（参考 qlib-zh/api_utils.py 的 RateLimiter）
class _RateLimiter:
    def __init__(self, max_per_sec=18, max_per_min=280):
        self.max_per_sec = max_per_sec
        self.max_per_min = max_per_min
        self._timestamps = []

    def acquire(self):
        now = time.time()
        cutoff = now - 60
        self._timestamps = [t for t in self._timestamps if t > cutoff]

        # 秒级限流
        sec_calls = sum(1 for t in self._timestamps if t > now - 1)
        if sec_calls >= self.max_per_sec:
            recent = sorted(t for t in self._timestamps if t > now - 1)
            wait = recent[0] + 1.001 - now
            if wait > 0:
                time.sleep(wait)
                now = time.time()

        # 分钟级限流
        self._timestamps = [t for t in self._timestamps if t > now - 60]
        min_calls = len(self._timestamps)
        if min_calls >= self.max_per_min:
            wait = self._timestamps[0] + 60.001 - now
            if wait > 0:
                time.sleep(wait)

        self._timestamps.append(time.time())


class TushareBackend(BaseStockBackend):
    """Tushare Pro 数据后端（基于 sxsc_tushare）"""

    def __init__(self, token: Optional[str] = None):
        from config import settings
        self.token = token or os.environ.get('TUSHARE_TOKEN') or settings.TUSHARE_TOKEN or ''
        self._api = None
        self._limiter = _RateLimiter()

    def _get_api(self):
        if self._api is None:
            import sxsc_tushare as sx
            if not self.token:
                raise ValueError("TUSHARE_TOKEN 未配置")
            sx.set_token(self.token)
            self._api = sx.get_api(env="prd")
        return self._api

    def _query(self, api_name: str, **kwargs) -> pd.DataFrame:
        """带限速和重试的 API 查询"""
        for attempt in range(3):
            try:
                self._limiter.acquire()
                api = self._get_api()
                df = api.query(api_name, **kwargs)
                if df is not None and isinstance(df, pd.DataFrame):
                    return df
                return pd.DataFrame()
            except Exception as e:
                msg = str(e)
                if any(kw in msg for kw in ("每分钟最多访问", "每小时最多访问")):
                    wait = 65 if "小时" in msg else 65
                    logger.warning(f"[{api_name}] 限流，等待 {wait}s")
                    time.sleep(wait)
                elif attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"[{api_name}] 查询失败: {e}")
                    return pd.DataFrame()
        return pd.DataFrame()

    @staticmethod
    def _ts_code(symbol: str) -> str:
        """600519 -> 600519.SH"""
        code = symbol.strip().zfill(6)
        if code.startswith(('6', '9')):
            return f"{code}.SH"
        elif code.startswith(('0', '3')):
            return f"{code}.SZ"
        return f"{code}.SH"

    # ---- 实时行情 ----

    def get_quote(self, symbol: str) -> Optional[Quote]:
        try:
            ts_code = self._ts_code(symbol)
            today = datetime.now().strftime('%Y%m%d')

            # 最新日线
            df = self._query("daily", ts_code=ts_code, start_date=today, end_date=today)
            if df.empty:
                yesterday = (datetime.now() - timedelta(days=5)).strftime('%Y%m%d')
                df = self._query("daily", ts_code=ts_code, start_date=yesterday, end_date=today)
            if df.empty:
                return None

            row = df.iloc[0]

            # 股票名称
            basic = self._query("stock_basic", ts_code=ts_code)
            name = basic.iloc[0]['name'] if not basic.empty else symbol

            # 每日基本面
            pe = pb = turnover_rate = total_mv = None
            trade_date = str(row.get('trade_date', today))
            db = self._query("daily_basic", ts_code=ts_code, trade_date=trade_date)
            if not db.empty:
                r = db.iloc[0]
                pe = float(r['pe']) if pd.notna(r.get('pe')) else None
                pb = float(r['pb']) if pd.notna(r.get('pb')) else None
                tr = float(r['turnover_rate']) if pd.notna(r.get('turnover_rate')) else None
                turnover_rate = round(tr * 100, 2) if tr else None
                mv = float(r['total_mv']) if pd.notna(r.get('total_mv')) else None
                total_mv = round(mv / 10000, 2) if mv else None

            price = float(row['close'])
            pre_close = float(row.get('pre_close', price))
            change = price - pre_close
            change_pct = (change / pre_close * 100) if pre_close else 0

            return Quote(
                symbol=ts_code, name=name,
                price=round(price, 2), change=round(change, 2),
                change_pct=round(change_pct, 2),
                volume=float(row.get('vol', 0)),
                amount=float(row.get('amount', 0)),
                high=float(row.get('high', 0)),
                low=float(row.get('low', 0)),
                open_=float(row.get('open', 0)),
                prev_close=round(pre_close, 2),
                turnover_rate=turnover_rate, pe=pe, pb=pb,
                market_cap=total_mv,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            )
        except Exception as e:
            logger.error(f"Tushare 行情失败 [{symbol}]: {e}")
            return None

    # ---- 历史 K 线 ----

    def get_historical(self, symbol: str, days: int = 365) -> Optional[List[Dict]]:
        try:
            ts_code = self._ts_code(symbol)
            end = datetime.now()
            start = end - timedelta(days=days)
            df = self._query("daily", ts_code=ts_code,
                             start_date=start.strftime('%Y%m%d'),
                             end_date=end.strftime('%Y%m%d'))
            if df.empty:
                return None
            df = df.sort_values('trade_date')
            return [
                {
                    'date': str(r['trade_date']),
                    'open': float(r['open']), 'high': float(r['high']),
                    'low': float(r['low']), 'close': float(r['close']),
                    'volume': float(r['vol']), 'amount': float(r.get('amount', 0)),
                }
                for _, r in df.iterrows()
            ]
        except Exception as e:
            logger.error(f"Tushare 历史数据失败 [{symbol}]: {e}")
            return None

    # ---- 基本面 ----

    def get_financials(self, symbol: str) -> Optional[FinancialSummary]:
        try:
            ts_code = self._ts_code(symbol)
            basic = self._query("stock_basic", ts_code=ts_code)
            name = basic.iloc[0]['name'] if not basic.empty else symbol

            df = self._query("fina_indicator", ts_code=ts_code, limit=1)
            if df.empty:
                return FinancialSummary(symbol=ts_code, name=name)

            r = df.iloc[0]

            # 参考 qlib-zh: 财务字段可能有不同命名
            eps = float(r['eps']) if pd.notna(r.get('eps')) else None
            roe = float(r['roe']) if pd.notna(r.get('roe')) else None

            # revenue 和 net_profit 可能在 fina_indicator 中叫不同字段名
            revenue = None
            for col in ['revenue', 'operating_revenue', 'b_income']:
                if col in r and pd.notna(r[col]):
                    revenue = float(r[col]) / 1e8
                    break

            net_profit = None
            for col in ['net_profit', 'net_profit_is', 'c_income']:
                if col in r and pd.notna(r[col]):
                    net_profit = float(r[col]) / 1e8
                    break

            return FinancialSummary(
                symbol=ts_code, name=name, eps=eps, roe=roe,
                revenue=revenue, net_profit=net_profit,
                report_date=str(r.get('end_date', '')),
            )
        except Exception as e:
            logger.warning(f"Tushare 财务数据失败 [{symbol}]: {e}")
            return None

    def get_news(self, symbol: str) -> List[NewsItem]:
        return []

    def get_guba(self, symbol: str) -> List[GubaPost]:
        return []

    def get_historical_pe(self, symbol: str, days: int = 365 * 3) -> List[float]:
        try:
            ts_code = self._ts_code(symbol)
            end = datetime.now()
            start = end - timedelta(days=days)
            df = self._query("daily_basic", ts_code=ts_code,
                             start_date=start.strftime('%Y%m%d'),
                             end_date=end.strftime('%Y%m%d'))
            if df.empty:
                return []
            pe_vals = []
            for _, row in df.iterrows():
                pe = row.get('pe')
                if pd.notna(pe):
                    pe_vals.append(float(pe))
            return pe_vals
        except Exception as e:
            logger.warning(f"Tushare 历史PE获取失败 [{symbol}]: {e}")
            return []
