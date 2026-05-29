"""
A 股数据提供者

分层架构：
- AbstractStockProvider: 抽象基类
- AStockProvider: 主入口，自动选择可用后端
  - AkShareBackend: AKShare（东方财富数据，需中国大陆网络）
  - YahooBackend: yfinance（全球覆盖，部分中国数据可能受限）
  - SinaBackend: 新浪财经（轻量实时行情，需中国大陆网络）
  - MockBackend: 模拟数据（开发/演示用）
"""
import json
import os
import random
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Type
from dataclasses import dataclass, field, asdict
from pathlib import Path

from loguru import logger


# ========== 统一数据结构 ==========

@dataclass
class Quote:
    """实时行情"""
    symbol: str
    name: str
    price: float
    change: float
    change_pct: float
    volume: float
    amount: float
    high: float
    low: float
    open_: float
    prev_close: float
    turnover_rate: Optional[float] = None
    pe: Optional[float] = None
    pb: Optional[float] = None
    market_cap: Optional[float] = None
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FinancialSummary:
    """基本面摘要"""
    symbol: str
    name: str
    revenue: Optional[float] = None
    revenue_yoy: Optional[float] = None
    net_profit: Optional[float] = None
    net_profit_yoy: Optional[float] = None
    eps: Optional[float] = None
    roe: Optional[float] = None
    gross_margin: Optional[float] = None
    debt_ratio: Optional[float] = None
    report_date: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TechnicalIndicators:
    """技术指标"""
    symbol: str
    price: float
    ma5: Optional[float] = None
    ma10: Optional[float] = None
    ma20: Optional[float] = None
    ma60: Optional[float] = None
    rsi_14: Optional[float] = None
    macd_dif: Optional[float] = None
    macd_dea: Optional[float] = None
    macd_hist: Optional[float] = None
    kdj_k: Optional[float] = None
    kdj_d: Optional[float] = None
    kdj_j: Optional[float] = None
    boll_upper: Optional[float] = None
    boll_middle: Optional[float] = None
    boll_lower: Optional[float] = None
    volume_ratio: Optional[float] = None
    amplitude: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NewsItem:
    """新闻"""
    title: str
    url: str = ""
    publish_time: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GubaPost:
    """股吧帖子"""
    title: str
    author: str = ""
    publish_time: str = ""
    read_count: int = 0
    comment_count: int = 0
    content: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ========== 抽象基类 ==========

class BaseStockBackend(ABC):
    """数据后端抽象基类"""

    @abstractmethod
    def get_quote(self, symbol: str) -> Optional[Quote]:
        ...

    @abstractmethod
    def get_historical(self, symbol: str, days: int = 365) -> Optional[List[dict]]:
        ...

    @abstractmethod
    def get_financials(self, symbol: str) -> Optional[FinancialSummary]:
        ...

    @abstractmethod
    def get_news(self, symbol: str) -> List[NewsItem]:
        ...

    @abstractmethod
    def get_guba(self, symbol: str) -> List[GubaPost]:
        ...

    def get_historical_pe(self, symbol: str, days: int = 365) -> List[float]:
        """获取历史 PE 序列，用于估值分位计算。默认实现返回空列表。"""
        return []


# ========== 模拟后端（开发/演示） ==========

class MockBackend(BaseStockBackend):
    """模拟数据后端，无需网络连接"""

    def __init__(self, base_price: Optional[float] = None):
        self.base_price = base_price

    def _code_to_name(self, symbol: str) -> str:
        names = {
            "600519": "贵州茅台", "000858": "五粮液", "000333": "美的集团",
            "600036": "招商银行", "601318": "中国平安", "600900": "长江电力",
            "AAPL": "Apple Inc.", "TSLA": "Tesla Inc.", "MSFT": "Microsoft",
        }
        return names.get(symbol, f"股票{symbol}")

    def _rand_price(self, base: float) -> tuple:
        change_pct = random.uniform(-3.0, 3.0)
        change = base * change_pct / 100
        return round(base + change, 2), round(change, 2), round(change_pct, 2)

    def get_quote(self, symbol: str) -> Quote:
        base = self.base_price or random.uniform(10, 200)
        price, change, change_pct = self._rand_price(base)
        return Quote(
            symbol=symbol, name=self._code_to_name(symbol),
            price=price, change=change, change_pct=change_pct,
            volume=random.uniform(1e6, 1e8), amount=random.uniform(1e8, 1e10),
            high=round(price * 1.02, 2), low=round(price * 0.98, 2),
            open_=round(price - change, 2), prev_close=round(price - change, 2),
            turnover_rate=round(random.uniform(0.5, 5), 2),
            pe=round(random.uniform(10, 50), 2),
            pb=round(random.uniform(1, 10), 2),
            market_cap=round(random.uniform(100, 20000), 2),
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        )

    def get_historical(self, symbol: str, days: int = 365) -> List[dict]:
        rows = []
        price = self.base_price or 100.0
        now = datetime.now()
        for i in range(days):
            dt = now - timedelta(days=days - i)
            change = random.uniform(-3, 3)
            price *= (1 + change / 100)
            rows.append({
                "date": dt.strftime('%Y-%m-%d'),
                "open": round(price * 0.99, 2),
                "high": round(price * 1.01, 2),
                "low": round(price * 0.98, 2),
                "close": round(price, 2),
                "volume": random.randint(100000, 10000000),
            })
        return rows

    def get_financials(self, symbol: str) -> FinancialSummary:
        return FinancialSummary(
            symbol=symbol, name=self._code_to_name(symbol),
            revenue=round(random.uniform(10, 1000), 2),
            revenue_yoy=round(random.uniform(-10, 30), 2),
            net_profit=round(random.uniform(1, 200), 2),
            net_profit_yoy=round(random.uniform(-20, 40), 2),
            eps=round(random.uniform(0.5, 10), 2),
            roe=round(random.uniform(5, 25), 2),
            gross_margin=round(random.uniform(20, 80), 2),
            debt_ratio=round(random.uniform(20, 70), 2),
            report_date=(datetime.now() - timedelta(days=random.randint(30, 180))).strftime('%Y-%m-%d'),
        )

    def get_news(self, symbol: str) -> List[NewsItem]:
        templates = [
            f"{self._code_to_name(symbol)}发布最新财报，业绩超预期",
            f"机构上调{symbol}目标价，看好未来增长",
            f"行业利好政策出台，{symbol}有望受益",
            f"{self._code_to_name(symbol)}宣布回购计划",
            f"市场震荡中{symbol}表现抗跌",
        ]
        return [NewsItem(
            title=t, source="模拟数据",
            publish_time=(datetime.now() - timedelta(hours=i)).strftime('%Y-%m-%d %H:%M'),
        ) for i, t in enumerate(random.sample(templates, min(3, len(templates))))]

    def get_guba(self, symbol: str) -> List[GubaPost]:
        templates = [
            "下周必涨，立帖为证！",
            "主力资金正在吸筹",
            "技术面看多，目标价翻倍",
            "风险提示：短期涨幅过大",
            "利好出尽是利空？",
        ]
        return [GubaPost(
            title=t, author=f"股民{i}", read_count=random.randint(100, 10000),
            comment_count=random.randint(10, 500),
            publish_time=(datetime.now() - timedelta(hours=i)).strftime('%Y-%m-%d %H:%M'),
        ) for i, t in enumerate(templates)]

    def get_historical_pe(self, symbol: str, days: int = 365 * 3) -> List[float]:
        import numpy as np
        base_pe = random.uniform(15, 40)
        return list(np.random.normal(base_pe, base_pe * 0.3, min(days, 500)))


# ========== AKShare 后端（需中国大陆网络） ==========

class AkShareBackend(BaseStockBackend):
    """AKShare 后端，使用东方财富数据"""

    def __init__(self, proxy: Optional[str] = None):
        self.proxy = proxy
        self._ak = None

    def _get_ak(self):
        if self._ak is None:
            import akshare as ak
            if self.proxy:
                import os
                os.environ['AKSHARE_PROXY'] = self.proxy
            self._ak = ak
        return self._ak

    def get_quote(self, symbol: str) -> Optional[Quote]:
        ak = self._get_ak()
        try:
            df = ak.stock_zh_a_spot_em()
            code = symbol.zfill(6)
            match = df[df['代码'] == code]
            if match.empty:
                return None
            row = match.iloc[0]
            return Quote(
                symbol=code, name=row.get('名称', ''),
                price=float(row.get('最新价', 0)),
                change=float(row.get('涨跌额', 0)),
                change_pct=float(row.get('涨跌幅', 0)),
                volume=float(row.get('成交量', 0)),
                amount=float(row.get('成交额', 0)),
                high=float(row.get('最高', 0)),
                low=float(row.get('最低', 0)),
                open_=float(row.get('今开', 0)),
                prev_close=float(row.get('昨收', 0)),
                turnover_rate=float(row.get('换手率', 0)) if '换手率' in row else None,
                pe=float(row.get('市盈率-动态', 0)) if '市盈率-动态' in row else None,
                pb=float(row.get('市净率', 0)) if '市净率' in row else None,
                market_cap=float(row.get('总市值', 0)) if '总市值' in row else None,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            )
        except Exception as e:
            logger.error(f"AkShare 行情获取失败 [{symbol}]: {e}")
            return None

    def get_historical(self, symbol: str, days: int = 365) -> Optional[List[dict]]:
        ak = self._get_ak()
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            df = ak.stock_zh_a_hist(
                symbol=symbol.zfill(6), period="daily",
                start_date=start.strftime('%Y%m%d'),
                end_date=end.strftime('%Y%m%d'), adjust="qfq",
            )
            if df is None or df.empty:
                return None
            df = df.sort_values('日期')
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"AkShare 历史数据获取失败 [{symbol}]: {e}")
            return None

    def get_financials(self, symbol: str) -> Optional[FinancialSummary]:
        ak = self._get_ak()
        try:
            code = symbol.zfill(6)
            quote = self.get_quote(code)
            name = quote.name if quote else ""
            df = ak.stock_financial_abstract(symbol=code, indicator="按年度")
            if df is None or df.empty:
                return FinancialSummary(symbol=code, name=name)
            row = df.iloc[0]
            return FinancialSummary(
                symbol=code, name=name,
                revenue=self._safe_div(row.get('营业收入', None), 1e8),
                net_profit=self._safe_div(row.get('净利润', None), 1e8),
                eps=self._safe_float(row.get('基本每股收益', None)),
                report_date=str(row.get('报告期', '')),
            )
        except Exception as e:
            logger.error(f"AkShare 财务数据获取失败 [{symbol}]: {e}")
            return None

    def get_news(self, symbol: str) -> List[NewsItem]:
        """插件式新闻聚合：遍历所有已启用的新闻源"""
        from market_data.news_sources import NEWS_SOURCES
        results = []
        for src in NEWS_SOURCES:
            if not src.enabled:
                continue
            try:
                items = src.fetch(symbol)
                if items:
                    logger.debug(f"[{symbol}] {src.name}: {len(items)} 条")
                results.extend(items)
            except Exception as e:
                logger.warning(f"[{symbol}] 新闻源 {src.name} 失败: {e}")
        if not results:
            logger.warning(f"[{symbol}] 所有新闻源均无数据")
        return results

    def get_guba(self, symbol: str) -> List[GubaPost]:
        """插件式股吧聚合"""
        from market_data.news_sources import GUBA_SOURCES
        results = []
        for src in GUBA_SOURCES:
            if not src.enabled:
                continue
            try:
                items = src.fetch(symbol)
                if items:
                    logger.debug(f"[{symbol}] {src.name}: {len(items)} 条")
                results.extend(items)
            except Exception as e:
                logger.warning(f"[{symbol}] 股吧源 {src.name} 失败: {e}")
        return results

    @staticmethod
    def _safe_float(val) -> Optional[float]:
        if val is None:
            return None
        try:
            return round(float(val), 4)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_div(val, divisor) -> Optional[float]:
        if val is None:
            return None
        try:
            return round(float(val) / divisor, 4)
        except (ValueError, TypeError):
            return None


# ========== BaoStock 后端（免费、无需token） ==========

class BaoStockBackend(BaseStockBackend):
    """BaoStock 数据后端，免费无需token，作为 Tushare/AkShare 失败时的兜底"""

    def __init__(self):
        self._logged_in = False

    def _login(self):
        if self._logged_in:
            return
        import baostock as bs
        bs.login()
        self._logged_in = True
        self._bs = bs

    def _bs_prefix(self, symbol: str) -> str:
        code = symbol.strip().zfill(6)
        return 'sh.' + code if code.startswith(('6', '9')) else 'sz.' + code

    def get_quote(self, symbol: str) -> Optional[Quote]:
        try:
            self._login()
            prefix = self._bs_prefix(symbol)
            code = symbol.strip().zfill(6)
            # 日K线最新一条作为行情
            rs = self._bs.query_history_k_data_plus(prefix,
                'date,open,high,low,close,volume,amount,peTTM,pbMRQ,turn',
                frequency='d', adjustflag='2')
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return None
            r = rows[-1]
            price = float(r[3]) if r[3] else 0
            prev_close = float(r[3]) if len(rows) > 1 and rows[-2][3] else price
            change_pct = (price - prev_close) / prev_close * 100 if prev_close else 0

            # 股票名称
            name = code
            try:
                rs_name = self._bs.query_stock_basic(code=prefix)
                if rs_name.error_code == '0':
                    while rs_name.next():
                        name = rs_name.get_row_data()[1]
            except Exception:
                pass

            pe = float(r[6]) if r[6] else None
            pb = float(r[7]) if r[7] else None
            turnover = float(r[8]) if r[8] else None

            return Quote(
                symbol=code, name=name, price=round(price, 2),
                change=round(price - prev_close, 2), change_pct=round(change_pct, 2),
                volume=float(r[4]) if r[4] else 0, amount=float(r[5]) if r[5] else 0,
                high=float(r[1]) if r[1] else 0, low=float(r[2]) if r[2] else 0,
                open_=float(r[0]) if r[0] else 0, prev_close=round(prev_close, 2),
                turnover_rate=round(turnover, 2) if turnover else None,
                pe=round(pe, 2) if pe else None, pb=round(pb, 2) if pb else None,
                market_cap=None, timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            )
        except Exception as e:
            logger.error(f"BaoStock 行情失败 [{symbol}]: {e}")
            return None

    def get_historical(self, symbol: str, days: int = 365) -> Optional[List[dict]]:
        try:
            self._login()
            prefix = self._bs_prefix(symbol)
            end = datetime.now()
            start = end - timedelta(days=days)
            rs = self._bs.query_history_k_data_plus(prefix,
                'date,open,high,low,close,volume,amount',
                start_date=start.strftime('%Y-%m-%d'), end_date=end.strftime('%Y-%m-%d'),
                frequency='d', adjustflag='2')
            rows = []
            while rs.next():
                r = rs.get_row_data()
                if r[0] and r[3]:
                    rows.append({
                        'date': r[0], 'open': float(r[1]), 'high': float(r[2]),
                        'low': float(r[3]), 'close': float(r[4]),
                        'volume': float(r[5]) if r[5] else 0,
                        'amount': float(r[6]) if r[6] else 0,
                    })
            return rows if rows else None
        except Exception as e:
            logger.error(f"BaoStock 历史数据失败 [{symbol}]: {e}")
            return None

    def get_financials(self, symbol: str) -> Optional[FinancialSummary]:
        try:
            self._login()
            prefix = self._bs_prefix(symbol)
            code = symbol.strip().zfill(6)
            name = code
            try:
                rs_name = self._bs.query_stock_basic(code=prefix)
                if rs_name.error_code == '0':
                    while rs_name.next():
                        name = rs_name.get_row_data()[1]
            except Exception:
                pass
            # 季度利润表
            eps = roe = None
            year = datetime.now().year
            rs = self._bs.query_profit_data(code=prefix, year=year, quarter=1)
            if rs.error_code == '0':
                fields = rs.get_field_names()
                while rs.next():
                    r = rs.get_row_data()
                    rd = dict(zip(fields, r))
                    if rd.get('roeAvg'):
                        try: roe = float(rd['roeAvg'])
                        except: pass
                    if rd.get('epsEnyu'):
                        try: eps = float(rd['epsEnyu'])
                        except: pass
            return FinancialSummary(
                symbol=code, name=name, eps=eps, roe=roe,
                report_date=f'{year}Q1',
            )
        except Exception as e:
            logger.warning(f"BaoStock 财务失败 [{symbol}]: {e}")
            return None

    def get_historical_pe(self, symbol: str, days: int = 365 * 3) -> List[float]:
        try:
            self._login()
            prefix = self._bs_prefix(symbol)
            end = datetime.now()
            start = end - timedelta(days=days)
            rs = self._bs.query_history_k_data_plus(prefix,
                'date,peTTM', start_date=start.strftime('%Y-%m-%d'),
                end_date=end.strftime('%Y-%m-%d'), frequency='d', adjustflag='2')
            pe_vals = []
            while rs.next():
                r = rs.get_row_data()
                if r[1]:
                    try: pe_vals.append(float(r[1]))
                    except: pass
            return pe_vals
        except Exception:
            return []

    def get_news(self, symbol: str) -> List[NewsItem]:
        return []

    def get_guba(self, symbol: str) -> List[GubaPost]:
        return []


# ========== 技术指标计算（与后端无关） ==========

class TechnicalIndicatorCalculator:
    """技术指标计算器，依赖 numpy"""

    @staticmethod
    def compute(hist_data: List[dict]) -> TechnicalIndicators:
        """从历史 K 线数据计算技术指标"""
        import numpy as np

        closes = np.array([r['close'] for r in hist_data], dtype=float)
        highs = np.array([r['high'] for r in hist_data], dtype=float)
        lows = np.array([r['low'] for r in hist_data], dtype=float)
        volumes = np.array([r['volume'] for r in hist_data], dtype=float)
        price = float(closes[-1]) if len(closes) > 0 else 0.0

        result = TechnicalIndicators(symbol=hist_data[0].get('symbol', ''), price=price)

        if len(closes) < 5:
            return result

        # 均线
        result.ma5 = _sma(closes, 5)
        result.ma10 = _sma(closes, 10)
        result.ma20 = _sma(closes, 20)
        result.ma60 = _sma(closes, 60)

        # RSI
        result.rsi_14 = _rsi(closes, 14)

        # MACD
        dif, dea, bar = _macd(closes)
        result.macd_dif = dif
        result.macd_dea = dea
        result.macd_hist = bar

        # KDJ
        k, d, j = _kdj(highs, lows, closes)
        result.kdj_k = k
        result.kdj_d = d
        result.kdj_j = j

        # 布林带
        u, m, l = _bollinger(closes)
        result.boll_upper = u
        result.boll_middle = m
        result.boll_lower = l

        # 量比
        if len(volumes) >= 5:
            avg_v = np.mean(volumes[-5:])
            result.volume_ratio = round(float(volumes[-1] / avg_v), 2) if avg_v > 0 else None

        # 振幅
        if len(highs) > 0 and price > 0:
            result.amplitude = round(float((highs[-1] - lows[-1]) / price * 100), 2)

        return result


def _sma(arr, period):
    import numpy as np
    if len(arr) < period:
        return None
    return round(float(np.mean(arr[-period:])), 2)


def _ema(arr, period):
    import numpy as np
    result = np.zeros_like(arr)
    multiplier = 2.0 / (period + 1)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = (arr[i] - result[i - 1]) * multiplier + result[i - 1]
    return result


def _rsi(arr, period=14):
    import numpy as np
    if len(arr) < period + 1:
        return None
    deltas = np.diff(arr)
    gains = np.maximum(deltas, 0)
    losses = np.abs(np.minimum(deltas, 0))
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(float(100 - (100 / (1 + rs))), 2)


def _macd(arr, fast=12, slow=26, signal=9):
    if len(arr) < slow:
        return None, None, None
    fast_ema = _ema(arr, fast)
    slow_ema = _ema(arr, slow)
    dif = fast_ema - slow_ema
    dea = _ema(dif, signal)
    bar = 2 * (dif - dea)
    return round(float(dif[-1]), 4), round(float(dea[-1]), 4), round(float(bar[-1]), 4)


def _kdj(high, low, close, period=9):
    import numpy as np
    if len(high) < period:
        return None, None, None
    rh = np.max(high[-period:])
    rl = np.min(low[-period:])
    if rh == rl:
        return 50.0, 50.0, 50.0
    rsv = (close[-1] - rl) / (rh - rl) * 100
    k, d = rsv, rsv
    for _ in range(3):
        k = 2 / 3 * k + 1 / 3 * rsv
        d = 2 / 3 * d + 1 / 3 * k
    j = 3 * k - 2 * d
    return round(float(k), 2), round(float(d), 2), round(float(j), 2)


def _bollinger(arr, period=20, num_std=2):
    import numpy as np
    if len(arr) < period:
        return None, None, None
    ma = float(np.mean(arr[-period:]))
    std = float(np.std(arr[-period:]))
    return (
        round(ma + num_std * std, 2),
        round(ma, 2),
        round(ma - num_std * std, 2),
    )


# ========== 主入口 ==========

class AStockProvider:
    """
    A 股数据提供者（主入口）

    自动选择可用后端（逐级降级）：
    1. TushareBackend（Tushare Pro，推荐，需token）
    2. AkShareBackend（东方财富，需中国大陆网络）
    3. BaoStockBackend（免费，无需token）
    4. MockBackend（内置模拟数据，保底）
    """

    def __init__(self, backend: Optional[str] = None, proxy: Optional[str] = None,
                 tushare_token: Optional[str] = None):
        from config import settings
        self.proxy = proxy or os.environ.get('AKSHARE_PROXY')
        self.tushare_token = tushare_token or os.environ.get('TUSHARE_TOKEN') or settings.TUSHARE_TOKEN or ''
        self._backend = None
        self._news_bk = None  # AkShare fallback for news/guba
        self._requested_backend = backend or os.environ.get('STOCK_BACKEND') or settings.STOCK_BACKEND or 'auto'
        self._active_backend_name = self._requested_backend

    @property
    def backend_name(self) -> str:
        return self._active_backend_name

    @property
    def backend(self) -> BaseStockBackend:
        if self._backend is not None:
            return self._backend

        if self._requested_backend == 'mock':
            self._backend = MockBackend()
            self._active_backend_name = 'mock'
            logger.info("使用 MockBackend（模拟数据）")
            return self._backend

        # 1. Tushare（优先）
        if self._requested_backend in ('tushare', 'auto') and self.tushare_token:
            try:
                from .tushare_provider import TushareBackend
                bk = TushareBackend(token=self.tushare_token)
                test = bk.get_quote("600519")
                if test is not None and test.price > 0:
                    self._backend = bk
                    self._active_backend_name = 'tushare'
                    logger.info("使用 TushareBackend（Tushare Pro）")
                    return self._backend
            except Exception as e:
                logger.warning(f"TushareBackend 不可用: {e}")

        # 2. AkShare
        if self._requested_backend in ('akshare', 'auto'):
            try:
                bk = AkShareBackend(proxy=self.proxy)
                test = bk.get_quote("600519")
                if test is not None and test.price > 0:
                    self._backend = bk
                    self._active_backend_name = 'akshare'
                    logger.info("使用 AkShareBackend（东方财富）")
                    return self._backend
            except Exception as e:
                logger.warning(f"AkShareBackend 不可用: {e}")

        # 3. BaoStock（免费兜底）
        if self._requested_backend in ('baostock', 'auto'):
            try:
                bk = BaoStockBackend()
                test = bk.get_quote("600519")
                if test is not None and test.price > 0:
                    self._backend = bk
                    self._active_backend_name = 'baostock'
                    logger.info("使用 BaoStockBackend（免费数据）")
                    return self._backend
            except Exception as e:
                logger.warning(f"BaoStockBackend 不可用: {e}")

        # 4. Mock（最终保底）
        logger.info("回退到 MockBackend（模拟数据）")
        self._backend = MockBackend()
        self._active_backend_name = 'mock'
        return self._backend

    @property
    def _news_backend(self):
        """Lazy AkShare 后端，专门用于新闻/股吧舆情数据"""
        if self._news_bk is not None:
            return self._news_bk if self._news_bk is not False else None
        if self._active_backend_name == 'akshare':
            self._news_bk = False  # 主后端就是 AkShare，不需要独立 fallback
            return None
        try:
            bk = AkShareBackend(proxy=self.proxy)
            logger.info("使用 AkShareBackend 作为新闻/舆情数据源")
            self._news_bk = bk
            return bk
        except Exception as e:
            logger.warning(f"AkShare 新闻后端不可用: {e}")
            self._news_bk = False
            return None

    def get_quote(self, symbol: str) -> Optional[Quote]:
        return self.backend.get_quote(symbol)

    def get_historical(self, symbol: str, days: int = 120) -> Optional[List[dict]]:
        return self.backend.get_historical(symbol, days)

    def get_technical_indicators(self, symbol: str) -> Optional[TechnicalIndicators]:
        hist = self.get_historical(symbol, days=120)
        if hist and len(hist) >= 20:
            return TechnicalIndicatorCalculator.compute(hist)
        logger.warning(f"历史数据不足，无法计算技术指标: {symbol}")
        return None

    def get_financial_summary(self, symbol: str) -> Optional[FinancialSummary]:
        return self.backend.get_financials(symbol)

    def get_news(self, symbol: str) -> List[NewsItem]:
        news = self.backend.get_news(symbol)
        if not news:
            nkb = self._news_backend
            if nkb:
                news = nkb.get_news(symbol)
        return news

    def get_guba_posts(self, symbol: str) -> List[GubaPost]:
        posts = self.backend.get_guba(symbol)
        if not posts:
            nkb = self._news_backend
            if nkb:
                posts = nkb.get_guba(symbol)
        return posts

    def get_historical_pe(self, symbol: str, days: int = 365 * 3) -> List[float]:
        return self.backend.get_historical_pe(symbol, days)

    def get_all_market_data(self, symbol: str) -> Dict[str, Any]:
        """一站式获取所有市场数据"""
        logger.info(f"开始获取 [{symbol}] 全量市场数据...")

        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'name': '',
            'quote': None,
            'technical_indicators': None,
            'financial_summary': None,
            'news': [],
            'guba_posts': [],
        }
        quote = self.get_quote(symbol)
        result['quote'] = quote
        if quote:
            result['name'] = quote.name

        result['technical_indicators'] = self.get_technical_indicators(symbol)
        result['financial_summary'] = self.get_financial_summary(symbol)
        result['news'] = self.get_news(symbol)
        result['guba_posts'] = self.get_guba_posts(symbol)

        # 转 dict 方便序列化
        for key in ('quote', 'technical_indicators', 'financial_summary'):
            if hasattr(result[key], 'to_dict'):
                result[key] = result[key].to_dict()

        result['news'] = [n.to_dict() for n in result['news']]
        result['guba_posts'] = [p.to_dict() for p in result['guba_posts']]

        logger.info(f"[{symbol}] 数据获取完成")
        return result


if __name__ == '__main__':
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "600519"
    backend = sys.argv[2] if len(sys.argv) > 2 else "mock"

    provider = AStockProvider(backend=backend)
    data = provider.get_all_market_data(symbol)

    print(f"\n=== {data['name']}({symbol}) ===")
    q = data['quote']
    if q:
        print(f"价格: {q['price']}  涨跌幅: {q['change_pct']}%")
    ti = data['technical_indicators']
    if ti:
        print(f"MA5: {ti['ma5']}  MA10: {ti['ma10']}  RSI: {ti['rsi_14']}")
        print(f"MACD: {ti['macd_hist']}  KDJ: {ti['kdj_k']}/{ti['kdj_d']}/{ti['kdj_j']}")
    n = data['financial_summary']
    if n:
        print(f"营收: {n['revenue']}亿  净利: {n['net_profit']}亿  EPS: {n['eps']}")
    print(f"新闻: {len(data['news'])}条  股吧: {len(data['guba_posts'])}条")
