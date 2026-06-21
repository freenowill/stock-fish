"""
AdvancedBackend: 适配 daily_stock_analysis 的 DataFetcherManager 到 StockFish 的 BaseStockBackend 接口。

通过 STOCK_BACKEND=advanced 激活，提供 11 个行情数据源 + 多源自动切换能力。
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from market_data.a_stock_provider import (
    BaseStockBackend,
    Quote,
    FinancialSummary,
    NewsItem,
    GubaPost,
)

logger = logging.getLogger(__name__)


def _yf_safe_float(value) -> Optional[float]:
    """yfinance 安全浮点数转换"""
    if value is None:
        return None
    try:
        result = float(value)
        if result != result:  # NaN
            return None
        return result
    except (TypeError, ValueError):
        return None


def _yf_pct(value) -> Optional[float]:
    """yfinance 比率转百分比（0.1 → 10.0）"""
    raw = _yf_safe_float(value)
    if raw is None:
        return None
    return round(raw * 100.0, 4)


class AdvancedBackend(BaseStockBackend):
    """
    使用 DataFetcherManager 的多源策略后端。

    数据源优先级（A 股）：
    EfinanceFetcher(0) > AkshareFetcher(1) > TushareFetcher(2,有Token动态提升)
    > PytdxFetcher(2) > BaostockFetcher(3) > YfinanceFetcher(4)

    美股：Finnhub > AlphaVantage > Yfinance > Longbridge
    港股：Akshare > Longbridge > Yfinance
    """

    def __init__(self):
        self._manager = None
        self._init_error = None
        self._initialized = False

    @property
    def manager(self):
        """Lazy initialization of DataFetcherManager."""
        if self._manager is not None:
            return self._manager
        if self._init_error is not None:
            return None
        try:
            from market_data.data_fetchers.base import DataFetcherManager

            self._manager = DataFetcherManager()
            self._initialized = True
            logger.info("AdvancedBackend: DataFetcherManager 初始化成功")
            return self._manager
        except Exception as e:
            self._init_error = e
            logger.warning(f"AdvancedBackend: DataFetcherManager 初始化失败: {e}")
            return None

    def _is_available(self) -> bool:
        return self.manager is not None

    # ---- BaseStockBackend interface ----

    def get_quote(self, symbol: str) -> Optional[Quote]:
        if not self._is_available():
            return None
        try:
            unified = self.manager.get_realtime_quote(symbol)
            if unified is None or unified.price is None or unified.price <= 0:
                return None
            dividend = self._get_dividend(symbol)
            return Quote(
                symbol=unified.code,
                name=unified.name or symbol,
                price=float(unified.price),
                change=float(unified.change_amount or 0),
                change_pct=float(unified.change_pct or 0),
                volume=float(unified.volume or 0),
                amount=float(unified.amount or 0),
                high=float(unified.high or 0),
                low=float(unified.low or 0),
                open_=float(unified.open_price or 0),
                prev_close=float(unified.pre_close or 0),
                turnover_rate=unified.turnover_rate,
                pe=unified.pe_ratio,
                pb=unified.pb_ratio,
                market_cap=round(unified.total_mv / 1e8, 2) if unified.total_mv else None,
                dividend=dividend,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            )
        except Exception as e:
            logger.warning(f"AdvancedBackend.get_quote({symbol}) failed: {e}")
            return None

    def _get_dividend(self, symbol: str) -> Optional[float]:
        """查询最近一次每股分红（税后，元/股）"""
        # 1. 先尝试 Tushare（A 股精确数据）
        div = self._get_tushare_dividend(symbol)
        if div is not None:
            return div

        # 2. Tushare 无结果，尝试 yfinance（港股/美股/A 股通用）
        try:
            import yfinance as yf
            code = symbol.strip().upper()
            if code.startswith('HK'):
                digits = code[2:].lstrip('0') or '0'
                yf_symbol = f"{digits}.HK"
            elif code in ('TSLA', 'AAPL', 'MSFT') or not code.isdigit():
                yf_symbol = code
            elif len(code) == 6:
                suffix = 'SS' if code.startswith(('6', '9', '5')) else 'SZ'
                yf_symbol = f"{code}.{suffix}"
            else:
                return None

            ticker = yf.Ticker(yf_symbol)
            import pandas as _pd
            div_series = ticker.dividends
            if div_series is not None and not div_series.empty:
                cutoff = _pd.Timestamp.now(tz=div_series.index.tz) - _pd.Timedelta(days=365)
                ttm_divs = [v for ts, v in div_series.items() if _pd.Timestamp(ts) >= cutoff]
                if ttm_divs:
                    total = sum(_yf_safe_float(v) or 0 for v in ttm_divs)
                    if total > 0:
                        logger.info(f"AdvancedBackend: yfinance 每股分红 {symbol}={total:.4f}")
                        return round(total, 4)
                last_div = _yf_safe_float(div_series.iloc[-1])
                if last_div is not None and last_div > 0:
                    return round(last_div, 4)
        except Exception as e:
            logger.debug(f"yfinance 分红查询失败 {symbol}: {e}")

        return None

    def _get_tushare_dividend(self, symbol: str) -> Optional[float]:
        """通过 Tushare 查询最近一次每股分红"""
        try:
            from market_data.compat import get_config
            cfg = get_config()
            token = cfg.tushare_token
            if not token:
                return None
            import sxsc_tushare as sx
            import pandas as _pd
            sx.set_token(token)
            api = sx.get_api(env='prd')

            code = symbol.strip().upper()
            if code.startswith('SH') or code.startswith('SZ') or code.startswith('BJ'):
                ts_code = code
            elif code.isdigit() and len(code) == 6:
                if code.startswith(('92', '43', '83', '87', '88')):
                    ts_code = f"{code}.BJ"
                elif code.startswith(('6', '9', '5')):
                    ts_code = f"{code}.SH"
                else:
                    ts_code = f"{code}.SZ"
            else:
                return None

            df = api.dividend(ts_code=ts_code, limit=3)
            if df is None or df.empty:
                return None

            for _, r in df.iterrows():
                cash_div = float(r['cash_div']) if _pd.notna(r.get('cash_div')) else 0
                cash_div_tax = float(r['cash_div_tax']) if _pd.notna(r.get('cash_div_tax')) else 0
                div = cash_div_tax or cash_div
                if div > 0:
                    logger.debug(f"AdvancedBackend: Tushare 每股分红 {symbol}={div}元")
                    return round(div, 3)
            return None
        except Exception as e:
            logger.debug(f"_get_tushare_dividend failed: {e}")
            return None

    def get_historical(self, symbol: str, days: int = 365) -> Optional[List[dict]]:
        if not self._is_available():
            return None
        try:
            df, source = self.manager.get_daily_data(symbol, days=days)
            if df is None or df.empty:
                return None
            df = df.copy()
            if 'date' in df.columns:
                df['date'] = df['date'].astype(str)

            # Ensure consistent column names expected by TechnicalIndicatorCalculator
            col_map = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'amount': 'amount',
            }
            rows = []
            for _, row in df.iterrows():
                rec = {}
                for col in df.columns:
                    val = row[col]
                    # Convert numpy types to Python native types
                    if hasattr(val, 'item'):
                        val = val.item()
                    rec[col] = val
                rows.append(rec)
            return rows
        except Exception as e:
            logger.warning(f"AdvancedBackend.get_historical({symbol}) failed: {e}")
            return None

    def get_financials(self, symbol: str) -> Optional[FinancialSummary]:
        if not self._is_available():
            return None
        name = self._resolve_name(symbol)

        # 1. 尝试 DataFetcherManager 基本面管道（akshare/yfinance 多源）
        try:
            ctx = self.manager.get_fundamental_context(symbol)
            if ctx and ctx.get('status') not in ('failed', 'not_supported'):
                earnings = ctx.get('earnings', {})
                earn_payload = earnings.get('payload', {}) or {}
                # financial_report 子结构包含核心财务数据
                fin_report = earn_payload.get('financial_report', {}) or {}
                growth = ctx.get('growth', {})
                growth_payload = growth.get('payload', {}) or {}

                def _safe_float(v, default=None):
                    try:
                        if v is None:
                            return default
                        return float(v)
                    except (TypeError, ValueError):
                        return default

                dividend_payload = earnings.get('dividend', {}) or {}
                div_per_share = dividend_payload.get('ttm_cash_dividend_per_share')
                # 计算股息率 = 每股股息 / 股价
                div_yield = None
                price = _safe_float(ctx.get('quote', {}).get('price') or fin_report.get('price'))
                if div_per_share and price and price > 0:
                    div_yield = round(div_per_share / price * 100, 2)

                fin = FinancialSummary(
                    symbol=symbol, name=name,
                    revenue=_safe_float(fin_report.get('revenue')),
                    revenue_yoy=_safe_float(growth_payload.get('revenue_yoy')),
                    net_profit=_safe_float(fin_report.get('net_profit_parent')),
                    net_profit_yoy=_safe_float(growth_payload.get('net_profit_yoy')),
                    eps=_safe_float(fin_report.get('eps')),
                    roe=_safe_float(growth_payload.get('roe')),
                    gross_margin=_safe_float(growth_payload.get('gross_margin')),
                    debt_ratio=_safe_float(growth_payload.get('debt_ratio')),
                    operating_cash_flow=_safe_float(fin_report.get('operating_cash_flow')),
                    free_cash_flow=_safe_float(fin_report.get('free_cash_flow')),
                    dividend_per_share=_safe_float(div_per_share),
                    dividend_yield=div_yield,
                    report_date=str(fin_report.get('report_date', '')),
                )
                if fin.eps or fin.roe or fin.revenue or fin.net_profit:
                    return fin
        except Exception as e:
            logger.debug(f"AdvancedBackend: 基本面适配器不可用 ({e})")

        # 2. Tushare 直接查询兜底（仅 A 股）
        try:
            fin = self._get_tushare_financials(symbol, name)
            if fin and (fin.eps or fin.roe):
                return fin
        except Exception as e:
            logger.debug(f"AdvancedBackend: Tushare 基本面兜底失败 ({e})")

        # 3. yfinance 直接兜底（适用于 A 股/港股/美股）
        try:
            fin = self._get_yfinance_financials(symbol, name)
            if fin and (fin.eps or fin.roe or fin.revenue):
                logger.info(f"AdvancedBackend: yfinance 基本面兜底成功 {symbol}")
                return fin
        except Exception as e:
            logger.debug(f"AdvancedBackend: yfinance 基本面兜底失败 ({e})")

        return FinancialSummary(symbol=symbol, name=name)

    def _get_yfinance_financials(self, symbol: str, name: str) -> Optional[FinancialSummary]:
        """通过 yfinance 获取基本面数据（港股/美股/A 股通用）"""
        try:
            import yfinance as yf
            import pandas as pd

            code = symbol.strip().upper()
            # 转为 yfinance 格式
            if code.startswith('HK'):
                digits = code[2:].lstrip('0') or '0'
                yf_symbol = f"{digits}.HK"
            elif code in ('TSLA', 'AAPL', 'MSFT') or not code.isdigit():
                yf_symbol = code  # 美股代码
            elif len(code) == 6:
                suffix = 'SS' if code.startswith(('6', '9', '5')) else 'SZ'
                yf_symbol = f"{code}.{suffix}"
            else:
                return None

            ticker = yf.Ticker(yf_symbol)
            info = {}
            try:
                info = ticker.get_info() if hasattr(ticker, 'get_info') else (ticker.info or {})
            except Exception:
                pass
            if not isinstance(info, dict):
                info = {}

            fin = FinancialSummary(symbol=symbol, name=name)

            fin.eps = _yf_safe_float(info.get('trailingEps'))
            fin.roe = _yf_pct(info.get('returnOnEquity'))
            fin.revenue = _yf_safe_float(info.get('totalRevenue'))
            fin.net_profit = _yf_safe_float(info.get('netIncomeToCommon'))
            fin.revenue_yoy = _yf_pct(info.get('revenueGrowth'))
            fin.net_profit_yoy = _yf_pct(info.get('earningsGrowth'))
            fin.gross_margin = _yf_pct(info.get('grossMargins'))
            fin.debt_ratio = _yf_pct(info.get('debtToEquity'))

            # 尝试从财务报表提取更精确的值
            try:
                bs = ticker.balance_sheet
                if bs is not None and not bs.empty:
                    if 'Total Debt' in bs.index:
                        total_debt = _yf_safe_float(bs.loc['Total Debt'].iloc[0])
                        total_equity = _yf_safe_float(bs.loc['Stockholders Equity'].iloc[0]) if 'Stockholders Equity' in bs.index else None
                        if total_debt is not None and total_equity not in (None, 0):
                            fin.debt_ratio = round(total_debt / total_equity * 100, 2)
            except Exception:
                pass

            # 尝试从季度报表获取最新营收/净利润
            try:
                income = ticker.quarterly_income_stmt
                if income is not None and not income.empty:
                    # Total Revenue
                    for key in ('Total Revenue', 'TotalRevenue', 'Revenue'):
                        if key in income.index:
                            val = _yf_safe_float(income.loc[key].iloc[0])
                            if val is not None:
                                fin.revenue = val
                                break
                    # Net Income
                    for key in ('Net Income', 'NetIncome', 'Net Income Common Stockholders'):
                        if key in income.index:
                            val = _yf_safe_float(income.loc[key].iloc[0])
                            if val is not None:
                                fin.net_profit = val
                                break
            except Exception:
                pass

            return fin
        except Exception:
            return None

    def _get_tushare_financials(self, symbol: str, name: str) -> Optional[FinancialSummary]:
        """通过 sxsc_tushare 查询 fina_indicator + daily_basic 补全基本面"""
        from market_data.compat import get_config
        cfg = get_config()
        token = cfg.tushare_token
        if not token:
            return None

        try:
            import sxsc_tushare as sx
            import pandas as pd
            sx.set_token(token)
            api = sx.get_api(env='prd')

            code = symbol.strip().upper()
            if code.startswith('SH') or code.startswith('SZ') or code.startswith('BJ'):
                ts_code = code
            elif code.isdigit() and len(code) == 6:
                if code.startswith(('92', '43', '83', '87', '88')):
                    ts_code = f"{code}.BJ"
                elif code.startswith(('6', '9', '5')):
                    ts_code = f"{code}.SH"
                else:
                    ts_code = f"{code}.SZ"
            else:
                return None

            # 取最新年报（end_date 为 1231 的最近一期）
            df = api.fina_indicator(ts_code=ts_code, limit=6)
            if df is None or df.empty:
                return None
            annual = df[df['end_date'].str.endswith('1231')]
            if annual.empty:
                annual = df  # 退市/新股用最新一期
            r = annual.iloc[0]

            eps = float(r['eps']) if pd.notna(r.get('eps')) else None
            roe = float(r['roe']) if pd.notna(r.get('roe')) else None

            # fina_indicator 字段: op_income=营业总收入, grossprofit_margin=毛利率,
            #   debt_to_assets=资产负债率, or_yoy=营收同比, netprofit_yoy=净利润同比
            revenue = round(float(r['op_income']) / 1e8, 2) if pd.notna(r.get('op_income')) else None
            gross_margin = float(r['grossprofit_margin']) if pd.notna(r.get('grossprofit_margin')) else None
            debt_ratio = float(r['debt_to_assets']) if pd.notna(r.get('debt_to_assets')) else None
            revenue_yoy = float(r['or_yoy']) if pd.notna(r.get('or_yoy')) else None
            net_profit_yoy = float(r['netprofit_yoy']) if pd.notna(r.get('netprofit_yoy')) else None

            # 净利润: EPS × 总股本 (Tushare total_share 单位是万股)
            net_profit = None
            try:
                db = api.daily_basic(ts_code=ts_code, limit=1)
                if db is not None and not db.empty:
                    total_share = float(db.iloc[0]['total_share']) if pd.notna(db.iloc[0].get('total_share')) else None
                    if total_share and eps:
                        net_profit = round(eps * total_share / 10000, 2)  # EPS×万股/10000 = 净利润(亿)
            except Exception:
                pass

            # 毛利率字段名可能是 grossprofit_margin 或 gross_margin
            if gross_margin is None:
                gross_margin = float(r['gross_margin']) if pd.notna(r.get('gross_margin')) else None

            logger.info(f"AdvancedBackend: Tushare 基本面 — eps={eps}, roe={roe}%, revenue={revenue}亿, net_profit={net_profit}亿")
            return FinancialSummary(
                symbol=symbol, name=name,
                eps=eps, roe=roe,
                revenue=revenue, revenue_yoy=revenue_yoy,
                net_profit=net_profit, net_profit_yoy=net_profit_yoy,
                gross_margin=gross_margin, debt_ratio=debt_ratio,
                report_date=str(r.get('end_date', '')),
            )
        except Exception as e:
            logger.debug(f"_get_tushare_financials failed: {e}")
            return None

    def get_news(self, symbol: str) -> List[NewsItem]:
        """获取新闻：优先 Tavily/Bocha 等搜索，兜底旧新闻源"""
        items: List[NewsItem] = []
        name = self._resolve_name(symbol)

        # 1. 优先使用搜索服务（Tavily 等已配置 API Key）
        if self._has_search_keys():
            try:
                svc = self._build_search_service()
                resp = svc.search_stock_news(symbol, name, max_results=10)
                if resp and resp.results:
                    for r in resp.results:
                        items.append(NewsItem(
                            title=r.title or '',
                            url=r.url or '',
                            publish_time=r.published_date or '',
                            source=f'{r.source or "web"}',
                        ))
                    logger.info(f"AdvancedBackend: 搜索服务获取到 {len(resp.results)} 条新闻")
            except Exception as e:
                logger.debug(f"AdvancedBackend: 搜索服务不可用 ({e})")

        # 2. 补充 A 股新闻源
        if not items:
            try:
                from market_data.news_sources import NEWS_SOURCES
                for src in NEWS_SOURCES:
                    try:
                        src_items = src.fetch(symbol)
                        if src_items:
                            items.extend(src_items)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"AdvancedBackend.get_news error: {e}")

        return items

    def _has_search_keys(self) -> bool:
        """Check if any search API keys are configured."""
        try:
            from market_data.compat import get_config
            cfg = get_config()
            keys = [
                cfg.anspire_api_keys, cfg.bocha_api_keys, cfg.tavily_api_keys,
                cfg.brave_api_keys, cfg.serpapi_keys, cfg.minimax_api_keys,
                cfg.searxng_base_urls,
            ]
            return any(keys) or getattr(cfg, 'searxng_public_instances_enabled', False)
        except Exception:
            return False

    def _build_search_service(self):
        """Build a SearchService with keys from config."""
        from market_data.search.search_service import SearchService
        from market_data.compat import get_config
        cfg = get_config()
        return SearchService(
            bocha_keys=cfg.bocha_api_keys or None,
            tavily_keys=cfg.tavily_api_keys or None,
            anspire_keys=cfg.anspire_api_keys or None,
            brave_keys=cfg.brave_api_keys or None,
            serpapi_keys=cfg.serpapi_keys or None,
            minimax_keys=cfg.minimax_api_keys or None,
            searxng_base_urls=cfg.searxng_base_urls or None,
            searxng_public_instances_enabled=cfg.searxng_public_instances_enabled,
            news_max_age_days=getattr(cfg, 'news_max_age_days', 3),
            news_strategy_profile=getattr(cfg, 'news_strategy_profile', 'short'),
        )

    def search_news(self, symbol: str, stock_name: str = '', max_results: int = 10) -> List[NewsItem]:
        """增强搜索：使用 7 引擎搜索，支持中文/英文多源。需要至少配置一个搜索 API Key。"""
        items: List[NewsItem] = []
        if not self._has_search_keys():
            logger.debug("AdvancedBackend: 未配置搜索 API Key，跳过搜索服务")
            return items
        name = stock_name or self._resolve_name(symbol)
        try:
            import threading
            result_container = {'resp': None, 'error': None}

            def _do_search():
                try:
                    svc = self._build_search_service()
                    result_container['resp'] = svc.search_comprehensive_intel(
                        stock_code=symbol, stock_name=name, max_searches=max_results
                    )
                except Exception as e:
                    result_container['error'] = e

            t = threading.Thread(target=_do_search, daemon=True)
            t.start()
            t.join(timeout=20)
            if t.is_alive():
                logger.debug("AdvancedBackend: 搜索超时（20s），跳过")
                return items

            resp = result_container['resp']
            # search_comprehensive_intel 返回 Dict[str, SearchResponse]
            if resp and isinstance(resp, dict):
                seen = set()
                for dim_name, dim_resp in resp.items():
                    if hasattr(dim_resp, 'results') and dim_resp.results:
                        for r in dim_resp.results:
                            key = r.url or r.title
                            if key and key not in seen:
                                seen.add(key)
                                items.append(NewsItem(
                                    title=r.title or '',
                                    url=r.url or '',
                                    publish_time=getattr(r, 'published_date', '') or '',
                                    source=f'{dim_name}/{r.source or "web"}',
                                ))
                logger.info(f"AdvancedBackend: 搜索服务获取到 {len(items)} 条结果 ({len(resp)} 个维度)")
            if result_container['error']:
                raise result_container['error']
        except Exception as e:
            logger.warning(f"AdvancedBackend.search_news: {e}")
        return items

    def get_social_sentiment(self, symbol: str) -> Optional[Dict]:
        """获取社交情感数据（Reddit/X/Polymarket，仅美股）"""
        try:
            from market_data.social_sentiment.social_sentiment_service import SocialSentimentService
            from market_data.compat import get_config
            cfg = get_config()
            svc = SocialSentimentService(
                api_key=cfg.social_sentiment_api_key,
                api_url=cfg.social_sentiment_api_url,
            )
            if svc.is_available():
                return svc.get_social_context(symbol)
        except Exception as e:
            logger.debug(f"AdvancedBackend: 社交情感不可用 ({e})")
        return None

    def get_guba(self, symbol: str) -> List[GubaPost]:
        posts: List[GubaPost] = []
        try:
            from market_data.news_sources import GUBA_SOURCES
            for src in GUBA_SOURCES:
                try:
                    src_posts = src.fetch(symbol)
                    if src_posts:
                        posts.extend(src_posts)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"AdvancedBackend.get_guba error: {e}")
        return posts

    def get_historical_pe(self, symbol: str, days: int = 365 * 3) -> List[float]:
        # PE is not typically available in daily OHLCV from most free sources.
        # Tushare provides it via daily_basic, but through DataFetcherManager the daily
        # data is standardized OHLCV only. Return empty list — callers should handle this.
        return []

    def _resolve_name(self, symbol: str) -> str:
        """Resolve stock name from DataFetcherManager or fallback to stock index."""
        try:
            if self.manager:
                name = self.manager.get_stock_name(symbol)
                if name and name != symbol:
                    return name
        except Exception:
            pass
        try:
            from market_data.stock_index.stock_index_loader import get_index_stock_name
            name = get_index_stock_name(symbol)
            if name:
                return name
        except Exception:
            pass
        return symbol
