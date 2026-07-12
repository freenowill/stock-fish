"""
StockEngine Agent

全流程分析 Agent：
1. 采集行情/技术指标/基本面
2. 情感分析
3. 信号生成
4. 估值与买入价计算
5. LLM 综合预测
"""
import os
import json
import math
import time
from typing import Optional, Dict, Any
from datetime import datetime

from loguru import logger

from config import settings
from market_data.a_stock_provider import AStockProvider
from market_data.sentiment_collector import SentimentCollector
from analysis.state.state import AnalysisState
from analysis.nodes.prediction_node import PredictionNode
from analysis.scoring import ScoringEngine

# 记忆系统集成
from memory.cache.cache_manager import cache_manager
from memory.analysis.analysis_store import AnalysisStore
from memory.stocks.stock_library import StockLibrary
from memory.masters.master_track import MasterTrackDB

# SearchService 缓存（5分钟有效期，支持key热更新）
_search_service_cache = None
_search_service_cache_time = 0


def _get_search_service():
    """获取 SearchService 实例（5分钟缓存，自动读取最新 env）"""
    global _search_service_cache, _search_service_cache_time
    now = time.time()
    if _search_service_cache is not None and (now - _search_service_cache_time) < 300:
        return _search_service_cache if _search_service_cache is not False else None

    try:
        from market_data.search.search_service import SearchService
        import os as _os
        bocha_key = _os.environ.get('BOCHA_API_KEY') or getattr(settings, 'BOCHA_API_KEY', None)
        bocha_keys = [k.strip() for k in bocha_key.split(',') if k.strip()] if bocha_key else None
        tavily_key = _os.environ.get('TAVILY_API_KEY') or getattr(settings, 'TAVILY_API_KEY', None)
        tavily_keys = [k.strip() for k in tavily_key.split(',') if k.strip()] if tavily_key else None
        svc = SearchService(bocha_keys=bocha_keys, tavily_keys=tavily_keys)
        _search_service_cache = svc
        _search_service_cache_time = now
        return svc
    except Exception as e:
        logger.warning(f"SearchService 初始化失败: {e}")
        _search_service_cache = False
        _search_service_cache_time = now
        return None


class StockAnalysisAgent:

    def __init__(self, backend: Optional[str] = None):
        bk = backend or os.environ.get('STOCK_BACKEND') or getattr(settings, 'STOCK_BACKEND', None) or 'auto'
        self.provider = AStockProvider(backend=bk)
        self.sentiment = SentimentCollector(enable_sentiment=True)
        self.scoring = ScoringEngine()
        self.prediction_node = PredictionNode(
            api_key=os.environ.get('LLM_API_KEY') or getattr(settings, 'LLM_API_KEY', None),
            base_url=os.environ.get('LLM_BASE_URL') or getattr(settings, 'LLM_BASE_URL', None),
            model=os.environ.get('LLM_MODEL_NAME') or getattr(settings, 'LLM_MODEL_NAME', None),
        )

    def analyze(self, symbol: str, cost_price: float = 0.0, master: str = "",
                shares: int = 0, total_assets: float = 0.0, available_cash: float = 0.0) -> Dict[str, Any]:
        """执行一次完整分析，返回结构化结果。master 非空时启用大师决策模式。"""
        state = AnalysisState(symbol=symbol, cost_price=cost_price,
                              shares=shares, total_assets=total_assets, available_cash=available_cash,
                              created_at=datetime.now().isoformat())

        try:
            # Step 0: 验证该股票过去预测的准确率（14天/3个月/12个月后回测）
            try:
                from memory.masters.accuracy import verify_predictions
                verified = verify_predictions(symbol=symbol)
                if verified:
                    logger.info(f"[{symbol}] 已验证 {len(verified)} 条历史预测结果")
            except Exception:
                pass

            # Step 1: 采集市场数据（通过缓存层）
            state.status = "gathering"

            def _dictify(obj):
                """将 dataclass 对象转为 dict，兼容已有 dict 和 None"""
                if obj is None:
                    return {}
                if isinstance(obj, dict):
                    return obj
                if hasattr(obj, 'to_dict'):
                    return obj.to_dict()
                if hasattr(obj, '_asdict'):
                    return obj._asdict()
                return dict(obj)

            # 整包缓存: get_all_market_data() 只在缓存未命中时真正调用 API
            market = cache_manager.get_market_data(
                symbol,
                lambda: self.provider.get_all_market_data(symbol)
            )
            state.stock_name = market.get('name', symbol)
            state.quote = _dictify(market.get('quote'))
            state.technical_indicators = _dictify(market.get('technical_indicators'))
            state.financial_summary = _dictify(market.get('financial_summary'))
            state.news = market.get('news', [])
            state.guba_posts = market.get('guba_posts', [])

            # 宏观/行业有独立的 TTL（4h），单独缓存
            state.macro_context = cache_manager.get_macro_context(
                lambda: market.get('macro_context') or self._fetch_macro_context())
            state.industry_context = cache_manager.get_industry_context(
                symbol, lambda: market.get('industry_context') or self._fetch_industry_context(symbol))

            # Step 1b: Web 搜索（多维度情报 — 新闻/研报/风险/业绩）
            try:
                svc = _get_search_service()
                if svc:
                    stock_name = state.stock_name or symbol
                    intel = svc.search_comprehensive_intel(
                        stock_code=symbol,
                        stock_name=stock_name,
                        max_searches=3,
                    )
                    # 聚合多维度搜索结果
                    all_results = []
                    dim_contexts = []
                    for dim_name, dim_resp in intel.items():
                        if dim_resp and dim_resp.results:
                            for r in dim_resp.results:
                                all_results.append({
                                    'title': r.title, 'url': r.url, 'snippet': r.snippet,
                                    'source': r.source, 'date': getattr(r, 'date', ''),
                                    'dimension': dim_name,
                                })
                        dim_contexts.append(
                            f"[{dim_resp.desc if hasattr(dim_resp, 'desc') else dim_name}]\n"
                            f"{dim_resp.to_context(max_results=5) if dim_resp else ''}"
                        )
                    state.search_results = {
                        'query': f'{stock_name}({symbol}) 综合情报',
                        'provider': '|'.join(set(r.get('source', '') for r in all_results)) or 'comprehensive',
                        'results': all_results,
                        'context': '\n\n'.join(dim_contexts),
                        'dimensions': list(intel.keys()),
                    }
                    logger.info(f"[{symbol}] 综合情报搜索完成: {len(all_results)} 条结果 ({len(intel)} 个维度)")
            except Exception as e:
                logger.warning(f"[{symbol}] 综合情报搜索失败, 降级到基础搜索: {e}")
                # 降级: 基础新闻搜索
                try:
                    if svc:
                        resp = svc.search_stock_news(
                            stock_code=symbol, stock_name=stock_name, max_results=8)
                        state.search_results = {
                            'query': resp.query, 'provider': resp.provider,
                            'results': [{'title': r.title, 'url': r.url, 'snippet': r.snippet,
                                         'source': r.source, 'date': getattr(r, 'date', '')}
                                        for r in (resp.results or [])],
                            'context': resp.to_context(max_results=8),
                            'dimensions': ['latest_news'],
                        }
                except Exception as e2:
                    logger.warning(f"[{symbol}] 基础搜索也失败: {e2}")
                    state.search_results = {'results': [], 'context': '', 'error': str(e)}

            logger.info(f"[{symbol}] Step 1/4: 数据采集完成")

            # ── 缓存原始市场数据 ──
            self._cache_raw_data(symbol, state)

            # Step 2: 舆情情感分析
            state.status = "analyzing"
            news_texts = [n.get('title', '') for n in state.news]
            guba_texts = [p.get('title', '') for p in state.guba_posts]
            if news_texts:
                ss = self.sentiment.analyze_batch(news_texts)
                state.sentiment_news = self._sent_to_dict(ss)
                state.important_bullish_news = self._extract_top_items(state.news, ss, 'positive', 3)
                state.important_bearish_news = self._extract_top_items(state.news, ss, 'negative', 3)
            if guba_texts:
                ss = self.sentiment.analyze_batch(guba_texts)
                state.sentiment_guba = self._sent_to_dict(ss)
                state.important_bullish_guba = self._extract_top_items(state.guba_posts, ss, 'positive', 3)
                state.important_bearish_guba = self._extract_top_items(state.guba_posts, ss, 'negative', 3)
            logger.info(f"[{symbol}] Step 2/4: 情感分析完成")

            # ── 情感历史百分位 + 关注热度追踪 ──
            self._compute_sentiment_percentile(symbol, state)
            self._compute_attention_volume(symbol, state)

            # Step 3: 估值分析 + 综合信号
            state.status = "analyzing"
            self._compute_valuation(symbol, state)
            score_result = self.scoring.compute(state)
            state.signals = self._score_to_signals(score_result)
            state.score_breakdown = {
                'final': score_result.final,
                'label': score_result.label,
                'technical': score_result.technical,
                'fundamental': score_result.fundamental,
                'sentiment': score_result.sentiment,
                'regime': score_result.regime,
                'confidence': score_result.confidence,
                'weights': score_result.weights,
                'breakdown': [
                    {'factor': d.factor, 'impact': d.impact,
                     'contribution': d.contribution, 'description': d.description}
                    for d in score_result.breakdown
                ],
            }
            logger.info(f"[{symbol}] Step 3/4: 信号生成完成 "
                       f"(总分: {score_result.final}, 评级: {score_result.label}, "
                       f"技术: {score_result.technical}, 基本面: {score_result.fundamental}, "
                       f"舆情: {score_result.sentiment}, 市场状态: {score_result.regime})")

            # ── 数据补充：从现有接口提取更多分析维度 ──
            self._extract_detailed_financials(symbol, state)

            # ── 缓存给LLM/大师前的全量数据 ──
            self._cache_pre_llm_data(symbol, master, state)
            self._compute_risk_metrics(symbol, state)
            self._compute_equity_risk_premium(state)
            # 长期PE分位在 _compute_valuation 中已完成(如已有数据)
            # 管理层质量（董监高增减持）
            try:
                mgmt = self._search_management_quality(symbol)
                if mgmt is not None:
                    state.management_quality = mgmt
            except Exception:
                pass

            # ── Web 搜索补充（仅大师模式启用，节省API配额）──
            if master:
                try:
                    svc = _get_search_service()
                    if svc:
                        stock_name = state.stock_name or symbol
                        try:
                            peer = self._search_peer_valuation(svc, symbol, stock_name)
                            if peer:
                                state.peer_valuation = peer
                        except Exception:
                            pass
                        try:
                            moat = self._search_economic_moat(svc, symbol, stock_name)
                            if moat:
                                state.moat_assessment = moat
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"[{symbol}] Web搜索补充失败: {e}")

            # Step 4: LLM 综合预测
            state.status = "predicting"
            state_dict = state.to_dict()

            if master:
                # ── 大师决策模式 ──
                logger.info(f"[{symbol}] Step 4/4: 启用大师决策模式 (master={master})")
                prediction = self.prediction_node.predict_with_master(state_dict, master)
                logger.info(f"[{symbol}] Step 4/4: 大师决策完成 → {prediction.outlook}")
            else:
                # ── Legacy 3+1 模式 ──
                prediction = self.prediction_node.predict(state_dict)
                logger.info(f"[{symbol}] Step 4/4: 多Agent辩论预测完成 → {prediction.outlook}")

            state.llm_analysis = prediction.analysis_text
            state.prediction_summary = {
                'outlook': prediction.outlook,
                'confidence': prediction.confidence,
                'price_target_current': prediction.price_target_current,
                'price_target_low': prediction.price_target_low,
                'price_target_high': prediction.price_target_high,
                'reason': prediction.reason,
                # 多 Agent 辩论观点
                'tech_view': prediction.tech_view,
                'fund_view': prediction.fund_view,
                'sent_view': prediction.sent_view,
                # 多周期预测 + 操作建议
                'short_term': prediction.short_term,
                'mid_term': prediction.mid_term,
                'long_term': prediction.long_term,
                'suggested_action': prediction.suggested_action,
                # 大师决策扩展
                'cio_decision': getattr(prediction, 'cio_decision', None),
                'employee_reports': getattr(prediction, 'employee_reports', []),
            }
            state.price_target = {
                'current': prediction.price_target_current,
                'low': prediction.price_target_low,
                'high': prediction.price_target_high,
            }
            state.risk_factors = [{'factor': f} for f in prediction.risk_factors]

            state.mark_complete()

            # ── 记忆系统：保存分析结果 ──
            try:
                state_dict = state.to_dict()
                timestamp = state.created_at or datetime.now().isoformat()

                # 保存分析归档
                analysis_store = AnalysisStore()
                employee_reports_list = getattr(prediction, 'employee_reports', None) if master else None
                cio_decision_dict = getattr(prediction, 'cio_decision', None) if master else None
                analysis_store.save(
                    symbol=symbol,
                    state=state_dict,
                    timestamp=timestamp,
                    employee_reports=employee_reports_list,
                    cio_decision=cio_decision_dict,
                    prediction=state.prediction_summary,
                )

                # 更新股票数据仓库
                StockLibrary().update(symbol, state_dict)

                # 记录大师决策（master 模式）
                if master:
                    MasterTrackDB().record_decision(
                        master_key=master,
                        symbol=symbol,
                        analysis_timestamp=timestamp,
                        state=state_dict,
                        prediction_summary=state.prediction_summary,
                        cio_decision=cio_decision_dict,
                    )

                logger.info(f"[{symbol}] 记忆系统保存完成")
            except Exception as mem_err:
                logger.warning(f"[{symbol}] 记忆系统保存失败 (非关键): {mem_err}")

        except Exception as e:
            logger.error(f"[{symbol}] 分析失败: {e}")
            state.mark_error(str(e))

        return state.to_dict()

    # ---- 估值计算 ----

    def _compute_valuation(self, symbol: str, state: AnalysisState):
        """计算 PE 历史分位数、估值等级、建议买入价 + 长期PE分位"""
        try:
            # 1年PE分位 (缓存)
            pe_values = cache_manager.get_historical_pe(
                symbol, 365,
                lambda: self.provider.get_historical_pe(symbol, days=365)
            )
            quote = state.quote or {}
            current_pe = quote.get('pe') if isinstance(quote, dict) else None
            current_price = quote.get('price', 0) if isinstance(quote, dict) else 0

            if not pe_values or not current_pe or current_pe <= 0:
                state.valuation_level = '正常'
                state.valuation_percentile = None
                state.historical_pe_avg = None
                state.suggested_buy_price = round(current_price * 0.95, 2) if current_price else 0
                # 仍尝试长期PE
                self._compute_long_term_pe(symbol, state, current_pe, current_price)
                return

            import numpy as np
            pe_array = np.array(pe_values, dtype=float)
            pe_array = pe_array[pe_array > 0]
            if len(pe_array) < 30:
                state.valuation_level = '正常'
                state.valuation_percentile = None
                state.historical_pe_avg = None
                state.suggested_buy_price = round(current_price * 0.95, 2) if current_price else 0
                self._compute_long_term_pe(symbol, state, current_pe, current_price)
                return

            avg_pe = float(np.mean(pe_array))
            state.historical_pe_avg = round(avg_pe, 2)
            percentile = float(np.sum(pe_array <= current_pe) / len(pe_array) * 100)
            state.valuation_percentile = round(percentile, 1)

            if percentile < 10:
                state.valuation_level = '很低'
            elif percentile < 30:
                state.valuation_level = '偏低'
            elif percentile < 70:
                state.valuation_level = '正常'
            elif percentile < 90:
                state.valuation_level = '偏高'
            else:
                state.valuation_level = '很高'

            # 建议买入价 = 当前价 × (历史PE均值 / 当前PE)
            fair_value = current_price * (avg_pe / current_pe)
            ti = state.technical_indicators or {}
            boll_lower = ti.get('boll_lower') if isinstance(ti, dict) else None
            if current_pe <= avg_pe:
                buy_price = current_price
            else:
                buy_price = min(fair_value, current_price * 0.95)
            if boll_lower and boll_lower > 0:
                buy_price = max(boll_lower, buy_price)
            state.suggested_buy_price = round(buy_price, 2)

            logger.info(f"[{symbol}] 估值: {state.valuation_level} (PE分位{percentile:.1f}%, "
                       f"当前PE{current_pe} vs 均值{avg_pe:.1f}), 建议买入价: {state.suggested_buy_price}")

            # 5年/10年长期PE分位
            self._compute_long_term_pe(symbol, state, current_pe, current_price)

        except Exception as e:
            logger.warning(f"[{symbol}] 估值计算失败: {e}")
            state.valuation_level = '正常'
            state.valuation_percentile = None
            state.historical_pe_avg = None
            state.suggested_buy_price = round((state.quote or {}).get('price', 100) * 0.95, 2)

    def _compute_long_term_pe(self, symbol: str, state, current_pe, current_price):
        """计算5年和10年PE分位"""
        try:
            import numpy as np
            for label, days in [('5y', 1825), ('10y', 3650)]:
                pe_vals = cache_manager.get_historical_pe(
                    symbol, days,
                    lambda d=days: self.provider.get_historical_pe(symbol, days=d)
                )
                if pe_vals and len(pe_vals) >= 120:
                    arr = np.array(pe_vals, dtype=float)
                    arr = arr[arr > 0]
                    if len(arr) >= 120:
                        pct = round(float(np.sum(arr <= current_pe) / len(arr) * 100), 1)
                        avg = round(float(np.mean(arr)), 2)
                        if label == '5y':
                            state.valuation_percentile_5y = pct
                            state.pe_avg_5y = avg
                        else:
                            state.valuation_percentile_10y = pct
                            state.pe_avg_10y = avg
                        logger.info(f"[{symbol}] {label} PE分位={pct}%, 均值={avg}")
        except Exception as e:
            logger.debug(f"[{symbol}] 长期PE分位计算失败: {e}")

    # ---- 信号生成（新版 ScoringEngine）----

    @staticmethod
    def _score_to_signals(result) -> dict:
        """将 ScoreResult 映射为兼容旧格式的 signals dict"""
        outlook_map = {
            '强烈看多': 'bullish', '看多': 'bullish', '偏多': 'bullish',
            '中性': 'neutral',
            '偏空': 'bearish', '看空': 'bearish', '强烈看空': 'bearish',
        }
        return {
            'overall': outlook_map.get(result.label, 'neutral'),
            'score': result.final,
            'label': result.label,
            'technical': result.technical,
            'fundamental': result.fundamental,
            'sentiment': result.sentiment,
            'regime': result.regime,
            'confidence': result.confidence,
            'details': [
                {
                    'factor': d.factor,
                    'impact': d.impact,
                    'weight': round(abs(d.contribution), 2),
                    'contribution': d.contribution,
                    'description': d.description,
                }
                for d in result.breakdown
            ],
        }

    # ---- 数据补充：ROIC / FCF / 多期趋势 / VaR / 收益率对比 ----

    def _extract_detailed_financials(self, symbol: str, state) -> None:
        """从 ak.stock_financial_abstract 提取 ROIC、FCF、多期趋势等维度"""
        try:
            import akshare as ak
            import numpy as np
            df = ak.stock_financial_abstract(symbol)
            if df is None or df.empty:
                return

            # 获取'指标'列名 (第1列)
            indicator_col = '指标' if '指标' in df.columns else df.columns[1]
            # 获取季度列 (第2列起) — 原始是时间倒序(最新在前)
            quarters_raw = [c for c in df.columns if c != indicator_col and not c.startswith('选项')]
            quarters = quarters_raw[::-1]  # 转为时间正序(最早→最新)
            latest_q = quarters[-1] if quarters else None  # 正序的最后一个=最新
            annual_cols = [c for c in quarters if str(c).endswith('1231')]  # 年报列

            def _get_val(row_name, col=None):
                """取指定行指标的值"""
                row = df[df[indicator_col].str.contains(row_name, na=False)]
                if row.empty:
                    return None
                try:
                    val = row.iloc[0][col or latest_q]
                    return float(val) if pd.notna(val) else None
                except (IndexError, ValueError, TypeError):
                    return None

            import pandas as pd

            # 1. ROIC (Row 42: 投入资本回报率)
            # 注意: 东方财富财务摘要中 ROIC 是期间值(非年化)。
            # 如果最新列为中报/季报，必须年化后才能与评分阈值(>15%/+2, <5%/-1)比较。
            # 策略: 优先取最近一期年报列(1231)，否则取最新列并年化。
            roic_raw = _get_val('投入资本回报率')
            if roic_raw is None:
                roic_raw = _get_val('ROIC')
            if roic_raw is not None and latest_q is not None:
                col_str = str(latest_q)
                if col_str.endswith('0331'):
                    # Q1 单季 → 年化 ×4
                    state.roic = round(roic_raw * 4, 2)
                elif col_str.endswith('0630'):
                    # 中报(半年) → 年化 ×2
                    state.roic = round(roic_raw * 2, 2)
                elif col_str.endswith('0930'):
                    # 三季报(9个月) → 年化 ×4/3
                    state.roic = round(roic_raw * 4 / 3, 2)
                else:
                    # 年报列(1231)或其他 → 已经是全年值
                    state.roic = roic_raw
            else:
                state.roic = roic_raw

            # 2. FCF per share (Row 25: 每股企业自由现金流量)
            state.fcf_per_share = _get_val('每股企业自由现金流量')
            # fallback: try 每股股东自由现金流量
            if state.fcf_per_share is None:
                state.fcf_per_share = _get_val('每股股东自由现金流量')
            # fallback: try每股现金流量净额
            if state.fcf_per_share is None:
                state.fcf_per_share = _get_val('每股现金流量净额')

            # 3. 每股经营现金流 (Row 23)
            state.operating_cash_flow_per_share = _get_val('每股经营现金流')

            # 4. 多期财务趋势
            trends = {}
            for metric_name, row_keyword in [
                ('roe', '净资产收益率'),
                ('gross_margin', '毛利率'),
                ('debt_ratio', '资产负债率'),
                ('eps', '基本每股收益'),
                ('revenue', '营业总收入'),
                ('net_profit', '归母净利润'),
                ('roic', '投入资本回报率'),
            ]:
                row = df[df[indicator_col].str.contains(row_keyword, na=False)]
                if not row.empty:
                    vals = []
                    for q in annual_cols[-6:]:  # 最近6期年报
                        try:
                            v = float(row.iloc[0][q]) if pd.notna(row.iloc[0].get(q)) else None
                            if v is not None:
                                vals.append(v)
                        except (ValueError, TypeError):
                            pass
                    if vals:
                        trends[metric_name] = vals

            # 计算EPS CAGR(近5年)
            if 'eps' in trends and len(trends['eps']) >= 2:
                first, last = trends['eps'][0], trends['eps'][-1]
                if first > 0 and last > 0:
                    years = len(trends['eps']) - 1
                    trends['eps_cagr_5y'] = round((last / first) ** (1 / years) - 1, 4)

            # ROE稳定性(标准差/均值)
            if 'roe' in trends and len(trends['roe']) >= 3:
                arr = np.array(trends['roe'])
                mean_roe = np.mean(arr)
                if mean_roe > 0:
                    trends['roe_stability'] = round(float(np.std(arr) / mean_roe), 3)

            # 毛利率趋势
            if 'gross_margin' in trends and len(trends['gross_margin']) >= 3:
                arr = np.array(trends['gross_margin'])
                trends['gross_margin_trend'] = '上升' if arr[-1] > arr[0] else '下降' if arr[-1] < arr[0] else '稳定'

            # 5. 从 FinancialSummary 兜底计算 ROIC 和 FCF/股
            fs = getattr(state, 'financial_summary', None) or {}
            q = getattr(state, 'quote', None) or {}
            # FCF/股
            if state.fcf_per_share is None and fs.get('free_cash_flow'):
                price = q.get('price', 0)
                mcap = q.get('market_cap', 0)
                if price and mcap and price > 0:
                    total_shares = mcap / price  # 亿股（mcap 单位亿, price 单位元）
                    fcf_val = fs['free_cash_flow']  # 亿
                    if total_shares > 0:
                        state.fcf_per_share = round(fcf_val / total_shares, 4)
            # ROIC ≈ NOPAT / (净利为正时才可信)
            if state.roic is None and fs.get('net_profit') and fs.get('roe'):
                np_profit = fs['net_profit']  # 亿
                roe_val = fs['roe']  # %
                debt = fs.get('debt_ratio', 50)  # %
                if np_profit > 0:
                    # 简化: ROIC ≈ ROE * (1 - debt_ratio/200)
                    state.roic = round(roe_val * (1 - debt / 200), 4)

            if trends:
                state.financial_trends = trends
                logger.info(f"[{symbol}] 详细财务数据: ROIC={state.roic}% FCF/股={state.fcf_per_share} "
                           f"趋势行数={len(trends)}")

        except ImportError:
            logger.debug(f"[{symbol}] akshare 不可用，跳过详细财务提取")
        except Exception as e:
            logger.debug(f"[{symbol}] 详细财务提取失败: {e}")

    def _compute_risk_metrics(self, symbol: str, state) -> None:
        """从历史价格计算 VaR/最大回撤/年化波动率/Beta"""
        try:
            hist = self.provider.get_historical(symbol, days=365)
            if not hist or len(hist) < 30:
                return

            import numpy as np
            closes = np.array([r.get('close', 0) for r in hist], dtype=float)
            if len(closes) < 30:
                return

            returns = np.diff(closes) / closes[:-1]

            # VaR 95%
            state.var_95 = round(float(np.percentile(returns, 5)) * 100, 2)

            # 最大回撤
            peak = np.maximum.accumulate(closes)
            drawdown = (peak - closes) / peak
            state.max_drawdown = round(float(np.max(drawdown)) * 100, 2)

            # 年化波动率
            state.annualized_volatility = round(float(np.std(returns) * np.sqrt(252) * 100), 2)

            logger.info(f"[{symbol}] 风险指标: VaR95={state.var_95}% MaxDD={state.max_drawdown}% "
                       f"Vol={state.annualized_volatility}%")
        except Exception as e:
            logger.debug(f"[{symbol}] 风险指标计算失败: {e}")

    def _compute_equity_risk_premium(self, state) -> None:
        """计算E/P收益率 vs 债券收益率的对比"""
        try:
            q = state.quote or {}
            fs = state.financial_summary or {}
            price = q.get('price', 0) if isinstance(q, dict) else 0
            eps = fs.get('eps', 0) if isinstance(fs, dict) else 0
            if price > 0 and eps > 0:
                state.earnings_yield = round(eps / price * 100, 2)
            # bond_yield 在 _fetch_macro_context 中已填充
            mc = state.macro_context or {}
            bond_yield = mc.get('bond_yield_10y')
            if state.earnings_yield is not None and bond_yield is not None:
                state.equity_risk_premium = round(state.earnings_yield - float(bond_yield), 2)
        except Exception as e:
            logger.debug(f"收益率对比计算失败: {e}")

    def _compute_sentiment_percentile(self, symbol: str, state) -> None:
        """记录当日情感并计算历史百分位"""
        try:
            from market_data.sentiment_collector import SentimentHistory
            sn = state.sentiment_news or {}
            sg = state.sentiment_guba or {}
            news_avg = sn.get('avg_score', 0) or 0
            guba_avg = sg.get('avg_score', 0) or 0
            news_count = sn.get('total_count', 0) or 0
            guba_count = sg.get('total_count', 0) or 0

            today = datetime.now().strftime('%Y-%m-%d')
            hist = SentimentHistory(symbol)
            hist.record(today, news_avg=news_avg, guba_avg=guba_avg,
                        news_count=news_count, guba_count=guba_count)

            pct = hist.get_percentile(news_avg, guba_avg)
            if pct.get('news_percentile') is not None:
                state.sentiment_percentile = pct['news_percentile']
                state.sentiment_history_days = pct['total_days']
                logger.info(f"[{symbol}] 情感百分位: news={pct['news_percentile']}% "
                           f"guba={pct.get('guba_percentile')}% (共{pct['total_days']}天)")
        except Exception as e:
            logger.debug(f"[{symbol}] 情感百分位计算失败: {e}")

    def _compute_attention_volume(self, symbol: str, state) -> None:
        """追踪新闻/股吧的关注热度变化"""
        try:
            from market_data.sentiment_collector import SentimentHistory
            sn = state.sentiment_news or {}
            sg = state.sentiment_guba or {}
            today = datetime.now().strftime('%Y-%m-%d')

            hist = SentimentHistory(symbol)
            # 读取已有历史，计算当前量在历史中的分位
            history = hist.history
            if len(history) >= 5:
                news_counts = sorted([h.get('news_count', 0) for h in history if h.get('news_count') is not None])
                guba_counts = sorted([h.get('guba_count', 0) for h in history if h.get('guba_count') is not None])
                current_news = sn.get('total_count', 0) or 0
                current_guba = sg.get('total_count', 0) or 0

                def _pct(val, arr):
                    return round(sum(1 for v in arr if v <= val) / len(arr) * 100, 1) if arr else None

                state.attention_news_percentile = _pct(current_news, news_counts)
                state.attention_guba_percentile = _pct(current_guba, guba_counts)
        except Exception as e:
            logger.debug(f"[{symbol}] 关注热度追踪失败: {e}")

    @staticmethod
    def _search_management_quality(symbol: str) -> Optional[Dict]:
        """从现有API获取管理层质量数据（董监高增减持）"""
        try:
            import akshare as ak
            import pandas as pd
            df = ak.stock_inner_trade_xq()
            if df is None or df.empty:
                return None

            # 尝试匹配股票代码（兼容600519和'600519'）
            code_col = '股票代码' if '股票代码' in df.columns else df.columns[0]
            insider = df[df[code_col].astype(str).str.contains(symbol, na=False)]
            if insider.empty:
                return {'insider_trades_count': 0, 'note': '近期无董监高交易记录'}

            net_shares = int(insider['变动股数'].sum()) if '变动股数' in df.columns else 0
            net_flow = '净增持' if net_shares > 0 else '净减持' if net_shares < 0 else '无变动'
            return {
                'insider_trades_count': len(insider),
                'net_shares_change': net_shares,
                'net_flow': net_flow,
                'latest_date': str(insider.iloc[-1].get('变动日期', '')),
            }
        except Exception as e:
            logger.debug(f"[{symbol}] 管理层数据获取失败: {e}")
            return None

    @staticmethod
    def _search_peer_valuation(svc, symbol: str, stock_name: str) -> Optional[Dict]:
        """搜索全球同行估值对比（邓普顿核心需求）"""
        try:
            # 搜索同行估值信息
            resp = svc.search_stock_news(
                stock_code=symbol, stock_name=stock_name, max_results=6,
                focus_keywords=['同行', '竞争对手', '估值', 'PE', '对比', '排名'])
            if not resp or not resp.results:
                return None

            peers = []
            snippets = []
            for r in resp.results[:6]:
                snippets.append(f"{r.title}: {r.snippet[:300]}")
                peers.append({'name': r.title[:40], 'snippet': r.snippet[:200],
                              'url': r.url, 'source': r.source})

            # 尝试从 yfinance 获取全球同行估值
            global_pe_data = {}
            try:
                import yfinance as yf
                # 搜索该行业全球代表性公司PE（如白酒→帝亚吉欧）
                industry_etfs = {
                    '白酒': 'DEO', '银行': 'KBE', '半导体': 'SMH',
                    '新能源': 'TAN', '医药': 'XLV', '消费': 'XLP',
                }
                for keyword, ticker in industry_etfs.items():
                    if keyword in stock_name or keyword in ''.join(snippets):
                        tk = yf.Ticker(ticker)
                        info = tk.info
                        global_pe_data[ticker] = {
                            'name': info.get('shortName', ticker),
                            'pe': info.get('trailingPE'),
                            'market_cap': info.get('marketCap'),
                        }
            except Exception:
                pass

            return {
                'peers': peers[:5],
                'snippets': snippets[:3],
                'global_pe_data': global_pe_data,
                'conclusion': '搜索到{}家同业公司信息'.format(len(peers)),
                'source': 'SearchService',
            }
        except Exception as e:
            logger.debug(f"[{symbol}] 同行估值搜索失败: {e}")
            return None

    @staticmethod
    def _search_economic_moat(svc, symbol: str, stock_name: str) -> Optional[Dict]:
        """搜索经济护城河评估（巴菲特核心需求）"""
        try:
            # 多维度搜索护城河相关信息
            all_results = []

            # 维度1: 护城河/竞争优势
            try:
                r1 = svc.search_stock_news(
                    stock_code=symbol, stock_name=stock_name, max_results=4,
                    focus_keywords=['护城河', '竞争优势', '行业地位', '龙头', '壁垒'])
                if r1 and r1.results:
                    all_results.extend(r1.results[:3])
            except Exception:
                pass

            # 维度2: 市场份额/竞争格局
            try:
                r2 = svc.search_stock_news(
                    stock_code=symbol, stock_name=stock_name, max_results=4,
                    focus_keywords=['市场份额', '竞争格局', '品牌力', '龙头地位'])
                if r2 and r2.results:
                    all_results.extend(r2.results[:3])
            except Exception:
                pass

            if not all_results:
                return None

            findings = []
            for r in all_results[:6]:
                findings.append({
                    'title': r.title[:60],
                    'snippet': r.snippet[:300],
                    'url': r.url,
                })

            # 从搜索结果中提取护城河信号（关键词匹配）
            moat_signals = {'品牌力': 0, '技术壁垒': 0, '网络效应': 0,
                          '规模经济': 0, '转换成本': 0, '监管壁垒': 0}
            combined_text = ' '.join(r.snippet for r in all_results)
            for signal in moat_signals:
                for kw in [signal, signal.replace('力', ''), signal.replace('效应', '')]:
                    if kw in combined_text:
                        moat_signals[signal] += 1

            detected_sources = [k for k, v in moat_signals.items() if v > 0]
            moat_level = '宽护城河' if len(detected_sources) >= 3 else \
                        '窄护城河' if len(detected_sources) >= 1 else '无明显护城河'

            return {
                'moat_level': moat_level,
                'moat_sources': detected_sources,
                'evidence': findings[:4],
                'search_summary': combined_text[:500],
            }
        except Exception as e:
            logger.debug(f"[{symbol}] 护城河评估失败: {e}")
            return None

    # ---- 工具 ----

    @staticmethod
    def _sent_to_dict(ss) -> dict:
        if ss is None:
            return {'avg_score': 0, 'positive_ratio': 0, 'negative_ratio': 0,
                    'total_count': 0, 'positive_count': 0, 'negative_count': 0}
        return {
            'avg_score': getattr(ss, 'avg_score', 0),
            'positive_ratio': getattr(ss, 'positive_ratio', 0),
            'negative_ratio': getattr(ss, 'negative_ratio', 0),
            'total_count': getattr(ss, 'total_count', 0),
            'positive_count': getattr(ss, 'positive_count', 0),
            'negative_count': getattr(ss, 'negative_count', 0),
        }

    @staticmethod
    def _extract_top_items(items: list, summary, direction: str, top_n: int) -> list:
        """从 items 中提取情感得分最高/最低的 top_n 条"""
        scores = getattr(summary, 'scores', []) or []
        if not scores or not items:
            return []
        pairs = list(zip(items, scores))
        reverse = direction == 'positive'
        sorted_pairs = sorted(pairs, key=lambda x: x[1].score, reverse=reverse)
        result = []
        for item, s in sorted_pairs:
            if direction == 'positive' and s.score <= 0.2:
                continue
            if direction == 'negative' and s.score >= -0.2:
                continue
            if len(result) >= top_n:
                break
            result.append({
                'title': item.get('title', ''),
                'source': item.get('source', item.get('author', '')),
                'publish_time': item.get('publish_time', ''),
                'sentiment_score': round(s.score, 2),
                'sentiment_label': s.label,
            })
        return result

    # ---- 数据缓存 (供审计使用，覆盖执行) ----

    @staticmethod
    def _cache_raw_data(symbol: str, state) -> None:
        """Step 1 完成后缓存原始市场数据，用于审计现有接口提供了什么"""
        try:
            path = os.path.join('batch_results', f'{symbol}_raw_data.json')
            data = {
                'symbol': symbol,
                'stock_name': state.stock_name,
                'quote_fields': list(state.quote.keys()) if state.quote else [],
                'quote_sample': {k: state.quote[k] for k in list(state.quote)[:6]} if state.quote else {},
                'technical_indicators_fields': list(state.technical_indicators.keys()) if state.technical_indicators else [],
                'technical_indicators_sample': {k: state.technical_indicators[k]
                                                for k in list(state.technical_indicators)[:8]} if state.technical_indicators else {},
                'financial_summary_fields': list(state.financial_summary.keys()) if state.financial_summary else [],
                'financial_summary': {k: v for k, v in (state.financial_summary or {}).items()
                                      if not k.startswith('_')},
                'news_count': len(state.news),
                'guba_count': len(state.guba_posts),
                'macro_context_fields': list(state.macro_context.keys()) if state.macro_context else [],
                'macro_context_sample': {k: v for k, v in (state.macro_context or {}).items()
                                         if isinstance(v, (str, int, float, bool)) or v is None},
                'industry_context_fields': list(state.industry_context.keys()) if state.industry_context else [],
                'industry_context': {k: v for k, v in (state.industry_context or {}).items()
                                     if isinstance(v, (str, int, float, bool)) or v is None},
                'search_results_dimensions': (state.search_results or {}).get('dimensions', []),
                'search_results_count': len((state.search_results or {}).get('results', [])),
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"[{symbol}] 原始市场数据已缓存 → {path}")
        except Exception as e:
            logger.warning(f"[{symbol}] 缓存原始数据失败: {e}")

    @staticmethod
    def _cache_pre_llm_data(symbol: str, master: str, state) -> None:
        """LLM/大师调用前缓存全量状态数据，用于审计给大师的分析输入是否充足"""
        try:
            suffix = f'{master}_' if master else ''
            path = os.path.join('batch_results', f'{symbol}_{suffix}pre_llm.json')
            state_dict = state.to_dict()
            # 排除过长的列表数据
            compact = {k: v for k, v in state_dict.items()
                       if k not in ('news', 'guba_posts', 'search_results')}
            # 对过长字段截取长度信息
            if 'prediction_summary' in compact and compact['prediction_summary']:
                for agent_key in ('cio_decision', 'employee_reports'):
                    if agent_key in compact.get('prediction_summary', {}):
                        del compact['prediction_summary'][agent_key]  # 尚无数据
            if 'score_breakdown' in compact and compact['score_breakdown']:
                if 'breakdown' in compact['score_breakdown']:
                    compact['score_breakdown']['breakdown_count'] = len(compact['score_breakdown']['breakdown'])
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(compact, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"[{symbol}] 预LLM状态已缓存 → {path} ({len(json.dumps(compact, default=str))} chars)")
        except Exception as e:
            logger.warning(f"[{symbol}] 缓存预LLM数据失败: {e}")

    # ---- 宏观/行业数据获取 (逐个API独立try/except, 单一失败不阻塞) ----

    @staticmethod
    def _safe_float(val):
        """将值转为 float，NaN/Infinity 返回 None，确保 JSON 可序列化。"""
        import math
        try:
            v = float(val)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _sanitize_context(ctx: dict) -> dict:
        """递归清理 dict 中的 NaN/Infinity 值，确保 JSON 可序列化。"""
        import math
        for k, v in ctx.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                ctx[k] = None
            elif isinstance(v, dict):
                StockAnalysisAgent._sanitize_context(v)
        return ctx

    @staticmethod
    def _safe_ak_call(fn, *args, **kwargs):
        """带重试的 akshare 调用，处理网络波动。失败返回 None。"""
        import time
        for attempt in range(2):
            try:
                time.sleep(0.4)  # 避免触发东方财富反爬限流
                return fn(*args, **kwargs)
            except Exception as e:
                if attempt == 0 and ('Connection' in str(e) or 'RemoteDisconnected' in str(e)):
                    time.sleep(1.5)
                    continue
        return None

    @staticmethod
    def _fetch_macro_context() -> dict:
        """获取宏观数据上下文。逐个调用 akshare API，任一失败不影响其他。"""
        ctx = {'source': 'akshare'}
        try:
            import akshare as ak
        except ImportError:
            logger.debug("akshare 不可用，宏观数据使用 placeholder")
            return {'source': 'placeholder', '_note': 'akshare 未安装'}

        # --- PMI ---
        try:
            df = StockAnalysisAgent._safe_ak_call(ak.macro_china_pmi)
            if df is not None and len(df) > 0:
                latest = df.iloc[-1]
                ctx['pmi'] = float(latest.get('制造业', latest.iloc[1])) if len(latest) > 1 else None
        except Exception as e:
            logger.debug(f"PMI 获取失败: {e}")

        # --- CPI ---
        try:
            df = StockAnalysisAgent._safe_ak_call(ak.macro_china_cpi)
            if df is not None and len(df) > 0:
                ctx['cpi_yoy'] = float(df.iloc[-1].get('全国-同比增长', df.iloc[-1, 2])) if df.shape[1] > 2 else None
        except Exception as e:
            logger.debug(f"CPI 获取失败: {e}")

        # --- SHIBOR (rate_interbank 已废弃, 改用 macro_china_shibor_all) ---
        try:
            df = StockAnalysisAgent._safe_ak_call(ak.macro_china_shibor_all)
            if df is not None and len(df) > 0:
                ctx['shibor'] = float(df.iloc[-1]['O/N-定价']) if 'O/N-定价' in df.columns else None
        except Exception as e:
            logger.debug(f"SHIBOR 获取失败: {e}")

        # --- LPR ---
        try:
            df = StockAnalysisAgent._safe_ak_call(ak.macro_china_lpr)
            if df is not None and len(df) > 0:
                latest = df.iloc[-1]
                if 'LPR1Y' in df.columns:
                    ctx['lpr_1y'] = float(latest['LPR1Y'])
                if 'LPR5Y' in df.columns:
                    ctx['lpr_5y'] = float(latest['LPR5Y'])
                # 推断政策倾向: 比较最新 LPR1Y 与前值
                if len(df) >= 2:
                    prev = df.iloc[-2]
                    if 'LPR1Y' in df.columns:
                        if float(latest['LPR1Y']) < float(prev['LPR1Y']):
                            ctx['policy_tilt'] = '宽松 (LPR下调)'
                        elif float(latest['LPR1Y']) > float(prev['LPR1Y']):
                            ctx['policy_tilt'] = '收紧 (LPR上调)'
                        else:
                            ctx['policy_tilt'] = '中性 (LPR不变)'
        except Exception as e:
            logger.debug(f"LPR 获取失败: {e}")

        # --- M2 货币供应 ---
        try:
            df = StockAnalysisAgent._safe_ak_call(ak.macro_china_money_supply)
            if df is not None and len(df) > 0:
                latest = df.iloc[-1]
                for col in df.columns:
                    if 'M2' in str(col) and '同比' in str(col):
                        ctx['m2_yoy'] = float(latest[col])
                        break
        except Exception as e:
            logger.debug(f"M2 获取失败: {e}")

        # --- 社融 ---
        try:
            df = StockAnalysisAgent._safe_ak_call(ak.macro_china_shrzgm)
            if df is not None and len(df) > 0:
                latest = df.iloc[-1]
                for col in df.columns:
                    if '增量' in str(col) or '规模' in str(col):
                        ctx['social_financing'] = float(latest[col])
                        break
        except Exception as e:
            logger.debug(f"社融获取失败: {e}")

        # --- 北向资金 (使用 stock_hsgt_hist_em 获取历史序列，避免当日汇总数据为0) ---
        try:
            df = StockAnalysisAgent._safe_ak_call(ak.stock_hsgt_hist_em)
            if df is not None and len(df) >= 2:
                # 按日期降序排列，取最近 5 个交易日
                df_sorted = df.sort_values('日期', ascending=False)
                latest_5 = df_sorted.head(5)
                ctx['northbound_flow'] = float(latest_5.iloc[0]['当日成交净买额'])
                ctx['northbound_5d_avg'] = round(float(latest_5['当日成交净买额'].mean()), 1)
                logger.info(f"北向资金: 当日={ctx['northbound_flow']:.1f}亿, 5日均={ctx['northbound_5d_avg']:.1f}亿")
        except Exception as e:
            logger.debug(f"北向资金获取失败: {e}")

        # --- 美元/人民币汇率 ---
        try:
            df = StockAnalysisAgent._safe_ak_call(ak.fx_spot_quote)
            if df is not None and len(df) > 0:
                usd_row = df[df['货币对'].str.contains('USD', na=False)] if '货币对' in df.columns else None
                if usd_row is not None and len(usd_row) > 0:
                    quote = usd_row.iloc[0]
                    v = quote.get('买报价') if '买报价' in quote.index else None
                    if v is None:
                        v = quote.get('卖报价') if '卖报价' in quote.index else None
                    ctx['usd_cny'] = StockAnalysisAgent._safe_float(v)
        except Exception as e:
            logger.debug(f"汇率获取失败: {e}")

        # --- 大盘状态推断 ---
        # --- 宏观经济日历 (国内外头部事件) ---
        try:
            df = StockAnalysisAgent._safe_ak_call(ak.news_economic_baidu)
            if df is not None and len(df) > 0:
                # 只取近期高重要性事件
                recent = df.head(20)
                events = []
                for _, row in recent.iterrows():
                    importance = row.get('重要性', 1)
                    if int(importance) >= 1:  # 中等重要以上
                        events.append({
                            'date': str(row.get('日期', '')),
                            'region': str(row.get('地区', '')),
                            'event': str(row.get('事件', '')),
                            'importance': int(importance),
                        })
                ctx['macro_events'] = events[:10]  # 最多 10 条
        except Exception as e:
            logger.debug(f"宏观经济日历获取失败: {e}")

        # --- 政策新闻 (CCTV 头条) ---
        try:
            df = StockAnalysisAgent._safe_ak_call(ak.news_cctv)
            if df is not None and len(df) > 0:
                headlines = []
                for _, row in df.head(5).iterrows():
                    title = str(row.get('title', ''))
                    if len(title) > 10:  # 过滤过短的标题
                        headlines.append(title)
                ctx['policy_headlines'] = headlines
        except Exception as e:
            logger.debug(f"CCTV 新闻获取失败: {e}")

        # --- 中国10年期国债收益率 ---
        try:
            df = StockAnalysisAgent._safe_ak_call(ak.bond_zh_us_rate)
            if df is not None and len(df) > 0:
                for _, row in df.iterrows():
                    rate_type = str(row.get('rate_type', ''))
                    term = str(row.get('期限', '')) if '期限' in df.columns else ''
                    if ('中国国债' in rate_type or '中国' in rate_type) and '10' in term:
                        val = float(row.get('收益率', row.iloc[-1]))
                        ctx['bond_yield_10y'] = val
                        logger.info(f"10年期国债收益率: {val}%")
                        break
        except Exception as e:
            logger.debug(f"国债收益率获取失败: {e}")

        # --- 大盘状态推断 ---
        try:
            df = StockAnalysisAgent._safe_ak_call(ak.stock_zh_index_daily_em, symbol="sh000001")
            if df is not None and len(df) >= 20:
                close = df['close'].astype(float)
                ma20 = close.rolling(20).mean().iloc[-1]
                current = close.iloc[-1]
                if current > ma20 * 1.03:
                    ctx['market_regime'] = '上升趋势'
                elif current < ma20 * 0.97:
                    ctx['market_regime'] = '下降趋势'
                else:
                    ctx['market_regime'] = '震荡'
        except Exception as e:
            logger.debug(f"大盘状态获取失败: {e}")

        logger.info(f"宏观数据采集完成: {len(ctx)} 个字段")
        return StockAnalysisAgent._sanitize_context(ctx)

    @staticmethod
    def _fetch_industry_context(symbol: str) -> dict:
        """获取行业数据上下文。逐个调用 akshare API, 任一失败不影响其他。"""
        ctx = {'source': 'akshare'}
        try:
            import akshare as ak
        except ImportError:
            return {'source': 'placeholder', '_note': 'akshare 未安装'}

        # --- 行业板块概况 ---
        try:
            df = StockAnalysisAgent._safe_ak_call(ak.stock_board_industry_name_em)
            if df is not None and len(df) > 0:
                ctx['industry_count'] = len(df)
                ctx['industry_names'] = ', '.join(df['板块名称'].head(10).tolist())
                # 整体行业涨跌比
                up_count = int((df['涨跌幅'] > 0).sum()) if '涨跌幅' in df.columns else 0
                ctx['industry_up_ratio'] = round(up_count / len(df) * 100, 1)
                # 平均涨跌幅
                if '涨跌幅' in df.columns:
                    ctx['industry_avg_change'] = round(float(df['涨跌幅'].mean()), 2)
        except Exception as e:
            logger.debug(f"行业板块概况获取失败: {e}")

        # --- 行业资金流 ---
        try:
            df = StockAnalysisAgent._safe_ak_call(ak.stock_sector_fund_flow_rank, indicator="5日", sector_type="行业资金流")
            if df is not None and len(df) > 0:
                # 主力净流入总额
                flow_col = None
                for col in df.columns:
                    if '主力净流入' in str(col) and '净额' in str(col):
                        flow_col = col
                        break
                if flow_col:
                    total_flow = float(df[flow_col].sum())
                    ctx['industry_fund_flow'] = round(total_flow, 1)
                    ctx['industry_fund_flow_bullish'] = total_flow > 0
        except Exception as e:
            logger.debug(f"行业资金流获取失败: {e}")

        # --- 行业动量 (选取代表性板块如白酒/银行对比) ---
        try:
            momentum_values = []
            # 尝试几个典型板块
            for board_name in ['白酒', '银行', '半导体']:
                try:
                    hist = StockAnalysisAgent._safe_ak_call(
                        ak.stock_board_industry_hist_em, symbol=board_name, period="日k",
                        start_date="20240101", end_date="20260101")
                    if hist is not None and len(hist) >= 20:
                        hist['收盘'] = hist['收盘'].astype(float)
                        pct_20d = (hist['收盘'].iloc[-1] / hist['收盘'].iloc[-20] - 1) * 100
                        momentum_values.append(pct_20d)
                except Exception:
                    pass
            if momentum_values:
                ctx['industry_momentum'] = round(sum(momentum_values) / len(momentum_values), 2)
        except Exception as e:
            logger.debug(f"行业动量获取失败: {e}")

        # --- 政策新闻(宏观层面) ---
        try:
            news_df = StockAnalysisAgent._safe_ak_call(ak.stock_info_global_em)
            if news_df is not None and len(news_df) > 0:
                policy_keywords = ['政策', '监管', '央行', '发改委', '证监会', '国常会', '国务院', '工信部', '降准', '降息', 'LPR']
                policy_news = news_df[news_df['标题'].str.contains('|'.join(policy_keywords), na=False)]
                if len(policy_news) > 0:
                    ctx['policy_events'] = policy_news['标题'].head(3).tolist()
                    # 简单判断政策影响方向
                    positive_words = ['利好', '支持', '鼓励', '放松', '降准', '降息', '减税', '补贴']
                    negative_words = ['利空', '收紧', '监管', '处罚', '加税', '限制']
                    pos_count = policy_news['标题'].str.contains('|'.join(positive_words), na=False).sum()
                    neg_count = policy_news['标题'].str.contains('|'.join(negative_words), na=False).sum()
                    if pos_count > neg_count * 1.5:
                        ctx['policy_impact'] = '偏利好'
                    elif neg_count > pos_count * 1.5:
                        ctx['policy_impact'] = '偏利空'
                    else:
                        ctx['policy_impact'] = '中性'
                else:
                    ctx['policy_events'] = '近期无重大政策新闻'
                    ctx['policy_impact'] = '中性'
        except Exception as e:
            logger.debug(f"政策新闻获取失败: {e}")

        logger.info(f"行业数据采集完成: {len(ctx)} 个字段")
        return StockAnalysisAgent._sanitize_context(ctx)
