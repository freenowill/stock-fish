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
import math
from typing import Optional, Dict, Any
from datetime import datetime

from loguru import logger

from config import settings
from market_data.a_stock_provider import AStockProvider
from market_data.sentiment_collector import SentimentCollector
from analysis.state.state import AnalysisState
from analysis.nodes.prediction_node import PredictionNode
from analysis.scoring import ScoringEngine


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

    def analyze(self, symbol: str, cost_price: float = 0.0, master: str = "") -> Dict[str, Any]:
        """执行一次完整分析，返回结构化结果。master 非空时启用大师决策模式。"""
        state = AnalysisState(symbol=symbol, cost_price=cost_price, created_at=datetime.now().isoformat())

        try:
            # Step 1: 采集市场数据
            state.status = "gathering"
            market = self.provider.get_all_market_data(symbol)
            state.stock_name = market.get('name', symbol)
            state.quote = market.get('quote')
            state.technical_indicators = market.get('technical_indicators')
            state.financial_summary = market.get('financial_summary')
            state.news = market.get('news', [])
            state.guba_posts = market.get('guba_posts', [])
            state.macro_context = market.get('macro_context') or self._fetch_macro_context()
            state.industry_context = market.get('industry_context') or self._fetch_industry_context(symbol)
            logger.info(f"[{symbol}] Step 1/4: 数据采集完成")

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

        except Exception as e:
            logger.error(f"[{symbol}] 分析失败: {e}")
            state.mark_error(str(e))

        return state.to_dict()

    # ---- 估值计算 ----

    def _compute_valuation(self, symbol: str, state: AnalysisState):
        """计算 PE 历史分位数、估值等级、建议买入价"""
        try:
            pe_values = self.provider.get_historical_pe(symbol, days=365)
            quote = state.quote or {}
            current_pe = quote.get('pe') if isinstance(quote, dict) else None
            current_price = quote.get('price', 0) if isinstance(quote, dict) else 0

            if not pe_values or not current_pe or current_pe <= 0:
                state.valuation_level = '正常'
                state.suggested_buy_price = round(current_price * 0.95, 2) if current_price else 0
                return

            import numpy as np
            pe_array = np.array(pe_values, dtype=float)
            pe_array = pe_array[pe_array > 0]
            if len(pe_array) < 30:
                state.valuation_level = '正常'
                state.suggested_buy_price = round(current_price * 0.95, 2) if current_price else 0
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
            # 低估时建议现价，高估时按公允价值打折；不低于布林下轨作为技术支撑
            if current_pe <= avg_pe:
                buy_price = current_price
            else:
                buy_price = min(fair_value, current_price * 0.95)
            if boll_lower and boll_lower > 0:
                buy_price = max(boll_lower, buy_price)
            state.suggested_buy_price = round(buy_price, 2)

            logger.info(f"[{symbol}] 估值: {state.valuation_level} (PE分位{percentile:.1f}%, "
                       f"当前PE{current_pe} vs 均值{avg_pe:.1f}), 建议买入价: {state.suggested_buy_price}")
        except Exception as e:
            logger.warning(f"[{symbol}] 估值计算失败: {e}")
            state.valuation_level = '正常'
            state.suggested_buy_price = round((state.quote or {}).get('price', 100) * 0.95, 2)

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

    # ---- 宏观/行业数据获取 (轻量级, 失败不阻塞) ----

    @staticmethod
    def _fetch_macro_context() -> dict:
        """获取宏观数据上下文。尝试 akshare，失败返回 placeholder。"""
        try:
            import akshare as ak
            # 尝试获取 PMI
            pmi = None
            try:
                pmi_df = ak.macro_china_pmi()
                if pmi_df is not None and len(pmi_df) > 0:
                    latest = pmi_df.iloc[-1]
                    pmi = float(latest.get('制造业', latest.iloc[1])) if len(latest) > 1 else None
            except Exception:
                pass

            # 尝试获取 SHIBOR
            shibor = None
            try:
                shibor_df = ak.rate_interbank(market="上海银行间同业拆放利率", indicator="隔夜")
                if shibor_df is not None and len(shibor_df) > 0:
                    shibor = float(shibor_df.iloc[-1, 1]) if shibor_df.shape[1] > 1 else None
            except Exception:
                pass

            # 尝试获取北向资金
            northbound = None
            try:
                nb_df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
                if nb_df is not None and len(nb_df) > 0:
                    northbound = float(nb_df.iloc[-1, 1]) if nb_df.shape[1] > 1 else None
            except Exception:
                pass

            # 尝试获取 LPR
            lpr_1y = None
            try:
                lpr_df = ak.macro_china_lpr()
                if lpr_df is not None and len(lpr_df) > 0:
                    lpr_1y = float(lpr_df.iloc[-1].get('1年期', lpr_df.iloc[-1, 1])) if len(lpr_df.iloc[-1]) > 1 else None
            except Exception:
                pass

            return {
                'shibor': shibor, 'lpr_1y': lpr_1y,
                'pmi': pmi, 'northbound_flow': northbound,
                'source': 'akshare',
            }
        except ImportError:
            logger.debug("akshare 不可用，宏观数据使用 placeholder")
        except Exception as e:
            logger.debug(f"宏观数据获取失败: {e}")

        return {'source': 'placeholder', '_note': '宏观数据源不可用，使用默认假设'}

    @staticmethod
    def _fetch_industry_context(symbol: str) -> dict:
        """获取行业数据上下文。尝试 akshare，失败返回 placeholder。"""
        try:
            import akshare as ak
            industry_name = None
            industry_pe = None
            try:
                # 尝试获取行业板块数据
                board_df = ak.stock_board_industry_name_em()
                if board_df is not None and len(board_df) > 0:
                    # 简单取前几个行业作为参考
                    industry_name = "行业数据已获取"
            except Exception:
                pass

            # 尝试获取行业 PE
            try:
                pe_df = ak.stock_board_industry_pe_em()
                if pe_df is not None and len(pe_df) > 0:
                    avg_pe = float(pe_df['平均市盈率'].mean()) if '平均市盈率' in pe_df.columns else None
                    industry_pe = avg_pe
            except Exception:
                pass

            return {
                'industry_name': industry_name, 'industry_pe_percentile': industry_pe,
                'source': 'akshare',
            }
        except ImportError:
            logger.debug("akshare 不可用，行业数据使用 placeholder")
        except Exception as e:
            logger.debug(f"行业数据获取失败: {e}")

        return {'source': 'placeholder', '_note': '行业数据源不可用，使用默认假设'}
