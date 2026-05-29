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


class StockAnalysisAgent:

    def __init__(self, backend: Optional[str] = None):
        bk = backend or os.environ.get('STOCK_BACKEND') or getattr(settings, 'STOCK_BACKEND', None) or 'auto'
        self.provider = AStockProvider(backend=bk)
        self.sentiment = SentimentCollector(enable_sentiment=True)
        self.prediction_node = PredictionNode(
            api_key=os.environ.get('LLM_API_KEY') or getattr(settings, 'LLM_API_KEY', None),
            base_url=os.environ.get('LLM_BASE_URL') or getattr(settings, 'LLM_BASE_URL', None),
            model=os.environ.get('LLM_MODEL_NAME') or getattr(settings, 'LLM_MODEL_NAME', None),
        )

    def analyze(self, symbol: str, cost_price: float = 0.0) -> Dict[str, Any]:
        """执行一次完整分析，返回结构化结果"""
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
            signals = self._generate_signals(state)
            state.signals = signals
            logger.info(f"[{symbol}] Step 3/4: 信号生成完成 (总体: {signals.get('overall')}, 评分: {signals.get('score')})")

            # Step 4: LLM 综合预测
            state.status = "predicting"
            state_dict = state.to_dict()
            prediction = self.prediction_node.predict(state_dict)
            state.llm_analysis = prediction.analysis_text
            state.prediction_summary = {
                'outlook': prediction.outlook,
                'confidence': prediction.confidence,
                'price_target_current': prediction.price_target_current,
                'price_target_low': prediction.price_target_low,
                'price_target_high': prediction.price_target_high,
                'reason': prediction.reason,
            }
            state.price_target = {
                'current': prediction.price_target_current,
                'low': prediction.price_target_low,
                'high': prediction.price_target_high,
            }
            state.short_term_pred = prediction.short_term
            state.mid_term_pred = prediction.mid_term
            state.long_term_pred = prediction.long_term
            state.suggested_action = prediction.suggested_action
            state.risk_factors = [{'factor': f} for f in prediction.risk_factors]
            logger.info(f"[{symbol}] Step 4/4: LLM 预测完成")

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

    # ---- 信号生成 ----

    def _generate_signals(self, state: AnalysisState) -> dict:
        """根据技术面 + 舆情 + 估值生成综合信号"""
        signals = {'overall': 'neutral', 'score': 0, 'details': []}
        score = 0
        ti = state.technical_indicators or {}
        quote = state.quote or {}
        price = quote.get('price', 0) if isinstance(quote, dict) else 0

        # RSI
        rsi = ti.get('rsi_14')
        if rsi is not None:
            if rsi > 70:
                signals['details'].append({'factor': 'RSI超买', 'impact': 'negative', 'weight': 1})
                score -= 1
            elif rsi < 30:
                signals['details'].append({'factor': 'RSI超卖', 'impact': 'positive', 'weight': 1})
                score += 1

        # MACD
        macd = ti.get('macd_hist')
        if macd is not None:
            if macd > 0:
                signals['details'].append({'factor': 'MACD金叉', 'impact': 'positive', 'weight': 1})
                score += 1
            else:
                signals['details'].append({'factor': 'MACD死叉', 'impact': 'negative', 'weight': 1})
                score -= 1

        # 价格 vs 均线
        for ma in ['ma5', 'ma10', 'ma20']:
            val = ti.get(ma)
            if price and val:
                if price > val:
                    score += 0.5
                    signals['details'].append({'factor': f'价格>MA{ma.upper()}', 'impact': 'positive', 'weight': 0.5})
                else:
                    score -= 0.5
                    signals['details'].append({'factor': f'价格<MA{ma.upper()}', 'impact': 'negative', 'weight': 0.5})

        # KDJ
        kj = ti.get('kdj_j')
        if kj is not None:
            if kj > 80:
                signals['details'].append({'factor': 'KDJ超买', 'impact': 'negative', 'weight': 0.5})
                score -= 0.5
            elif kj < 20:
                signals['details'].append({'factor': 'KDJ超卖', 'impact': 'positive', 'weight': 0.5})
                score += 0.5

        # 布林带
        bu = ti.get('boll_upper')
        bl = ti.get('boll_lower')
        if price and bu and bl:
            if price >= bu:
                signals['details'].append({'factor': '价格触及布林上轨', 'impact': 'negative', 'weight': 0.5})
                score -= 0.5
            elif price <= bl:
                signals['details'].append({'factor': '价格触及布林下轨', 'impact': 'positive', 'weight': 0.5})
                score += 0.5

        # 估值信号
        vl = state.valuation_level
        if vl == '很低':
            signals['details'].append({'factor': '估值极低', 'impact': 'positive', 'weight': 2})
            score += 2
        elif vl == '偏低':
            signals['details'].append({'factor': '估值偏低', 'impact': 'positive', 'weight': 1})
            score += 1
        elif vl == '偏高':
            signals['details'].append({'factor': '估值偏高', 'impact': 'negative', 'weight': 1})
            score -= 1
        elif vl == '很高':
            signals['details'].append({'factor': '估值极高', 'impact': 'negative', 'weight': 2})
            score -= 2

        # 舆情
        ns = state.sentiment_news or {}
        gs = state.sentiment_guba or {}
        news_score = ns.get('avg_score', 0) or 0
        guba_score = gs.get('avg_score', 0) or 0
        sent_avg = (news_score + guba_score) / 2
        score += sent_avg * 2
        if sent_avg != 0:
            signals['details'].append({
                'factor': '舆情情感',
                'impact': 'positive' if sent_avg > 0 else 'negative',
                'weight': 2,
            })

        signals['score'] = round(score, 2)
        if score > 3:
            signals['overall'] = 'bullish'
        elif score < -3:
            signals['overall'] = 'bearish'
        else:
            signals['overall'] = 'neutral'

        return signals

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
