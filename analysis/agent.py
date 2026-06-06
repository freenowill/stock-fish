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

    def analyze(self, symbol: str, cost_price: float = 0.0, master: str = "",
                shares: int = 0, total_assets: float = 0.0, available_cash: float = 0.0) -> Dict[str, Any]:
        """执行一次完整分析，返回结构化结果。master 非空时启用大师决策模式。"""
        state = AnalysisState(symbol=symbol, cost_price=cost_price,
                              shares=shares, total_assets=total_assets, available_cash=available_cash,
                              created_at=datetime.now().isoformat())

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

        # --- 北向资金 (stock_hsgt_north_net_flow_in_em 已废弃, 改用 stock_hsgt_fund_flow_summary_em) ---
        try:
            df = StockAnalysisAgent._safe_ak_call(ak.stock_hsgt_fund_flow_summary_em)
            if df is not None and len(df) > 0:
                nb = df[df['资金方向'] == '北向']
                if len(nb) > 0:
                    ctx['northbound_flow'] = float(nb['成交净买额'].sum())
                    ctx['northbound_5d_avg'] = round(float(nb['成交净买额'].sum()) / max(1, len(nb)), 1)
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
