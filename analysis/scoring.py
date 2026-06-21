"""
Scoring Engine — -5 ~ +5 综合评分体系

三层架构：
  技术面 (50%): RSI + MACD + 均线排列 + 布林带 + 量价关系 + 动量
  基本面 (30%): PE分位(行业修正) + ROE(现金流验证) + 利润增长 + 股息率
  舆情面 (20%): 新闻情感 + 股吧情感 + 情绪一致性

市场状态自适应：趋势市放大均线/动量权重，震荡市放大 RSI/布林带权重
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
import math


@dataclass
class FactorDetail:
    factor: str
    impact: str           # 'positive' | 'negative' | 'neutral'
    contribution: float   # 该因子对总分的实际贡献
    raw_value: Any = None
    description: str = ""


@dataclass
class ScoreResult:
    final: float                    # -5 ~ +5
    label: str                      # 强烈看多/看多/偏多/中性/偏空/看空/强烈看空
    technical: float                # 技术面分项 -5 ~ +5
    fundamental: float              # 基本面分项 -5 ~ +5
    sentiment: float                # 舆情面分项 -5 ~ +5
    regime: str                     # 'trending_up' | 'trending_down' | 'ranging'
    weights: Dict[str, float]       # 实际使用的权重
    breakdown: List[FactorDetail]   # 逐因子明细
    confidence: str                 # '高' | '中' | '低'


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


class ScoringEngine:
    """-5 ~ +5 综合评分引擎"""

    # ---- 公开 API ----

    def compute(self, state) -> ScoreResult:
        """
        从 AnalysisState 计算综合评分。

        输入: AnalysisState (包含 quote, technical_indicators,
              financial_summary, sentiment_news, sentiment_guba,
              valuation_percentile, valuation_level, news, guba_posts)
        输出: ScoreResult
        """
        ti = self._unwrap(state.technical_indicators)
        quote = self._unwrap(state.quote)
        fin = self._unwrap(state.financial_summary)
        sent_news = getattr(state, 'sentiment_news', None) or {}
        sent_guba = getattr(state, 'sentiment_guba', None) or {}

        price = safe_float(quote.get('price', 0))
        prev_close = safe_float(quote.get('prev_close', price))
        change_pct = safe_float(quote.get('change_pct', 0))

        # 可用性检测
        available = {
            'technical': price > 0 and ti.get('rsi_14') is not None,
            'fundamental': len(fin) > 0,
            'sentiment': bool(sent_news.get('total_count', 0) or sent_guba.get('total_count', 0)),
            'pe': getattr(state, 'valuation_percentile', None) is not None,
        }

        # 市场状态检测
        regime = self._detect_regime(ti, price)

        # 动态权重
        weights = self._dynamic_weights(regime, available)

        # 各层计算
        technical = self._score_technical(ti, quote, price, regime)
        fundamental = self._score_fundamental(fin, state, available)
        sentiment = self._score_sentiment(sent_news, sent_guba)

        # 加权汇总
        raw = (
            weights['technical'] * technical +
            weights['fundamental'] * fundamental +
            weights['sentiment'] * sentiment
        )
        final = clamp(round(raw, 1), -5.0, 5.0)

        # 明细
        breakdown = self._build_breakdown(
            ti, quote, price, fin, state, sent_news, sent_guba,
            technical, fundamental, sentiment, weights, regime
        )

        return ScoreResult(
            final=final,
            label=self._score_to_label(final),
            technical=round(technical, 1),
            fundamental=round(fundamental, 1),
            sentiment=round(sentiment, 1),
            regime=regime,
            weights=weights,
            breakdown=breakdown,
            confidence=self._confidence(final, abs(technical), abs(fundamental)),
        )

    # ---- 市场状态检测 ----

    def _detect_regime(self, ti: dict, price: float) -> str:
        """
        用 ADX 和 MA 关系判断市场状态。
        返回: 'trending_up' | 'trending_down' | 'ranging'
        """
        ma5 = safe_float(ti.get('ma5', 0))
        ma20 = safe_float(ti.get('ma20', 0))
        ma60 = safe_float(ti.get('ma60', 0))

        # 简易 ADX 替代: 用 MA 排列强度（均线发散程度）
        if ma5 > 0 and ma20 > 0 and ma60 > 0:
            spread = abs(ma5 - ma20) / ma20 + abs(ma20 - ma60) / ma60
            if spread > 0.04 and price > ma5 > ma20:
                return 'trending_up'
            elif spread > 0.04 and price < ma5 < ma20:
                return 'trending_down'

        return 'ranging'

    # ---- 动态权重 ----

    def _dynamic_weights(self, regime: str, available: dict) -> Dict[str, float]:
        w = {'technical': 0.50, 'fundamental': 0.30, 'sentiment': 0.20}

        # 缺失数据重新分配
        if not available['sentiment']:
            lost = w['sentiment']
            w['sentiment'] = 0
            w['technical'] += lost * 0.5
            w['fundamental'] += lost * 0.5
        if not available['fundamental'] and not available['pe']:
            lost = w['fundamental']
            w['fundamental'] = 0.10  # 保留一点给估值分位
            w['technical'] += lost * 0.6
            w['sentiment'] += lost * 0.4

        # 趋势市中技术面更有效，震荡市中基本面权重提升
        if regime in ('trending_up', 'trending_down'):
            w['technical'] = min(0.60, w['technical'] + 0.05)
            w['fundamental'] = max(0.20, w['fundamental'] - 0.05)
        else:  # ranging
            w['technical'] = max(0.40, w['technical'] - 0.05)
            w['fundamental'] = min(0.35, w['fundamental'] + 0.05)

        # 归一化
        total = w['technical'] + w['fundamental'] + w['sentiment']
        return {k: round(v / total, 2) for k, v in w.items()}

    # ================================================================
    #  技术面 (-5 ~ +5)
    # ================================================================

    def _score_technical(self, ti: dict, quote: dict, price: float, regime: str) -> float:
        rsi = self._rsi_score(ti)            # [-2, +2]
        macd = self._macd_score(ti, price)    # [-1.5, +1.5]
        ma = self._ma_alignment_score(ti, price)  # [-1.5, +1.5]
        bb = self._bollinger_score(ti, price) # [-1, +1]
        vol = self._volume_score(ti, quote)   # [-1, +1]
        mom = self._momentum_score(ti, price) # [-1.5, +1.5]

        # 趋势市: MA + 动量权重提升; 震荡市: RSI + 布林带权重提升
        if regime in ('trending_up', 'trending_down'):
            raw = rsi * 0.9 + macd * 1.0 + ma * 1.3 + bb * 0.6 + vol * 0.8 + mom * 1.2
        else:
            raw = rsi * 1.3 + macd * 0.8 + ma * 0.7 + bb * 1.2 + vol * 0.8 + mom * 0.7

        return clamp(round(raw, 1), -5.0, 5.0)

    def _rsi_score(self, ti: dict) -> float:
        """RSI 非线性映射: 超卖/超买区加速，中性区平缓 [-2, +2]"""
        rsi = safe_float(ti.get('rsi_14', 50), 50)
        if rsi <= 0:
            return 0
        if rsi <= 25:
            return +2.0                 # 极度超卖
        elif rsi <= 35:
            return +0.5 + (35 - rsi) * 0.15  # [+0.5, +2.0]
        elif rsi <= 45:
            return (rsi - 40) * 0.05    # [-0.25, +0.25]
        elif rsi <= 55:
            return -(rsi - 50) * 0.1    # [-0.5, +0.5]
        elif rsi <= 65:
            return -0.5 - (rsi - 55) * 0.15  # [-0.5, -2.0]
        elif rsi <= 75:
            return -2.0 - (rsi - 65) * 0.05  # [-2.0, -2.5]
        else:
            return -2.0                 # 极度超买

    def _macd_score(self, ti: dict, price: float) -> float:
        """MACD 柱状线强度: 用 ATR 做波动率归一化 [-1.5, +1.5]"""
        hist = safe_float(ti.get('macd_hist', 0))
        if hist == 0 or price <= 0:
            return 0

        boll_upper = safe_float(ti.get('boll_upper', 0))
        boll_lower = safe_float(ti.get('boll_lower', 0))
        atr_est = (boll_upper - boll_lower) / 4 if boll_upper > boll_lower else price * 0.02
        if atr_est <= 0:
            return 0

        normalized = hist / atr_est
        return clamp(normalized * 0.3, -1.5, 1.5)

    def _ma_alignment_score(self, ti: dict, price: float) -> float:
        """均线排列: 越整齐的多头/空头排列分越高 [-1.5, +1.5]"""
        mas = {
            'MA5': safe_float(ti.get('ma5', 0)),
            'MA10': safe_float(ti.get('ma10', 0)),
            'MA20': safe_float(ti.get('ma20', 0)),
        }
        valid_mas = {k: v for k, v in mas.items() if v > 0}
        if len(valid_mas) < 2 or price <= 0:
            return 0

        above = sum(1 for v in valid_mas.values() if price > v)
        below = len(valid_mas) - above

        # 检查严格多头排列 (price > MA5 > MA10 > MA20)
        values = [price] + sorted(valid_mas.values(), reverse=True)
        is_bullish_align = all(values[i] > values[i+1] for i in range(len(values)-1))
        # 检查严格空头排列
        values_rev = [price] + sorted(valid_mas.values())
        is_bearish_align = all(values_rev[i] < values_rev[i+1] for i in range(len(values_rev)-1))

        if is_bullish_align:
            return +1.5
        elif is_bearish_align:
            return -1.5
        elif above == len(valid_mas):
            return +0.8
        elif below == len(valid_mas):
            return -0.8
        elif above > below:
            return +0.3
        elif below > above:
            return -0.3
        return 0

    def _bollinger_score(self, ti: dict, price: float) -> float:
        """布林带 %B 位置: 均值回归逻辑 [-1, +1]"""
        upper = safe_float(ti.get('boll_upper', 0))
        lower = safe_float(ti.get('boll_lower', 0))
        if upper <= lower or price <= 0:
            return 0

        pct_b = (price - lower) / (upper - lower)
        return -clamp((pct_b - 0.5) * 2.5, -1.0, 1.0)

    def _volume_score(self, ti: dict, quote: dict) -> float:
        """量价关系: 结合价格位置判断 [-1, +1]"""
        vol_ratio = safe_float(ti.get('volume_ratio', 1), 1)
        change_pct = safe_float(quote.get('change_pct', 0))
        boll_mid = safe_float(ti.get('boll_middle', 0))

        # 价格位置（布林带中轨附近为中性）
        if boll_mid > 0:
            price_level = safe_float(quote.get('price', 0)) / boll_mid
        else:
            price_level = 1.0

        if vol_ratio > 1.8:
            if change_pct > 2 and price_level < 0.95:    # 低位放量大涨 → 突破确认
                return +1.0
            elif change_pct > 2:                          # 高位放量大涨 → 警惕出货
                return +0.3
            elif change_pct < -2 and price_level < 0.95: # 低位放量大跌 → 恐慌盘，可能见底
                return +0.3
            elif change_pct < -2:                         # 高位放量大跌 → 危险
                return -1.0
            else:
                return 0.2 if change_pct > 0 else -0.2
        elif vol_ratio < 0.5:
            return -0.3                                    # 缩量，方向不明
        return 0

    def _momentum_score(self, ti: dict, price: float) -> float:
        """20 日动量: 用 MA20 估算 [-1.5, +1.5]"""
        ma20 = safe_float(ti.get('ma20', 0))
        if ma20 <= 0:
            return 0

        roc = (price - ma20) / ma20 * 100  # 20日涨跌幅百分比
        return clamp(roc * 0.15, -1.5, 1.5)

    # ================================================================
    #  基本面 (-5 ~ +5)
    # ================================================================

    def _score_fundamental(self, fin: dict, state, available: dict) -> float:
        pe = self._pe_score(state, fin)
        roe = self._roe_score(state, fin)
        growth = self._growth_score(fin)
        div = self._dividend_score(fin)

        # 无 PE 数据时降 PE 权重
        if not available.get('pe'):
            pe = 0
            roe_weight = 0.45
            growth_weight = 0.35
            div_weight = 0.20
        else:
            roe_weight = 0.30
            growth_weight = 0.25
            div_weight = 0.15
            # pe 占 0.30

        raw = pe * 0.30 + roe * roe_weight + growth * growth_weight + div * div_weight
        return clamp(round(raw, 1), -5.0, 5.0)

    def _pe_score(self, state, fin: dict) -> float:
        """PE 分位: 加入行业相对估值修正 [-3, +3]"""
        percentile = getattr(state, 'valuation_percentile', None)
        if percentile is None:
            return 0

        # PE 分位基础分
        if percentile < 5:
            base = +3.0
        elif percentile < 15:
            base = +2.0
        elif percentile < 30:
            base = +1.0
        elif percentile < 70:
            base = 0.0
        elif percentile < 85:
            base = -1.0
        elif percentile < 95:
            base = -2.0
        else:
            base = -3.0

        # 盈利趋势修正: 利润正增长 → PE 低是真实的便宜; 利润负增长 → PE 低可能是价值陷阱
        net_profit_yoy = safe_float(fin.get('net_profit_yoy', 0))
        if net_profit_yoy > 20:
            adjustment = +0.5
        elif net_profit_yoy > 0:
            adjustment = +0.2
        elif net_profit_yoy < -10:
            adjustment = -0.5
        else:
            adjustment = 0

        return clamp(base + adjustment, -3.0, 3.0)

    def _roe_score(self, state, fin: dict) -> float:
        """ROE + 现金流验证 [-1.5, +1.5]"""
        roe = safe_float(fin.get('roe', 0))
        debt_ratio = safe_float(fin.get('debt_ratio', 0))
        operating_cf = safe_float(fin.get('operating_cash_flow', None))
        free_cf = safe_float(fin.get('free_cash_flow', None))
        net_profit = safe_float(fin.get('net_profit', None))
        is_financial = _is_financial_industry(state) if hasattr(state, 'get') else False

        if roe > 25:
            base = +1.5
        elif roe > 15:
            base = +1.0
        elif roe > 8:
            base = +0.5
        elif roe > 3:
            base = 0.0
        elif roe > 0:
            base = -0.5
        else:
            base = -1.5

        # 高负债扣分（金融行业除外：银行/保险高负债是结构性特征）
        if debt_ratio > 70 and not is_financial:
            base -= 0.3

        # 现金流验证：经营现金流为正 → +0.2，自由现金流为负 → -0.4
        if operating_cf is not None and operating_cf > 0:
            base += 0.2
        if free_cf is not None and free_cf < 0:
            base -= 0.4
        # 利润含金量：FCF/净利 > 0.8 → +0.2
        if free_cf is not None and net_profit is not None and net_profit > 0:
            fcf_ratio = free_cf / net_profit
            if fcf_ratio > 0.8:
                base += 0.2
            elif fcf_ratio < 0:
                base -= 0.3

        return clamp(base, -1.5, 1.5)

    def _growth_score(self, fin: dict) -> float:
        """利润增长 + 营收增长 [-0.5, +1]"""
        profit_yoy = safe_float(fin.get('net_profit_yoy', 0))
        revenue_yoy = safe_float(fin.get('revenue_yoy', 0))

        profit_score = clamp(profit_yoy * 0.025, -0.5, 0.7)
        revenue_score = clamp(revenue_yoy * 0.01, -0.2, 0.3)

        return clamp(profit_score + revenue_score, -0.5, 1.0)

    def _dividend_score(self, fin: dict) -> float:
        """股息率: A 股高股息是加分项 [-0.3, +0.5]"""
        # 股息率通常不在 financial_summary 中直接提供，从 PE/PB 反推或跳过
        eps = safe_float(fin.get('eps', 0))
        roe = safe_float(fin.get('roe', 0))
        debt_ratio = safe_float(fin.get('debt_ratio', 0))

        # 用 ROE 和负债率间接估计分红意愿
        if eps > 0 and roe > 15 and debt_ratio < 50:
            return +0.5
        elif eps > 0 and roe > 10:
            return +0.3
        return 0

    # ================================================================
    #  舆情面 (-5 ~ +5)
    # ================================================================

    def _score_sentiment(self, sent_news: dict, sent_guba: dict) -> float:
        """新闻 + 股吧情感 + 一致性 [-5, +5]"""
        news_avg = sent_news.get('avg_score', 0) or 0
        guba_avg = sent_guba.get('avg_score', 0) or 0

        # 映射到 [-2.5, +2.5]
        news_scaled = clamp(news_avg * (2.5 / 0.9), -2.5, 2.5)
        guba_scaled = clamp(guba_avg * (2.5 / 0.9), -2.5, 2.5)

        # 一致性调整
        if abs(news_avg) > 0.4 and abs(guba_avg) > 0.4:
            if (news_avg > 0) == (guba_avg > 0):
                consistency = +0.5
            else:
                consistency = -0.5
        else:
            consistency = 0

        # 新闻权重略高（机构资金 vs 散户情绪）
        raw = news_scaled * 0.55 + guba_scaled * 0.45 + consistency
        return clamp(round(raw, 1), -5.0, 5.0)

    # ================================================================
    #  结果解析
    # ================================================================

    def _score_to_label(self, score: float) -> str:
        if score >= 4.0:
            return '强烈看多'
        elif score >= 2.0:
            return '看多'
        elif score >= 0.5:
            return '偏多'
        elif score > -0.5:
            return '中性'
        elif score > -2.0:
            return '偏空'
        elif score > -4.0:
            return '看空'
        else:
            return '强烈看空'

    def _confidence(self, final: float, tech_abs: float, fund_abs: float) -> str:
        if abs(final) > 3.5 and tech_abs > 2.5 and fund_abs > 1.5:
            return '高'
        elif abs(final) > 1.5:
            return '中'
        return '低'

    # ---- 明细构建 ----

    def _build_breakdown(
        self, ti: dict, quote: dict, price: float, fin: dict, state,
        sent_news: dict, sent_guba: dict,
        tech: float, fund: float, sent: float,
        weights: dict, regime: str,
    ) -> List[FactorDetail]:
        details: List[FactorDetail] = []

        # 市场状态
        regime_label = {'trending_up': '上升趋势', 'trending_down': '下降趋势', 'ranging': '震荡盘整'}

        # 技术面明细
        rsi = safe_float(ti.get('rsi_14', 50), 50)
        rsi_contrib = self._rsi_score(ti)
        details.append(FactorDetail(
            factor=f'RSI(14)={rsi:.0f}', raw_value=rsi,
            impact='positive' if rsi_contrib > 0 else 'negative' if rsi_contrib < 0 else 'neutral',
            contribution=rsi_contrib, description='非线性加速映射',
        ))

        macd_hist = safe_float(ti.get('macd_hist', 0))
        macd_contrib = self._macd_score(ti, price)
        details.append(FactorDetail(
            factor=f'MACD柱={macd_hist:.3f}', raw_value=macd_hist,
            impact='positive' if macd_contrib > 0 else 'negative' if macd_contrib < 0 else 'neutral',
            contribution=macd_contrib, description='ATR归一化',
        ))

        ma_contrib = self._ma_alignment_score(ti, price)
        details.append(FactorDetail(
            factor='均线排列', raw_value={k: safe_float(ti.get(k, 0)) for k in ['ma5','ma10','ma20']},
            impact='positive' if ma_contrib > 0 else 'negative' if ma_contrib < 0 else 'neutral',
            contribution=ma_contrib, description=f'多头排列' if ma_contrib > 0 else '空头排列' if ma_contrib < 0 else '交叉缠绕',
        ))

        bb_contrib = self._bollinger_score(ti, price)
        pct_b = (price - safe_float(ti.get('boll_lower', 0))) / max(safe_float(ti.get('boll_upper', 0)) - safe_float(ti.get('boll_lower', 0)), 0.01)
        details.append(FactorDetail(
            factor=f'布林带(%B={pct_b:.2f})', raw_value=pct_b,
            impact='positive' if bb_contrib > 0 else 'negative' if bb_contrib < 0 else 'neutral',
            contribution=bb_contrib, description='触下轨反弹' if pct_b < 0.2 else '触上轨回调' if pct_b > 0.8 else '中轨附近',
        ))

        vol_contrib = self._volume_score(ti, quote)
        details.append(FactorDetail(
            factor=f'量价关系(量比={safe_float(ti.get("volume_ratio",1)):.1f})',
            raw_value=safe_float(ti.get('volume_ratio', 1)),
            impact='positive' if vol_contrib > 0 else 'negative' if vol_contrib < 0 else 'neutral',
            contribution=vol_contrib,
        ))

        mom_contrib = self._momentum_score(ti, price)
        details.append(FactorDetail(
            factor='20日动量',
            raw_value=round((price - safe_float(ti.get('ma20', price))) / max(price, 0.01) * 100, 1),
            impact='positive' if mom_contrib > 0 else 'negative' if mom_contrib < 0 else 'neutral',
            contribution=mom_contrib,
        ))

        # 基本面明细
        pe_contrib = self._pe_score(state, fin)
        pct = getattr(state, 'valuation_percentile', None)
        details.append(FactorDetail(
            factor=f'PE估值分位({pct:.0f}%)' if pct is not None else 'PE估值(无数据)',
            raw_value=pct,
            impact='positive' if pe_contrib > 0 else 'negative' if pe_contrib < 0 else 'neutral',
            contribution=pe_contrib,
            description=getattr(state, 'valuation_level', '') or '',
        ))

        roe = safe_float(fin.get('roe', 0))
        roe_contrib = self._roe_score(state, fin)
        details.append(FactorDetail(
            factor=f'ROE({roe:.1f}%)', raw_value=roe,
            impact='positive' if roe_contrib > 0 else 'negative' if roe_contrib < 0 else 'neutral',
            contribution=roe_contrib,
        ))

        profit_yoy = safe_float(fin.get('net_profit_yoy', 0))
        growth_contrib = self._growth_score(fin)
        details.append(FactorDetail(
            factor=f'利润增长({profit_yoy:+.1f}%)', raw_value=profit_yoy,
            impact='positive' if growth_contrib > 0 else 'negative' if growth_contrib < 0 else 'neutral',
            contribution=growth_contrib,
        ))

        div_contrib = self._dividend_score(fin)
        details.append(FactorDetail(
            factor='分红潜力', raw_value=None,
            impact='positive' if div_contrib > 0 else 'neutral',
            contribution=div_contrib,
        ))

        # 舆情面明细
        news_avg = sent_news.get('avg_score', 0) or 0
        guba_avg = sent_guba.get('avg_score', 0) or 0
        sent_contrib = self._score_sentiment(sent_news, sent_guba)
        details.append(FactorDetail(
            factor=f'新闻情感(avg={news_avg:.2f})', raw_value=news_avg,
            impact='positive' if news_avg > 0 else 'negative' if news_avg < 0 else 'neutral',
            contribution=clamp(news_avg * (2.5/0.9) * 0.55, -2.5, 2.5),
        ))
        details.append(FactorDetail(
            factor=f'股吧情感(avg={guba_avg:.2f})', raw_value=guba_avg,
            impact='positive' if guba_avg > 0 else 'negative' if guba_avg < 0 else 'neutral',
            contribution=clamp(guba_avg * (2.5/0.9) * 0.45, -2.5, 2.5),
        ))

        return details

    # ---- 工具 ----

    @staticmethod
    def _unwrap(obj: Any) -> dict:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        return {}

def _is_financial_industry(state) -> bool:
    """判断行业是否金融业（银行/保险/证券），高负债是结构性特征"""
    try:
        names = ''
        stock_name = ''
        if hasattr(state, 'industry_context') and state.industry_context:
            ctx = state.industry_context
            if isinstance(ctx, dict):
                names = str(ctx.get('industry_names', ''))
        if hasattr(state, 'stock_name') and state.stock_name:
            stock_name = str(state.stock_name)
        return any(k in names or k in stock_name for k in ('银行', '保险', '证券', '金融', '券商'))
    except Exception:
        return False
