"""
LLM 预测节点

将所有分析数据（技术面 + 基本面 + 舆情 + 成本 + 信号）输入 LLM，
生成多周期预测（短/中/长期）和操作建议。
"""
import json
import os
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class PredictionResult:
    """LLM 预测输出"""
    analysis_text: str = ""
    outlook: str = "中性"
    reason: str = ""
    risk_factors: List[str] = field(default_factory=list)
    positive_factors: List[str] = field(default_factory=list)

    # 多周期预测
    short_term: Optional[Dict] = None   # {direction, change_pct, confidence, reason}
    mid_term: Optional[Dict] = None
    long_term: Optional[Dict] = None

    # 操作建议
    suggested_action: Optional[Dict] = None  # {action, reason, stop_loss, take_profit}

    # 兼容旧字段
    price_target_current: Optional[float] = None
    price_target_low: Optional[float] = None
    price_target_high: Optional[float] = None
    confidence: str = "低"

    raw_llm_output: str = ""


class PredictionNode:
    """预测节点"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None):
        from config import settings
        self.api_key = api_key or os.environ.get('LLM_API_KEY') or getattr(settings, 'LLM_API_KEY', None) or ''
        self.base_url = base_url or os.environ.get('LLM_BASE_URL') or getattr(settings, 'LLM_BASE_URL', None) or 'https://api.openai.com/v1'
        self.model = model or os.environ.get('LLM_MODEL_NAME') or getattr(settings, 'LLM_MODEL_NAME', None) or 'gpt-4o-mini'

    def predict(self, state: dict) -> PredictionResult:
        if self.api_key:
            return self._llm_predict(state)
        else:
            return self._rule_predict(state)

    def _llm_predict(self, state: dict) -> PredictionResult:
        prompt = self._build_prompt(state)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的A股分析师。请综合技术面、基本面、估值、舆情、成本价格，给出多周期预测和操作建议。必须输出严格JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            return self._parse_llm_response(raw)
        except Exception as e:
            logger.warning(f"LLM 预测失败, 使用规则降级: {e}")
            return self._rule_predict(state)

    def _rule_predict(self, state: dict) -> PredictionResult:
        signals = state.get('signals', {}) or {}
        score = signals.get('score', 0)
        quote = state.get('quote', {}) or {}
        price = quote.get('price', 0) if isinstance(quote, dict) else 0
        cost = state.get('cost_price', 0) or 0

        if score > 2:
            outlook = "看多"
        elif score < -2:
            outlook = "看空"
        else:
            outlook = "中性"

        # 短期：±2-5%
        st_dir = "上涨" if score > 0 else "下跌"
        st_pct = round(abs(score) * 1.5, 1)
        st_conf = "高" if abs(score) > 4 else "中" if abs(score) > 2 else "低"
        # 中期：±5-15%
        mt_dir = "上涨" if score > 0 else "下跌"
        mt_pct = round(abs(score) * 3, 1)
        mt_conf = "中" if abs(score) > 3 else "低"
        # 长期：±10-25%
        lt_dir = "上涨" if score > 0 else "下跌"
        lt_pct = round(abs(score) * 5, 1)
        lt_conf = "低"

        # 操作建议
        if cost and cost > 0:
            pnl = (price - cost) / cost * 100
            if pnl > 10:
                action = "建议减仓"
                action_reason = f"已盈利{pnl:.0f}%，可部分止盈锁定利润"
            elif pnl > 0:
                action = "持有观望"
                action_reason = f"浮盈{pnl:.0f}%，趋势尚可，继续持有"
            elif pnl > -5:
                action = "持有等待"
                action_reason = f"小幅浮亏{abs(pnl):.0f}%，尚未触及止损，观察反弹信号"
            else:
                action = "建议止损"
                action_reason = f"浮亏{abs(pnl):.0f}%，建议设止损位控制风险"
        else:
            action = "参考估值买入"
            action_reason = "无成本价格，建议参考估值分位和布林下轨分批建仓"

        stop_loss = round(price * 0.93, 2) if price else 0
        take_profit = round(price * 1.15, 2) if price else 0

        return PredictionResult(
            outlook=outlook,
            analysis_text=f"## {outlook}信号\n\n综合评分 {score:.1f}，当前走势偏{outlook}。",
            reason=f"综合评分{score:.1f}",
            short_term={"direction": st_dir, "change_pct": st_pct, "confidence": st_conf,
                        "reason": f"基于技术信号强度{score:.1f}的短期推算"},
            mid_term={"direction": mt_dir, "change_pct": mt_pct, "confidence": mt_conf,
                      "reason": f"基于趋势延续和估值{state.get('valuation_level', '正常')}的中期推算"},
            long_term={"direction": lt_dir, "change_pct": lt_pct, "confidence": lt_conf,
                       "reason": f"基于PE分位{state.get('valuation_percentile', 50):.0f}%的长期均值回归推算"},
            suggested_action={"action": action, "reason": action_reason,
                              "stop_loss": stop_loss, "take_profit": take_profit},
            price_target_current=price,
            price_target_low=round(price * 0.95, 2) if price else None,
            price_target_high=round(price * 1.10, 2) if price else None,
            confidence=st_conf,
            risk_factors=[],
            positive_factors=[],
        )

    def _build_prompt(self, state: dict) -> str:
        quote = state.get('quote', {}) or {}
        ti = state.get('technical_indicators', {}) or {}
        fs = state.get('financial_summary', {}) or {}
        sn = state.get('sentiment_news', {}) or {}
        sg = state.get('sentiment_guba', {}) or {}
        signals = state.get('signals', {}) or {}
        symbol = state.get('symbol', '')
        name = state.get('stock_name', symbol)
        cost = state.get('cost_price', 0) or 0
        val = state.get('valuation_level', '正常')
        val_pct = state.get('valuation_percentile', 50)
        sug_buy = state.get('suggested_buy_price', 0) or 0
        avg_pe = state.get('historical_pe_avg', 0) or 0
        price = quote.get('price', 0) or 0

        # 利好利空摘要
        bull_news = state.get('important_bullish_news', []) or []
        bear_news = state.get('important_bearish_news', []) or []
        bull_guba = state.get('important_bullish_guba', []) or []
        bear_guba = state.get('important_bearish_guba', []) or []
        bull_lines = "\n".join([f"  - {n['title']}" for n in bull_news[:3] + bull_guba[:2]])
        bear_lines = "\n".join([f"  - {n['title']}" for n in bear_news[:3] + bear_guba[:2]])

        cost_line = f"- 成本价格: {cost}\n- 当前盈亏: {((price - cost) / cost * 100) if cost and price else 'N/A'}" if cost else "- 成本价格: 未提供"

        return f"""请分析以下A股数据，输出多周期预测和操作建议的JSON。

## 股票信息
- 代码: {symbol}  名称: {name}
{cost_line}

## 行情
- 现价: {price}  涨跌幅: {quote.get('change_pct', 'N/A')}%
- PE: {quote.get('pe', 'N/A')}  PB: {quote.get('pb', 'N/A')}
- 市值: {quote.get('market_cap', 'N/A')}亿  换手率: {quote.get('turnover_rate', 'N/A')}%

## 估值
- 等级: {val} (PE历史分位: {val_pct}%)
- 历史PE均值: {avg_pe}
- 建议买入价: {sug_buy}

## 技术指标
- RSI(14): {ti.get('rsi_14', 'N/A')}
- MACD柱: {ti.get('macd_hist', 'N/A')}
- KDJ: {ti.get('kdj_k', 'N/A')}/{ti.get('kdj_d', 'N/A')}/{ti.get('kdj_j', 'N/A')}
- MA5:{ti.get('ma5','N/A')} MA10:{ti.get('ma10','N/A')} MA20:{ti.get('ma20','N/A')}
- 布林: {ti.get('boll_lower','N/A')} ~ {ti.get('boll_middle','N/A')} ~ {ti.get('boll_upper','N/A')}

## 基本面
- EPS: {fs.get('eps','N/A')}  ROE: {fs.get('roe','N/A')}%
- 营收: {fs.get('revenue','N/A')}亿  净利: {fs.get('net_profit','N/A')}亿

## 舆情
- 新闻: avg={sn.get('avg_score','N/A')} pos={sn.get('positive_count',0)} neg={sn.get('negative_count',0)} total={sn.get('total_count',0)}
- 股吧: avg={sg.get('avg_score','N/A')} pos={sg.get('positive_count',0)} neg={sg.get('negative_count',0)} total={sg.get('total_count',0)}

## 利好因素
{bull_lines or '  无'}

## 利空因素
{bear_lines or '  无'}

## 综合信号
- 评分: {signals.get('score','N/A')}  方向: {signals.get('overall','N/A')}

## 要求
请基于现价、{'成本价'+str(cost)+'、' if cost else ''}技术面、估值、舆情，输出JSON:

{{
  "analysis_text": "综合分析(200字内)",
  "outlook": "看多/看空/中性",
  "reason": "核心逻辑(50字内)",
  "short_term": {{
    "direction": "上涨/下跌/震荡",
    "change_pct": 3.5,
    "confidence": "高/中/低",
    "reason": "1~2周预测依据(30字内)"
  }},
  "mid_term": {{
    "direction": "上涨/下跌/震荡",
    "change_pct": 8.0,
    "confidence": "高/中/低",
    "reason": "1~3月预测依据(30字内)"
  }},
  "long_term": {{
    "direction": "上涨/下跌/震荡",
    "change_pct": 15.0,
    "confidence": "高/中/低",
    "reason": "6~12月预测依据(30字内)"
  }},
  "suggested_action": {{
    "action": "买入/加仓/持有/减仓/卖出",
    "reason": "操作理由(50字内)",
    "stop_loss": {price * 0.93:.1f},
    "take_profit": {price * 1.15:.1f}
  }},
  "risk_factors": ["风险1", "风险2"],
  "positive_factors": ["积极因素1", "积极因素2"]
}}

注意:
- change_pct 为正表示上涨幅度，为负表示下跌幅度
- 短期(1~2周)侧重技术信号和舆情热点
- 中期(1~3月)侧重趋势和估值回归
- 长期(6~12月)侧重基本面和行业前景
- 操作建议综合考虑{'成本盈亏、' if cost else ''}估值、技术面和风险承受"""

    def _parse_llm_response(self, raw: str) -> PredictionResult:
        try:
            if '```json' in raw:
                raw = raw.split('```json')[1].split('```')[0]
            elif '```' in raw:
                raw = raw.split('```')[1].split('```')[0]
            raw = raw.strip()
            d = json.loads(raw)

            return PredictionResult(
                analysis_text=d.get('analysis_text', ''),
                outlook=d.get('outlook', '中性'),
                reason=d.get('reason', ''),
                risk_factors=d.get('risk_factors', []),
                positive_factors=d.get('positive_factors', []),
                short_term=d.get('short_term'),
                mid_term=d.get('mid_term'),
                long_term=d.get('long_term'),
                suggested_action=d.get('suggested_action'),
                raw_llm_output=raw,
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"LLM 输出解析失败: {e}")
            return PredictionResult(
                analysis_text=raw,
                raw_llm_output=raw,
            )
