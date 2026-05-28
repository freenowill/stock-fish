"""
LLM 预测节点

将所有分析数据（技术面 + 基本面 + 舆情 + 信号）输入 LLM，
生成自然语言分析、价格目标、风险点。
"""
import json
import os
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class PredictionResult:
    """LLM 预测输出"""
    analysis_text: str                          # 自然语言分析
    price_target_current: Optional[float] = None
    price_target_low: Optional[float] = None
    price_target_high: Optional[float] = None
    confidence: str = "低"                       # 高/中/低
    outlook: str = "中性"                        # 看多/看空/中性
    risk_factors: List[str] = field(default_factory=list)
    positive_factors: List[str] = field(default_factory=list)
    reason: str = ""
    raw_llm_output: str = ""


class PredictionNode:
    """
    预测节点：将分析数据送入 LLM，生成综合预测。

    支持两种模式：
    1. LLM 模式（需要 API Key）→ 调用 OpenAI 兼容 API
    2. 规则模式（无 API Key） → 基于信号评分的简单预测
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None):
        from config import settings
        self.api_key = api_key or os.environ.get('LLM_API_KEY') or getattr(settings, 'LLM_API_KEY', None) or ''
        self.base_url = base_url or os.environ.get('LLM_BASE_URL') or getattr(settings, 'LLM_BASE_URL', None) or 'https://api.openai.com/v1'
        self.model = model or os.environ.get('LLM_MODEL_NAME') or getattr(settings, 'LLM_MODEL_NAME', None) or 'gpt-4o-mini'

    def predict(self, state: dict) -> PredictionResult:
        """执行预测"""
        if self.api_key:
            return self._llm_predict(state)
        else:
            return self._rule_predict(state)

    def _llm_predict(self, state: dict) -> PredictionResult:
        """LLM 模式：将结构化数据转为提示词，调用 LLM"""
        prompt = self._build_prompt(state)

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的 A 股股票分析师。请基于提供的技术面、基本面、舆情数据，给出客观的分析和预测。"},
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
        """规则模式：根据评分和阈值预测"""
        signals = state.get('signals', {}) or {}
        score = signals.get('score', 0)
        quote = state.get('quote', {}) or {}
        price = quote.get('price', 0) if isinstance(quote, dict) else 0
        ti = state.get('technical_indicators', {}) or {}

        if score > 2:
            outlook = "看多"
            confidence = "中" if score > 4 else "低"
            change_pct = round(abs(score) * 2, 1)
        elif score < -2:
            outlook = "看空"
            confidence = "中" if score < -4 else "低"
            change_pct = round(abs(score) * 2, 1)
        else:
            outlook = "中性"
            confidence = "低"
            change_pct = round(abs(score) * 1.5, 1)

        price_target_current = price
        price_target_low = round(price * (1 - 0.03 * max(abs(score), 1)), 2)
        price_target_high = round(price * (1 + 0.03 * max(abs(score), 1)), 2)

        factors = []
        for s in (signals.get('details') or []):
            impact = s.get('impact', '')
            factor = s.get('factor', '')
            if impact == 'positive':
                factors.append(factor)

        risk_factors_text = []
        for s in (signals.get('details') or []):
            if s.get('impact') == 'negative':
                risk_factors_text.append(s.get('factor', ''))

        analysis = (
            f"## {outlook}信号\n\n"
            f"综合评分 {score:.1f}，技术面与舆情分析显示当前走势偏{outlook}。\n\n"
        )
        if ti:
            rsi = ti.get('rsi_14', 'N/A')
            macd = ti.get('macd_hist', 'N/A')
            analysis += f"**技术面**: RSI={rsi}, MACD={'金叉' if macd and macd > 0 else '死叉'}\n\n"
        if factors:
            analysis += "**积极因素**: " + ", ".join(factors) + "\n\n"
        if risk_factors_text:
            analysis += "**风险因素**: " + ", ".join(risk_factors_text) + "\n\n"
        analysis += f"**预测区间**: {price_target_low} - {price_target_high} (当前 {price})\n"
        analysis += f"**置信度**: {confidence}"

        return PredictionResult(
            analysis_text=analysis,
            price_target_current=price_target_current,
            price_target_low=price_target_low,
            price_target_high=price_target_high,
            confidence=confidence,
            outlook=outlook,
            risk_factors=risk_factors_text,
            positive_factors=factors,
        )

    def _build_prompt(self, state: dict) -> str:
        """构建 LLM 提示词"""
        quote = state.get('quote', {}) or {}
        ti = state.get('technical_indicators', {}) or {}
        fs = state.get('financial_summary', {}) or {}
        news = state.get('news', []) or []
        sn = state.get('sentiment_news', {}) or {}
        sg = state.get('sentiment_guba', {}) or {}
        signals = state.get('signals', {}) or {}
        symbol = state.get('symbol', '')
        name = state.get('stock_name', state.get('name', symbol))

        return f"""请分析以下 A 股数据，输出 JSON 格式预测报告。

## 股票信息
- 代码: {symbol}
- 名称: {name}

## 实时行情
- 价格: {quote.get('price', 'N/A')}
- 涨跌幅: {quote.get('change_pct', 'N/A')}%
- PE: {quote.get('pe', 'N/A')}
- PB: {quote.get('pb', 'N/A')}
- 市值: {quote.get('market_cap', 'N/A')}亿

## 技术指标
- RSI(14): {ti.get('rsi_14', 'N/A')}
- MACD: {ti.get('macd_hist', 'N/A')}
- KDJ: {ti.get('kdj_k', 'N/A')}/{ti.get('kdj_d', 'N/A')}/{ti.get('kdj_j', 'N/A')}
- MA5: {ti.get('ma5', 'N/A')}  MA10: {ti.get('ma10', 'N/A')}  MA20: {ti.get('ma20', 'N/A')}
- 布林上下轨: {ti.get('boll_upper', 'N/A')} / {ti.get('boll_lower', 'N/A')}
- 量比: {ti.get('volume_ratio', 'N/A')}

## 基本面
- 营收: {fs.get('revenue', 'N/A')}亿  营收同比: {fs.get('revenue_yoy', 'N/A')}%
- 净利润: {fs.get('net_profit', 'N/A')}亿  净利同比: {fs.get('net_profit_yoy', 'N/A')}%
- EPS: {fs.get('eps', 'N/A')}  ROE: {fs.get('roe', 'N/A')}%

## 舆情数据
- 新闻情感得分: {sn.get('avg_score', 'N/A')} (共{sn.get('total_count', 0)}条)
- 股吧情感得分: {sg.get('avg_score', 'N/A')} (共{sg.get('total_count', 0)}条)

## 综合信号
- 评分: {signals.get('score', 'N/A')}
- 方向: {signals.get('overall', 'N/A')}

## 要求
请输出 JSON，格式如下:
{{
  "analysis_text": "综合分析段落",
  "price_target_current": 当前价,
  "price_target_low": 下限目标价,
  "price_target_high": 上限目标价,
  "confidence": "高/中/低",
  "outlook": "看多/看空/中性",
  "risk_factors": ["风险点1", "风险点2"],
  "positive_factors": ["积极因素1", "积极因素2"],
  "reason": "核心逻辑"
}}"""

    def _parse_llm_response(self, raw: str) -> PredictionResult:
        """解析 LLM JSON 响应"""
        try:
            # 清理可能的 markdown 包裹
            if '```json' in raw:
                raw = raw.split('```json')[1].split('```')[0]
            elif '```' in raw:
                raw = raw.split('```')[1].split('```')[0]
            raw = raw.strip()

            d = json.loads(raw)
            return PredictionResult(
                analysis_text=d.get('analysis_text', ''),
                price_target_current=d.get('price_target_current'),
                price_target_low=d.get('price_target_low'),
                price_target_high=d.get('price_target_high'),
                confidence=d.get('confidence', '低'),
                outlook=d.get('outlook', '中性'),
                risk_factors=d.get('risk_factors', []),
                positive_factors=d.get('positive_factors', []),
                reason=d.get('reason', ''),
                raw_llm_output=raw,
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"LLM 输出解析失败: {e}")
            return PredictionResult(
                analysis_text=raw,
                raw_llm_output=raw,
            )
