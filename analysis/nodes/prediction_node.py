"""
LLM 预测节点 — 多 Agent 并行辩论模式

3 个专职 Agent 并行分析各自领域数据:
  - 技术面 Agent: 只看 K 线/指标
  - 基本面 Agent: 只看估值/财务
  - 舆情 Agent:   只看新闻/股吧/情感

Moderator 阅读三方观点后综合裁决，输出最终预测。

借鉴 BettaFish ForumEngine 的多 Agent 辩论模式。
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class AgentView:
    """单个 Agent 的观点"""
    role: str = ""            # tech / fundamental / sentiment
    outlook: str = "中性"     # 看多/看空/中性
    confidence: str = "低"    # 高/中/低
    score: float = 0          # -10 ~ 10
    key_points: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    raw_output: str = ""


@dataclass
class PredictionResult:
    """LLM 预测输出（含多 Agent 观点 + 主持人裁决）"""
    analysis_text: str = ""
    outlook: str = "中性"
    reason: str = ""
    risk_factors: List[str] = field(default_factory=list)
    positive_factors: List[str] = field(default_factory=list)

    # 各 Agent 独立观点
    tech_view: Optional[Dict] = None
    fund_view: Optional[Dict] = None
    sent_view: Optional[Dict] = None

    # 主持人多周期预测
    short_term: Optional[Dict] = None   # {direction, change_pct, confidence, reason}
    mid_term: Optional[Dict] = None
    long_term: Optional[Dict] = None

    # 兼容旧字段
    price_target_current: Optional[float] = None
    price_target_low: Optional[float] = None
    price_target_high: Optional[float] = None
    confidence: str = "低"

    raw_llm_output: str = ""


class PredictionNode:
    """多 Agent 并行预测节点"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None):
        from config import settings
        self.api_key = api_key or os.environ.get('LLM_API_KEY') or getattr(settings, 'LLM_API_KEY', None) or ''
        self.base_url = base_url or os.environ.get('LLM_BASE_URL') or getattr(settings, 'LLM_BASE_URL', None) or 'https://api.openai.com/v1'
        self.model = model or os.environ.get('LLM_MODEL_NAME') or getattr(settings, 'LLM_MODEL_NAME', None) or 'gpt-4o-mini'

    # ── 主入口 ──

    def predict(self, state: dict) -> PredictionResult:
        if self.api_key:
            return self._multi_agent_predict(state)
        else:
            return self._rule_predict(state)

    # ── 多 Agent 并行 ──

    def _multi_agent_predict(self, state: dict) -> PredictionResult:
        """3 Agent 并行分析 → Moderator 综合裁决"""
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # 并行调用 3 个 Agent
        agents = {
            'tech':         (self._tech_prompt, self._build_tech_data(state)),
            'fundamental':  (self._fund_prompt, self._build_fund_data(state)),
            'sentiment':    (self._sent_prompt, self._build_sent_data(state)),
        }

        views: Dict[str, AgentView] = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._call_agent, client, role, prompt, data): role
                for role, (prompt, data) in agents.items()
            }
            for future in as_completed(futures):
                role = futures[future]
                try:
                    views[role] = future.result(timeout=30)
                except Exception as e:
                    logger.warning(f"Agent [{role}] 失败: {e}")
                    views[role] = AgentView(role=role, outlook="中性", confidence="低",
                                            key_points=[f"Agent 调用失败: {str(e)[:50]}"])

        # 主持人综合裁决
        debate_text = self._format_debate(state, views)
        final = self._call_moderator(client, state, views, debate_text)

        # 组装结果
        q = state.get('quote', {}) or {}
        price = q.get('price', 0) if isinstance(q, dict) else 0

        return PredictionResult(
            analysis_text=final.get('analysis_text', ''),
            outlook=final.get('outlook', views.get('tech', AgentView()).outlook),
            reason=final.get('reason', ''),
            risk_factors=final.get('risk_factors', []),
            positive_factors=final.get('positive_factors', []),
            tech_view=self._view_to_dict(views.get('tech')),
            fund_view=self._view_to_dict(views.get('fundamental')),
            sent_view=self._view_to_dict(views.get('sentiment')),
            short_term=final.get('short_term'),
            mid_term=final.get('mid_term'),
            long_term=final.get('long_term'),
            price_target_current=price,
            price_target_low=final.get('price_target_low'),
            price_target_high=final.get('price_target_high'),
            confidence=final.get('confidence', '低'),
            raw_llm_output=final.get('raw', ''),
        )

    # ── Agent 调用 ──

    def _call_agent(self, client, role: str, prompt: str, data: str) -> AgentView:
        full_prompt = f"{prompt}\n\n## 分析数据\n{data}"
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"你是A股{role}分析专家。请仅基于提供的数据给出独立判断。输出严格JSON。"},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        d = self._parse_json(raw)
        return AgentView(
            role=role,
            outlook=d.get('outlook', '中性'),
            confidence=d.get('confidence', '低'),
            score=float(d.get('score', 0)),
            key_points=d.get('key_points', []),
            risks=d.get('risks', []),
            raw_output=raw,
        )

    def _call_moderator(self, client, state: dict, views: Dict[str, AgentView], debate_text: str) -> dict:
        """主持人阅读三方辩论后给出最终判断"""
        prompt = self._moderator_prompt(state, debate_text)
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是A股投资委员会主席。三位分析师（技术面、基本面、舆情）已给出独立判断。请你审阅三方观点，辩论、裁决，输出最终预测JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            result = self._parse_json(raw)
            result['raw'] = raw
            return result
        except Exception as e:
            logger.warning(f"Moderator 调用失败: {e}")
            # 降级：取多数 Agent 的观点
            outlooks = [v.outlook for v in views.values()]
            majority = max(set(outlooks), key=outlooks.count)
            return {
                'outlook': majority, 'confidence': '低',
                'analysis_text': f"主持人调用失败({e})，取多数Agent观点: {majority}",
                'reason': f"Agent投票: " + ", ".join(f"{r}={o}" for r, o in zip(views.keys(), outlooks)),
            }

    # ── 提示词 ──

    @property
    def _tech_prompt(self) -> str:
        return """你是技术分析专家。仅根据K线指标给出独立判断，不要考虑基本面或消息面。

输出JSON:
{
  "outlook": "看多/看空/中性",
  "confidence": "高/中/低",
  "score": -5,
  "key_points": ["MACD死叉且柱状线扩大，短期动能偏空", "RSI=19进入超卖区，技术性反弹概率增加"],
  "risks": ["均线空头排列，趋势尚未扭转", "若跌破布林下轨可能加速下跌"]
}

评分规则:
- 每项看多信号+1分，强看多+2分；看空-1分，强看空-2分
- 最终score为各项加总，范围约-10~10
- outlook: score>2→看多, score<-2→看空, 否则中性
- 仅输出JSON，不要额外文字"""

    @property
    def _fund_prompt(self) -> str:
        return """你是基本面/估值分析专家。仅根据财务数据和PE分位给出独立判断，不要考虑技术面或消息面。

输出JSON:
{
  "outlook": "看多/看空/中性",
  "confidence": "高/中/低",
  "score": 3,
  "key_points": ["PE处于历史1%分位，远低于均值，估值极低", "ROE=10.6%盈利能力稳健"],
  "risks": ["行业景气度下行可能压制估值修复", "营收增速放缓需关注"]
}

评分规则:
- PE分位<10%: +3分; <30%: +1分; >70%: -1分; >90%: -3分
- ROE>15%: +2分; ROE>10%: +1分; ROE<5%: -1分
- EPS同比正增长: +1分; 负增长: -1分
- outlook: score>2→看多, score<-2→看空, 否则中性
- 仅输出JSON，不要额外文字"""

    @property
    def _sent_prompt(self) -> str:
        return """你是舆情分析专家。仅根据新闻和股吧数据给出独立判断，不要考虑技术面或基本面。

输出JSON:
{
  "outlook": "看多/看空/中性",
  "confidence": "高/中/低",
  "score": 2,
  "key_points": ["茅台30亿回购完成，注销股份提振信心", "白酒板块集体下挫，市场情绪偏谨慎"],
  "risks": ["主力资金净流出", "股吧看空帖子多于看多"]
}

评分规则:
- 情感avg>0.3: +2分; >0.1: +1分; <-0.1: -1分; <-0.3: -2分
- 正面占比>50%: +1分; 负面占比>50%: -1分
- outlook: score>2→看多, score<-2→看空, 否则中性
- 仅输出JSON，不要额外文字"""

    def _moderator_prompt(self, state: dict, debate_text: str) -> str:
        q = state.get('quote', {}) or {}
        price = q.get('price', 0) if isinstance(q, dict) else 0
        return f"""## 股票信息
{state.get('stock_name', '')}({state.get('symbol', '')})  现价: {price}

## 三位分析师独立观点

{debate_text}

## 你的任务

作为投资委员会主席，请审阅上述三方观点后:

1. **指出共识** — 三位分析师在哪些判断上一致？
2. **指出分歧** — 哪些判断相互矛盾？你更认同一方的理由是什么？
3. **综合裁决** — 给出最终的多空判断和置信度
4. **多周期预测** — 综合技术面(短期)、估值(中长期)、舆情(情绪面)给出短/中/长期涨跌预测

输出JSON:
{{
  "analysis_text": "综合三位分析师观点的完整分析(200字内)",
  "outlook": "看多/看空/中性",
  "confidence": "高/中/低",
  "reason": "核心裁决逻辑(80字内)",
  "short_term": {{
    "direction": "上涨/下跌/震荡",
    "change_pct": 3.5,
    "confidence": "高/中/低",
    "reason": "1~2周预测依据(40字内)"
  }},
  "mid_term": {{
    "direction": "上涨/下跌/震荡",
    "change_pct": 8.0,
    "confidence": "高/中/低",
    "reason": "1~3月预测依据(40字内)"
  }},
  "long_term": {{
    "direction": "上涨/下跌/震荡",
    "change_pct": 15.0,
    "confidence": "高/中/低",
    "reason": "6~12月预测依据(40字内)"
  }},
  "price_target_low": {price * 0.93:.1f},
  "price_target_high": {price * 1.10:.1f},
  "risk_factors": ["风险1", "风险2"],
  "positive_factors": ["积极因素1", "积极因素2"]
}}

注意:
- short_term.change_pct: 预计1-2周内的涨跌幅度，正数上涨负数下跌
- mid_term.change_pct: 预计1-3月内的涨跌幅度，侧重估值回归
- long_term.change_pct: 预计6-12月内的涨跌幅度，侧重基本面和行业趋势"""

    # ── 数据构造 (每个 Agent 只看自己的领域) ──

    def _build_tech_data(self, state: dict) -> str:
        ti = state.get('technical_indicators', {}) or {}
        q = state.get('quote', {}) or {}
        price = q.get('price', 0) if isinstance(q, dict) else 0
        lines = [
            f"现价: {price}  涨跌: {q.get('change_pct','?')}%",
            f"RSI(14): {ti.get('rsi_14','?')}  MACD柱: {ti.get('macd_hist','?')}",
            f"KDJ: {ti.get('kdj_k','?')}/{ti.get('kdj_d','?')}/{ti.get('kdj_j','?')}",
            f"MA5:{ti.get('ma5','?')} MA10:{ti.get('ma10','?')} MA20:{ti.get('ma20','?')} MA60:{ti.get('ma60','?')}",
            f"布林: {ti.get('boll_lower','?')} ~ {ti.get('boll_middle','?')} ~ {ti.get('boll_upper','?')}",
            f"量比: {ti.get('volume_ratio','?')}  振幅: {ti.get('amplitude','?')}%",
        ]
        return "\n".join(lines)

    def _build_fund_data(self, state: dict) -> str:
        fs = state.get('financial_summary', {}) or {}
        q = state.get('quote', {}) or {}
        lines = [
            f"PE: {q.get('pe','?')}  PB: {q.get('pb','?')}  市值: {q.get('market_cap','?')}亿",
            f"估值等级: {state.get('valuation_level','?')} (PE分位: {state.get('valuation_percentile','?')}%)",
            f"历史PE均值: {state.get('historical_pe_avg','?')}  建议买入价: {state.get('suggested_buy_price','?')}",
            f"EPS: {fs.get('eps','?')}  ROE: {fs.get('roe','?')}%",
            f"营收: {fs.get('revenue','?')}亿  净利: {fs.get('net_profit','?')}亿",
            f"毛利率: {fs.get('gross_margin','?')}%  负债率: {fs.get('debt_ratio','?')}%",
        ]
        return "\n".join(lines)

    def _build_sent_data(self, state: dict) -> str:
        sn = state.get('sentiment_news', {}) or {}
        sg = state.get('sentiment_guba', {}) or {}
        bull_news = state.get('important_bullish_news', []) or []
        bear_news = state.get('important_bearish_news', []) or []
        bull_guba = state.get('important_bullish_guba', []) or []
        bear_guba = state.get('important_bearish_guba', []) or []

        lines = [
            f"新闻: {sn.get('total_count',0)}条  正面{sn.get('positive_count',0)}  负面{sn.get('negative_count',0)}  avg={sn.get('avg_score',0):.2f}",
            f"股吧: {sg.get('total_count',0)}条  正面{sg.get('positive_count',0)}  负面{sg.get('negative_count',0)}  avg={sg.get('avg_score',0):.2f}",
            "",
            "利好:",
        ]
        for n in bull_news[:3]:
            lines.append(f"  + {n['title']}")
        for g in bull_guba[:2]:
            lines.append(f"  + [股吧] {g['title']}")
        lines.append("")
        lines.append("利空:")
        for n in bear_news[:3]:
            lines.append(f"  - {n['title']}")
        for g in bear_guba[:2]:
            lines.append(f"  - [股吧] {g['title']}")
        return "\n".join(lines)

    def _format_debate(self, state: dict, views: Dict[str, AgentView]) -> str:
        """格式化三方观点供主持人阅读"""
        names = {'tech': '技术面分析师', 'fundamental': '基本面分析师', 'sentiment': '舆情分析师'}
        parts = []
        for role, v in views.items():
            name = names.get(role, role)
            parts.append(
                f"### {name}\n"
                f"判断: {v.outlook}  置信度: {v.confidence}  评分: {v.score}\n"
                f"看多理由:\n" + "\n".join(f"  - {p}" for p in v.key_points) + "\n"
                f"风险点:\n" + "\n".join(f"  - {r}" for r in v.risks)
            )
        return "\n\n".join(parts)

    # ── 规则降级 ──

    def _rule_predict(self, state: dict) -> PredictionResult:
        signals = state.get('signals', {}) or {}
        score = signals.get('score', 0)
        q = state.get('quote', {}) or {}
        price = q.get('price', 0) if isinstance(q, dict) else 0

        if score > 2:
            outlook = "看多"
        elif score < -2:
            outlook = "看空"
        else:
            outlook = "中性"

        conf = "高" if abs(score) > 4 else "中" if abs(score) > 2 else "低"

        return PredictionResult(
            outlook=outlook,
            confidence=conf,
            analysis_text=f"## {outlook}信号 (规则模式)\n\n综合评分 {score:.1f}",
            reason=f"规则引擎: 综合评分{score:.1f}",
            price_target_current=price,
            price_target_low=round(price * 0.95, 2) if price else None,
            price_target_high=round(price * 1.10, 2) if price else None,
            risk_factors=[],
            positive_factors=[],
        )

    # ── 工具 ──

    @staticmethod
    def _view_to_dict(v: Optional[AgentView]) -> Optional[dict]:
        if v is None:
            return None
        return {
            'role': v.role, 'outlook': v.outlook, 'confidence': v.confidence,
            'score': v.score, 'key_points': v.key_points, 'risks': v.risks,
        }

    @staticmethod
    def _parse_json(raw: str) -> dict:
        try:
            raw = raw.strip()
            if raw.startswith('```json'):
                raw = raw.split('```json')[1].split('```')[0]
            elif raw.startswith('```'):
                raw = raw.split('```')[1].split('```')[0]
            return json.loads(raw)
        except (json.JSONDecodeError, KeyError):
            logger.warning(f"JSON 解析失败: {raw[:100]}")
            return {}
