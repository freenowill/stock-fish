"""
E6 舆情分析师 — 交易部

分析新闻/股吧 NLP 情感 + 资金流向 + 情绪极端程度判断。
"""
from .base import BaseAgent, EmployeeReport


class SentimentAgent(BaseAgent):
    """舆情分析师: 新闻情感、股吧情绪、资金流向"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.employee_id = "sentiment"
        self.role = "舆情分析师"
        self.department = "交易部"

    def analyze(self, state: dict) -> EmployeeReport:
        data = self.build_sent_context(state)
        return self._llm_analyze(data) if self.has_llm else self._rule_analyze(state)

    def _llm_analyze(self, data: str) -> EmployeeReport:
        prompt = """你是A股舆情分析专家。仅根据新闻和股吧数据给出独立判断。

输出JSON:
{
  "outlook": "看多/看空/中性",
  "confidence": "高/中/低",
  "score": 2,
  "key_points": ["回购消息正面，利好股价", "市场恐慌情绪处于极值，或为反向指标"],
  "risks": ["主力资金持续流出", "股吧看空比例过高，注意踩踏风险"]
}

评分规则:
- 新闻平均情感>0.3: +2分; >0.1: +1分; <-0.1: -1分; <-0.3: -2分
- 股吧平均情感>0.3: +2分; >0.1: +1分; <-0.1: -1分; <-0.3: -2分
- 正面占比>60%: +1分; 负面占比>60%: -1分
- 新闻+股吧情感同向极端时: 额外±1分(一致性加成/惩罚)
- 若利好数量>3且有实质性利好: +1分
- 最终score范围为-10到10
- outlook: score>2→看多, score<-2→看空, 否则中性"""

        result = self._call_llm(
            f"你是{self.role}。请仅基于提供的数据给出独立判断。输出严格JSON。",
            f"{prompt}\n\n## 舆情数据\n{data}",
            temperature=0.3)
        return self._build_report(result)

    def _rule_analyze(self, state: dict) -> EmployeeReport:
        sn = state.get('sentiment_news', {}) or {}
        sg = state.get('sentiment_guba', {}) or {}

        news_avg = sn.get('avg_score', 0) or 0
        guba_avg = sg.get('avg_score', 0) or 0
        news_pos_ratio = sn.get('positive_ratio', 0.5) or 0.5
        guba_pos_ratio = sg.get('positive_ratio', 0.5) or 0.5

        score = 0.0
        points, risks = [], []

        if news_avg > 0.3:
            score += 2; points.append(f"新闻情感偏正面(avg={news_avg:.2f})")
        elif news_avg > 0.1:
            score += 1
        elif news_avg < -0.3:
            score -= 2; risks.append(f"新闻情感极负面(avg={news_avg:.2f})")
        elif news_avg < -0.1:
            score -= 1

        if guba_avg > 0.3:
            score += 2; points.append(f"股吧情绪偏正面(avg={guba_avg:.2f})")
        elif guba_avg < -0.3:
            score -= 2; risks.append(f"股吧情绪极悲观(avg={guba_avg:.2f})")
        elif guba_avg < -0.1:
            score -= 1

        # 极端情绪 → 反向指标
        if guba_avg < -0.5 and news_avg < -0.3:
            score += 1  # 极度悲观可能是买入时机
            points.append("⚠️ 市场情绪处于极端悲观区域，可能是反向指标(邓普顿信号)")
        elif guba_avg > 0.5 and news_avg > 0.3:
            score -= 1  # 极度乐观可能是卖出信号
            risks.append("⚠️ 市场情绪处于极端乐观区域，谨防逆转")

        # 情感百分位 — 历史对比信号（邓普顿核心）
        sp = state.get('sentiment_percentile')
        if sp is not None:
            if sp < 5:
                score += 2  # 极端悲观是强烈反向买入信号
                points.append(f"🔥 情感处于历史最低{sp:.0f}%分位 — 邓普顿式'最大悲观点'信号")
            elif sp < 15:
                score += 1  # 偏悲观是温和反向信号
                points.append(f"情感处于历史第{sp:.0f}%分位 — 偏悲观，关注反向机会")
            elif sp > 95:
                score -= 2  # 极端乐观
                risks.append(f"情感处于历史最高{sp:.0f}%分位 — 过度乐观，警惕反转")
            elif sp > 85:
                score -= 1

        # 关注热度 — 低关注可能意味着被遗忘
        ap = state.get('attention_news_percentile')
        if ap is not None and ap < 10:
            points.append(f"📰 新闻关注度处于历史最低{ap:.0f}%分位 — 该股被市场遗忘(邓普顿青睐)")

        if abs(news_avg - guba_avg) > 0.4:
            points.append("新闻与股吧情绪严重背离，市场信息不对称")

        outlook = "看多" if score >= 2 else "看空" if score <= -2 else "中性"
        conf = "高" if abs(score) > 4 else "中" if abs(score) > 2 else "低"

        return EmployeeReport(employee_id=self.employee_id, role=self.role,
                              department=self.department, outlook=outlook, confidence=conf,
                              score=score, key_points=points, risks=risks)

    def _build_report(self, result: dict) -> EmployeeReport:
        score = float(result.get('score', 0))
        return EmployeeReport(employee_id=self.employee_id, role=self.role,
                              department=self.department,
                              outlook=result.get('outlook', '中性'),
                              confidence=result.get('confidence', '低'),
                              score=score,
                              key_points=result.get('key_points', []),
                              risks=result.get('risks', []),
                              raw_output=str(result))
