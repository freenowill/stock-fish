"""
E3 估值分析师 — 研究部

专注 PE/PB 分位、DCF 估值、同行对比、安全边际计算。
从原有基本面分析师中独立出来，专注"贵不贵"的问题。
"""
from .base import BaseAgent, EmployeeReport


class ValuationAgent(BaseAgent):
    """估值分析师: DCF、PE/PB 分位、安全边际"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.employee_id = "valuation"
        self.role = "估值分析师"
        self.department = "研究部"

    def analyze(self, state: dict) -> EmployeeReport:
        data = self.build_valuation_context(state)
        return self._llm_analyze(data) if self.has_llm else self._rule_analyze(state)

    def _llm_analyze(self, data: str) -> EmployeeReport:
        prompt = """你是A股估值分析专家。根据估值数据给出独立判断。

输出JSON:
{
  "outlook": "看多/看空/中性",
  "confidence": "高/中/低",
  "score": -5,
  "key_points": ["PE处于历史1%分位，远低于均值，估值极低", "长期PE分位同样偏低，估值有安全边际"],
  "risks": ["低估值可能反映基本面恶化，需警惕价值陷阱"]
}

评分规则:
- PE分位<10%: +3分; <30%: +2分; <50%: +1分; >70%: -1分; >90%: -3分
- 5年PE分位<10%: +2分(长期低估信号); 10年PE分位<10%: +3分(极端低估)
- 股票收益率(E/P) > 国债收益率+3%: +2分(高权益溢价); <国债收益率: -2分(股票偏贵)
- PB<1(破净): +2分; PB<1.5: +1分
- ROE>15%且PE<行业均值: +2分 (低估值+高质量=双击机会)
- PE分位>90%且ROE<5%: -2分 (高估值+低质量=风险)
- 最终score范围为-10到10
- outlook: score>2→看多, score<-2→看空, 否则中性"""

        result = self._call_llm(
            f"你是{self.role}。请仅基于提供的数据给出独立判断。输出严格JSON。",
            f"{prompt}\n\n## 估值数据\n{data}",
            temperature=0.3)
        return self._build_report(result)

    def _rule_analyze(self, state: dict) -> EmployeeReport:
        pe_pct = state.get('valuation_percentile', 50) or 50
        q = state.get('quote', {}) or {}
        pb = q.get('pb', 999)

        score = 0.0
        points, risks = [], []

        if pe_pct < 10:
            score += 3; points.append(f"PE处于历史{pe_pct:.0f}%分位，估值极低")
        elif pe_pct < 30:
            score += 2; points.append(f"PE处于历史{pe_pct:.0f}%分位，估值偏低")
        elif pe_pct < 50:
            score += 1; points.append(f"PE处于历史{pe_pct:.0f}%分位，估值合理偏低")
        elif pe_pct < 70:
            points.append(f"PE处于历史{pe_pct:.0f}%分位，估值正常")
        elif pe_pct < 90:
            score -= 1; points.append(f"PE处于历史{pe_pct:.0f}%分位，估值偏高"); risks.append("高估值回调风险")
        else:
            score -= 3; points.append(f"PE处于历史{pe_pct:.0f}%分位，估值极高"); risks.append("估值泡沫风险")

        # 长期PE分位 (5年)
        pe_5y = state.get('valuation_percentile_5y')
        if pe_5y is not None:
            if pe_5y < 10:
                score += 2; points.append(f"5年PE分位={pe_5y:.0f}%，长期低估")
            elif pe_5y > 80:
                score -= 1; risks.append(f"5年PE分位={pe_5y:.0f}%，长期偏高")

        # 股票收益率 vs 国债收益率 (巴菲特指标)
        eq = state.get('equity_risk_premium')
        if eq is not None:
            yield_val = state.get('earnings_yield')
            bond = state.get('bond_yield_10y')
            if eq > 3:
                score += 2; points.append(f"E/P={yield_val:.1f}% > 国债{bond:.1f}%+3%，权益溢价显著")
            elif eq < 0:
                score -= 2; risks.append(f"E/P={yield_val:.1f}% < 国债{bond:.1f}%，股票相对债券偏贵")

        if pb < 1:
            score += 2; points.append("PB<1，低于净资产(破净)")
        elif pb < 1.5:
            score += 1; points.append(f"PB={pb:.1f}，估值相对合理")

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
