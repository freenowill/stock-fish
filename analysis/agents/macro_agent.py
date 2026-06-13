"""
E1 宏观分析师 — 宏观部

分析宏观经济周期、利率/M2/CPI/PMI 趋势、北向资金方向。
为 CIO 回答："当前大的宏观环境是否支持权益投资？"
"""
from .base import BaseAgent, EmployeeReport


class MacroAgent(BaseAgent):
    """宏观分析师: 经济周期、货币政策、资本流动"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.employee_id = "macro"
        self.role = "宏观分析师"
        self.department = "宏观部"

    def analyze(self, state: dict) -> EmployeeReport:
        macro = state.get('macro_context') or {}
        data = self._format_macro_data(state, macro)
        return self._llm_analyze(data) if self.has_llm else self._rule_analyze(state, macro)

    def _format_macro_data(self, state: dict, macro: dict) -> str:
        q = state.get('quote', {}) or {}
        lines = [
            "# 宏观背景数据",
            "",
            "## 利率与货币政策",
            f"SHIBOR 隔夜: {macro.get('shibor', 'N/A')}%",
            f"1年期 LPR: {macro.get('lpr_1y', 'N/A')}%",
            f"5年期 LPR: {macro.get('lpr_5y', 'N/A')}%",
            f"政策倾向: {macro.get('policy_tilt', 'N/A')}",
            "",
            "## 经济指标",
            f"制造业 PMI: {macro.get('pmi', 'N/A')}",
            f"CPI 同比: {macro.get('cpi_yoy', 'N/A')}%",
            f"社融增速: {macro.get('social_financing', 'N/A')}%",
            "",
            "## 资本流动",
            f"北向资金当日净流入: {macro.get('northbound_flow', 'N/A')}亿元",
            f"北向资金 5 日均值: {macro.get('northbound_5d_avg', 'N/A')}亿元",
            f"美元/人民币: {macro.get('usd_cny', 'N/A')}",
            "",
            "## 市场整体",
            f"大盘状态: {macro.get('market_regime', 'N/A')}",
            f"标的现价: {q.get('price', '?')}",
        ]

        # 国内外宏观经济事件日历
        macro_events = macro.get('macro_events', [])
        if macro_events:
            lines.append("")
            lines.append("## 近期国内外重要经济事件")
            for ev in macro_events[:8]:
                stars = '★' * ev.get('importance', 1)
                lines.append(f"- [{ev.get('region', '')}] {ev.get('date', '')}: {ev.get('event', '')} {stars}")

        # 政策新闻(CCTV头条)
        headlines = macro.get('policy_headlines', [])
        if headlines:
            lines.append("")
            lines.append("## 政策与宏观新闻头条")
            for h in headlines[:5]:
                lines.append(f"- {h}")

        return "\n".join(lines)

    def _llm_analyze(self, data: str) -> EmployeeReport:
        prompt = """你是宏观经济学家。根据宏观数据给出对A股权益投资的整体判断。

输出JSON:
{
  "outlook": "看多/看空/中性",
  "confidence": "高/中/低",
  "score": 2,
  "key_points": ["货币政策宽松，LPR下调利好股市", "PMI连续3月高于50，经济复苏确认"],
  "risks": ["北向资金持续流出，外资信心不足", "CPI上升可能制约宽松空间"]
}

评分规则:
- 货币政策宽松(LPR下调/降准): +2分; 收紧: -2分
- PMI>50且上升: +2分; PMI<50且下降: -2分
- 北向资金净流入>20亿/日: +1分; 净流出>20亿/日: -1分
- 人民币升值: +1分; 贬值>5%: -1分
- 大盘处于上升趋势: +1分; 下降趋势: -1分
- 最终score范围为-10到10
- outlook: score>2→看多, score<-2→看空, 否则中性"""

        result = self._call_llm(
            f"你是{self.role}。请仅基于提供的宏观数据给出独立判断。输出严格JSON。",
            f"{prompt}\n\n{data}", temperature=0.3)
        return self._build_report(result)

    def _rule_analyze(self, state: dict, macro: dict) -> EmployeeReport:
        score = 0.0
        points, risks = [], []
        available_fields = 0
        total_fields = 6  # pmi, northbound, regime, policy, cpi, shibor

        pmi = macro.get('pmi')
        northbound = macro.get('northbound_flow')
        regime = macro.get('market_regime', '')
        policy = macro.get('policy_tilt', '')
        cpi = macro.get('cpi_yoy')
        shibor = macro.get('shibor')

        if pmi is not None and pmi != 'N/A':
            available_fields += 1
            pmi = float(pmi)
            if pmi > 50:
                score += 2; points.append(f"PMI={pmi}，制造业处于扩张区间")
            elif pmi > 48:
                points.append(f"PMI={pmi}，接近荣枯线")
            else:
                score -= 2; risks.append(f"PMI={pmi}，制造业收缩")

        if northbound is not None and northbound != 'N/A':
            available_fields += 1
            nb = float(northbound)
            if nb > 20:
                score += 1; points.append(f"北向资金大幅流入({nb}亿)，外资看多")
            elif nb > 0:
                points.append(f"北向资金小幅流入({nb}亿)")
            elif nb < -20:
                score -= 1; risks.append(f"北向资金大幅流出({nb}亿)")
            elif nb < 0:
                score -= 0.5

        if policy and '宽松' in str(policy):
            available_fields += 1
            score += 2; points.append("货币政策偏宽松，利好权益资产")
        elif policy and '收紧' in str(policy):
            available_fields += 1
            score -= 2; risks.append("货币政策收紧，不利于权益资产")
        elif policy:
            available_fields += 1

        if regime and ('上升' in str(regime) or 'bull' in str(regime).lower()):
            available_fields += 1
            score += 1; points.append("大盘处于上升趋势")
        elif regime and ('下降' in str(regime) or 'bear' in str(regime).lower()):
            available_fields += 1
            score -= 1; risks.append("大盘处于下降趋势")
        elif regime:
            available_fields += 1  # 震荡也算有数据

        if cpi is not None and cpi != 'N/A':
            available_fields += 1
        if shibor is not None and shibor != 'N/A':
            available_fields += 1

        # 数据完整性标注
        completeness = available_fields / total_fields if total_fields > 0 else 0
        if completeness < 0.5:
            risks.append(f"宏观数据完整度{completeness:.0%}（{available_fields}/{total_fields}），部分指标不可用，本报告置信度应下调")
        elif completeness < 0.8:
            points.append(f"宏观数据部分缺失（{available_fields}/{total_fields}），结论需谨慎对待")
        else:
            points.append(f"宏观数据较为完整（{available_fields}/{total_fields}）")

        if not points and not risks:
            points.append("宏观数据不完整，无法做出全面判断")
            risks.append("数据缺失风险：部分宏观指标不可用")

        outlook = "看多" if score >= 2 else "看空" if score <= -2 else "中性"
        conf = "高" if abs(score) > 4 else "中" if abs(score) > 2 else "低"
        # 数据不完整时下调置信度
        if completeness < 0.5:
            conf = "低"

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
