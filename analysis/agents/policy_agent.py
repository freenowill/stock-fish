"""
E2 行业政策分析师 — 宏观部

分析产业政策导向、监管变化、竞争格局、行业景气度。
为 CIO 回答："该行业政策风向是顺是逆？行业处于什么周期阶段？"
"""
from .base import BaseAgent, EmployeeReport


class PolicyAgent(BaseAgent):
    """行业政策分析师: 产业政策、监管、行业周期"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.employee_id = "policy"
        self.role = "行业政策分析师"
        self.department = "宏观部"

    def analyze(self, state: dict) -> EmployeeReport:
        industry = state.get('industry_context') or {}
        data = self._format_policy_data(state, industry)
        return self._llm_analyze(data) if self.has_llm else self._rule_analyze(state, industry)

    def _format_policy_data(self, state: dict, industry: dict) -> str:
        q = state.get('quote', {}) or {}
        fs = state.get('financial_summary', {}) or {}

        # 追加 build_industry_context() 的结构化行业数据（cross-reference）
        ind_ctx = self.build_industry_context(state)

        flow_direction = '流入' if industry.get('industry_fund_flow_bullish') else '流出'
        lines = [
            "# 行业与政策背景数据",
            "",
            "## 行业概况",
            f"板块概况: {industry.get('industry_names', 'N/A')}",
            f"板块涨跌比: {industry.get('industry_up_ratio', 'N/A')}% 上涨",
            f"板块平均涨跌幅: {industry.get('industry_avg_change', 'N/A')}%",
            f"板块资金流向: {industry.get('industry_fund_flow', 'N/A')}亿元 ({flow_direction})",
            f"行业 20 日动量: {industry.get('industry_momentum', 'N/A')}%",
            "",
            "## 政策环境",
            f"政策影响评估: {industry.get('policy_impact', 'N/A')}",
            f"近期政策事件: {industry.get('policy_events', '无')}",
            "",
            "## 标的信息",
            f"股票: {state.get('stock_name', '?')} ({state.get('symbol', '?')})",
            f"现价: {q.get('price', '?')}元  PE: {q.get('pe', '?')}",
            f"ROE: {fs.get('roe', '?')}%  营收同比: {fs.get('revenue_yoy', '?')}%",
            "",
            "## 原始行业数据 (供交叉验证)",
            ind_ctx,
        ]
        return "\n".join(lines)

    def _llm_analyze(self, data: str) -> EmployeeReport:
        prompt = """你是行业政策研究专家。根据行业和政策数据给出独立判断。

输出JSON:
{
  "outlook": "看多/看空/中性",
  "confidence": "高/中/低",
  "score": 3,
  "key_points": ["新能源行业政策利好持续释放，补贴加码", "行业竞争格局改善，龙头集中度提升"],
  "risks": ["技术路线存在不确定性", "国际贸易摩擦可能影响出口"]
}

评分规则:
- 政策明确利好(补贴/税收优惠/放松管制): +3分; 政策收紧(监管加压/取消补贴): -3分
- 行业景气上行(需求增长>15%): +2分; 下行: -2分
- 行业PE分位<30%(低估): +1分; >70%(高估): -1分
- 行业集中度提升(龙头份额扩大): +1分
- 行业动量>10%(近期强势): +1分; <-10%: -1分
- 最终score范围为-10到10
- outlook: score>2→看多, score<-2→看空, 否则中性"""

        result = self._call_llm(
            f"你是{self.role}。请仅基于提供的行业和政策数据给出独立判断。输出严格JSON。",
            f"{prompt}\n\n{data}", temperature=0.3)
        return self._build_report(result)

    def _rule_analyze(self, state: dict, industry: dict) -> EmployeeReport:
        score = 0.0
        points, risks = [], []

        impact = industry.get('policy_impact', '')
        momentum = industry.get('industry_momentum')
        fund_flow = industry.get('industry_fund_flow')
        up_ratio = industry.get('industry_up_ratio')

        if impact and '利好' in str(impact):
            score += 3; points.append(f"政策面利好: {impact}")
        elif impact and '利空' in str(impact):
            score -= 3; risks.append(f"政策面利空: {impact}")
        elif impact and '中性' in str(impact):
            points.append("政策面中性，无明确方向")
        else:
            points.append("政策数据不完整，需进一步关注")

        if momentum is not None and momentum != 'N/A':
            m = float(momentum)
            if m > 10:
                score += 1; points.append(f"行业动量+{m}%，近期强势")
            elif m < -10:
                score -= 1; risks.append(f"行业动量{m}%，近期弱势")

        if fund_flow is not None and fund_flow != 'N/A':
            f = float(fund_flow)
            if f > 0:
                points.append(f"板块资金净流入{f}亿元")
            else:
                risks.append(f"板块资金净流出{abs(f)}亿元")

        if up_ratio is not None and up_ratio != 'N/A':
            u = float(up_ratio)
            if u > 60:
                points.append(f"板块普涨 ({u:.0f}%上涨)")
            elif u < 40:
                risks.append(f"板块普跌 ({u:.0f}%上涨)")

        if not points and not risks:
            points.append("行业政策数据不完整，无法做深度判断")

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
