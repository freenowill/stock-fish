"""
E8 独立监察员 — 监察部

站在所有人观点反面提问，寻找被忽略的盲点。
不看多也不看空，任务是挑战每个主流假设。
"""
from .base import BaseAgent, EmployeeReport


class Overseer(BaseAgent):
    """独立监察员 / 魔鬼代言人: 挑战所有假设，找盲点"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.employee_id = "overseer"
        self.role = "独立监察员"
        self.department = "监察部"

    def analyze(self, state: dict, other_reports: list = None) -> EmployeeReport:
        """
        基于其他员工的报告 + 原始数据，找出被忽视的盲点和替代假设。

        Args:
            state: 分析状态 dict
            other_reports: 其他员工的 EmployeeReport 列表
        """
        data = self.build_overseer_context(state, other_reports or [])
        return self._llm_analyze(data) if self.has_llm else self._rule_analyze(state, other_reports or [])

    def _llm_analyze(self, data: str) -> EmployeeReport:
        prompt = """你是A股"独立监察员"——你的任务是挑战投资团队的所有主流假设，找出被忽略的盲点。

**你不是来看多或看空的**。无论团队一致看多还是看空，你的工作都是站在反方向提问。

你的具体职责:
1. 【挑战共识】如果大多数分析师都看多，你要问: 他们可能集体忽略了什么？有没有一波人都掉进的认知陷阱？
2. 【替代情景】提出至少2个被团队忽略的"替代情景"——事情可能朝完全不同的方向发展
3. 【证据质疑】质疑关键证据的质量: 这个数据是否可靠？结论是否过度外推？是否有幸存者偏差？
4. 【黑天鹅提问】有什么极低概率但极高影响的事件可能彻底改变结论？
5. 【认知偏差检查】团队是否陷入了以下偏差: 确认偏误、近因偏误、羊群效应、过度自信？
6. 【数据完整性检查】是否有关键数据缺失（如现金流、分红数据、行业对比等）而未标注？分析师是否基于不完整的数据做出了过度确定的结论？
7. 【大股东/管理层检查】是否讨论了大股东增减持动态？是否有股权质押风险或管理层变动？
8. 【利润质量检查】净利润增长是否与非经常性损益相关？营收和毛利率趋势是否支持净利润变动？

输出JSON:
{
  "outlook": "中性",
  "confidence": "高/中/低",
  "score": 0,
  "key_points": [
    "盲点1: 团队过度关注PE分位低，但可能忽略了盈利下修风险（价值陷阱）",
    "盲点2: 所有人都假设行业政策不变，但实际政策风险正在积累",
    "替代情景: 若消费降级加速，即使PE处于低位，价格仍可能继续下跌30%+"
  ],
  "risks": [
    "认知偏差: 团队可能陷入'锚定效应'——过度依赖历史PE均值",
    "数据质量: 散户情感数据样本量太小，可能不具代表性",
    "遗漏因素: 团队未讨论大股东减持压力"
  ]
}

评分规则:
- 监察员不给出多空方向判断。outlook 始终为"中性"
- score 始终为 0（你不是分析师，不参与多空投票）
- 你的价值在于 blind_spots 和 alternative_scenarios，不在方向判断"""

        result = self._call_llm(
            f"你是{self.role}。你的任务是挑战所有分析师的假设，找出盲点和替代情景。输出严格JSON。",
            f"{prompt}\n\n{data}",
            temperature=0.5)  # 稍高温度以鼓励创造性思维
        return self._build_report(result)

    def _rule_analyze(self, state: dict, other_reports: list) -> EmployeeReport:
        """规则模式下无法真正做监察，输出通用警示"""
        points, risks = [], []

        # 统计其他员工的方向
        bullish = sum(1 for r in other_reports if not r.error and r.outlook in ('看多', '偏多'))
        bearish = sum(1 for r in other_reports if not r.error and r.outlook in ('看空', '偏空'))
        neutral = len([r for r in other_reports if not r.error]) - bullish - bearish

        if bullish > bearish + 2:
            points.append("⚠️ 团队一致性过强: 多数分析师看多，谨防集体误判")
            risks.append(f"羊群效应: {bullish}人看多 vs {bearish}人看空，请CIO确认不盲目从众")
        elif bearish > bullish + 2:
            points.append("⚠️ 团队一致性过强: 多数分析师看空，是否过度悲观？")
            risks.append(f"恐慌蔓延: {bearish}人看空 vs {bullish}人看多，是否存在被忽视的利好？")

        points.append("规则模式: 无法深度挑战假设。建议启用LLM以获取完整监察报告。")
        risks.extend([
            "确认偏误: 分析师可能选择性使用支持自己结论的数据",
            "锚定效应: 过度依赖历史平均值做判断",
            "遗漏风险: 地缘政治/监管政策变化未纳入分析",
            "数据质量: 检查经营现金流/自由现金流是否缺失——若缺失则无法验证净利润质量",
            "利润质量: 检查净利润增长是否可能来自非经常性损益（营收与毛利率趋势不支持时）",
        ])

        return EmployeeReport(employee_id=self.employee_id, role=self.role,
                              department=self.department, outlook="中性", confidence="中",
                              score=0, key_points=points, risks=risks)

    def _build_report(self, result: dict) -> EmployeeReport:
        return EmployeeReport(employee_id=self.employee_id, role=self.role,
                              department=self.department,
                              outlook="中性",  # 监察员不参与多空投票
                              confidence=result.get('confidence', '中'),
                              score=0.0,  # 监察员不打分
                              key_points=result.get('key_points', []),
                              risks=result.get('risks', []),
                              raw_output=str(result))
