"""
E7 风险经理 — 风控部

评估下行风险: VaR、最大回撤、黑天鹅识别、流动性风险。
具有软否决权: 当风险评分超过阈值时，标记 veto=true 要求 CIO 回应。
"""
from .base import BaseAgent, EmployeeReport


class RiskManager(BaseAgent):
    """风险经理: 下行风险评估、软否决权"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.employee_id = "risk"
        self.role = "风险经理"
        self.department = "风控部"
        self.veto_threshold = 7.0  # 风险评分 >= 7 时触发软否决

    def analyze(self, state: dict) -> EmployeeReport:
        data = self.build_risk_context(state)
        if self.has_llm:
            return self._llm_analyze(data, state)
        else:
            return self._rule_analyze(state)

    def _llm_analyze(self, data: str, state: dict) -> EmployeeReport:
        q = state.get('quote', {}) or {}
        price = q.get('price', 0) if isinstance(q, dict) else 0
        ti = state.get('technical_indicators', {}) or {}
        vol_ratio = ti.get('volume_ratio', 1) or 1
        amplitude = ti.get('amplitude', 3) or 3

        prompt = f"""你是A股风险管理专家。评估该持仓的下行风险并决定是否行使否决权。

## 市场背景
- 现价: {price}元  量比: {vol_ratio}  振幅: {amplitude}%
{data}

## 软否决权规则
- 风险评分 1-10 (10=最高风险)
- 风险评分 >= 7 时，你必须设置 veto=true，要求CIO书面回应
- veto_reason 必须清晰说明: 为什么风险大到需要否决? 什么条件下可以解除否决?

输出JSON:
{{
  "outlook": "看多/看空/中性",
  "confidence": "高/中/低",
  "score": 3,
  "risk_score": 4,
  "key_points": ["短期波动率偏高但未超过历史阈值", "基本面无恶化迹象"],
  "risks": ["流动性风险: 量比偏低", "下行空间: 约-8%"],
  "veto": false,
  "veto_reason": ""
}}"""

        result = self._call_llm(
            f"你是{self.role}。评估下行风险，风险评分>=7时行使软否决权。输出严格JSON。",
            prompt, temperature=0.2)
        return self._build_report(result)

    def _rule_analyze(self, state: dict) -> EmployeeReport:
        ti = state.get('technical_indicators', {}) or {}
        q = state.get('quote', {}) or {}
        price = q.get('price', 0) if isinstance(q, dict) else 0
        vol_ratio = ti.get('volume_ratio', 1) or 1
        amplitude = ti.get('amplitude', 3) or 3
        signals = state.get('signals', {}) or {}
        score = signals.get('score', 0) or 0

        risk_score = 5.0  # 基准
        points, risks = [], []
        veto, veto_reason = False, ""

        # 波动率评估
        if amplitude > 8:
            risk_score += 2; risks.append(f"振幅={amplitude}%极度偏高，日内波动风险大")
        elif amplitude > 5:
            risk_score += 1; risks.append(f"振幅={amplitude}%偏高")

        # 流动性评估
        if vol_ratio < 0.5:
            risk_score += 2; risks.append(f"量比={vol_ratio}极低，流动性风险显著")
        elif vol_ratio < 0.7:
            risk_score += 1; risks.append(f"量比={vol_ratio}偏低")

        # 信号方向
        if score < -3:
            risk_score += 2; risks.append(f"综合信号={score}，强烈看空")
        elif score < -1:
            risk_score += 1

        # 估值保护
        val_level = state.get('valuation_level', '正常')
        if val_level == '很低':
            risk_score -= 2; points.append("估值很低，下行空间有限")
        elif val_level == '很高':
            risk_score += 2; risks.append("估值很高，回调风险加大")

        # VaR 量化风险
        var_95 = state.get('var_95')
        if var_95 is not None:
            if abs(var_95) > 5:
                risk_score += 2; risks.append(f"VaR(95%)={var_95:.1f}%，日风险较高")
            elif abs(var_95) > 3:
                risk_score += 1; risks.append(f"VaR(95%)={var_95:.1f}%，日风险中等")
        max_dd = state.get('max_drawdown')
        if max_dd is not None:
            if max_dd > 40:
                risk_score += 2; risks.append(f"近1年最大回撤{max_dd:.0f}%，下行极深")
            elif max_dd > 25:
                risk_score += 1

        risk_score = max(1, min(10, risk_score))

        if risk_score >= self.veto_threshold:
            veto = True
            veto_reason = f"风险评分={risk_score:.0f}/10，超过否决阈值。建议暂停操作并重新评估。"
            points.append(f"🚫 行使软否决权: {veto_reason}")

        if not veto:
            points.append(f"风险评分={risk_score:.0f}/10，在可控范围内")

        # 评分标准化: -10 (高风险) ~ +10 (低风险)，与其他 agent 统一量纲
        normalized_score = round((5 - risk_score) * 2.0, 1)
        outlook = "看多" if normalized_score >= 2 else "看空" if normalized_score <= -2 else "中性"
        conf = "高" if abs(risk_score - 5) > 3 else "中"

        return EmployeeReport(employee_id=self.employee_id, role=self.role,
                              department=self.department, outlook=outlook,
                              confidence=conf,
                              score=normalized_score,
                              key_points=points, risks=risks)

    def _build_report(self, result: dict) -> EmployeeReport:
        risk_score = float(result.get('risk_score', 5))
        return EmployeeReport(employee_id=self.employee_id, role=self.role,
                              department=self.department,
                              outlook=result.get('outlook', '中性'),
                              confidence=result.get('confidence', '低'),
                              score=float(result.get('score', 0)),
                              key_points=result.get('key_points', []),
                              risks=result.get('risks', []),
                              raw_output=str(result))
