"""
E4 基本面分析师 — 研究部

专注财务健康度、盈利质量、现金流、护城河、管理层评估。
不再包含估值（估值已拆分给 ValuationAgent）。
"""
from .base import BaseAgent, EmployeeReport


class FundamentalAgent(BaseAgent):
    """基本面分析师: ROE、盈利质量、护城河、管理层"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.employee_id = "fundamental"
        self.role = "基本面分析师"
        self.department = "研究部"

    def analyze(self, state: dict) -> EmployeeReport:
        data = self.build_fund_context(state)
        return self._llm_analyze(data) if self.has_llm else self._rule_analyze(state)

    def _llm_analyze(self, data: str) -> EmployeeReport:
        prompt = """你是A股基本面分析专家。分析财务数据给出独立判断。

    输出JSON:
    {
      "outlook": "看多/看空/中性",
      "confidence": "高/中/低",
      "score": 3,
      "key_points": ["ROE=26%远高于15%门槛，盈利能力卓越", "ROIC=18%显示强韧护城河", "自由现金流连续5年为正，利润质量高"],
      "risks": ["营收增速放缓，需关注成长瓶颈", "资产负债率上升"]
    }

    评分规则:
    - ROE>20%: +3分; >15%: +2分; >10%: +1分; <5%: -1分; <0%: -2分
    - ROIC>15%: +2分 (强护城河信号); ROIC>10%: +1分; ROIC<5%: -1分
    - 营收同比正增长>20%: +2分; >10%: +1分; 负增长: -2分
    - 净利润同比正增长>20%: +2分; >10%: +1分; 负增长: -2分
    - 毛利率>行业均值: +1分; 毛利率趋势下跌: -1分
    - 负债率<40%: +1分; 负债率>70%: -1分
    - 最终score范围为-10到10
    - outlook: score>2→看多, score<-2→看空, 否则中性"""

        result = self._call_llm(
            f"你是{self.role}。请仅基于提供的数据给出独立判断。输出严格JSON。",
            f"{prompt}\n\n## 财务数据\n{data}",
            temperature=0.3)
        return self._build_report(result)

    def _rule_analyze(self, state: dict) -> EmployeeReport:
        fs = state.get('financial_summary', {}) or {}
        roe = fs.get('roe', 0) or 0
        rev_yoy = fs.get('revenue_yoy', 0) or 0
        profit_yoy = fs.get('net_profit_yoy', 0) or 0
        gross_margin = fs.get('gross_margin', 0) or 0
        debt_ratio = fs.get('debt_ratio', 0) or 0
        is_financial = BaseAgent._is_financial_industry(state)

        score = 0.0
        points, risks = [], []

        if roe > 20:
            score += 3; points.append(f"ROE={roe:.1f}%，盈利能力卓越")
        elif roe > 15:
            score += 2; points.append(f"ROE={roe:.1f}%，盈利良好")
        elif roe > 10:
            score += 1; points.append(f"ROE={roe:.1f}%，盈利尚可")
        elif roe < 5:
            score -= 1; risks.append(f"ROE={roe:.1f}%偏低")

        # ROIC 评估 (巴菲特护城河指标)
        roic = state.get('roic')
        if roic is not None:
            if roic > 15:
                score += 2; points.append(f"ROIC={roic:.1f}%，护城河强劲")
            elif roic > 10:
                score += 1; points.append(f"ROIC={roic:.1f}%，护城河稳固")
            elif roic < 5:
                score -= 1; risks.append(f"ROIC={roic:.1f}%，资本回报效率偏低")

        # 自由现金流验证 (巴菲特核心指标)
        oper_cf = fs.get('operating_cash_flow', 0) or 0
        fcf = fs.get('free_cash_flow')
        rev = fs.get('revenue', 0) or 0
        if fcf is not None:
            if fcf > 0 and rev > 0:
                fcf_margin = fcf / rev * 100
                if fcf_margin > 10:
                    score += 2; points.append(f"FCF/营收={fcf_margin:.1f}%，现金流极强")
                else:
                    score += 1; points.append(f"自由现金流={fcf:.1f}亿，现金流健康")
            elif fcf < 0:
                score -= 2; risks.append(f"自由现金流为负({fcf:.1f}亿)，盈利质量存疑")
        elif oper_cf > 0 and rev > 0:
            # 无 FCF 时用经营现金流近似
            score += 0.5; points.append(f"经营现金流为正({oper_cf:.1f}亿)")
        else:
            # 现金流数据完全缺失
            risks.append("经营现金流/自由现金流数据缺失，净利润质量无法验证")

        # 多期趋势
        ft = state.get('financial_trends') or {}
        gm_trend = ft.get('gross_margin_trend')
        if gm_trend == '下降':
            risks.append("毛利率呈下降趋势，需关注竞争恶化")
        elif gm_trend == '上升':
            points.append("毛利率持续改善，定价权增强")
        eps_cagr = ft.get('eps_cagr_5y')
        if eps_cagr is not None and eps_cagr < -0.05:
            risks.append(f"EPS 5年CAGR={eps_cagr*100:.1f}%，盈利增长乏力")

        if rev_yoy > 20:
            score += 2; points.append(f"营收同比+{rev_yoy:.1f}%，高速增长")
        elif rev_yoy > 10:
            score += 1; points.append(f"营收同比+{rev_yoy:.1f}%，稳步增长")
        elif rev_yoy < 0:
            score -= 2; risks.append(f"营收同比{rev_yoy:.1f}%，收入收缩")

        if profit_yoy > 20:
            score += 2; points.append(f"净利润同比+{profit_yoy:.1f}%，利润高增")
            # 利润质量检查: 利润增速远高于收入增速时可能有非经常性损益
            if rev_yoy is not None and profit_yoy is not None:
                try:
                    if float(profit_yoy) > float(rev_yoy) * 3 and float(rev_yoy) < 10:
                        risks.append("净利润增速远超营收增速，可能存在非经常性损益")
                except (ValueError, TypeError):
                    pass
        elif profit_yoy > 10:
            score += 1
        elif profit_yoy < 0:
            score -= 2; risks.append(f"净利润同比{profit_yoy:.1f}%，盈利下滑")

        if gross_margin > 40:
            score += 1; points.append(f"毛利率{gross_margin:.1f}%，护城河体现")
        if debt_ratio < 40 and not is_financial:
            score += 1; points.append(f"负债率{debt_ratio:.1f}%，财务稳健")
        elif debt_ratio > 70 and not is_financial:
            score -= 1; risks.append(f"负债率{debt_ratio:.1f}%，财务杠杆偏高")
        elif is_financial:
            points.append(f"负债率{debt_ratio:.1f}%（金融行业，高负债为结构性特征）")

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
