"""
E5 技术分析师 — 研究部

专注 K 线形态、趋势指标、动量、成交量、支撑阻力。
"""
from .base import BaseAgent, EmployeeReport


class TechnicalAgent(BaseAgent):
    """技术分析师: 趋势、动量、形态、量价"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.employee_id = "technical"
        self.role = "技术分析师"
        self.department = "研究部"

    def analyze(self, state: dict) -> EmployeeReport:
        data = self.build_tech_context(state)
        return self._llm_analyze(data) if self.has_llm else self._rule_analyze(state)

    def _llm_analyze(self, data: str) -> EmployeeReport:
        prompt = """你是A股技术分析专家。仅根据K线指标给出独立判断。

输出JSON:
{
  "outlook": "看多/看空/中性",
  "confidence": "高/中/低",
  "score": -3,
  "key_points": ["MACD死叉且柱状线扩大，短期动能偏空", "RSI=19进入超卖区，技术性反弹概率增加"],
  "risks": ["均线空头排列，趋势尚未扭转", "若跌破布林下轨可能加速下跌"]
}

评分规则:
- RSI<20超卖: +2分; RSI>80超买: -2分; RSI正常: 0~±1分
- MACD金叉/DIF>DEA: +1分; MACD死叉/DIF<DEA: -1分
- 均线多头排列(MA5>MA10>MA20>MA60): +2分; 空头排列: -2分; 粘连: 0分
- 价格>MA20: +1分; 价格<MA20: -1分
- %B(布林位置)<0.1: +1分(抄底区); %B>0.9: -1分(顶部区)
- 量比>1.5(放量上涨): +1分; 放量下跌: -1分; 缩量: -0.5分
- 最终score范围为-10到10
- outlook: score>2→看多, score<-2→看空, 否则中性"""

        result = self._call_llm(
            f"你是{self.role}。请仅基于提供的数据给出独立判断。输出严格JSON。",
            f"{prompt}\n\n## 技术数据\n{data}",
            temperature=0.3)
        return self._build_report(result)

    def _rule_analyze(self, state: dict) -> EmployeeReport:
        ti = state.get('technical_indicators', {}) or {}
        q = state.get('quote', {}) or {}
        price = q.get('price', 0) if isinstance(q, dict) else 0
        rsi = ti.get('rsi_14', 50) or 50
        macd_hist = ti.get('macd_hist', 0) or 0
        ma5, ma10, ma20 = ti.get('ma5', 0) or 0, ti.get('ma10', 0) or 0, ti.get('ma20', 0) or 0
        vol_ratio = ti.get('volume_ratio', 1) or 1

        score = 0.0
        points, risks = [], []

        if rsi < 20:
            score += 2; points.append(f"RSI={rsi:.0f}，严重超卖，反弹概率高")
        elif rsi < 30:
            score += 1; points.append(f"RSI={rsi:.0f}，超卖区域")
        elif rsi > 80:
            score -= 2; points.append(f"RSI={rsi:.0f}，严重超买，回调风险大"); risks.append("过度投机信号")
        elif rsi > 70:
            score -= 1; points.append(f"RSI={rsi:.0f}，超买区域")
        else:
            points.append(f"RSI={rsi:.0f}，处于中性区间")

        if macd_hist > 0:
            score += 1; points.append(f"MACD柱={macd_hist:.3f}>0，动能偏多")
        else:
            score -= 1; points.append(f"MACD柱={macd_hist:.3f}<0，动能偏空")

        if ma5 > ma10 > ma20 and price > ma5:
            score += 2; points.append("均线多头排列，趋势向上")
        elif ma5 < ma10 < ma20 and price < ma5:
            score -= 2; risks.append("均线空头排列，趋势向下")
        else:
            points.append("均线交织，方向不明朗")

        if price > ma20:
            score += 1
        elif price < ma20:
            score -= 1; risks.append("价格运行于MA20下方")

        if vol_ratio > 1.5:
            points.append(f"量比={vol_ratio:.1f}，交易活跃")
        elif vol_ratio < 0.5:
            score -= 0.5; points.append("缩量，市场情绪低迷")

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
