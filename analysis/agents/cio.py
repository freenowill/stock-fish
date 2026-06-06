"""
CIO Agent — 最终决策人执行器

接收用户选择的大师名称和所有员工报告，
加载大师专属 System Prompt，调用 LLM 输出结构化决策。

用法:
    cio = CIOAgent()
    decision = cio.decide(
        master_key="buffett",
        employee_reports=[report1, report2, ...],
        state=analysis_state_dict,
    )
    # decision 是 CIODecision 实例
"""
import json
from typing import Optional, Dict, List
from loguru import logger

from .base import BaseAgent, CIODecision, EmployeeReport
from .cio_prompts import get_master_prompt, get_master_info


class CIOAgent(BaseAgent):
    """最终决策人 — 以大师的投资哲学对员工报告做出最终裁决"""

    def decide(self, master_key: str, employee_reports: List[EmployeeReport],
               state: dict) -> CIODecision:
        """
        执行大师决策流程

        Args:
            master_key: 大师标识 (buffett/graham/fisher/lynch/templeton/soros/dalio)
            employee_reports: 所有员工的分析报告列表
            state: 分析状态 dict (AnalysisState.to_dict() 的输出)

        Returns:
            CIODecision 结构化决策
        """
        master_info = get_master_info(master_key)
        if not master_info:
            return CIODecision(
                master_key=master_key,
                error=f"未知的大师: {master_key}",
            )

        system_prompt = get_master_prompt(master_key)
        if not system_prompt:
            return CIODecision(
                master_name=master_info['name'],
                master_key=master_key,
                error=f"大师 prompt 未找到: {master_key}",
            )

        # 组装用户提示：员工报告 + 关键市场数据
        user_prompt = self._build_cio_user_prompt(employee_reports, state, master_info)

        # 调用 LLM
        if not self.has_llm:
            logger.warning(f"LLM 未配置，CIO 降级为规则模式")
            return self._fallback_decision(master_info, employee_reports, state)

        result = self._call_llm(system_prompt, user_prompt, temperature=0.4)

        if not result:
            logger.warning(f"LLM 调用失败，CIO 降级为规则决策")
            return self._fallback_decision(master_info, employee_reports, state)

        # 组装 CIODecision
        return self._parse_cio_result(result, master_info)

    # ── Prompt 构建 ──

    def _build_cio_user_prompt(self, reports: List[EmployeeReport],
                                state: dict, master_info: dict) -> str:
        """构建发送给 CIO 的用户 prompt: 包含所有员工报告 + 市场背景"""
        q = state.get('quote', {}) or {}
        price = q.get('price', 0) if isinstance(q, dict) else 0
        cost = state.get('cost_price', 0) or 0
        symbol = state.get('symbol', '')
        name = state.get('stock_name', '')

        lines = [
            f"# {master_info['name']} 先生，以下是您的投资研究团队提交的分析报告。",
            f"",
            f"## 标的信息",
            f"- 股票: {name}({symbol})",
            f"- 现价: {price} 元",
        ]
        if cost > 0:
            pnl_pct = (price - cost) / cost * 100
            lines.append(f"- 用户成本价: {cost} 元 (浮动盈亏: {pnl_pct:+.1f}%)")

        lines.append(f"- 估值等级: {state.get('valuation_level', 'N/A')}")
        lines.append(f"- 系统建议买入价: {state.get('suggested_buy_price', 'N/A')}")
        lines.append("")

        # 用户持仓/资金信息
        shares = state.get('shares', 0) or 0
        total_assets = state.get('total_assets', 0) or 0
        available_cash = state.get('available_cash', 0) or 0
        has_portfolio = shares > 0 or total_assets > 0 or available_cash > 0
        if has_portfolio:
            lines.append("## 用户持仓与资金状况")
            position_value = shares * price if shares > 0 and price > 0 else 0
            position_pct = round(position_value / total_assets * 100, 1) if total_assets > 0 and position_value > 0 else 0
            if shares > 0:
                lines.append(f"- 持仓数量: {shares} 股  (市值约 {position_value:.0f} 元)")
                if cost > 0:
                    cost_total = shares * cost
                    pnl_total = position_value - cost_total
                    lines.append(f"- 持仓成本: {cost_total:.0f} 元  浮动盈亏: {pnl_total:+.0f} 元")
            if total_assets > 0:
                lines.append(f"- 总资产: {total_assets:.0f} 元")
                if position_pct > 0:
                    lines.append(f"- 该股仓位占比: {position_pct}%")
            if available_cash > 0:
                lines.append(f"- 可用资金: {available_cash:.0f} 元")
                if total_assets > 0:
                    cash_pct = round(available_cash / total_assets * 100, 1)
                    lines.append(f"- 现金占比: {cash_pct}%")
            lines.append("")
            lines.append("**重要**: 以上持仓数据是你做决策的核心约束。你给出的 `position_size_pct`、`entry_conditions`、`stop_loss` 必须与用户的资金状况匹配——例如可用资金不足时应分批建仓，仓位过重时应优先控风险。")
            lines.append("")

        # 员工报告
        lines.append("## 部门分析报告")
        lines.append("")

        dept_names = {
            'macro': '宏观部', 'policy': '宏观部', 'valuation': '研究部',
            'fundamental': '研究部', 'technical': '研究部', 'sentiment': '交易部',
            'risk': '风控部', 'overseer': '监察部',
        }

        for r in reports:
            dept = dept_names.get(r.employee_id, '其他')
            if r.error:
                lines.append(f"### {r.role} ({dept}) [⚠️ 报告生成失败]")
                lines.append(f"错误: {r.error}")
            else:
                lines.append(f"### {r.role} ({dept})")
                lines.append(f"判断: {r.outlook}  置信度: {r.confidence}  评分: {r.score:.1f}")
                if r.key_points:
                    lines.append("关键观点:")
                    for p in r.key_points:
                        lines.append(f"  - {p}")
                if r.risks:
                    lines.append("风险提示:")
                    for risk in r.risks:
                        lines.append(f"  - {risk}")
            lines.append("")

        # 附加关键原始数据供 CIO 交叉验证
        lines.append("## 关键原始数据 (供交叉验证)")
        ti = state.get('technical_indicators', {}) or {}
        fs = state.get('financial_summary', {}) or {}
        lines.append(f"PE: {q.get('pe', 'N/A')}  PB: {q.get('pb', 'N/A')}  市值: {q.get('market_cap', 'N/A')}亿")
        lines.append(f"ROE: {fs.get('roe', 'N/A')}%  EPS: {fs.get('eps', 'N/A')}")
        lines.append(f"RSI(14): {ti.get('rsi_14', 'N/A')}  MACD柱: {ti.get('macd_hist', 'N/A')}")
        lines.append(f"PE 分位: {state.get('valuation_percentile', 'N/A')}%")

        return "\n".join(lines)

    # ── 结果解析 ──

    def _parse_cio_result(self, result: dict, master_info: dict) -> CIODecision:
        """将 LLM 返回的 JSON 解析为 CIODecision"""
        return CIODecision(
            master_name=master_info['name'],
            master_key=master_info['key'],
            decision_summary=result.get('decision_summary', ''),
            evidence_chain=result.get('evidence_chain', []),
            base_case=result.get('base_case'),
            bull_case=result.get('bull_case'),
            bear_case=result.get('bear_case'),
            order=result.get('order'),
            short_term=result.get('short_term'),
            mid_term=result.get('mid_term'),
            long_term=result.get('long_term'),
            risk_monitoring=result.get('risk_monitoring', []),
            decision_quality=result.get('decision_quality'),
            veto_response=result.get('veto_response', ''),
            raw_llm_output=json.dumps(result, ensure_ascii=False),
        )

    # ── 降级: 规则决策 ──

    def _fallback_decision(self, master_info: dict,
                           reports: List[EmployeeReport],
                           state: dict) -> CIODecision:
        """
        当 LLM 不可用时，基于员工报告的规则降级决策。
        取 majority outlook + 风险加权。
        """
        # 统计各员工的方向
        valid_reports = [r for r in reports if not r.error]
        if not valid_reports:
            outlook = "中性"
            confidence = "低"
            summary = "所有员工报告生成失败，无法做出决策。"
        else:
            bullish = sum(1 for r in valid_reports if r.outlook in ('看多', '偏多'))
            bearish = sum(1 for r in valid_reports if r.outlook in ('看空', '偏空'))
            neutral = len(valid_reports) - bullish - bearish

            if bullish > bearish and bullish > neutral:
                outlook = "看多"
            elif bearish > bullish and bearish > neutral:
                outlook = "看空"
            else:
                outlook = "中性"

            confidence = "高" if abs(bullish - bearish) >= 3 else "中" if abs(bullish - bearish) >= 1 else "低"
            summary = f"(规则降级) 基于{len(valid_reports)}份有效报告: 看多{bullish}/看空{bearish}/中性{neutral} → {outlook}"

        q = state.get('quote', {}) or {}
        price = q.get('price', 0) if isinstance(q, dict) else 0

        return CIODecision(
            master_name=master_info['name'],
            master_key=master_info['key'],
            decision_summary=summary,
            evidence_chain=[f"{r.role}: {r.outlook} (confidence={r.confidence})" for r in valid_reports],
            short_term={'direction': outlook, 'change_pct': 0, 'confidence': confidence,
                       'reason': 'LLM 不可用，规则降级'},
            mid_term={'direction': outlook, 'change_pct': 0, 'confidence': '低',
                     'reason': 'LLM 不可用，规则降级'},
            long_term={'direction': '震荡', 'change_pct': 0, 'confidence': '低',
                      'reason': 'LLM 不可用，无法做长期预测'},
            order={'action': '观望', 'position_size_pct': 0, 'entry_conditions': 'N/A',
                   'stop_loss': {'level': 0, 'type': 'N/A'},
                   'take_profit': {'level_1': 0, 'level_2': 0, 'type': 'N/A'}},
            decision_quality={'confidence': '低', 'key_uncertainties': ['LLM 不可用，决策精度下降'],
                             'next_review': '1个交易日后'},
        )
