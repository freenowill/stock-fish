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
               state: dict, track_context: Optional[str] = None) -> CIODecision:
        """
        执行大师决策流程

        Args:
            master_key: 大师标识 (buffett/graham/fisher/lynch/templeton/soros/dalio)
            employee_reports: 所有员工的分析报告列表
            state: 分析状态 dict (AnalysisState.to_dict() 的输出)
            track_context: 可选，历史准确率上下文（用于回注大师提示词）

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

        # 组装用户提示：员工报告 + 关键市场数据 + 历史准确率
        user_prompt = self._build_cio_user_prompt(employee_reports, state, master_info, track_context=track_context)

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
                                state: dict, master_info: dict,
                                track_context: Optional[str] = None) -> str:
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
        # 评分引擎信号
        sb = state.get('score_breakdown') or {}
        if isinstance(sb, dict):
            score_val = sb.get('final', 'N/A')
            score_label = sb.get('label', '')
            lines.append(f"- 系统综合评分: {score_val} ({score_label})")
            if sb.get('technical') is not None:
                lines.append(f"  技术{sb.get('technical', '')} / 基本面{sb.get('fundamental', '')} / 舆情{sb.get('sentiment', '')}")
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
        pe_pct = state.get('valuation_percentile')
        pe_pct_str = f"{pe_pct:.1f}%" if pe_pct is not None else "N/A"
        lines.append(f"PE: {q.get('pe', 'N/A')}  PB: {q.get('pb', 'N/A')}  市值: {q.get('market_cap', 'N/A')}亿")
        lines.append(f"ROE: {fs.get('roe', 'N/A')}%  EPS: {fs.get('eps', 'N/A')}  "
                     f"经营现金流: {fs.get('operating_cash_flow', 'N/A')}亿  "
                     f"自由现金流: {fs.get('free_cash_flow', 'N/A')}亿  "
                     f"股息率: {fs.get('dividend_yield') or q.get('dividend_yield', 'N/A')}%")
        lines.append(f"RSI(14): {ti.get('rsi_14', 'N/A')}  MACD柱: {ti.get('macd_hist', 'N/A')}")
        lines.append(f"PE 分位: {pe_pct_str}")

        # 新增数据字段
        roic = state.get('roic')
        if roic is not None:
            lines.append(f"ROIC: {roic:.2f}% (投入资本回报率)")
        fcf_ps = state.get('fcf_per_share')
        if fcf_ps is not None:
            lines.append(f"每股自由现金流: {fcf_ps:.2f}元")
        pe_5y = state.get('valuation_percentile_5y')
        if pe_5y is not None:
            lines.append(f"5年PE分位: {pe_5y:.1f}%")
        ey = state.get('earnings_yield')
        bond = state.get('bond_yield_10y')
        if ey is not None:
            lines.append(f"E/P收益率: {ey:.1f}%" + (f" (vs 国债{bond:.1f}%)" if bond is not None else ""))
        var_95 = state.get('var_95')
        if var_95 is not None:
            lines.append(f"VaR(95%): {var_95:.1f}%  最大回撤: {state.get('max_drawdown', 'N/A')}%")

        # 护城河评估（巴菲特关注）
        moat = state.get('moat_assessment')
        if moat and moat.get('moat_level'):
            lines.append(f"护城河评估: {moat['moat_level']}" +
                        (f" (来源: {', '.join(moat['moat_sources'][:3])})" if moat.get('moat_sources') else ""))

        # 同行估值对比（邓普顿关注）
        peer = state.get('peer_valuation')
        if peer and peer.get('peers'):
            lines.append(f"同行搜索: {len(peer['peers'])}家同业公司信息")
            if peer.get('global_pe_data'):
                for t, d in peer['global_pe_data'].items():
                    if d.get('pe'):
                        lines.append(f"  {d['name']} PE={d['pe']:.1f}")

        # ── 数据质量告警 ──
        lines.append("")
        warnings = []

        # 现金流数据缺失告警
        ocf = fs.get('operating_cash_flow')
        fcf = fs.get('free_cash_flow')
        if ocf is None or ocf == 0 or ocf == 'N/A' or ocf == '?':
            warnings.append("⚠️ 数据缺失告警: 经营现金流数据不可用（null/0），净利润质量无法通过现金流验证")
        if fcf is None or fcf == 0 or fcf == 'N/A' or fcf == '?':
            warnings.append("⚠️ 数据缺失告警: 自由现金流数据不可用（null/0），无法计算FCF收益率")
        net_profit = fs.get('net_profit')
        np_yoy = fs.get('net_profit_yoy')
        rev = fs.get('revenue')
        rev_yoy = fs.get('revenue_yoy')
        if net_profit is not None and rev is not None and net_profit != 'N/A' and rev != 'N/A':
            try:
                np_val = float(net_profit) if net_profit is not None else 0
                rev_val = float(rev) if rev is not None else 0
                if np_val > 0 and rev_val > 0 and np_val / rev_val > 0.3:
                    warnings.append(f"⚠️ 净利率异常: 净利润({net_profit}亿)/营收({rev}亿)>30%，可能存在非经常性损益")
            except (ValueError, TypeError):
                pass

        if warnings:
            lines.append("## ⚠️ 数据完整性问题 — 你的决策中必须回应以下告警")
            lines.append("")
            for w in warnings:
                lines.append(w)
            lines.append("")
            lines.append("请在 rationale 或 extraordinary_items_note 中针对以上告警做出说明。")

        # 注入历史准确率追踪（如果可用）
        if track_context:
            lines.append("")
            lines.append(track_context)
            lines.append("")
            lines.append("请参考以上历史表现，思考你在哪些情境下容易判断失误，并在本次决策中有意识地避免同样错误。")

        return "\n".join(lines)

    # ── 结果解析 ──

    def _parse_cio_result(self, result: dict, master_info: dict) -> CIODecision:
        """将 LLM 返回的 JSON 解析为 CIODecision，含输出质量校验"""
        # ── 校验 1: evidence_chain 必须包含全部 8 名员工 ──
        evidence_chain = result.get('evidence_chain', [])
        EXPECTED_EMPLOYEES = 8  # 宏观/行业/估值/基本面/技术/舆情/风险/监察员
        EMPLOYEE_NAMES = [
            "宏观分析师", "行业政策分析师", "估值分析师", "基本面分析师",
            "技术分析师", "舆情分析师", "风险经理", "独立监察员",
        ]
        ec_len_original = len(evidence_chain)
        if ec_len_original < EXPECTED_EMPLOYEES:
            logger.warning(
                f"CIO evidence_chain 不完整: 期望 {EXPECTED_EMPLOYEES} 条, "
                f"实际 {ec_len_original} 条 — 缺失 {EXPECTED_EMPLOYEES - ec_len_original} 位员工, "
                f"自动补全中..."
            )
            # 检测已覆盖的员工角色
            covered = set()
            for entry in evidence_chain:
                for emp_name in EMPLOYEE_NAMES:
                    if emp_name in entry:
                        covered.add(emp_name)
                        break
            # 补全缺失的员工
            for emp in EMPLOYEE_NAMES:
                if emp not in covered:
                    evidence_chain.append(
                        f"{emp}: [系统自动补注 — CIO 原始输出未引用此员工，"
                        f"请参考该员工报告自行补充判断]"
                    )
            logger.warning(f"已自动补全 {len(evidence_chain) - ec_len_original} 条缺失的员工引用")
        elif len(evidence_chain) > EXPECTED_EMPLOYEES:
            logger.warning(
                f"CIO evidence_chain 超长: 期望 {EXPECTED_EMPLOYEES} 条, "
                f"实际 {len(evidence_chain)} 条 — 自动截断至 {EXPECTED_EMPLOYEES} 条"
            )
            evidence_chain = evidence_chain[:EXPECTED_EMPLOYEES]

        # ── 校验 2: scenario probability 之和必须为 1.0 ──
        for case_key in ('base_case', 'bull_case', 'bear_case'):
            case = result.get(case_key)
            if case and isinstance(case, dict):
                prob = case.get('probability')
                if prob is not None and (prob < 0 or prob > 1):
                    logger.warning(f"CIO {case_key}.probability={prob} 超出 [0,1] 范围")

        return CIODecision(
            master_name=master_info['name'],
            master_key=master_info['key'],
            decision_summary=result.get('decision_summary', ''),
            rationale=result.get('rationale', ''),
            evidence_chain=evidence_chain,
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
            extraordinary_items_note=result.get('extraordinary_items_note', ''),
            raw_llm_output=json.dumps(result, ensure_ascii=False),
        )

    # ── 降级: 规则决策 ──

    def _fallback_decision(self, master_info: dict,
                           reports: List[EmployeeReport],
                           state: dict) -> CIODecision:
        """
        当 LLM 不可用时，增强型规则降级决策。
        置信度加权投票 + 评分引擎融合 + 基本情景分析 + 监察员风险纳入。
        """
        valid_reports = [r for r in reports if not r.error]
        if not valid_reports:
            outlook = "中性"
            confidence = "低"
            summary = "所有员工报告生成失败，无法做出决策。"
            ec_info = "无有效员工报告"
            overseer_risks = []
        else:
            # ── 置信度加权投票 ──
            confidence_map = {'高': 1.0, '中': 0.6, '低': 0.3}
            weighted_bullish = 0.0
            weighted_bearish = 0.0
            weighted_neutral = 0.0
            for r in valid_reports:
                w = confidence_map.get(r.confidence, 0.5)
                if r.outlook in ('看多', '偏多'):
                    weighted_bullish += w
                elif r.outlook in ('看空', '偏空'):
                    weighted_bearish += w
                else:
                    weighted_neutral += w

            total_w = weighted_bullish + weighted_bearish + weighted_neutral
            if total_w > 0:
                bull_pct = weighted_bullish / total_w
                bear_pct = weighted_bearish / total_w
            else:
                bull_pct = bear_pct = 0

            if bull_pct >= 0.5:
                outlook = "看多"
            elif bear_pct >= 0.5:
                outlook = "看空"
            elif bull_pct > bear_pct:
                outlook = "偏多"
            elif bear_pct > bull_pct:
                outlook = "偏空"
            else:
                outlook = "中性"

            diff = abs(bull_pct - bear_pct)
            confidence = "高" if diff >= 0.4 else "中" if diff >= 0.15 else "低"

            # ── 融合评分引擎信号 ──
            sb = state.get('score_breakdown') or {}
            score_val = sb.get('final') if isinstance(sb, dict) else None
            score_outlook = sb.get('label') if isinstance(sb, dict) else None

            score_note = ""
            if score_val is not None and score_outlook:
                # 如果评分与投票方向矛盾，保守处理
                score_bullish = score_val >= 0.5
                score_bearish = score_val <= -0.5
                vote_bullish = outlook in ('看多', '偏多')
                vote_bearish = outlook in ('看空', '偏空')
                if (vote_bullish and score_bearish) or (vote_bearish and score_bullish):
                    score_note = f"（注意：投票方向与评分{score_outlook}({score_val})矛盾，最终取保守方向）"
                    outlook = "中性"

            # ── 提取监察员风险点 ──
            overseer_risks = []
            for r in valid_reports:
                if r.employee_id == 'overseer' and r.risks:
                    overseer_risks = r.risks

            q = state.get('quote', {}) or {}
            price = q.get('price', 0) if isinstance(q, dict) else 0

            # ── 基本情景分析基于波动率 ──
            var_95 = state.get('var_95', 3.0) or 3.0
            max_dd = state.get('max_drawdown', 20.0) or 20.0
            vol = state.get('annualized_volatility', 20.0) or 20.0
            change_pct = round(price * var_95 / 100, 2) if price > 0 else 0

            summary = f"(规则降级·增强) 置信度加权投票→{outlook}，评分信号{score_outlook or 'N/A'}({score_val}) {score_note}".strip()

            # 构建详细的证据链 + 监察员纳入
            ec_info = []
            for r in valid_reports:
                entry = f"{r.role}: {r.outlook} (置信度={r.confidence})"
                if r.key_points:
                    entry += f" — {'; '.join(r.key_points[:2])}"
                ec_info.append(entry)
            # 确保监察员在证据链中（重要）
            has_overseer = any(r.employee_id == 'overseer' for r in valid_reports)
            if not has_overseer:
                ec_info.append("独立监察员: 报告缺失，未参与规则决策")

            # 构建详细 rationale
            rationale_parts = [
                f"(LLM不可用·规则降级·增强模式) 基于{len(valid_reports)}份员工报告置信度加权投票: "
                f"看多权重{bull_pct:.0%} / 看空权重{bear_pct:.0%} / 中性权重{weighted_neutral/total_w:.0%}",
                f"→ {outlook}（置信度{confidence}）",
                f"评分引擎: {score_outlook or 'N/A'}（{score_val}）",
            ]
            if overseer_risks:
                rationale_parts.append(f"监察员风险提示: {'; '.join(overseer_risks[:3])}（已纳入综合判断）")
            if score_note:
                rationale_parts.append(score_note)
            rationale_parts.append("建议启用LLM获取完整深度推理。")
            rationale = " | ".join(rationale_parts)

            # 生成基本操作建议
            if outlook == "看多":
                action = "买入"
                pos_pct = 10
                sl = round(price * (1 - var_95 / 100 * 1.5), 2) if price > 0 else 0
                tp = round(price * (1 + var_95 / 100 * 2), 2) if price > 0 else 0
            elif outlook == "偏多":
                action = "加仓"
                pos_pct = 5
                sl = round(price * (1 - var_95 / 100 * 1.5), 2) if price > 0 else 0
                tp = round(price * (1 + var_95 / 100 * 1.5), 2) if price > 0 else 0
            elif outlook in ("看空", "偏空"):
                action = "卖出"
                pos_pct = 0
                sl = 0
                tp = 0
            else:
                action = "观望"
                pos_pct = 0
                sl = 0
                tp = 0

            # 有监察员极端风险时预警
            risk_monitoring = []
            if overseer_risks:
                for r in overseer_risks[:3]:
                    risk_monitoring.append({"trigger": f"监察员提示: {r}", "action": "重新评估持仓"})

            return CIODecision(
                master_name=master_info['name'],
                master_key=master_info['key'],
                decision_summary=summary,
                rationale=rationale,
                evidence_chain=ec_info,
                short_term={'direction': outlook, 'change_pct': change_pct, 'confidence': confidence,
                           'reason': f'规则降级模式: 置信度加权投票{outlook}'},
                mid_term={'direction': outlook, 'change_pct': change_pct * 2 if price > 0 else 0,
                         'confidence': confidence if confidence != '高' else '中',
                         'reason': '规则降级模式: 基于加权投票趋势外推'},
                long_term={'direction': '震荡', 'change_pct': 0, 'confidence': '低',
                          'reason': 'LLM不可用，无法做长期预测'},
                order={'action': action, 'position_size_pct': pos_pct,
                       'entry_conditions': f'参考价{price}元',
                       'stop_loss': {'level': sl, 'type': '固定止损', 'trigger': f'跌破{sl}'},
                       'take_profit': {'level_1': tp, 'level_2': 0, 'type': '一次性止盈'}},
                risk_monitoring=risk_monitoring if risk_monitoring else [
                    {'trigger': '价格波动超过VaR预测', 'action': '关注是否触发止损'}
                ],
                decision_quality={'confidence': '低', 'key_uncertainties': ['LLM不可用，决策精度下降'],
                                 'next_review': '1个交易日后'},
                veto_response='规则降级模式下自动跳过风险否决评估',
            )

        # 无有效报告时的兜底
        q = state.get('quote', {}) or {}
        price = q.get('price', 0) if isinstance(q, dict) else 0
        return CIODecision(
            master_name=master_info['name'],
            master_key=master_info['key'],
            decision_summary=summary,
            rationale=f"LLM不可用且所有员工报告均失败，无法做出有效决策。",
            evidence_chain=[ec_info] if isinstance(ec_info, str) else ec_info if isinstance(ec_info, list) else [],
            short_term={'direction': '震荡', 'change_pct': 0, 'confidence': '低', 'reason': '所有报告失败'},
            mid_term={'direction': '震荡', 'change_pct': 0, 'confidence': '低', 'reason': '所有报告失败'},
            long_term={'direction': '震荡', 'change_pct': 0, 'confidence': '低', 'reason': '所有报告失败'},
            order={'action': '观望', 'position_size_pct': 0, 'entry_conditions': 'N/A',
                   'stop_loss': {'level': 0, 'type': 'N/A'},
                   'take_profit': {'level_1': 0, 'level_2': 0, 'type': 'N/A'}},
            decision_quality={'confidence': '低', 'key_uncertainties': ['LLM不可用，所有员工报告失败'],
                             'next_review': '1个交易日后'},
        )
