"""
Seed Document Builder

将 StockEngine 分析结果转换为 MiroFish 兼容的种子文档，
供 Zep GraphRAG 构建知识图谱。

使用自然语言叙事结构，包含 7 个专业 Agent 角色描述，
便于 Zep 提取实体和关系。
"""
import json
from datetime import datetime
from typing import Dict, Any, List


class SeedDocumentBuilder:
    """分析结果 → MiroFish 种子文档"""

    @staticmethod
    def build(analysis_result: Dict[str, Any]) -> str:
        """将分析结果转换为叙事风格的种子文档文本"""
        symbol = analysis_result.get('symbol', '')
        name = analysis_result.get('stock_name', analysis_result.get('name', ''))
        signals = analysis_result.get('signals', {}) or {}
        quote = analysis_result.get('quote', {}) or {}
        ti = analysis_result.get('technical_indicators', {}) or {}
        fs = analysis_result.get('financial_summary', {}) or {}
        ps = analysis_result.get('prediction_summary', {}) or {}
        news = analysis_result.get('news', []) or []
        guba = analysis_result.get('guba_posts', []) or []
        search_news = analysis_result.get('search_news', []) or []

        lines = []
        lines.append(f"# {name}({symbol}) 市场分析报告")
        lines.append("")

        # 公司概况（叙事段落）
        price = quote.get('price', 'N/A')
        change_pct = quote.get('change_pct', 'N/A')
        pe = quote.get('pe', 'N/A')
        pb = quote.get('pb', 'N/A')
        market_cap = quote.get('market_cap', 'N/A')
        score = signals.get('score', 0)
        label = signals.get('label', '')

        lines.append(f"{name}({symbol}) 是一家在上海证券交易所上市的白酒龙头企业。当前股价为 {price} 元，"
                     f"今日涨跌幅为 {change_pct}%。公司总市值约 {market_cap} 亿元，"
                     f"市盈率 PE 为 {pe}，市净率 PB 为 {pb}。")
        if score:
            lines.append(f"综合评分系统给出 {label} 信号，评分为 {score} 分（范围-5到+5）。")
        lines.append("")

        # 技术分析（叙事段落）
        lines.append("## 技术分析")
        rsi = ti.get('rsi_14', 'N/A')
        macd_hist = ti.get('macd_hist', 'N/A')
        kdj_k = ti.get('kdj_k', 'N/A')
        kdj_d = ti.get('kdj_d', 'N/A')
        kdj_j = ti.get('kdj_j', 'N/A')
        ma5 = ti.get('ma5', 'N/A')
        ma10 = ti.get('ma10', 'N/A')
        ma20 = ti.get('ma20', 'N/A')
        boll_lower = ti.get('boll_lower', 'N/A')
        boll_middle = ti.get('boll_middle', 'N/A')
        boll_upper = ti.get('boll_upper', 'N/A')
        vol_ratio = ti.get('volume_ratio', 'N/A')

        lines.append(f"技术指标显示，RSI(14)为 {rsi}，MACD柱状线为 {macd_hist}，"
                     f"KDJ指标为 {kdj_k}/{kdj_d}/{kdj_j}。")
        lines.append(f"移动平均线：MA5 = {ma5}，MA10 = {ma10}，MA20 = {ma20}。"
                     f"布林带通道：下轨 {boll_lower}，中轨 {boll_middle}，上轨 {boll_upper}。"
                     f"量比为 {vol_ratio}。")
        lines.append("")

        # 基本面（叙事段落）
        lines.append("## 基本面分析")
        revenue = fs.get('revenue', 'N/A')
        net_profit = fs.get('net_profit', 'N/A')
        eps = fs.get('eps', 'N/A')
        roe = fs.get('roe', 'N/A')
        gross_margin = fs.get('gross_margin', 'N/A')
        debt_ratio = fs.get('debt_ratio', 'N/A')
        revenue_yoy = fs.get('revenue_yoy', 'N/A')
        net_profit_yoy = fs.get('net_profit_yoy', 'N/A')

        lines.append(f"公司最新财务数据显示，营业收入为 {revenue} 亿元，净利润为 {net_profit} 亿元。"
                     f"每股收益 EPS 为 {eps} 元，净资产收益率 ROE 为 {roe}%。"
                     f"毛利率为 {gross_margin}%，资产负债率为 {debt_ratio}%。")
        lines.append(f"营收同比增长 {revenue_yoy}%，净利润同比增长 {net_profit_yoy}%。")
        lines.append("")

        # 预测展望（叙事段落）
        if ps:
            outlook = ps.get('outlook', 'N/A')
            confidence = ps.get('confidence', 'N/A')
            reason = ps.get('reason', '')
            lines.append("## 预测展望")
            lines.append(f"分析师团队通过多Agent辩论，给出 {outlook} 展望，置信度为 {confidence}。")
            if reason:
                lines.append(f"核心判断逻辑：{reason}")
            lines.append("")

            # 多周期预测
            lines.append("### 多周期价格预测")
            for p_name, p_label in [('short_term', '短期1-2周'), ('mid_term', '中期1-3月'), ('long_term', '长期6-12月')]:
                p = ps.get(p_name) or {}
                if p:
                    direction = p.get('direction', '震荡')
                    change_pct = p.get('change_pct', 0)
                    p_conf = p.get('confidence', '中')
                    lines.append(f"- {p_label}：预期 {direction}，变动幅度 {change_pct:+.1f}%，置信度 {p_conf}")
            lines.append("")

            # 操作建议
            act = ps.get('suggested_action') or {}
            if act:
                action = act.get('action', '持有')
                stop_loss = act.get('stop_loss', 'N/A')
                take_profit = act.get('take_profit', 'N/A')
                lines.append(f"操作建议：{action}，建议止损价 {stop_loss} 元，止盈价 {take_profit} 元。")
                lines.append("")

        # ---- 大师决策 (当用户选择了最终决策人时) ----
        ps = analysis_result.get('prediction_summary', {}) or {}
        cio = ps.get('cio_decision') or {}
        master_name = cio.get('master_name', '')
        if master_name:
            lines.append("## 最终决策人分析")
            lines.append(f"用户选择了 **{master_name}** 作为最终决策者。以下是 {master_name} 基于投资团队报告的最终裁决：")
            lines.append("")
            lines.append(f"### {master_name} 的核心结论")
            lines.append(cio.get('decision_summary', 'N/A'))
            lines.append("")

            # 证据链
            evidence = cio.get('evidence_chain', [])
            if evidence:
                lines.append("### 决策依据")
                for e in evidence:
                    lines.append(f"- {e}")
                lines.append("")

            # 三情景
            base = cio.get('base_case') or {}
            bull = cio.get('bull_case') or {}
            bear = cio.get('bear_case') or {}
            if base or bull or bear:
                lines.append("### 情景分析")
                for label, case in [('基准情景', base), ('乐观情景', bull), ('悲观情景', bear)]:
                    if case:
                        prob = case.get('probability', 0)
                        prob_pct = round(prob * 100) if isinstance(prob, float) else prob
                        lines.append(f"- **{label}** ({prob_pct}%): "
                                   f"{case.get('direction', 'N/A')} → 目标价 {case.get('target', 'N/A')}元")
                lines.append("")

            # 操作指令
            order = cio.get('order') or {}
            if order:
                action = order.get('action', 'N/A')
                position = order.get('position_size_pct', 0)
                entry = order.get('entry_conditions', '')
                sl = order.get('stop_loss', {}) or {}
                tp = order.get('take_profit', {}) or {}
                lines.append("### 操作指令")
                lines.append(f"- 操作: **{action}**  |  建议仓位: {position}%")
                if entry:
                    lines.append(f"- 入场条件: {entry}")
                if sl:
                    lines.append(f"- 止损: {sl.get('level', 'N/A')}元 ({sl.get('type', 'N/A')})")
                if tp:
                    tp1 = tp.get('level_1', tp) if isinstance(tp, dict) else tp
                    tp2 = tp.get('level_2', '') if isinstance(tp, dict) else ''
                    lines.append(f"- 止盈: {tp1}" + (f' / {tp2}' if tp2 else '') + '元')
                lines.append("")

            # 风险监控
            monitor = cio.get('risk_monitoring', [])
            if monitor:
                lines.append("### 风险监控指标")
                for rm in monitor:
                    lines.append(f"- 若 {rm.get('trigger', '')} → {rm.get('action', '')}")
                lines.append("")

            dq = cio.get('decision_quality') or {}
            if dq:
                conf = dq.get('confidence', '-')
                review = dq.get('next_review', '-')
                lines.append(f"决策置信度: {conf}  |  下次回顾: {review}")
                lines.append("")

            # 员工报告摘要
            emp_reports = ps.get('employee_reports', [])
            if emp_reports:
                lines.append("### 投资团队部门报告摘要")
                for r in emp_reports:
                    err = ' [⚠ 报告生成失败]' if r.get('error') else ''
                    lines.append(f"- [{r.get('department', '?')}] {r.get('role', '?')}: "
                               f"{r.get('outlook', '?')} (score={r.get('score', 0):+.1f}, conf={r.get('confidence', '?')}){err}")
                lines.append("")

        # ---- 7 位专业 Agent 角色定义 ----
        lines.append("## 市场参与者")

        # Agent 1: 沃伦·巴菲特的代理人
        lines.append(f"沃伦·巴菲特的代理人正在关注{name}。巴菲特的价值投资理念强调以合理的价格寻找优秀的公司，"
                     f"拥有持久竞争优势和优秀管理层的企业。该代理人正在评估{name}是否具备\"经济护城河\"，"
                     f"以及当前价格是否提供了足够的安全边际。")

        # Agent 2: 查理·芒格的经纪人
        lines.append(f"查理·芒格的经纪人与巴菲特一道审视{name}。芒格更强调\"以合理的价格收购优质企业\"，"
                     f"关注企业的长期竞争格局和管理层的诚信与能力。他认为投资是\"等待最佳机会的艺术\"。")

        # Agent 3: 估值代理
        eps = fs.get('eps', 'N/A')
        pe_val = quote.get('pe', 'N/A')
        lines.append(f"估值代理正在计算{name}的内在价值。基于 DCF 模型、PE 估值分位、历史均值回归等工具，"
                     f"该股票当前 PE 为 {pe_val}，每股收益 EPS 为 {eps}。估值代理将综合多个估值模型给出公允价值区间，"
                     f"并与当前市场价格对比，生成买入/卖出/持有信号。")

        # Agent 4: 情绪代理
        news_count = len(news or []) + len(search_news or [])
        guba_count = len(guba or [])
        lines.append(f"情绪代理正在分析市场对{name}的整体情绪。通过监控新闻报道（{news_count}条）、"
                     f"股吧讨论（{guba_count}条）和社交媒体热度，判断市场是过度乐观还是过度悲观。"
                     f"情绪代理使用自然语言处理技术提取市场情绪倾向，生成基于情绪的反向或顺势交易信号。")

        # Agent 5: 基本面分析师
        roe_val = fs.get('roe', 'N/A')
        rev = fs.get('revenue', 'N/A')
        profit = fs.get('net_profit', 'N/A')
        gm = fs.get('gross_margin', 'N/A')
        dr = fs.get('debt_ratio', 'N/A')
        lines.append(f"基本面分析师正在深入研究{name}的财务健康度。最新数据显示 ROE 为 {roe_val}%，"
                     f"营收 {rev} 亿元，净利润 {profit} 亿元，毛利率 {gm}%，资产负债率 {dr}%。"
                     f"分析师关注盈利增长趋势、现金流质量和竞争优势的可持续性，"
                     f"结合行业对比和宏观经济环境，给出基本面维度的投资评级。")

        # Agent 6: 技术分析师
        rsi = ti.get('rsi_14', 'N/A')
        ma5 = ti.get('ma5', 'N/A')
        ma20 = ti.get('ma20', 'N/A')
        boll_l = ti.get('boll_lower', 'N/A')
        boll_u = ti.get('boll_upper', 'N/A')
        lines.append(f"技术分析师正在分析{name}的价格走势图表。RSI(14)为 {rsi}，"
                     f"MA5={ma5}，MA20={ma20}，布林带下轨 {boll_l}，上轨 {boll_u}。"
                     f"技术分析师综合运用趋势跟踪、动量指标、支撑阻力位和成交量分析，"
                     f"识别关键的价格模式和潜在的转折点，生成基于技术面的交易信号。")

        # Agent 7: 风险经理
        price = quote.get('price', 'N/A')
        vol_ratio = ti.get('volume_ratio', 'N/A')
        lines.append(f"风险经理正在评估持有{name}的风险敞口。基于当前股价 {price} 元、波动率指标和量比 {vol_ratio}，"
                     f"计算 VaR（在险价值）、最大回撤和仓位风险。风险经理设定持仓限额、止损位和风险调整后的回报目标，"
                     f"确保投资组合的整体风险在可接受范围内。")

        lines.append("")

        # 新闻和舆情
        all_news = (news or []) + (search_news or [])
        if all_news:
            lines.append("## 近期新闻与舆情")
            for n in all_news[:8]:
                title = n.get('title', '')
                source = n.get('source', '未知来源')
                if title:
                    lines.append(f"- 据{source}报道：{title}")
            lines.append("")

        if guba:
            lines.append("## 股吧讨论")
            lines.append("在东方财富股吧中，散户投资者正在热烈讨论该股。")
            for p in guba[:5]:
                title = p.get('title', '')
                author = p.get('author', '某股民')
                if title:
                    lines.append(f"- 股民{author}发帖称：{title}")
            lines.append("")

        # 实体关系显式说明（辅助 Zep 提取 7 个 Agent 类型）
        lines.append("## 实体关系总结")
        lines.append(f"[Entity] 沃伦·巴菲特的代理人(BuffettProxy) 正在使用价值投资策略评估 {name}(Company)。")
        lines.append(f"[Entity] 查理·芒格的经纪人(MungerProxy) 与 BuffettProxy 合作，共同分析 {name}(Company) 的长期竞争力。")
        lines.append(f"[Entity] 估值代理(ValuationAgent) 通过 DCF 和 PE 分位计算 {name}(Company) 的内在价值。")
        lines.append(f"[Entity] 情绪代理(SentimentAgent) 通过 NLP 分析市场对 {name}(Company) 的情绪倾向。")
        lines.append(f"[Entity] 基本面分析师(FundamentalAnalyst) 分析 {name}(Company) 的财务数据和增长前景。")
        lines.append(f"[Entity] 技术分析师(TechnicalAnalyst) 通过图表和技术指标分析 {name}(Company) 的价格趋势。")
        lines.append(f"[Entity] 风险经理(RiskManager) 评估 {name}(Company) 的投资风险并设定仓位限制。")
        lines.append(f"[Entity] 中国证监会(Regulator) 监管 {name}(Company) 的信息披露合规性。")
        lines.append(f"[Entity] 财经媒体(MediaOutlet) 发布了关于 {name}(Company) 的新闻报道。")

        return "\n".join(lines)

    @staticmethod
    def build_scenario_scenarios(analysis_result: Dict[str, Any], debug: bool = False) -> List[Dict[str, Any]]:
        """生成三种推演场景的参数。debug=True 时使用 2 个 Agent。"""
        signals = analysis_result.get('signals', {}) or {}
        base_score = signals.get('score', 0) or 0
        ps = analysis_result.get('prediction_summary', {}) or {}

        agent_count = 7  # 两种模式都是 7 Agent，轮数在 orchestrator 控制

        scenarios = [
            {
                "name": "base",
                "label": "基准场景",
                "description": "基于当前市场信号的自然演化",
                "sentiment_bias": round(base_score / 10, 2),
                "volatility": 0.3,
                "agent_count": agent_count,
            },
            {
                "name": "bull",
                "label": "乐观场景",
                "description": "假设利好消息催化",
                "sentiment_bias": min(0.8, round((base_score + 5) / 10, 2)),
                "volatility": 0.5,
                "agent_count": agent_count,
            },
            {
                "name": "bear",
                "label": "悲观场景",
                "description": "假设利空打击",
                "sentiment_bias": max(-0.8, round((base_score - 5) / 10, 2)),
                "volatility": 0.6,
                "agent_count": agent_count,
            },
        ]
        return scenarios
