"""
Seed Document Builder

将 StockEngine 分析结果转换为 MiroFish 兼容的种子文档，
供 Zep GraphRAG 构建知识图谱。

使用自然语言叙事结构，便于 Zep 提取实体和关系。
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

        # 市场参与者（叙事段落用于实体提取）
        lines.append("## 市场参与者")
        total_score = abs(score) if score else 3
        bullish_count = max(1, int(total_score + 3))
        bearish_count = max(1, int(5 - total_score)) if total_score < 5 else 1
        total_investors = bullish_count + bearish_count + max(1, 10 - bullish_count - bearish_count)

        lines.append(f"市场上约有 {total_investors} 位投资者关注 {name}。看多方约 {bullish_count} 人，"
                     f"认为技术面和基本面积极，预期股价上涨。看空方约 {bearish_count} 人，"
                     f"认为存在技术面风险和估值回调压力。另有部分投资者持中性观望态度。")
        lines.append(f"多位证券分析师（FinancialAnalyst）发布了研究报告，机构投资者（InstitutionalInvestor）"
                     f"正在密切关注该股票。财经媒体（MediaOutlet）对此进行了报道，"
                     f"社交媒体上的财经博主（SocialMediaInfluencer）也在讨论该股走势。"
                     f"中国证监会（Regulator）负责监管该公司的信息披露。")
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

        # 实体关系显式说明（辅助Zep提取）
        lines.append("## 实体关系总结")
        lines.append(f"[Entity] {name}(Company) 是上海证券交易所(StockExchange)的上市公司。")
        lines.append(f"[Entity] 机构投资者(InstitutionalInvestor) 和分析师(Analyst) 正在分析 {name} 的财务报表。")
        lines.append(f"[Entity] 财经媒体(MediaOutlet) 发布了关于 {name} 的新闻报道。")
        lines.append(f"[Entity] 散户投资者(Investor) 在东方财富股吧讨论 {name} 的股价走势。")
        lines.append(f"[Entity] 中国证监会(Regulator) 监管 {name} 的信息披露合规性。")
        lines.append(f"[Entity] 社交媒体财经博主(SocialMediaInfluencer) 发表了关于 {name} 投资价值的观点。")

        return "\n".join(lines)

    @staticmethod
    def build_scenario_scenarios(analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成三种推演场景的参数"""
        signals = analysis_result.get('signals', {}) or {}
        base_score = signals.get('score', 0) or 0
        ps = analysis_result.get('prediction_summary', {}) or {}

        # 基准场景：当前信号方向
        scenarios = [
            {
                "name": "base",
                "label": "基准场景",
                "description": "基于当前市场信号的自然演化",
                "sentiment_bias": round(base_score / 10, 2),  # -1.0 ~ 1.0
                "volatility": 0.3,
                "agent_count": 15,
            },
            {
                "name": "bull",
                "label": "乐观场景",
                "description": "假设利好消息催化",
                "sentiment_bias": min(0.8, round((base_score + 5) / 10, 2)),
                "volatility": 0.5,
                "agent_count": 20,
            },
            {
                "name": "bear",
                "label": "悲观场景",
                "description": "假设利空打击",
                "sentiment_bias": max(-0.8, round((base_score - 5) / 10, 2)),
                "volatility": 0.6,
                "agent_count": 20,
            },
        ]
        return scenarios
