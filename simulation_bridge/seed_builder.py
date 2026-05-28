"""
Seed Document Builder

将 StockEngine 分析结果转换为 MiroFish 兼容的种子文档，
供 Zep GraphRAG 构建知识图谱。
"""
import json
from datetime import datetime
from typing import Dict, Any, List


class SeedDocumentBuilder:
    """分析结果 → MiroFish 种子文档"""

    @staticmethod
    def build(analysis_result: Dict[str, Any]) -> str:
        """将分析结果转换为结构化种子文档文本"""
        symbol = analysis_result.get('symbol', '')
        name = analysis_result.get('stock_name', analysis_result.get('name', ''))
        signals = analysis_result.get('signals', {}) or {}
        quote = analysis_result.get('quote', {}) or {}
        ti = analysis_result.get('technical_indicators', {}) or {}
        fs = analysis_result.get('financial_summary', {}) or {}
        ps = analysis_result.get('prediction_summary', {}) or {}
        news = analysis_result.get('news', []) or []
        guba = analysis_result.get('guba_posts', []) or []

        lines = []
        lines.append(f"===MARKET OVERVIEW===")
        lines.append(f"股票: {name}({symbol})")
        lines.append(f"当前价格: {quote.get('price', 'N/A')}")
        lines.append(f"涨跌幅: {quote.get('change_pct', 'N/A')}%")
        lines.append(f"PE: {quote.get('pe', 'N/A')}  PB: {quote.get('pb', 'N/A')}")
        lines.append(f"总市值: {quote.get('market_cap', 'N/A')}亿")
        lines.append(f"综合信号: {signals.get('overall', 'neutral')}(评分: {signals.get('score', 0)})")
        lines.append("")

        lines.append(f"===TECHNICAL INDICATORS===")
        lines.append(f"RSI(14): {ti.get('rsi_14', 'N/A')}")
        lines.append(f"MACD: {ti.get('macd_hist', 'N/A')}")
        lines.append(f"KDJ: {ti.get('kdj_k', 'N/A')}/{ti.get('kdj_d', 'N/A')}/{ti.get('kdj_j', 'N/A')}")
        lines.append(f"MA5/10/20: {ti.get('ma5', 'N/A')}/{ti.get('ma10', 'N/A')}/{ti.get('ma20', 'N/A')}")
        lines.append(f"布林通道: {ti.get('boll_lower', 'N/A')} ~ {ti.get('boll_middle', 'N/A')} ~ {ti.get('boll_upper', 'N/A')}")
        lines.append(f"量比: {ti.get('volume_ratio', 'N/A')}")
        lines.append("")

        lines.append(f"===FINANCIAL DATA===")
        lines.append(f"营收: {fs.get('revenue', 'N/A')}亿  净利: {fs.get('net_profit', 'N/A')}亿")
        lines.append(f"EPS: {fs.get('eps', 'N/A')}  ROE: {fs.get('roe', 'N/A')}%")
        lines.append(f"毛利率: {fs.get('gross_margin', 'N/A')}%  负债率: {fs.get('debt_ratio', 'N/A')}%")
        lines.append("")

        if ps:
            lines.append(f"===PREDICTION===")
            lines.append(f"展望: {ps.get('outlook', 'N/A')}  置信度: {ps.get('confidence', 'N/A')}")
            lines.append(f"预测区间: {ps.get('price_target_low', 'N/A')} ~ {ps.get('price_target_high', 'N/A')}")
            if ps.get('reason'):
                lines.append(f"核心逻辑: {ps['reason']}")
            lines.append("")

        lines.append(f"===KEY PLAYERS===")
        # 根据信号生成虚拟市场参与者
        total_score = abs(signals.get('score', 0))
        bullish_count = max(1, int(total_score + 5))
        bearish_count = max(1, int(5 - total_score)) if total_score < 5 else 1
        lines.append(f"- 看多方: {bullish_count}个投资者 (理由: 技术面/基本面/舆情积极)")
        lines.append(f"- 看空方: {bearish_count}个投资者 (理由: 技术面/基本面/舆情风险)")
        lines.append(f"- 中立/观望: {max(1, 10 - bullish_count - bearish_count)}个投资者")
        lines.append("")

        if news:
            lines.append(f"===RECENT NEWS===")
            for n in news[:5]:
                lines.append(f"- {n.get('title', '')}")
            lines.append("")

        if guba:
            lines.append(f"===MARKET SENTIMENT===")
            for p in guba[:5]:
                lines.append(f"- {p.get('title', '')}")
            lines.append("")

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
