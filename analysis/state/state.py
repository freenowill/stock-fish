"""
StockFish 分析 Agent 状态定义
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class AnalysisState:
    """一次分析任务的完整状态"""
    symbol: str
    stock_name: str = ""
    status: str = "pending"
    error: Optional[str] = None
    created_at: str = ""
    completed_at: Optional[str] = None

    # 各阶段输出
    quote: Optional[Dict] = None
    technical_indicators: Optional[Dict] = None
    financial_summary: Optional[Dict] = None
    news: List[Dict] = field(default_factory=list)
    guba_posts: List[Dict] = field(default_factory=list)
    sentiment_news: Optional[Dict] = None
    sentiment_guba: Optional[Dict] = None
    signals: Optional[Dict] = None

    # LLM 预测输出
    llm_analysis: Optional[str] = None
    prediction_summary: Optional[Dict] = None
    risk_factors: List[Dict] = field(default_factory=list)
    price_target: Optional[Dict] = None

    # 估值分析
    valuation_level: str = ""          # 很低/偏低/正常/偏高/很高
    valuation_percentile: float = 0.0  # PE 历史分位数
    suggested_buy_price: float = 0.0   # 建议买入价
    historical_pe_avg: float = 0.0     # 历史平均 PE

    # 重要新闻/股吧摘要
    important_bullish_news: List[Dict] = field(default_factory=list)  # 利好新闻
    important_bearish_news: List[Dict] = field(default_factory=list)  # 利空新闻
    important_bullish_guba: List[Dict] = field(default_factory=list)  # 利好股吧
    important_bearish_guba: List[Dict] = field(default_factory=list)  # 利空股吧

    def to_dict(self) -> dict:
        d = asdict(self)
        d['created_at'] = self.created_at or datetime.now().isoformat()
        return d

    def mark_complete(self):
        self.status = "complete"
        self.completed_at = datetime.now().isoformat()

    def mark_error(self, msg: str):
        self.status = "error"
        self.error = msg
        self.completed_at = datetime.now().isoformat()
