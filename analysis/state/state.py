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

    # 用户输入
    cost_price: float = 0.0       # 用户成本价格
    shares: int = 0               # 持仓数量（股）
    total_assets: float = 0.0     # 总资产（元）
    available_cash: float = 0.0   # 可用资金（元）

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

    # 多周期预测
    short_term_pred: Optional[Dict] = None   # 短期 (1~2周): {direction, change_pct, confidence, reason}
    mid_term_pred: Optional[Dict] = None     # 中期 (1~3月): {direction, change_pct, confidence, reason}
    long_term_pred: Optional[Dict] = None    # 长期 (6~12月): {direction, change_pct, confidence, reason}
    suggested_action: Optional[Dict] = None  # {action, reason, stop_loss, take_profit}

    # 评分分解（新版 ScoringEngine）
    score_breakdown: Optional[Dict] = None

    # 估值分析
    valuation_level: str = ""
    valuation_percentile: Optional[float] = None  # None = 未计算/数据不可用
    suggested_buy_price: float = 0.0
    historical_pe_avg: Optional[float] = None     # None = 未计算/数据不可用

    # 重要新闻/股吧摘要
    important_bullish_news: List[Dict] = field(default_factory=list)
    important_bearish_news: List[Dict] = field(default_factory=list)
    important_bullish_guba: List[Dict] = field(default_factory=list)
    important_bearish_guba: List[Dict] = field(default_factory=list)

    # 宏观/行业上下文 (Phase 3 新增)
    macro_context: Optional[Dict] = None
    industry_context: Optional[Dict] = None

    # Web 搜索结果 (Phase 4 新增)
    search_results: Optional[Dict] = None  # {query, results: [{title, url, snippet, source}], summary}

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
