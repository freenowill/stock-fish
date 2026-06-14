"""
舆情情感采集器
复用 BettaFish 的多语言情感分析模型，对新闻/股吧内容进行情感打分
"""
import os
import sys
import json
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from loguru import logger


# ========== 数据结构 ==========

@dataclass
class SentimentScore:
    """单条文本的情感得分"""
    text: str
    label: str            # 非常负面 / 负面 / 中性 / 正面 / 非常正面
    confidence: float     # 置信度 0-1
    score: float          # 综合得分 -1.0 ~ 1.0
    success: bool = True


@dataclass
class SentimentSummary:
    """情感分析汇总"""
    total_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    avg_score: float           # 平均得分 -1.0 ~ 1.0
    positive_ratio: float      # 正面比例 0-1
    negative_ratio: float      # 负面比例 0-1
    scores: List[SentimentScore] = field(default_factory=list)


# ========== 情感分析器 ==========

class SentimentCollector:
    """
    舆情情感采集器。
    复用 BettaFish 的 WeiboMultilingualSentimentAnalyzer 进行 5 级情感分析。
    """

    def __init__(self, enable_sentiment: bool = True):
        self._analyzer = None
        self._initialized = False
        self._init_failed = False      # 初始化失败后不再重试，避免日志刷屏
        self._hf_checked = False
        self._hf_reachable = False
        self.enable_sentiment = enable_sentiment

    def _initialize(self):
        """延迟初始化情感分析模型（失败则用规则降级，仅尝试一次）"""
        if self._initialized:
            return True
        if self._init_failed:
            return False
        if not self.enable_sentiment:
            self._init_failed = True
            return False

        # 先检查 HuggingFace 是否可达（只检查一次），避免模型下载阻塞
        if not self._is_hf_available():
            logger.warning("HuggingFace 不可达，情感分析使用规则降级")
            self._init_failed = True
            return False

        try:
            _bf_path = str(Path(__file__).resolve().parent.parent.parent / "BettaFish")
            if _bf_path not in sys.path:
                sys.path.insert(0, _bf_path)

            module_path = Path(_bf_path) / "InsightEngine" / "tools" / "sentiment_analyzer.py"
            if not module_path.exists():
                logger.warning(f"BettaFish 情感模型未找到 ({module_path})，使用规则降级")
                self._init_failed = True
                return False

            spec = importlib.util.spec_from_file_location(
                "bettafish_sentiment", module_path,
                submodule_search_locations=[]
            )
            if spec is None or spec.loader is None:
                logger.warning("无法加载情感分析模块，使用规则降级")
                self._init_failed = True
                return False

            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._analyzer = mod.WeiboMultilingualSentimentAnalyzer()
            success = self._analyzer.initialize()
            if success:
                self._initialized = True
                logger.info("情感分析模型初始化成功")
            else:
                logger.warning("情感分析模型初始化失败，使用规则降级")
                self._init_failed = True
            return success
        except Exception as e:
            logger.warning(f"情感分析模型加载失败，使用规则降级: {e}")
            self._init_failed = True
            return False

    def _is_hf_available(self) -> bool:
        """缓存式检查 HuggingFace 是否可达（只检查一次）"""
        if self._hf_checked:
            return self._hf_reachable
        self._hf_checked = True
        import socket
        try:
            sock = socket.create_connection(("huggingface.co", 443), timeout=2)
            sock.close()
            self._hf_reachable = True
        except (OSError, socket.timeout):
            self._hf_reachable = False
        return self._hf_reachable

    def analyze_text(self, text: str) -> SentimentScore:
        """分析单条文本情感 — 混合策略: 模型 + 关键词规则互补"""
        if not text or not text.strip():
            return SentimentScore(text=text, label='中性', confidence=1.0, score=0.0)

        model_result = None
        # 尝试使用模型分析
        if self._initialize():
            try:
                result = self._analyzer.analyze_single_text(text)
                label = result.sentiment_label
                confidence = result.confidence
                score = self._label_to_score(label, confidence)
                model_result = SentimentScore(
                    text=text[:200],
                    label=label,
                    confidence=confidence,
                    score=score,
                )
            except Exception as e:
                logger.debug(f"情感分析失败，使用规则降级: {e}")

        # 规则降级: 基于关键词的简单情感判断（始终运行，与模型互补）
        rule_result = self._rule_based_sentiment(text)

        # 混合决策: 取信号更强的结果
        # 模型对中文财经新闻标题常返回中性(score=0.0)，而规则能识别关键词
        if model_result is None:
            return rule_result
        if abs(rule_result.score) > abs(model_result.score):
            return rule_result
        # 模型有非中性信号时优先用模型（置信度更高）
        if abs(model_result.score) > 0.2:
            return model_result
        # 两者都弱信号时偏好规则的判断（对中文财经更敏感）
        return rule_result

    def analyze_batch(self, texts: List[str]) -> SentimentSummary:
        """批量分析情感"""
        scores = [self.analyze_text(t) for t in texts]
        return self._build_summary(scores)

    def analyze_news(self, news_list: List[Dict[str, str]]) -> SentimentSummary:
        """分析新闻列表情感"""
        texts = []
        for item in news_list:
            title = item.get('title', '') or item.get('Title', '')
            # 优先用内容，没有则用标题
            content = item.get('content', '') or item.get('Content', '')
            texts.append(content or title)

        return self.analyze_batch(texts)

    def analyze_guba_posts(self, posts: List[Any]) -> SentimentSummary:
        """分析股吧帖子情感"""
        texts = []
        for post in posts:
            if hasattr(post, 'title'):
                texts.append(post.title)
            elif isinstance(post, dict):
                texts.append(post.get('title', ''))
            else:
                texts.append(str(post))

        return self.analyze_batch(texts)

    # ---- 内部工具 ----

    @staticmethod
    def _label_to_score(label: str, confidence: float) -> float:
        """将 5 级标签映射到 -1 ~ 1 得分"""
        mapping = {
            '非常正面': 0.9,
            '正面': 0.5,
            '中性': 0.0,
            '负面': -0.5,
            '非常负面': -0.9,
        }
        base = mapping.get(label, 0.0)
        return round(base * confidence, 4)

    @staticmethod
    def _rule_based_sentiment(text: str) -> SentimentScore:
        """规则降级：基于关键词的简单情感判断"""
        text_lower = text.lower()

        positive_words = ['涨', '涨停', '利好', '突破', '拉升', '买入', '推荐',
                          '增长', '盈利', '看好', '牛市', '反弹', '放量', '强势',
                          '降准', '降息', '放水', '宽松', '复苏',
                          'up', 'bullish', 'buy', 'positive', 'growth', 'rally']
        negative_words = ['跌', '跌停', '利空', '暴跌', '减持', '卖出', '风险',
                          '亏损', '暴雷', '崩盘', '熊市', '破位', '缩量', '弱势',
                          '加息', '收紧', '通缩', '违约', '退市', 'st', 'st\n*',
                          'down', 'bearish', 'sell', 'negative', 'crash', 'risk']

        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        if pos_count > neg_count:
            label = '正面'
            confidence = min(0.5 + 0.1 * (pos_count - neg_count), 0.8)
            score = 0.5
        elif neg_count > pos_count:
            label = '负面'
            confidence = min(0.5 + 0.1 * (neg_count - pos_count), 0.8)
            score = -0.5
        else:
            label = '中性'
            confidence = 0.5
            score = 0.0

        return SentimentScore(text=text[:200], label=label, confidence=round(confidence, 4), score=score)

    @staticmethod
    def _build_summary(scores: List[SentimentScore]) -> SentimentSummary:
        total = len(scores)
        if total == 0:
            return SentimentSummary(
                total_count=0, positive_count=0, negative_count=0, neutral_count=0,
                avg_score=0.0, positive_ratio=0.0, negative_ratio=0.0,
            )

        positive = sum(1 for s in scores if s.score > 0.2)
        negative = sum(1 for s in scores if s.score < -0.2)
        neutral = total - positive - negative
        avg = sum(s.score for s in scores) / total

        return SentimentSummary(
            total_count=total,
            positive_count=positive,
            negative_count=negative,
            neutral_count=neutral,
            avg_score=round(avg, 4),
            positive_ratio=round(positive / total, 4),
            negative_ratio=round(negative / total, 4),
            scores=scores,
        )


# ========== 情感历史缓存（用于计算百分位）==========

class SentimentHistory:
    """
    情感历史缓存 — 记录每日情感分值，用于计算当前情感在历史中的百分位。

    数据存储: data/sentiment_history/{symbol}.json
    """
    HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'sentiment_history')

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._ensure_dir()
        self.history = self._load()

    def _ensure_dir(self):
        os.makedirs(self.HISTORY_DIR, exist_ok=True)

    def _path(self) -> str:
        return os.path.join(self.HISTORY_DIR, f'{self.symbol}.json')

    def _load(self) -> list:
        try:
            if os.path.exists(self._path()):
                with open(self._path(), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('history', [])
        except Exception as e:
            logger.debug(f"情感历史加载失败 [{self.symbol}]: {e}")
        return []

    def _save(self, history: list):
        try:
            with open(self._path(), 'w', encoding='utf-8') as f:
                json.dump({'symbol': self.symbol, 'history': history}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"情感历史保存失败 [{self.symbol}]: {e}")

    def record(self, date_str: str, news_avg: float = 0.0, guba_avg: float = 0.0,
               news_count: int = 0, guba_count: int = 0):
        """记录当日情感数据，同日期覆盖"""
        # 移除同日期旧记录
        self.history = [h for h in self.history if h.get('date') != date_str]
        self.history.append({
            'date': date_str,
            'news_avg': news_avg,
            'guba_avg': guba_avg,
            'news_count': news_count,
            'guba_count': guba_count,
        })
        # 仅保留最近365天
        self.history = sorted(self.history, key=lambda x: x['date'])[-365:]
        self._save(self.history)

    def get_percentile(self, current_news_avg: float = 0.0, current_guba_avg: float = 0.0) -> dict:
        """计算当前情感在历史中的百分位"""
        if len(self.history) < 5:
            return {'news_percentile': None, 'guba_percentile': None, 'total_days': len(self.history)}

        news_scores = sorted([h.get('news_avg', 0) for h in self.history if h.get('news_avg') is not None])
        guba_scores = sorted([h.get('guba_avg', 0) for h in self.history if h.get('guba_avg') is not None])

        def _pct(val, arr):
            if not arr:
                return None
            return round(sum(1 for v in arr if v <= val) / len(arr) * 100, 1)

        return {
            'news_percentile': _pct(current_news_avg, news_scores),
            'guba_percentile': _pct(current_guba_avg, guba_scores),
            'total_days': len(self.history),
        }
    collector = SentimentCollector()

    texts = [
        "贵州茅台突破1800元，创历史新高！",
        "今天跌停了，亏麻了...",
        "市场整体平稳运行，成交量温和放大",
    ]

    for t in texts:
        result = collector.analyze_text(t)
        print(f"[{result.score:>5.2f}] {result.label:<6s} ({result.confidence:.0%}) {t}")

    news = [
        {"title": "AI概念股集体爆发，科创50大涨3%", "content": "今日AI概念股全面走强..."},
        {"title": "美联储加息预期升温，亚太市场承压", "content": "受美联储鹰派言论影响..."},
    ]
    summary = collector.analyze_news(news)
    print(f"\n新闻情感汇总: 平均 {summary.avg_score:.2f}, 正面 {summary.positive_ratio:.0%}, 负面 {summary.negative_ratio:.0%}")
