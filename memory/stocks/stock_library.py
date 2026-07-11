"""
StockLibrary — 按股票分类的持久化数据存储

每只股票一个子目录 memory/stocks/{symbol}/，包含:
  - meta.json          股票元信息
  - history/{date}.json 每日数据快照
  - financials/{}.json  季度财务数据
  - sentiment/{}.json   每日情绪快照
  - news/index.json     新闻索引（去重）
"""
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from memory import STOCKS_DIR, MEMORY_ROOT


class StockLibrary:
    """股票数据仓库 — 按股票分类的持久化存储"""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or STOCKS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ── 公开 API ──

    def update(self, symbol: str, state: Dict[str, Any]) -> None:
        """从 AnalysisState 提取数据，更新股票仓库"""
        try:
            meta = self._load_meta(symbol)
            stock_name = state.get("stock_name", "") or state.get("symbol", symbol)

            # 更新 meta
            now_iso = datetime.now().isoformat()
            if not meta.get("first_analyzed"):
                meta["first_analyzed"] = now_iso
            meta["symbol"] = symbol
            meta["name"] = stock_name
            meta["industry"] = state.get("industry_context", {}).get("industry") if isinstance(state.get("industry_context"), dict) else meta.get("industry", "")
            meta["last_analyzed"] = now_iso
            meta["analysis_count"] = meta.get("analysis_count", 0) + 1
            meta["last_analysis_timestamp"] = now_iso
            self._save_meta(symbol, meta)

            # 保存每日历史快照
            self._save_history(symbol, state)

            # 保存季度财务数据
            self._save_financials(symbol, state)

            # 保存每日情绪快照
            self._save_sentiment(symbol, state)

            # 保存新闻索引
            self._save_news_index(symbol, state)

            logger.debug(f"StockLibrary 更新完成: {symbol} ({meta['analysis_count']} 次分析)")

        except Exception as e:
            logger.warning(f"StockLibrary 更新失败 [{symbol}]: {e}")

    def get_meta(self, symbol: str) -> Dict[str, Any]:
        """获取股票元信息"""
        return self._load_meta(symbol)

    def get_price_at_date(self, symbol: str, target_date: str) -> Optional[float]:
        """
        获取某只股票在指定日期的收盘价。
        从 history 快照中读取，无快照时返回 None。
        Args:
            symbol: 股票代码
            target_date: 日期字符串 "2026-07-11"
        Returns:
            收盘价，或 None
        """
        history_dir = self._symbol_dir(symbol) / "history"
        if not history_dir.exists():
            return None
        path = history_dir / f"{target_date}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("price") or data.get("close") or data.get("prev_close")
        except Exception:
            return None

    def get_history(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的历史快照"""
        history_dir = self._symbol_dir(symbol) / "history"
        if not history_dir.exists():
            return []
        files = sorted(history_dir.glob("*.json"), reverse=True)
        result = []
        for f in files[:limit]:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    result.append(json.load(fh))
            except Exception:
                continue
        return result

    def list_symbols(self) -> List[str]:
        """列出所有已跟踪的股票代码"""
        if not self.base_dir.exists():
            return []
        return sorted(
            d.name for d in self.base_dir.iterdir()
            if d.is_dir() and (d / "meta.json").exists()
            and not d.name.startswith('__')
        )

    # ── 内部方法 ──

    def _symbol_dir(self, symbol: str) -> Path:
        return self.base_dir / symbol

    def _load_meta(self, symbol: str) -> Dict[str, Any]:
        path = self._symbol_dir(symbol) / "meta.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"symbol": symbol, "name": "", "industry": "", "analysis_count": 0}

    def _save_meta(self, symbol: str, meta: Dict[str, Any]) -> None:
        path = self._symbol_dir(symbol) / "meta.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, default=str, indent=2)
        tmp.rename(path)

    def _save_history(self, symbol: str, state: Dict[str, Any]) -> None:
        """保存每日行情+技术指标快照"""
        quote = state.get("quote") or {}
        ti = state.get("technical_indicators") or {}
        if not quote:
            return

        date = datetime.now().strftime("%Y-%m-%d")
        record = {
            "date": date,
            "symbol": symbol,
            "price": quote.get("price"),
            "change_pct": quote.get("change_pct"),
            "volume": quote.get("volume"),
            "amount": quote.get("amount"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "open": quote.get("open"),
            "prev_close": quote.get("prev_close"),
            "pe": quote.get("pe"),
            "pb": quote.get("pb"),
            "market_cap": quote.get("market_cap"),
            "turnover_rate": quote.get("turnover_rate"),
            "ma5": ti.get("ma5"),
            "ma10": ti.get("ma10"),
            "ma20": ti.get("ma20"),
            "ma60": ti.get("ma60"),
            "rsi_14": ti.get("rsi_14"),
            "macd_hist": ti.get("macd_hist"),
            "boll_lower": ti.get("boll_lower"),
            "boll_upper": ti.get("boll_upper"),
            "volume_ratio": ti.get("volume_ratio"),
            "amplitude": ti.get("amplitude"),
            "saved_at": datetime.now().isoformat(),
        }
        # 清理 None 字段
        record = {k: v for k, v in record.items() if v is not None}

        history_dir = self._symbol_dir(symbol) / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        path = history_dir / f"{date}.json"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, default=str, indent=2)
        tmp.rename(path)

    def _save_financials(self, symbol: str, state: Dict[str, Any]) -> None:
        """保存季度财务数据"""
        fs = state.get("financial_summary") or {}
        if not fs:
            return

        # 从财务摘要中推算季度标签
        quarter = datetime.now().strftime("%Y") + f"Q{(datetime.now().month - 1) // 3 + 1}"
        record = {
            "quarter": quarter,
            "report_date": datetime.now().strftime("%Y-%m-%d"),
            "revenue": fs.get("revenue"),
            "revenue_yoy": fs.get("revenue_yoy"),
            "net_profit": fs.get("net_profit"),
            "net_profit_yoy": fs.get("net_profit_yoy"),
            "eps": fs.get("eps"),
            "roe": fs.get("roe"),
            "gross_margin": fs.get("gross_margin"),
            "debt_ratio": fs.get("debt_ratio"),
            "operating_cash_flow": fs.get("operating_cash_flow"),
            "free_cash_flow": fs.get("free_cash_flow"),
            "dividend_per_share": fs.get("dividend_per_share"),
            "dividend_yield": fs.get("dividend_yield"),
            "saved_at": datetime.now().isoformat(),
        }
        record = {k: v for k, v in record.items() if v is not None}

        fin_dir = self._symbol_dir(symbol) / "financials"
        fin_dir.mkdir(parents=True, exist_ok=True)
        path = fin_dir / f"{quarter}.json"
        if path.exists():
            return  # 同一季度不覆盖
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, default=str, indent=2)
        tmp.rename(path)

    def _save_sentiment(self, symbol: str, state: Dict[str, Any]) -> None:
        """保存每日情绪快照"""
        sn = state.get("sentiment_news") or {}
        sg = state.get("sentiment_guba") or {}
        if not sn and not sg:
            return

        date = datetime.now().strftime("%Y-%m-%d")
        record = {
            "date": date,
            "symbol": symbol,
            "news_count": sn.get("count", 0) if isinstance(sn, dict) else 0,
            "news_avg_score": sn.get("avg_score") if isinstance(sn, dict) else None,
            "news_positive_ratio": sn.get("positive_ratio") if isinstance(sn, dict) else None,
            "news_negative_ratio": sn.get("negative_ratio") if isinstance(sn, dict) else None,
            "guba_count": sg.get("count", 0) if isinstance(sg, dict) else 0,
            "guba_avg_score": sg.get("avg_score") if isinstance(sg, dict) else None,
            "guba_positive_ratio": sg.get("positive_ratio") if isinstance(sg, dict) else None,
            "guba_negative_ratio": sg.get("negative_ratio") if isinstance(sg, dict) else None,
            "sentiment_percentile": state.get("sentiment_percentile"),
            "saved_at": datetime.now().isoformat(),
        }
        record = {k: v for k, v in record.items() if v is not None}

        sent_dir = self._symbol_dir(symbol) / "sentiment"
        sent_dir.mkdir(parents=True, exist_ok=True)
        path = sent_dir / f"{date}.json"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, default=str, indent=2)
        tmp.rename(path)

    def _save_news_index(self, symbol: str, state: Dict[str, Any]) -> None:
        """保存新闻索引（去重）"""
        news_list = state.get("news") or []
        if not news_list:
            return

        idx_path = self._symbol_dir(symbol) / "news" / "index.json"
        idx_path.parent.mkdir(parents=True, exist_ok=True)

        # 读取现有索引
        existing = {"articles": []}
        if idx_path.exists():
            try:
                with open(idx_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                pass

        seen_titles = set()
        for a in existing.get("articles", []):
            t = a.get("title", "")
            if t:
                seen_titles.add(t)

        # 添加新条目
        new_articles = []
        for n in news_list:
            title = n.get("title", "")
            if title and title not in seen_titles:
                seen_titles.add(title)
                new_articles.append({
                    "title": title,
                    "date": n.get("date", datetime.now().strftime("%Y-%m-%d")),
                    "source": n.get("source", ""),
                    "url": n.get("url", ""),
                    "saved_at": datetime.now().isoformat(),
                })

        if not new_articles:
            return

        existing["articles"].extend(new_articles)
        existing["symbol"] = symbol
        existing["last_updated"] = datetime.now().isoformat()
        existing["article_count"] = len(existing["articles"])

        tmp = idx_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, default=str, indent=2)
        tmp.rename(idx_path)
        logger.debug(f"StockLibrary 新闻索引 [{symbol}]: 新增 {len(new_articles)} 条")
