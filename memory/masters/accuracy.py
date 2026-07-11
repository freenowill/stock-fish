"""
准确率计算工具函数

提供验证脚本和报表使用。
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from loguru import logger

from memory.masters.master_track import MasterTrackDB
from memory.stocks.stock_library import StockLibrary


def calculate_accuracy(master_key: str) -> Dict:
    """计算某位大师的准确率"""
    db = MasterTrackDB()
    return db.get_accuracy(master_key)


def accuracy_by_master() -> Dict[str, Dict]:
    """计算所有大师的准确率"""
    MASTERS = ["graham", "buffett", "fisher", "lynch", "templeton", "soros", "dalio"]
    result = {}
    for master_key in MASTERS:
        result[master_key] = calculate_accuracy(master_key)
    return result


def accuracy_by_symbol(master_key: str) -> Dict[str, Dict]:
    """按股票统计某位大师的准确率"""
    db = MasterTrackDB()
    records = db._load_records(master_key)

    symbol_stats: Dict[str, Dict] = {}
    for r in records:
        sym = r.get("symbol", "")
        if not sym:
            continue
        if sym not in symbol_stats:
            symbol_stats[sym] = {"total": 0, "correct": 0, "wrong": 0}
        if r.get("was_correct_short") is not None:
            symbol_stats[sym]["total"] += 1
            if r["was_correct_short"]:
                symbol_stats[sym]["correct"] += 1
            else:
                symbol_stats[sym]["wrong"] += 1

    result = {}
    for sym, stats in symbol_stats.items():
        total = stats["total"]
        result[sym] = {
            "total": total,
            "correct": stats["correct"],
            "wrong": stats["wrong"],
            "accuracy": round(stats["correct"] / total * 100, 1) if total > 0 else None,
        }
    return result


def _get_price_at_date(symbol: str, target_date: str) -> Optional[float]:
    """
    从 StockLibrary 历史快照获取某日收盘价。
    Args:
        symbol: 股票代码
        target_date: "2026-07-25"
    Returns:
        收盘价，无数据返回 None
    """
    # 先从 StockLibrary 的每日快照读
    price = StockLibrary().get_price_at_date(symbol, target_date)
    if price is not None:
        logger.debug(f"验证价格 [{symbol}@{target_date}]: {price} (来自 StockLibrary)")
        return price

    # 快照中没有 → 尝试从 provider 获取历史日线
    try:
        from market_data.a_stock_provider import AStockProvider
        provider = AStockProvider()
        hist = provider.get_historical(symbol, days=365)
        if hist:
            # hist 是 list of dict/Quote，找 target_date 那天的收盘价
            target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
            for h in hist:
                h_date = None
                if hasattr(h, 'date'):
                    h_date = h.date
                elif isinstance(h, dict):
                    h_date = h.get('date') or h.get('trade_date')
                if h_date:
                    if isinstance(h_date, str):
                        h_date = h_date[:10]
                    if h_date == target_date or h_date == target_dt:
                        price = h.close if hasattr(h, 'close') else h.get('close') if isinstance(h, dict) else None
                        if price:
                            return float(price)
    except Exception as e:
        logger.debug(f"验证价格 fallback 失败 [{symbol}@{target_date}]: {e}")

    return None


def verify_predictions(symbol: Optional[str] = None) -> List[str]:
    """
    检查待验证的记录，更新实际结果。
    从 StockLibrary 历史快照读取实际价格，比较预测方向是否正确。

    Args:
        symbol: 若指定，只验证该股票的相关记录；否则验证所有

    Returns:
        已更新的记录 ID 列表
    """
    db = MasterTrackDB()
    unverified = db.get_unverified_records()

    if not unverified:
        return []

    updated = []
    for record in unverified:
        try:
            rec_symbol = record.get("symbol", "")
            if symbol and rec_symbol != symbol:
                continue

            analysis_date_str = record.get("analysis_timestamp", "")[:10]
            if not analysis_date_str:
                continue

            analysis_date = datetime.strptime(analysis_date_str, "%Y-%m-%d")
            now = datetime.now()
            price_at = record.get("price_at_analysis")

            if not price_at or price_at <= 0:
                continue

            days_since = (now - analysis_date).days

            # ── 短周期验证 (14 天) ──
            if record["was_correct_short"] is None and days_since >= 14:
                check_date = (analysis_date + timedelta(days=14)).strftime("%Y-%m-%d")
                actual_price = _get_price_at_date(rec_symbol, check_date)
                if actual_price and actual_price > 0:
                    actual_change = (actual_price - price_at) / price_at * 100
                    pred = record.get("short_term_pred", {}) or {}
                    pred_change = pred.get("change_pct", 0) or 0
                    same_direction = (actual_change > 0) == (pred_change > 0)
                    db.update_outcome(record["id"], "short", round(actual_change, 2), same_direction)
                    updated.append(record["id"])
                    logger.info(f"验证短周期 [{rec_symbol}] {record['id'][:20]}: "
                               f"pred={pred_change:+.1f}% actual={actual_change:+.1f}% "
                               f"{'✓' if same_direction else '✗'}")

            # ── 中周期验证 (3 个月) ──
            if record["was_correct_mid"] is None and days_since >= 90:
                check_date = (analysis_date + timedelta(days=90)).strftime("%Y-%m-%d")
                actual_price = _get_price_at_date(rec_symbol, check_date)
                if actual_price and actual_price > 0:
                    actual_change = (actual_price - price_at) / price_at * 100
                    pred = record.get("mid_term_pred", {}) or {}
                    pred_change = pred.get("change_pct", 0) or 0
                    same_direction = (actual_change > 0) == (pred_change > 0)
                    db.update_outcome(record["id"], "mid", round(actual_change, 2), same_direction)
                    updated.append(record["id"])
                    logger.info(f"验证中周期 [{rec_symbol}] {record['id'][:20]}: "
                               f"pred={pred_change:+.1f}% actual={actual_change:+.1f}% "
                               f"{'✓' if same_direction else '✗'}")

            # ── 长周期验证 (12 个月) ──
            if record["was_correct_long"] is None and days_since >= 365:
                check_date = (analysis_date + timedelta(days=365)).strftime("%Y-%m-%d")
                actual_price = _get_price_at_date(rec_symbol, check_date)
                if actual_price and actual_price > 0:
                    actual_change = (actual_price - price_at) / price_at * 100
                    pred = record.get("long_term_pred", {}) or {}
                    pred_change = pred.get("change_pct", 0) or 0
                    same_direction = (actual_change > 0) == (pred_change > 0)
                    db.update_outcome(record["id"], "long", round(actual_change, 2), same_direction)
                    updated.append(record["id"])
                    logger.info(f"验证长周期 [{rec_symbol}] {record['id'][:20]}: "
                               f"pred={pred_change:+.1f}% actual={actual_change:+.1f}% "
                               f"{'✓' if same_direction else '✗'}")

        except Exception as e:
            logger.warning(f"验证失败 [{record.get('id', '?')}]: {e}")

    return updated
