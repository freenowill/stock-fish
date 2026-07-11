"""
准确率计算工具函数

提供给验证脚本和报表使用。
"""
from typing import Dict, List

from memory.masters.master_track import MasterTrackDB


def calculate_accuracy(master_key: str) -> Dict:
    """计算某位大师的准确率"""
    db = MasterTrackDB()
    return db.get_accuracy(master_key)


def accuracy_by_master() -> Dict[str, Dict]:
    """计算所有大师的准确率"""
    from analysis.agents.cio_prompts import MASTERS
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


def verify_predictions():
    """
    检查所有待验证的记录，更新实际结果（由定时任务/脚本调用）

    这是一个占位函数 — 实际的验证逻辑需要根据市场数据 API 补充。
    验证时机:
      - short: 分析后 14 天
      - mid: 分析后 3 个月
      - long: 分析后 12 个月
    """
    db = MasterTrackDB()
    unverified = db.get_unverified_records()

    if not unverified:
        return []

    from datetime import datetime, timedelta

    updated = []
    for record in unverified:
        try:
            analysis_date = datetime.fromisoformat(record["analysis_timestamp"][:19])
            now = datetime.now()
            price_at = record.get("price_at_analysis")

            if not price_at:
                continue

            days_since = (now - analysis_date).days

            # 短周期验证 (14 天)
            if record["was_correct_short"] is None and days_since >= 14:
                # TODO: 从市场数据 API 获取实际价格
                # actual_price = get_price_at_date(record["symbol"], analysis_date + timedelta(days=14))
                # actual_change = (actual_price - price_at) / price_at * 100
                # pred_change = record.get("short_term_pred", {}).get("change_pct", 0)
                # same_direction = (actual_change > 0) == (pred_change > 0)
                # db.update_outcome(record["id"], "short", actual_change, same_direction)
                # updated.append(record["id"])
                pass

        except Exception as e:
            logger = __import__("loguru").logger
            logger.warning(f"验证失败 [{record.get('id')}]: {e}")

    return updated


# 已知的大师列表（用于 accuracy_by_master）
MASTERS = [
    "graham", "buffett", "fisher", "lynch",
    "templeton", "soros", "dalio",
]
