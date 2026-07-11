"""
MasterTrackDB — 大师决策记录、查询与准确率追踪

用于:
1. 记录每次大师决策 (record_decision)
2. 生成提示词上下文 (get_prompt_context) — 回注到 CIO 用户提示词
3. 更新预测结果验证 (update_outcome)
4. 统计准确率 (get_accuracy)

每条记录包含预测时的价格、多周期预测方向和幅度，
actual_outcome / was_correct 初始为 None，由验证脚本在
短/中/长期后更新。
"""
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger

from memory import MASTERS_DIR


class MasterTrackDB:
    """大师追踪数据库 — 基于文件的 append-only 存储"""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or MASTERS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ── 记录 ──

    def record_decision(self, master_key: str, symbol: str,
                        analysis_timestamp: str, state: Dict[str, Any],
                        prediction_summary: Optional[Dict] = None,
                        cio_decision: Optional[Dict] = None) -> None:
        """
        记录一次大师决策

        Args:
            master_key: 大师标识 (buffett/graham/...)
            symbol: 股票代码
            analysis_timestamp: 分析时间
            state: AnalysisState.to_dict() (用于获取 quote 价格)
            prediction_summary: state.prediction_summary
            cio_decision: CIODecision.to_dict() (可选，用于丰富记录)
        """
        quote = state.get("quote") or {}
        price = quote.get("price") if isinstance(quote, dict) else None

        short_pred = (prediction_summary or {}).get("short_term") or {}
        mid_pred = (prediction_summary or {}).get("mid_term") or {}
        long_pred = (prediction_summary or {}).get("long_term") or {}

        record = {
            "id": f"track_{master_key}_{symbol}_{analysis_timestamp[:10]}_{uuid4().hex[:8]}",
            "master_key": master_key,
            "symbol": symbol,
            "analysis_timestamp": analysis_timestamp,
            "price_at_analysis": price,

            # 决策摘要
            "decision_summary": (prediction_summary or {}).get("outlook", ""),
            "decision_outlook": (prediction_summary or {}).get("outlook", ""),

            # 多周期预测
            "short_term_pred": {
                "direction": short_pred.get("direction"),
                "change_pct": short_pred.get("change_pct"),
                "confidence": short_pred.get("confidence"),
            } if short_pred else None,
            "mid_term_pred": {
                "direction": mid_pred.get("direction"),
                "change_pct": mid_pred.get("change_pct"),
                "confidence": mid_pred.get("confidence"),
            } if mid_pred else None,
            "long_term_pred": {
                "direction": long_pred.get("direction"),
                "change_pct": long_pred.get("change_pct"),
                "confidence": long_pred.get("confidence"),
            } if long_pred else None,

            # CIO 额外字段
            "cio_rationale": cio_decision.get("rationale", "")[:200] if cio_decision else "",
            "base_case_target": (
                cio_decision.get("base_case", {}).get("target")
                if cio_decision and isinstance(cio_decision.get("base_case"), dict)
                else None
            ),

            # 结果验证字段（初始为 None）
            "actual_outcome_short": None,
            "actual_outcome_mid": None,
            "actual_outcome_long": None,
            "actual_change_pct_short": None,
            "actual_change_pct_mid": None,
            "actual_change_pct_long": None,
            "was_correct_short": None,
            "was_correct_mid": None,
            "was_correct_long": None,

            "recorded_at": datetime.now().isoformat(),
            "verified_short_at": None,
            "verified_mid_at": None,
            "verified_long_at": None,
        }

        self._append_record(master_key, record)
        logger.debug(f"MasterTrack 已记录: {master_key} → {symbol} ({prediction_summary.get('outlook', 'N/A') if prediction_summary else 'N/A'})")

    # ── 查询 ──

    def get_prompt_context(self, master_key: str, symbol: str,
                           limit: int = 5) -> str:
        """
        生成大师历史准确率的提示词上下文

        返回格式:
        ```
        ## Your Historical Performance

        Your last 5 predictions for 600519:
          [2026-06-27] 看多 (actual: +3.2%) -> CORRECT
          [2026-06-13] 看多 (actual: -1.5%) -> WRONG
        Symbol accuracy: 4/5 (80%)
        Overall accuracy (short-term): 12/18 (67%)
        ```
        """
        records = self._load_records(master_key)

        # 该股票的已验证记录
        symbol_records = [
            r for r in records
            if r.get("symbol") == symbol and r.get("was_correct_short") is not None
        ]

        # 所有已验证记录
        all_verified = [r for r in records if r.get("was_correct_short") is not None]

        lines = []

        if symbol_records:
            lines.append(f"\n## Your Historical Performance")
            lines.append(f"\nYour last {limit} predictions for {symbol}:")
            for r in symbol_records[-limit:]:
                actual = r.get("actual_change_pct_short")
                actual_str = f"{actual:+.1f}%" if actual is not None else "N/A"
                was_right = "CORRECT" if r.get("was_correct_short") else "WRONG"
                lines.append(
                    f"  [{r.get('analysis_timestamp', '')[:10]}] "
                    f"{r.get('decision_outlook', 'N/A')} "
                    f"(actual: {actual_str})"
                    f" -> {was_right}"
                )

            sym_total = len(symbol_records)
            sym_correct = sum(1 for r in symbol_records if r.get("was_correct_short"))
            sym_acc = round(sym_correct / sym_total * 100) if sym_total > 0 else 0
            lines.append(f"  Symbol accuracy: {sym_correct}/{sym_total} ({sym_acc}%)")

        # 总体准确率
        if all_verified:
            total = len(all_verified)
            correct = sum(1 for r in all_verified if r.get("was_correct_short"))
            acc = round(correct / total * 100) if total > 0 else 0
            lines.append(f"Overall accuracy (short-term): {correct}/{total} ({acc}%)")

        return "\n".join(lines) if lines else ""

    def get_symbol_records(self, master_key: str, symbol: str,
                           verified_only: bool = True) -> List[Dict]:
        """获取某个大师对某只股票的所有记录"""
        records = self._load_records(master_key)
        result = [r for r in records if r.get("symbol") == symbol]
        if verified_only:
            result = [r for r in result if r.get("was_correct_short") is not None]
        return result

    def get_unverified_records(self) -> List[Dict]:
        """
        获取所有待验证的记录（was_correct_short == None）
        用于验证脚本定期检查
        """
        result = []
        for master_file in self.base_dir.glob("*.json"):
            records = self._load_records(master_file.stem)
            for r in records:
                if r.get("was_correct_short") is None:
                    result.append(r)
        return result

    def get_accuracy(self, master_key: str) -> Dict[str, Any]:
        """统计某位大师的准确率"""
        records = self._load_records(master_key)
        verified = [r for r in records if r.get("was_correct_short") is not None]

        total = len(verified)
        short_correct = sum(1 for r in verified if r.get("was_correct_short"))
        mid_correct = sum(1 for r in verified if r.get("was_correct_mid"))
        mid_count = sum(1 for r in verified if r.get("was_correct_mid") is not None)
        long_correct = sum(1 for r in verified if r.get("was_correct_long"))
        long_count = sum(1 for r in verified if r.get("was_correct_long") is not None)

        return {
            "master_key": master_key,
            "total_predictions": len(records),
            "verified_short_term": total,
            "short_term_accuracy": round(short_correct / total * 100, 1) if total > 0 else None,
            "verified_mid_term": mid_count,
            "mid_term_accuracy": round(mid_correct / mid_count * 100, 1) if mid_count > 0 else None,
            "verified_long_term": long_count,
            "long_term_accuracy": round(long_correct / long_count * 100, 1) if long_count > 0 else None,
        }

    # ── 更新 ──

    def update_outcome(self, record_id: str, field: str,
                       actual_change_pct: float, was_correct: bool) -> None:
        """
        更新一条记录的实际结果

        Args:
            record_id: 记录 ID
            field: 'short' | 'mid' | 'long'
            actual_change_pct: 实际涨跌幅百分比
            was_correct: 预测方向是否正确
        """
        with self._lock:
            for master_file in self.base_dir.glob("*.json"):
                records = self._load_records(master_file.stem, acquire_lock=False)
                updated = False
                for r in records:
                    if r.get("id") == record_id:
                        r[f"actual_change_pct_{field}"] = actual_change_pct
                        r[f"was_correct_{field}"] = was_correct
                        r["actual_outcome_" + field] = "上涨" if actual_change_pct > 0 else "下跌" if actual_change_pct < 0 else "震荡"
                        r[f"verified_{field}_at"] = datetime.now().isoformat()
                        updated = True
                        break

                if updated:
                    self._save_records(master_file.stem, records, acquire_lock=False)
                    logger.info(f"MasterTrack 已更新 [{record_id}] {field}: {'✓' if was_correct else '✗'} ({actual_change_pct:+.1f}%)")
                    return

        logger.warning(f"MasterTrack 更新失败: 未找到记录 {record_id}")

    # ── 内部方法 ──

    def _records_path(self, master_key: str) -> Path:
        return self.base_dir / f"{master_key}.json"

    def _append_record(self, master_key: str, record: Dict) -> None:
        """追加一条记录到文件"""
        with self._lock:
            records = self._load_records(master_key, acquire_lock=False)
            records.append(record)
            self._save_records(master_key, records, acquire_lock=False)

    def _load_records(self, master_key: str,
                      acquire_lock: bool = True) -> List[Dict]:
        """加载某位大师的所有记录"""
        if acquire_lock:
            self._lock.acquire()
        try:
            path = self._records_path(master_key)
            if not path.exists():
                return []
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
        except Exception as e:
            logger.warning(f"MasterTrack 加载失败 [{master_key}]: {e}")
            return []
        finally:
            if acquire_lock:
                self._lock.release()

    def _save_records(self, master_key: str, records: List[Dict],
                      acquire_lock: bool = True) -> None:
        """保存某位大师的所有记录"""
        if acquire_lock:
            self._lock.acquire()
        try:
            path = self._records_path(master_key)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, default=str, indent=2)
            tmp.rename(path)
        except Exception as e:
            logger.warning(f"MasterTrack 保存失败 [{master_key}]: {e}")
        finally:
            if acquire_lock:
                self._lock.release()
