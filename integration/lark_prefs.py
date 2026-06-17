"""
用户偏好管理 — 每个用户一个 JSON 文件。

存储默认大师选择，后续扩展：推送开关、最大自选数等。

文件: data/user_prefs/<open_id>.json
映射: data/user_prefs/chat_map.json (open_id → chat_id)
"""

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from loguru import logger


class UserPrefManager:
    """线程安全的用户偏好读写。

    每个飞书用户一个 JSON 文件，存储在 data/user_prefs/ 下。
    """

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "data" / "user_prefs"
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _safe_id(self, open_id: Optional[str]) -> str:
        """将 open_id 转为安全的文件名。"""
        if not open_id:
            return "_anonymous"
        return re.sub(r"[^a-zA-Z0-9_-]", "_", open_id)

    def _path(self, open_id: Optional[str]) -> Path:
        return self._dir / f"{self._safe_id(open_id)}.json"

    def _lock(self, open_id: Optional[str]) -> threading.Lock:
        sid = self._safe_id(open_id)
        with self._global_lock:
            if sid not in self._locks:
                self._locks[sid] = threading.Lock()
            return self._locks[sid]

    # ── 大师偏好 ───────────────────────────────────────────

    def get_master(self, open_id: Optional[str]) -> str:
        """获取用户的默认大师，空字符串表示未设置。"""
        if not open_id:
            return ""
        lock = self._lock(open_id)
        with lock:
            data = self._read(open_id)
            return data.get("default_master", "")

    def set_master(self, open_id: str, master: str) -> None:
        """设置用户的默认大师。空字符串表示关闭大师模式。"""
        if not open_id:
            return
        lock = self._lock(open_id)
        with lock:
            data = self._read(open_id)
            data["open_id"] = open_id
            data["default_master"] = master
            data["updated_at"] = datetime.now().isoformat()
            self._write(open_id, data)
            logger.info(
                f"[UserPrefs] master={'off' if not master else master} for {open_id}"
            )

    # ── chat_id 映射（供后续推送使用）────────────────────────

    def save_chat_map(self, open_id: str, chat_id: str) -> None:
        """保存 open_id → chat_id 映射。"""
        chat_map_path = self._dir / "chat_map.json"
        lock = self._lock("_chat_map")
        with lock:
            data = {}
            if chat_map_path.exists():
                try:
                    data = json.loads(chat_map_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
            data[open_id] = chat_id
            chat_map_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def get_chat_id(self, open_id: str) -> Optional[str]:
        """获取用户的最近 chat_id。"""
        chat_map_path = self._dir / "chat_map.json"
        if not chat_map_path.exists():
            return None
        try:
            data = json.loads(chat_map_path.read_text(encoding="utf-8"))
            return data.get(open_id)
        except json.JSONDecodeError:
            return None

    # ── 内部读写 ────────────────────────────────────────────

    def _read(self, open_id: Optional[str]) -> dict:
        path = self._path(open_id)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning(f"[UserPrefs] 损坏的 JSON: {path}")
        return {}

    def _write(self, open_id: str, data: dict) -> None:
        path = self._path(open_id)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
