"""
内存 (Memory) 系统配置

从 config.settings 继承缓存 TTL，提供默认值。
"""
from config import settings as _settings


def _get(key: str, default):
    """从全局 Settings 取值，不存在则返回默认值"""
    return getattr(_settings, key, default)


# ── 开关 ──
MEMORY_ENABLED: bool = _get("MEMORY_ENABLED", True)

# ── 缓存 TTL (秒) ──
CACHE_TTL_QUOTE: int = _get("CACHE_TTL_QUOTE", 300)            # 5 min
CACHE_TTL_TECHNICAL: int = _get("CACHE_TTL_TECHNICAL", 900)    # 15 min
CACHE_TTL_HISTORICAL: int = _get("CACHE_TTL_HISTORICAL", 3600) # 1 h
CACHE_TTL_FINANCIAL: int = _get("CACHE_TTL_FINANCIAL", 7200)   # 2 h
CACHE_TTL_NEWS: int = _get("CACHE_TTL_NEWS", 600)              # 10 min
CACHE_TTL_GUBA: int = _get("CACHE_TTL_GUBA", 600)             # 10 min
CACHE_TTL_MACRO: int = _get("CACHE_TTL_MACRO", 14400)          # 4 h
CACHE_TTL_INDUSTRY: int = _get("CACHE_TTL_INDUSTRY", 14400)    # 4 h

# ── 缓存容量 ──
CACHE_MAX_ENTRIES_PER_TYPE: int = _get("CACHE_MAX_ENTRIES_PER_TYPE", 10000)

# ── 文件路径 ──
from pathlib import Path
MEMORY_ROOT = Path(__file__).resolve().parent
CACHE_DATA_DIR = str(MEMORY_ROOT / "cache" / "data")
