"""
CacheManager — 中央缓存编排器

为每种数据类型提供类型化封装方法，内部管理多个 TtlCache 实例。
单例模式，全局共享。
"""
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from loguru import logger

from memory.cache.ttl_cache import TtlCache
from memory.config import (
    MEMORY_ENABLED,
    CACHE_DATA_DIR,
    CACHE_TTL_QUOTE,
    CACHE_TTL_TECHNICAL,
    CACHE_TTL_HISTORICAL,
    CACHE_TTL_FINANCIAL,
    CACHE_TTL_NEWS,
    CACHE_TTL_GUBA,
    CACHE_TTL_MACRO,
    CACHE_TTL_INDUSTRY,
    CACHE_MAX_ENTRIES_PER_TYPE,
)


class CacheManager:
    """中央缓存编排器 — 单例，全局共享"""

    def __init__(self, cache_dir: str = CACHE_DATA_DIR):
        self.cache_dir = Path(cache_dir)
        self._lock = threading.RLock()
        self._caches: Dict[str, TtlCache] = {}

        if MEMORY_ENABLED:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"CacheManager 初始化, 缓存目录: {self.cache_dir}")

    # ── 内部: 获取/创建子缓存 ──

    def _get_cache(self, name: str, default_ttl: int = 300,
                   max_entries: int = CACHE_MAX_ENTRIES_PER_TYPE) -> TtlCache:
        with self._lock:
            if name not in self._caches:
                persist_dir = self.cache_dir / name
                persist_dir.mkdir(parents=True, exist_ok=True)
                cache = TtlCache(
                    name=name,
                    persist_dir=persist_dir,
                    default_ttl=default_ttl,
                    max_entries=max_entries,
                )
                self._caches[name] = cache
            return self._caches[name]

    # ── 类型化数据访问 ──

    def get_market_data(self, symbol: str, fetch_fn: Callable[[], Any]) -> Any:
        """获取完整市场数据包（TTL=5min）— 整包缓存，避免重复调用 get_all_market_data()"""
        if not MEMORY_ENABLED:
            return fetch_fn()
        return self._get_cache("market_data", CACHE_TTL_QUOTE).get_or_fetch(
            symbol, fetch_fn, data_type="market_data", symbol=symbol)

    def get_quote(self, symbol: str, fetch_fn: Callable[[], Any]) -> Any:
        """获取行情数据（TTL=5min）"""
        if not MEMORY_ENABLED:
            return fetch_fn()
        return self._get_cache("quote", CACHE_TTL_QUOTE).get_or_fetch(
            symbol, fetch_fn, data_type="quote", symbol=symbol)

    def get_historical_pe(self, symbol: str, days: int, fetch_fn: Callable[[], Any]) -> Any:
        """获取历史 PE 数据（TTL=1h），按 {symbol}:pe:{days} 缓存"""
        if not MEMORY_ENABLED:
            return fetch_fn()
        key = f"{symbol}:pe:{days}"
        return self._get_cache("historical_pe", CACHE_TTL_HISTORICAL).get_or_fetch(
            key, fetch_fn, data_type="historical_pe", symbol=symbol)

    def get_financial_abstract(self, symbol: str, fetch_fn: Callable[[], Any]) -> Any:
        """获取详细财务数据（TTL=2h）"""
        if not MEMORY_ENABLED:
            return fetch_fn()
        return self._get_cache("financial_abstract", CACHE_TTL_FINANCIAL).get_or_fetch(
            symbol, fetch_fn, data_type="financial_abstract", symbol=symbol)

    def get_technical_indicators(self, symbol: str, fetch_fn: Callable[[], Any]) -> Any:
        """获取技术指标（TTL=15min）"""
        if not MEMORY_ENABLED:
            return fetch_fn()
        return self._get_cache("technical", CACHE_TTL_TECHNICAL).get_or_fetch(
            symbol, fetch_fn, data_type="technical", symbol=symbol)

    def get_financial_summary(self, symbol: str, fetch_fn: Callable[[], Any]) -> Any:
        """获取财务摘要（TTL=2h）"""
        if not MEMORY_ENABLED:
            return fetch_fn()
        return self._get_cache("financial", CACHE_TTL_FINANCIAL).get_or_fetch(
            symbol, fetch_fn, data_type="financial", symbol=symbol)

    def get_historical(self, symbol: str, days: int, fetch_fn: Callable[[], Any]) -> Any:
        """获取历史 K 线（TTL=1h），按 {symbol}:{days} 缓存"""
        if not MEMORY_ENABLED:
            return fetch_fn()
        key = f"{symbol}:{days}"
        return self._get_cache("historical", CACHE_TTL_HISTORICAL).get_or_fetch(
            key, fetch_fn, data_type="historical", symbol=symbol)

    def get_news(self, symbol: str, fetch_fn: Callable[[], Any]) -> Any:
        """获取新闻（TTL=10min）"""
        if not MEMORY_ENABLED:
            return fetch_fn()
        return self._get_cache("news", CACHE_TTL_NEWS).get_or_fetch(
            symbol, fetch_fn, data_type="news", symbol=symbol)

    def get_guba(self, symbol: str, fetch_fn: Callable[[], Any]) -> Any:
        """获取股吧帖子（TTL=10min）"""
        if not MEMORY_ENABLED:
            return fetch_fn()
        return self._get_cache("guba", CACHE_TTL_GUBA).get_or_fetch(
            symbol, fetch_fn, data_type="guba", symbol=symbol)

    def get_macro_context(self, fetch_fn: Callable[[], Any]) -> Any:
        """获取宏观上下文（TTL=4h），全局共享"""
        if not MEMORY_ENABLED:
            return fetch_fn()
        return self._get_cache("macro", CACHE_TTL_MACRO, max_entries=10).get_or_fetch(
            "global", fetch_fn, data_type="macro", symbol="global")

    def get_industry_context(self, symbol: str, fetch_fn: Callable[[], Any]) -> Any:
        """获取行业上下文（TTL=4h）"""
        if not MEMORY_ENABLED:
            return fetch_fn()
        return self._get_cache("industry", CACHE_TTL_INDUSTRY).get_or_fetch(
            symbol, fetch_fn, data_type="industry", symbol=symbol)

    # ── 缓存管理 ──

    def warm_up(self, symbol: str, provider: object) -> None:
        """预检查所有缓存，只获取过期的（空实现 — 实际由 get_* 的 get_or_fetch 按需填充）"""
        if not MEMORY_ENABLED:
            return
        logger.debug(f"Cache warm-up 准备就绪: {symbol}")

    def invalidate_symbol(self, symbol: str) -> Dict[str, int]:
        """清除某只股票在所有子缓存中的条目"""
        result = {}
        with self._lock:
            for name, cache in self._caches.items():
                count = cache.invalidate_by_symbol(symbol)
                if count > 0:
                    result[name] = count
        if result:
            logger.debug(f"失效缓存 [{symbol}]: {result}")
        return result

    def invalidate_all(self) -> None:
        """清空所有缓存"""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
            logger.info("所有缓存已清空")

    def stats(self) -> Dict[str, dict]:
        """获取所有子缓存的统计"""
        result = {}
        with self._lock:
            for name, cache in self._caches.items():
                result[name] = cache.stats()
        return result


# 模块级单例
cache_manager = CacheManager()
