"""
Compatibility adapter: wraps StockFish pydantic Settings into the attribute-based
interface expected by ported daily_stock_analysis modules.

Usage in ported code (replaces `from src.config import get_config`):
    from market_data.compat import get_config
    cfg = get_config()
    cfg.tushare_token   # accesses settings.TUSHARE_TOKEN
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Constants ported from daily_stock_analysis src/config.py
AGENT_MAX_STEPS_DEFAULT = 10
FUNDAMENTAL_STAGE_TIMEOUT_SECONDS_DEFAULT = 8.0
SUPPORTED_LLM_CHANNEL_PROTOCOLS = ("openai", "anthropic", "gemini", "vertex_ai", "deepseek", "ollama")

NEWS_STRATEGY_WINDOWS: Dict[str, int] = {
    "ultra_short": 1,
    "short": 3,
    "medium": 7,
    "long": 30,
}


def normalize_news_strategy_profile(profile: Optional[str]) -> str:
    """Normalize news strategy profile to a known window key."""
    if profile in NEWS_STRATEGY_WINDOWS:
        return profile
    return "short"


def resolve_news_window_days(
    news_max_age_days: int = 3,
    news_strategy_profile: Optional[str] = None,
) -> int:
    """Return the effective news window in days.

    Uses the strategy profile's window as an upper bound, capped by max_age_days.
    """
    strategy_days = NEWS_STRATEGY_WINDOWS.get(
        normalize_news_strategy_profile(news_strategy_profile),
        3,
    )
    return min(news_max_age_days, strategy_days) if news_max_age_days > 0 else strategy_days


# ---- record_provider_run stub ----
# The ported base.py calls record_provider_run() for diagnostics tracking.
# In StockFish we don't need this subsystem, so we provide a no-op stub
# that accepts the same kwargs signature without doing anything.

def record_provider_run(
    data_type: str = "",
    provider: str = "",
    operation: str = "",
    success: bool = True,
    latency_ms: float = 0.0,
    error_type: str = "",
    fallback_from: str = "",
    fallback_to: str = "",
    **kwargs: Any,
) -> None:
    """No-op stub for diagnostics tracking (not used in StockFish)."""


# ---- Config adapter ----

class LegacyConfigAdapter:
    """
    Wraps StockFish's pydantic Settings and exposes attribute-style access
    matching the field names used by the ported daily_stock_analysis code.

    Attribute mapping rules:
    1. Try Settings field name with the EXACT same lowercased name
       (e.g., cfg.tushare_token → settings.TUSHARE_TOKEN)
    2. Try Settings field name with UPPER_SNAKE_CASE
    3. Explicit property overrides for known mismatches

    This adapter is created once and cached in _config_singleton.
    """

    def __init__(self):
        # Lazy import to avoid circular dependencies at module load time
        from config import settings

        self._settings = settings

    # ---- Explicit property mappings for fields used by ported code ----

    @property
    def tushare_token(self) -> Optional[str]:
        return self._settings.TUSHARE_TOKEN

    @property
    def akshare_sleep_min(self) -> float:
        return getattr(self._settings, "AKSHARE_SLEEP_MIN", 2.0)

    @property
    def akshare_sleep_max(self) -> float:
        return getattr(self._settings, "AKSHARE_SLEEP_MAX", 5.0)

    @property
    def max_retries(self) -> int:
        return getattr(self._settings, "MAX_RETRIES", 3)

    @property
    def retry_base_delay(self) -> float:
        return getattr(self._settings, "RETRY_BASE_DELAY", 1.0)

    @property
    def retry_max_delay(self) -> float:
        return getattr(self._settings, "RETRY_MAX_DELAY", 30.0)

    @property
    def debug(self) -> bool:
        return self._settings.DEBUG

    @property
    def tushare_rate_limit_per_minute(self) -> int:
        return getattr(self._settings, "TUSHARE_RATE_LIMIT_PER_MINUTE", 80)

    @property
    def enable_realtime_quote(self) -> bool:
        return getattr(self._settings, "ENABLE_REALTIME_QUOTE", True)

    @property
    def enable_realtime_technical_indicators(self) -> bool:
        return getattr(self._settings, "ENABLE_REALTIME_TECHNICAL_INDICATORS", True)

    @property
    def enable_chip_distribution(self) -> bool:
        return getattr(self._settings, "ENABLE_CHIP_DISTRIBUTION", True)

    @property
    def enable_eastmoney_patch(self) -> bool:
        return getattr(self._settings, "ENABLE_EASTMONEY_PATCH", False)

    @property
    def realtime_source_priority(self) -> str:
        return getattr(
            self._settings,
            "REALTIME_SOURCE_PRIORITY",
            "tencent,akshare_sina,efinance,akshare_em",
        )

    @property
    def realtime_cache_ttl(self) -> int:
        return getattr(self._settings, "REALTIME_CACHE_TTL", 600)

    @property
    def circuit_breaker_cooldown(self) -> int:
        return getattr(self._settings, "CIRCUIT_BREAKER_COOLDOWN", 300)

    @property
    def enable_fundamental_pipeline(self) -> bool:
        return getattr(self._settings, "ENABLE_FUNDAMENTAL_PIPELINE", True)

    @property
    def fundamental_stage_timeout_seconds(self) -> float:
        return getattr(
            self._settings,
            "FUNDAMENTAL_STAGE_TIMEOUT_SECONDS",
            FUNDAMENTAL_STAGE_TIMEOUT_SECONDS_DEFAULT,
        )

    @property
    def fundamental_fetch_timeout_seconds(self) -> float:
        return getattr(self._settings, "FUNDAMENTAL_FETCH_TIMEOUT_SECONDS", 3.0)

    @property
    def fundamental_retry_max(self) -> int:
        return getattr(self._settings, "FUNDAMENTAL_RETRY_MAX", 1)

    @property
    def fundamental_cache_ttl_seconds(self) -> int:
        return getattr(self._settings, "FUNDAMENTAL_CACHE_TTL_SECONDS", 120)

    @property
    def fundamental_cache_max_entries(self) -> int:
        return getattr(self._settings, "FUNDAMENTAL_CACHE_MAX_ENTRIES", 256)

    @property
    def prefetch_realtime_quotes(self) -> bool:
        return getattr(self._settings, "PREFETCH_REALTIME_QUOTES", True)

    @property
    def tickflow_api_key(self) -> Optional[str]:
        return getattr(self._settings, "TICKFLOW_API_KEY", None)

    @property
    def finnhub_api_key(self) -> Optional[str]:
        return getattr(self._settings, "FINNHUB_API_KEY", None)

    @property
    def alphavantage_api_key(self) -> Optional[str]:
        return getattr(self._settings, "ALPHAVANTAGE_API_KEY", None)

    @property
    def longbridge_app_key(self) -> Optional[str]:
        return getattr(self._settings, "LONGBRIDGE_APP_KEY", None)

    @property
    def longbridge_app_secret(self) -> Optional[str]:
        return getattr(self._settings, "LONGBRIDGE_APP_SECRET", None)

    @property
    def longbridge_access_token(self) -> Optional[str]:
        return getattr(self._settings, "LONGBRIDGE_ACCESS_TOKEN", None)

    @property
    def stock_index_remote_update_enabled(self) -> bool:
        return getattr(self._settings, "STOCK_INDEX_REMOTE_UPDATE_ENABLED", True)

    # ---- Search engine keys ----

    @property
    def bocha_api_keys(self) -> List[str]:
        val = getattr(self._settings, "BOCHA_API_KEYS", None)
        if val:
            return [k.strip() for k in val.split(",") if k.strip()]
        single = getattr(self._settings, "BOCHA_API_KEY", None)
        return [single] if single else []

    @property
    def tavily_api_keys(self) -> List[str]:
        val = getattr(self._settings, "TAVILY_API_KEYS", None)
        if val:
            return [k.strip() for k in val.split(",") if k.strip()]
        single = getattr(self._settings, "TAVILY_API_KEY", None)
        return [single] if single else []

    @property
    def brave_api_keys(self) -> List[str]:
        val = getattr(self._settings, "BRAVE_API_KEYS", None)
        if val:
            return [k.strip() for k in val.split(",") if k.strip()]
        single = getattr(self._settings, "BRAVE_API_KEY", None)
        return [single] if single else []

    @property
    def serpapi_keys(self) -> List[str]:
        val = getattr(self._settings, "SERPAPI_KEYS", None)
        if val:
            return [k.strip() for k in val.split(",") if k.strip()]
        single = getattr(self._settings, "SERPAPI_API_KEY", None)
        return [single] if single else []

    @property
    def anspire_api_keys(self) -> List[str]:
        val = getattr(self._settings, "ANSPIRE_API_KEYS", None)
        if val:
            return [k.strip() for k in val.split(",") if k.strip()]
        single = getattr(self._settings, "ANSPIRE_API_KEY", None)
        return [single] if single else []

    @property
    def minimax_api_keys(self) -> List[str]:
        val = getattr(self._settings, "MINIMAX_API_KEYS", None)
        if val:
            return [k.strip() for k in val.split(",") if k.strip()]
        single = getattr(self._settings, "MINIMAX_API_KEY", None)
        return [single] if single else []

    @property
    def searxng_base_urls(self) -> List[str]:
        val = getattr(self._settings, "SEARXNG_BASE_URLS", None)
        if val:
            return [u.strip() for u in val.split(",") if u.strip()]
        single = getattr(self._settings, "SEARXNG_BASE_URL", None)
        return [single] if single else []

    @property
    def searxng_public_instances_enabled(self) -> bool:
        return getattr(self._settings, "SEARXNG_PUBLIC_INSTANCES_ENABLED", True)

    # ---- Social sentiment ----

    @property
    def social_sentiment_api_key(self) -> Optional[str]:
        return getattr(self._settings, "SOCIAL_SENTIMENT_API_KEY", None)

    @property
    def social_sentiment_api_url(self) -> str:
        return getattr(
            self._settings,
            "SOCIAL_SENTIMENT_API_URL",
            "https://api.adanos.org",
        )

    # ---- News & analysis ----

    @property
    def news_max_age_days(self) -> int:
        return getattr(self._settings, "NEWS_MAX_AGE_DAYS", 3)

    @property
    def news_strategy_profile(self) -> str:
        return getattr(self._settings, "NEWS_STRATEGY_PROFILE", "short")

    @property
    def bias_threshold(self) -> float:
        return getattr(self._settings, "BIAS_THRESHOLD", 5.0)

    # ---- LLM ----

    @property
    def litellm_model(self) -> str:
        return getattr(self._settings, "LITELLM_MODEL", "")

    @property
    def litellm_fallback_models(self) -> List[str]:
        val = getattr(self._settings, "LITELLM_FALLBACK_MODELS", None)
        if val:
            return [m.strip() for m in val.split(",") if m.strip()]
        return []

    @property
    def llm_temperature(self) -> float:
        return getattr(self._settings, "LLM_TEMPERATURE", 0.7)

    # ---- Report ----

    @property
    def report_language(self) -> str:
        return getattr(self._settings, "REPORT_LANGUAGE", "zh")

    # ---- Generic fallback: snake_case → UPPER_SNAKE_CASE lookup ----

    def __getattr__(self, name: str) -> Any:
        """Fallback: try to look up the attribute as a UPPER_SNAKE_CASE Settings field."""
        if name.startswith("_"):
            raise AttributeError(name)
        env_name = name.upper()
        val = getattr(self._settings, env_name, None)
        if val is not None:
            return val
        # If still None, check os.environ directly
        val = os.environ.get(env_name)
        if val is not None:
            return val
        raise AttributeError(
            f"Config has no attribute '{name}' (tried Settings.{env_name})"
        )


# Singleton cache
_config_singleton: Optional[LegacyConfigAdapter] = None


def get_config() -> LegacyConfigAdapter:
    """Return the singleton LegacyConfigAdapter wrapping StockFish settings."""
    global _config_singleton
    if _config_singleton is None:
        _config_singleton = LegacyConfigAdapter()
    return _config_singleton
