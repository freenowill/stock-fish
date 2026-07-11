"""
StockFish 全局配置
pydantic-settings 从 .env 和环境变量自动加载
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

# 添加 BettaFish 和 MiroFish 到 Python 路径以便 import
PROJECT_ROOT = Path(__file__).resolve().parent

# 将 .env 加载到 os.environ（兼容直接读取 os.environ 的代码）
load_dotenv(PROJECT_ROOT / ".env")
BETTAFISH_DIR = str(PROJECT_ROOT.parent / "BettaFish")
MIROFISH_DIR = str(PROJECT_ROOT.parent / "MiroFish" / "backend")

for p in [BETTAFISH_DIR, MIROFISH_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


class Settings(BaseSettings):
    model_config = {"env_file": str(PROJECT_ROOT / ".env"), "extra": "allow"}

    # Flask
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # ===== LLM（通用，OpenAI 格式） =====
    LLM_API_KEY: Optional[str] = None
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL_NAME: str = "gpt-4o-mini"

    # ===== 数据源 =====
    AKSHARE_PROXY: Optional[str] = None
    TUSHARE_TOKEN: Optional[str] = None
    STOCK_BACKEND: str = "mock"

    # ---- 多源数据渠道 Token ----
    LONGBRIDGE_APP_KEY: Optional[str] = None
    LONGBRIDGE_APP_SECRET: Optional[str] = None
    LONGBRIDGE_ACCESS_TOKEN: Optional[str] = None
    LONGBRIDGE_REGION: Optional[str] = None  # cn|hk - 提升大陆连接稳定性
    FINNHUB_API_KEY: Optional[str] = None
    ALPHAVANTAGE_API_KEY: Optional[str] = None
    TICKFLOW_API_KEY: Optional[str] = None
    SOCIAL_SENTIMENT_API_KEY: Optional[str] = None
    SOCIAL_SENTIMENT_API_URL: str = "https://api.adanos.org"

    # ---- 搜索引擎 Key ----
    BOCHA_API_KEY: Optional[str] = None
    BOCHA_API_KEYS: Optional[str] = None  # comma-separated multi-key
    TAVILY_API_KEY: Optional[str] = None
    TAVILY_API_KEYS: Optional[str] = None
    BRAVE_API_KEY: Optional[str] = None
    BRAVE_API_KEYS: Optional[str] = None
    SERPAPI_API_KEY: Optional[str] = None
    SERPAPI_KEYS: Optional[str] = None
    ANSPIRE_API_KEY: Optional[str] = None
    ANSPIRE_API_KEYS: Optional[str] = None
    MINIMAX_API_KEY: Optional[str] = None
    MINIMAX_API_KEYS: Optional[str] = None
    SEARXNG_BASE_URL: Optional[str] = None
    SEARXNG_BASE_URLS: Optional[str] = None
    SEARXNG_PUBLIC_INSTANCES_ENABLED: bool = True

    # ---- 实时行情优先级 ----
    REALTIME_SOURCE_PRIORITY: str = "tencent,akshare_sina,efinance,akshare_em"
    REALTIME_CACHE_TTL: int = 600
    CIRCUIT_BREAKER_COOLDOWN: int = 300

    # ---- 特性开关 ----
    ENABLE_REALTIME_QUOTE: bool = True
    ENABLE_REALTIME_TECHNICAL_INDICATORS: bool = True
    ENABLE_CHIP_DISTRIBUTION: bool = True
    ENABLE_EASTMONEY_PATCH: bool = False
    ENABLE_FUNDAMENTAL_PIPELINE: bool = True
    PREFETCH_REALTIME_QUOTES: bool = True
    STOCK_INDEX_REMOTE_UPDATE_ENABLED: bool = True

    # ---- 基本面超时 ----
    FUNDAMENTAL_STAGE_TIMEOUT_SECONDS: float = 8.0
    FUNDAMENTAL_FETCH_TIMEOUT_SECONDS: float = 3.0
    FUNDAMENTAL_RETRY_MAX: int = 1
    FUNDAMENTAL_CACHE_TTL_SECONDS: int = 120
    FUNDAMENTAL_CACHE_MAX_ENTRIES: int = 256

    # ---- 流控 ----
    AKSHARE_SLEEP_MIN: float = 2.0
    AKSHARE_SLEEP_MAX: float = 5.0
    TUSHARE_RATE_LIMIT_PER_MINUTE: int = 80
    MAX_RETRIES: int = 3
    RETRY_BASE_DELAY: float = 1.0
    RETRY_MAX_DELAY: float = 30.0

    # ---- 新闻 ----
    NEWS_MAX_AGE_DAYS: int = 3
    NEWS_STRATEGY_PROFILE: str = "short"
    BIAS_THRESHOLD: float = 5.0

    # ===== 飞书 / Lark Bot =====
    LARK_APP_ID: Optional[str] = None
    LARK_APP_SECRET: Optional[str] = None
    LARK_BOT_NAME: str = "StockFish"
    STOCKFISH_API_URL: str = "http://127.0.0.1:8000"  # Bot 调用 Flask API 的地址

    # ===== BettaFish 路径（情感分析复用） =====
    BETTAFISH_PATH: str = BETTAFISH_DIR

    # ===== MiroFish 配置 =====
    MIROFISH_BACKEND_PATH: str = MIROFISH_DIR
    MIROFISH_HOST: str = "localhost"
    MIROFISH_PORT: int = 5001
    ZEP_API_KEY: Optional[str] = None
    OASIS_DEFAULT_MAX_ROUNDS: int = 20
    OASIS_SIMULATION_AGENT_COUNT: int = 15
    OASIS_DEBUG: bool = False  # debug模式：2 Agent / 2轮

    # ===== 行情缓存 =====
    CACHE_TTL_SECONDS: int = 60

    # ===== 记忆系统 (Memory) =====
    MEMORY_ENABLED: bool = True                       # 总开关
    CACHE_TTL_QUOTE: int = 300                        # 行情 5min
    CACHE_TTL_TECHNICAL: int = 900                    # 技术指标 15min
    CACHE_TTL_HISTORICAL: int = 3600                  # 历史 K 线 1h
    CACHE_TTL_FINANCIAL: int = 7200                   # 财务摘要 2h
    CACHE_TTL_NEWS: int = 600                         # 新闻 10min
    CACHE_TTL_GUBA: int = 600                         # 股吧 10min
    CACHE_TTL_MACRO: int = 14400                      # 宏观 4h
    CACHE_TTL_INDUSTRY: int = 14400                   # 行业 4h
    CACHE_MAX_ENTRIES_PER_TYPE: int = 10000            # 每类缓存最大条目


settings = Settings()
