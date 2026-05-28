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

    # ===== BettaFish 路径（情感分析复用） =====
    BETTAFISH_PATH: str = BETTAFISH_DIR

    # ===== MiroFish 配置 =====
    MIROFISH_BACKEND_PATH: str = MIROFISH_DIR
    MIROFISH_HOST: str = "localhost"
    MIROFISH_PORT: int = 5001
    ZEP_API_KEY: Optional[str] = None
    OASIS_DEFAULT_MAX_ROUNDS: int = 20
    OASIS_SIMULATION_AGENT_COUNT: int = 15

    # ===== 行情缓存 =====
    CACHE_TTL_SECONDS: int = 60


settings = Settings()
