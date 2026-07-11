"""
StockFish 记忆系统 (Memory System)

四层架构:
1. cache/     — 数据缓存 (避免重复 API 请求)
2. stocks/    — 股票数据仓库 (按股票分类持久化)
3. analysis/  — 分析归档 (每次分析结果的永久保存)
4. masters/   — 大师自优化 (决策记录 + 准确率追踪)
"""
import os
from pathlib import Path
from typing import Optional

# 内存系统根目录 (相对于项目根)
MEMORY_ROOT = Path(__file__).resolve().parent

# 运行时数据目录
CACHE_DATA_DIR = MEMORY_ROOT / "cache" / "data"
STOCKS_DIR = MEMORY_ROOT / "stocks"
ANALYSIS_DIR = MEMORY_ROOT / "analysis"
MASTERS_DIR = MEMORY_ROOT / "masters" / "records"


def ensure_dirs():
    """确保所有运行时数据目录存在"""
    for d in [CACHE_DATA_DIR, STOCKS_DIR, ANALYSIS_DIR, MASTERS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# 延迟加载的单例引用 (由各模块在首次导入时初始化)
_cache_manager: Optional[object] = None
_stock_library: Optional[object] = None
_analysis_store: Optional[object] = None
_master_track_db: Optional[object] = None


def get_cache_manager():
    global _cache_manager
    if _cache_manager is None:
        from memory.cache.cache_manager import cache_manager as cm
        _cache_manager = cm
    return _cache_manager


def get_stock_library():
    global _stock_library
    if _stock_library is None:
        from memory.stocks.stock_library import StockLibrary as sl
        _stock_library = sl()  # StockLibrary 无状态，可以新实例
    return _stock_library


def get_analysis_store():
    global _analysis_store
    if _analysis_store is None:
        from memory.analysis.analysis_store import AnalysisStore as as_
        _analysis_store = as_()  # AnalysisStore 无状态，可以新实例
    return _analysis_store


def get_master_track_db():
    global _master_track_db
    if _master_track_db is None:
        from memory.masters.master_track import MasterTrackDB as mt
        _master_track_db = mt()  # MasterTrackDB 无状态，可以新实例
    return _master_track_db


# 应用启动时创建目录
ensure_dirs()
