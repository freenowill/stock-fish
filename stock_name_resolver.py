# -*- coding: utf-8 -*-
"""
股票名称解析模块

将中文股票名称（如"贵州茅台"）解析为股票代码（如"600519"）。
支持多重数据源降级使用：
1. stocks.index.json（远程全量索引）
2. STOCK_NAME_MAP（内置 ~100 只常见股票）
3. 原始输入原样返回（向后兼容）
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from market_data.stock_index.stock_index_loader import (
    get_stock_name_index_map,
    find_existing_stock_index_path,
    clear_stock_index_cache as clear_index_cache,
)
from market_data.stock_index.stock_mapping import STOCK_NAME_MAP

logger = logging.getLogger(__name__)

# 尝试 import pypinyin（可选依赖）
try:
    from pypinyin import lazy_pinyin as _lazy_pinyin

    def _compute_pinyin(text: str) -> str:
        return "".join(_lazy_pinyin(text)).lower().replace(" ", "")
except ImportError:
    _lazy_pinyin = None  # type: ignore

    def _compute_pinyin(text: str) -> str:  # type: ignore
        return ""


# HK/US 股票代码后缀
_HK_SUFFIXES = (".HK",)
_US_SUFFIXES = (".US",)
_ALL_SUFFIXES = _HK_SUFFIXES + _US_SUFFIXES + (".SH", ".SZ", ".BJ", ".SS")


def _has_cjk(text: str) -> bool:
    """检查字符串是否包含 CJK 中文字符"""
    for ch in text:
        if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿':
            return True
    return False


def is_stock_code(text: str) -> bool:
    """检测输入是否为股票代码格式"""
    if not text:
        return False

    upper = text.upper().strip()

    # 包含中文字符 → 肯定是名称，不是代码
    if _has_cjk(upper):
        return False

    # 纯数字
    if upper.isdigit():
        # A 股 6 位代码 / 港股 5 位
        return len(upper) in (5, 6)

    # 带前缀 SH/SZ/BJ/HK
    for prefix in ("SH", "SZ", "BJ", "HK", "SS"):
        if upper.startswith(prefix) and upper[len(prefix):].isdigit():
            return True

    # 带后缀 .SH/.SZ/.HK/.BJ/.US
    for suffix in _ALL_SUFFIXES:
        if upper.endswith(suffix.upper()):
            base = upper[: -len(suffix.upper())]
            if base.isdigit():
                return True

    # US ticker: 1-5 个字母（纯字母，且不含 CJK）
    if upper.isalpha() and 1 <= len(upper) <= 5:
        return True

    return False


class StockNameResolver:
    """
    股票名称解析器

    将中文股票名称解析为股票代码，支持：
    - 精确名称匹配
    - 子串模糊匹配
    - 拼音匹配（可选，需 pypinyin）
    - 自动补全搜索

    数据源降级：远程索引 > 内置映射 > 原样返回
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._last_refresh: float = 0
        self._refresh_interval: float = 600.0  # 10 分钟刷新一次

        # name -> code 缓存
        self._name_to_code: Dict[str, str] = {}
        # code -> name 缓存（从 STOCK_NAME_MAP 构建）
        self._code_to_name: Dict[str, str] = {}
        # 拼音 -> code 缓存
        self._pinyin_to_code: Dict[str, str] = {}
        # 是否有完整索引
        self._has_full_index: bool = False

        self._init_from_builtin()
        self._try_load_index()

    def _init_from_builtin(self) -> None:
        """从内置 STOCK_NAME_MAP 构建初始映射"""
        for code, name in STOCK_NAME_MAP.items():
            normalized_code = code.strip().upper()
            normalized_name = name.strip()
            self._code_to_name[normalized_code] = normalized_name
            self._name_to_code[normalized_name] = normalized_code

            # 拼音索引
            if _lazy_pinyin is not None:
                pinyin = _compute_pinyin(normalized_name)
                if pinyin and pinyin not in self._pinyin_to_code:
                    self._pinyin_to_code[pinyin] = normalized_code

        logger.debug(
            "[股票名称解析器] 内置映射加载完成: %d 只股票",
            len(self._code_to_name),
        )

    def _try_load_index(self) -> bool:
        """尝试从远程 stocks.index.json 加载全量索引"""
        try:
            index_path = find_existing_stock_index_path()
            if index_path is None:
                logger.debug("[股票名称解析器] 未找到 stocks.index.json，使用内置映射")
                return False

            name_map = get_stock_name_index_map()
            if not name_map:
                logger.debug("[股票名称解析器] stocks.index.json 为空")
                return False

            # 反向构建 name -> code
            # name_map 当前是 code -> name，转成 name -> code
            # 但由于可能有多个 code 对应同一 name，需要处理去重
            name_count: int = 0
            for code_candidate, candidate_name in name_map.items():
                if not candidate_name or not code_candidate:
                    continue
                normalized_name = candidate_name.strip()
                if not normalized_name:
                    continue
                normalized_code = code_candidate.strip().upper()

                # 只在没有冲突或当前映射为 builtin 时覆盖
                existing = self._name_to_code.get(normalized_name)
                if existing is None or existing in STOCK_NAME_MAP:
                    self._name_to_code[normalized_name] = normalized_code

                # 更新拼音索引
                if _lazy_pinyin is not None:
                    pinyin = _compute_pinyin(normalized_name)
                    if pinyin and pinyin not in self._pinyin_to_code:
                        self._pinyin_to_code[pinyin] = normalized_code

                name_count += 1

            self._has_full_index = True
            logger.debug(
                "[股票名称解析器] 远程索引加载完成: %d 条",
                name_count,
            )
            return True
        except Exception as e:
            logger.debug("[股票名称解析器] 加载远程索引失败: %s，使用内置映射", e)
            return False

    def refresh_index(self, force: bool = False) -> bool:
        """刷新股票索引"""
        with self._lock:
            now = time.time()
            if not force and (now - self._last_refresh) < self._refresh_interval:
                return self._has_full_index

            self._last_refresh = now
            # 清空缓存再重新加载
            clear_index_cache()
            return self._try_load_index()

    def resolve(self, text: str) -> Dict[str, Any]:
        """解析用户输入，返回匹配结果

        Args:
            text: 用户输入的股票代码或名称

        Returns:
            {
                "resolved": bool,
                "input_type": "code" | "name" | "unknown",
                "exact": bool,
                "matches": [{"code": "...", "name": "...", "match_type": "..."}],
            }
        """
        if not text or not text.strip():
            return {
                "resolved": False,
                "input_type": "unknown",
                "exact": False,
                "matches": [],
            }

        raw = text.strip().upper()
        # 尝试刷新索引（非强制）
        self.refresh_index()

        # 如果是代码格式
        if is_stock_code(raw) and raw.isdigit() and len(raw) in (5, 6):
            code = raw
            name = self._code_to_name.get(code)
            if name:
                return {
                    "resolved": True,
                    "input_type": "code",
                    "exact": True,
                    "matches": [{"code": code, "name": name, "match_type": "exact"}],
                }
            # 代码格式但不在映射中，作为代码返回
            return {
                "resolved": True,
                "input_type": "code",
                "exact": True,
                "matches": [{"code": code, "name": code, "match_type": "exact"}],
            }

        # US/HK ticker 格式
        if is_stock_code(raw):
            code = raw
            # 尝试去掉后缀匹配
            base_code = raw
            for suffix in _ALL_SUFFIXES:
                if raw.endswith(suffix.upper()):
                    base_code = raw[: -len(suffix.upper())]
                    break
            name = self._code_to_name.get(base_code)
            if name:
                return {
                    "resolved": True,
                    "input_type": "code",
                    "exact": True,
                    "matches": [{"code": base_code, "name": name, "match_type": "exact"}],
                }
            # US ticker 原样返回
            return {
                "resolved": True,
                "input_type": "code",
                "exact": True,
                "matches": [{"code": code, "name": code, "match_type": "exact"}],
            }

        # 中文名称匹配
        original_text = text.strip()
        matches: List[Dict[str, Any]] = []

        # 1. 精确匹配
        code = self._name_to_code.get(original_text)
        if code:
            name = self._code_to_name.get(code, original_text)
            matches.append({"code": code, "name": name, "match_type": "exact"})
            return {
                "resolved": True,
                "input_type": "name",
                "exact": True,
                "matches": matches,
            }

        # 2. 子串匹配
        for name_candidate, code_candidate in self._name_to_code.items():
            if original_text in name_candidate or name_candidate in original_text:
                real_name = self._code_to_name.get(code_candidate, name_candidate)
                matches.append({
                    "code": code_candidate,
                    "name": real_name,
                    "match_type": "substring",
                })

        # 3. 拼音匹配
        if _lazy_pinyin is not None and not matches:
            input_pinyin = _compute_pinyin(original_text)
            if input_pinyin:
                for pinyin, code_candidate in self._pinyin_to_code.items():
                    if input_pinyin in pinyin or pinyin in input_pinyin:
                        real_name = self._code_to_name.get(code_candidate, "")
                        matches.append({
                            "code": code_candidate,
                            "name": real_name,
                            "match_type": "pinyin",
                        })

        if matches:
            # 去重
            seen = set()
            unique_matches = []
            for m in matches:
                if m["code"] not in seen:
                    seen.add(m["code"])
                    unique_matches.append(m)
            return {
                "resolved": True,
                "input_type": "name",
                "exact": False,
                "matches": unique_matches[:10],
            }

        # 4. 没找到，返回空
        return {
            "resolved": False,
            "input_type": "unknown",
            "exact": False,
            "matches": [],
        }

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """前端自动补全搜索接口

        支持：前缀匹配 > 子串匹配 > 拼音匹配
        """
        if not query or not query.strip():
            return []

        q = query.strip()
        results: List[Dict[str, Any]] = []
        seen_codes: set = set()

        # 如果输入是代码，直接返回
        if is_stock_code(q.upper()):
            code = q.upper()
            name = self._code_to_name.get(code, code)
            return [{"code": code, "name": name, "match_type": "exact"}]

        # 1. 前缀匹配（名称以输入开头）
        for name_candidate, code_candidate in sorted(
            self._name_to_code.items(), key=lambda x: len(x[0])
        ):
            if len(results) >= limit:
                break
            if code_candidate in seen_codes:
                continue
            if name_candidate.startswith(q):
                real_name = self._code_to_name.get(code_candidate, name_candidate)
                results.append({
                    "code": code_candidate,
                    "name": real_name,
                    "match_type": "prefix",
                })
                seen_codes.add(code_candidate)

        # 2. 子串匹配
        for name_candidate, code_candidate in sorted(
            self._name_to_code.items(), key=lambda x: len(x[0])
        ):
            if len(results) >= limit:
                break
            if code_candidate in seen_codes:
                continue
            if q in name_candidate:
                real_name = self._code_to_name.get(code_candidate, name_candidate)
                results.append({
                    "code": code_candidate,
                    "name": real_name,
                    "match_type": "substring",
                })
                seen_codes.add(code_candidate)

        # 3. 拼音匹配
        if _lazy_pinyin is not None and len(results) < limit:
            input_pinyin = _compute_pinyin(q)
            if input_pinyin:
                for pinyin, code_candidate in sorted(
                    self._pinyin_to_code.items(), key=lambda x: len(x[0])
                ):
                    if len(results) >= limit:
                        break
                    if code_candidate in seen_codes:
                        continue
                    if pinyin.startswith(input_pinyin) or input_pinyin in pinyin:
                        real_name = self._code_to_name.get(code_candidate, "")
                        results.append({
                            "code": code_candidate,
                            "name": real_name or code_candidate,
                            "match_type": "pinyin",
                        })
                        seen_codes.add(code_candidate)

        return results[:limit]


# 全局单例
_resolver: Optional[StockNameResolver] = None
_resolver_lock = threading.Lock()


def get_resolver() -> StockNameResolver:
    """获取全局 StockNameResolver 单例"""
    global _resolver
    with _resolver_lock:
        if _resolver is None:
            _resolver = StockNameResolver()
        return _resolver


def resolve_symbol(text: str) -> Optional[str]:
    """便利函数：将用户输入解析为股票代码

    规则：
    - 已经是代码格式 → 原样返回（大写）
    - 是中文名称 → 解析为代码
    - 无法解析 → 返回 None（调用方决定是否用原输入）

    Args:
        text: 用户输入

    Returns:
        解析后的股票代码，或 None
    """
    if not text or not text.strip():
        return None

    raw = text.strip()
    resolver = get_resolver()

    # 尝试刷新索引
    resolver.refresh_index()

    # 已经是代码格式
    if is_stock_code(raw.upper()):
        return raw.upper()

    # 名称解析
    result = resolver.resolve(raw)
    if result["resolved"] and result["matches"]:
        return result["matches"][0]["code"]

    return None
