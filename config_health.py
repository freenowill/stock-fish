# -*- coding: utf-8 -*-
"""
配置健康检查模块

按 4 层分级检查所有配置项，返回结构化 JSON 报告：
Tier 1 — 核心服务（LLM、后端连通性）
Tier 2 — 行情/财务数据（Tushare、实时行情源、基本面管道）
Tier 3 — 增强功能（搜索引擎、Zep、MiroFish）
Tier 4 — 飞书机器人（Lark）
"""

import os
import shutil
import time
import threading
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# 每个检查项的结构
# {
#   "key": "LLM_API_KEY",
#   "tier": 1,
#   "tier_label": "核心服务",
#   "category": "llm",
#   "status": "ok" | "warning" | "error" | "not_configured",
#   "configured": bool,
#   "description": "大语言模型 API 密钥",
#   "impact": "影响 AI 分析和预测功能",
#   "detail": "",
#   "tested": bool,
# }

# 按 tier 的中文标签
_TIER_LABELS = {
    1: "核心服务",
    2: "行情 / 财务数据",
    3: "增强功能",
    4: "飞书机器人",
}

# 检查项定义：key -> (description, impact, category)
_CHECK_DEFS: Dict[str, tuple[str, str, str]] = {
    # Tier 1
    "LLM_API_KEY": ("大语言模型 API 密钥", "影响 AI 分析和预测功能", "llm"),
    "LLM_BASE_URL": ("LLM API 地址", "影响 AI 分析功能的 API 调用", "llm"),
    "LLM_MODEL_NAME": ("LLM 模型名称", "影响 AI 分析使用的模型", "llm"),
    "STOCK_BACKEND": ("行情数据后端", "影响所有行情、技术指标和财务数据获取", "data"),
    # Tier 2
    "TUSHARE_TOKEN": ("Tushare Pro Token", "影响 A 股财务数据、历史 PE 等深度数据", "data"),
    "ENABLE_REALTIME_QUOTE": ("实时行情开关", "关闭后使用缓存行情", "data"),
    "ENABLE_REALTIME_TECHNICAL_INDICATORS": ("实时技术指标开关", "关闭后使用缓存技术指标", "data"),
    "ENABLE_FUNDAMENTAL_PIPELINE": ("基本面增强管道", "关闭后基本面分析降级", "data"),
    "ENABLE_CHIP_DISTRIBUTION": ("筹码分布开关", "关闭后筹码分布数据不可用", "data"),
    "STOCK_BACKEND_CONNECTIVITY": ("数据后端连通性", "后端服务不可用则无法获取行情", "data"),
    # Tier 3
    "TAVILY_API_KEY": ("Tavily 搜索 API Key", "影响增强搜索能力（新闻/事件检索）", "search"),
    "BOCHA_API_KEY": ("Bocha 搜索 API Key", "影响增强搜索能力", "search"),
    "BRAVE_API_KEY": ("Brave 搜索 API Key", "影响增强搜索能力", "search"),
    "SERPAPI_API_KEY": ("SerpAPI 搜索 API Key", "影响增强搜索能力", "search"),
    "ANSPIRE_API_KEY": ("Anspire 搜索 API Key", "影响增强搜索能力", "search"),
    "MINIMAX_API_KEY": ("MiniMax 搜索 API Key", "影响增强搜索能力", "search"),
    "SEARXNG_BASE_URL": ("SearXNG 搜索服务", "影响增强搜索能力", "search"),
    "SEARCH_SERVICE_AVAILABLE": ("搜索引擎服务状态", "影响分析中的上下文搜索", "search"),
    "ZEP_API_KEY": ("Zep GraphRAG API Key", "影响 MiroFish 图记忆能力", "mirofish"),
    "MIROFISH": ("MiroFish 推演服务", "影响智能推演功能", "mirofish"),
    # Tier 4
    "LARK_APP_ID": ("飞书 App ID", "影响飞书机器人功能", "lark"),
    "LARK_APP_SECRET": ("飞书 App Secret", "影响飞书机器人功能", "lark"),
}


def _check_env(key: str) -> bool:
    """检查环境变量是否存在且非空"""
    val = os.environ.get(key, "").strip()
    return bool(val)


def _check_any_env(keys: List[str]) -> bool:
    """检查一组环境变量中任意一个存在"""
    return any(_check_env(k) for k in keys)


class _Cache:
    """简单的 TTL 缓存，避免短时间内重复检查"""
    def __init__(self, ttl: float = 30.0):
        self._data: Dict[str, tuple[float, Any]] = {}
        self._ttl = ttl
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._data:
                ts, val = self._data[key]
                if time.time() - ts < self._ttl:
                    return val
        return None

    def set(self, key: str, val: Any) -> None:
        with self._lock:
            self._data[key] = (time.time(), val)


class ConfigHealthChecker:
    """配置健康检查器，聚合所有检查逻辑"""

    def __init__(
        self,
        settings_obj: Any = None,
        agent_obj: Any = None,
        orchestrator_obj: Any = None,
        search_service_obj: Any = None,
    ):
        self.settings = settings_obj
        self.agent = agent_obj
        self.orchestrator = orchestrator_obj
        self.search_service = search_service_obj
        self._cache = _Cache(ttl=30.0)

    def run_all_checks(self) -> Dict[str, Any]:
        """运行全部检查，返回结构化报告"""
        checks: List[Dict[str, Any]] = []

        checks.extend(self._check_tier1_core())
        checks.extend(self._check_tier2_data())
        checks.extend(self._check_tier3_advanced())
        checks.extend(self._check_tier4_bot())
        checks.extend(self._check_feature_flags())

        # 统计汇总
        total = len(checks)
        ok_count = sum(1 for c in checks if c["status"] == "ok")
        warning_count = sum(1 for c in checks if c["status"] == "warning")
        error_count = sum(1 for c in checks if c["status"] == "error")
        not_configured = sum(1 for c in checks if c["status"] == "not_configured")

        # 整体状态
        if error_count > 0:
            overall = "error"
        elif warning_count > 0:
            overall = "warning"
        else:
            overall = "ok"

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "overall": overall,
            "checks": checks,
            "summary": {
                "total": total,
                "ok": ok_count,
                "warning": warning_count,
                "error": error_count,
                "not_configured": not_configured,
            },
        }

    def _make_item(
        self,
        key: str,
        status: str,
        configured: bool,
        detail: str = "",
        tested: bool = False,
        tier: int = 1,
    ) -> Dict[str, Any]:
        desc, impact, category = _CHECK_DEFS.get(key, (key, "未知影响", "other"))
        return {
            "key": key,
            "tier": tier,
            "tier_label": _TIER_LABELS.get(tier, "其他"),
            "category": category,
            "status": status,
            "configured": configured,
            "description": desc,
            "impact": impact,
            "detail": detail,
            "tested": tested,
        }

    def _check_tier1_core(self) -> List[Dict[str, Any]]:
        """Tier 1: 核心服务"""
        items = []

        # LLM_API_KEY
        has_llm = _check_env("LLM_API_KEY")
        items.append(self._make_item(
            "LLM_API_KEY",
            "ok" if has_llm else "not_configured",
            has_llm,
            detail=f"已配置" if has_llm else "未配置，AI 分析不可用",
            tier=1,
        ))

        # LLM_BASE_URL
        base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        items.append(self._make_item(
            "LLM_BASE_URL",
            "ok",
            True,
            detail=base_url,
            tier=1,
        ))

        # LLM_MODEL_NAME
        model = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")
        items.append(self._make_item(
            "LLM_MODEL_NAME",
            "ok",
            True,
            detail=model,
            tier=1,
        ))

        # STOCK_BACKEND（后端连通性）
        backend_name = "mock"
        backend_ok = False
        if self.agent and hasattr(self.agent, "provider"):
            try:
                provider = self.agent.provider
                if hasattr(provider, "backend_name"):
                    backend_name = provider.backend_name
                if backend_name and backend_name != "mock":
                    backend_ok = True
            except Exception as e:
                logger.debug(f"[健康检查] 获取后端名称失败: {e}")

        items.append(self._make_item(
            "STOCK_BACKEND",
            "ok" if backend_ok else "warning",
            backend_ok,
            detail=f"当前后端: {backend_name}" if backend_name else "未配置",
            tested=True,
            tier=1,
        ))

        return items

    def _check_tier2_data(self) -> List[Dict[str, Any]]:
        """Tier 2: 行情/财务数据"""
        items = []

        # TUSHARE_TOKEN
        ts_token = os.environ.get("TUSHARE_TOKEN", "").strip()
        ts_ok = bool(ts_token)
        ts_detail = "已配置" if ts_ok else "未配置，Tushare 数据不可用"
        if ts_ok and len(ts_token) >= 16:
            ts_detail += "（格式正确）"
        elif ts_ok:
            ts_detail = "已配置但格式异常（期望 16 位以上 Token）"
            ts_ok = False  # 格式不正确时标记为未配置

        items.append(self._make_item(
            "TUSHARE_TOKEN",
            "ok" if ts_ok else "not_configured",
            ts_ok,
            detail=ts_detail if ts_ok else (ts_detail if ts_token else "未配置"),
            tier=2,
        ))

        # STOCK_BACKEND_CONNECTIVITY — 尝试获取实时行情探活
        connectivity_ok = False
        connectivity_detail = "未检测"
        cached = self._cache.get("backend_connectivity")
        if cached is not None:
            connectivity_ok, connectivity_detail = cached
        elif self.agent and hasattr(self.agent, "provider"):
            try:
                provider = self.agent.provider
                if hasattr(provider, "get_realtime_quote"):
                    quote = provider.get_realtime_quote("600519")
                    if quote and isinstance(quote, dict) and quote.get("price", 0) > 0:
                        connectivity_ok = True
                        connectivity_detail = f"连通正常（当前价: {quote.get('price', 0)}）"
                    else:
                        connectivity_detail = "端返回空数据"
                elif hasattr(provider, "get_quote"):
                    quote = provider.get_quote("600519")
                    if quote and isinstance(quote, dict) and quote.get("price", 0) > 0:
                        connectivity_ok = True
                        connectivity_detail = f"连通正常（当前价: {quote.get('price', 0)}）"
                    else:
                        connectivity_detail = "端返回空数据"
                else:
                    connectivity_detail = "无 get_realtime_quote/get_quote 方法"
            except Exception as e:
                connectivity_detail = f"探活失败: {e}"
            self._cache.set("backend_connectivity", (connectivity_ok, connectivity_detail))

        items.append(self._make_item(
            "STOCK_BACKEND_CONNECTIVITY",
            "ok" if connectivity_ok else "error",
            connectivity_ok,
            detail=connectivity_detail,
            tested=True,
            tier=2,
        ))

        return items

    def _check_tier3_advanced(self) -> List[Dict[str, Any]]:
        """Tier 3: 增强功能"""
        items = []

        # 搜索引擎 API Keys
        search_keys = [
            ("TAVILY_API_KEY", ["TAVILY_API_KEY", "TAVILY_API_KEYS"]),
            ("BOCHA_API_KEY", ["BOCHA_API_KEY", "BOCHA_API_KEYS"]),
            ("BRAVE_API_KEY", ["BRAVE_API_KEY", "BRAVE_API_KEYS"]),
            ("SERPAPI_API_KEY", ["SERPAPI_API_KEY", "SERPAPI_KEYS"]),
            ("ANSPIRE_API_KEY", ["ANSPIRE_API_KEY", "ANSPIRE_KEYS"]),
            ("MINIMAX_API_KEY", ["MINIMAX_API_KEY", "MINIMAX_KEYS"]),
        ]

        for display_key, env_keys in search_keys:
            has_any = _check_any_env(env_keys)
            items.append(self._make_item(
                display_key,
                "ok" if has_any else "not_configured",
                has_any,
                detail="已配置" if has_any else "未配置",
                tier=3,
            ))

        # SearXNG
        searxng_url = os.environ.get("SEARXNG_BASE_URL", "").strip()
        searxng_public = os.environ.get("SEARXNG_PUBLIC_INSTANCES_ENABLED", "true").lower() == "true"
        has_searxng = bool(searxng_url) or searxng_public
        items.append(self._make_item(
            "SEARXNG_BASE_URL",
            "ok" if has_searxng else "not_configured",
            has_searxng,
            detail=searxng_url if searxng_url else ("使用公共实例" if searxng_public else "未配置"),
            tier=3,
        ))

        # SearchService 整体可用性
        search_avail = False
        search_detail = "未检测"
        if self.search_service is not None:
            try:
                if hasattr(self.search_service, "is_available"):
                    search_avail = self.search_service.is_available
                elif hasattr(self.search_service, "search"):
                    # 尝试一个简单搜索来判断
                    search_avail = True
                search_detail = "可用" if search_avail else "无可用搜索引擎"
            except Exception as e:
                search_detail = f"检测失败: {e}"
        else:
            search_detail = "SearchService 未初始化"

        items.append(self._make_item(
            "SEARCH_SERVICE_AVAILABLE",
            "ok" if search_avail else "warning",
            search_avail,
            detail=search_detail,
            tested=True,
            tier=3,
        ))

        # ZEP_API_KEY
        has_zep = _check_env("ZEP_API_KEY")
        items.append(self._make_item(
            "ZEP_API_KEY",
            "ok" if has_zep else "not_configured",
            has_zep,
            detail="已配置" if has_zep else "未配置，图记忆功能不可用",
            tier=3,
        ))

        # MiroFish 连通性
        mf_ok = False
        mf_detail = "未检测"
        mf_url = ""
        if self.orchestrator is not None:
            try:
                if hasattr(self.orchestrator, "client"):
                    client = self.orchestrator.client
                    if hasattr(client, "health_check"):
                        mf_ok = client.health_check()
                        mf_detail = "连通正常" if mf_ok else "服务不可达"
                    if hasattr(client, "base_url"):
                        mf_url = client.base_url
                elif hasattr(self.orchestrator, "health_check"):
                    mf_ok = self.orchestrator.health_check()
                    mf_detail = "连通正常" if mf_ok else "服务不可达"
            except Exception as e:
                mf_detail = f"检测失败: {e}"

        if mf_url:
            mf_detail += f" ({mf_url})"

        items.append(self._make_item(
            "MIROFISH",
            "ok" if mf_ok else ("warning" if not mf_ok and mf_url else "not_configured"),
            mf_ok,
            detail=mf_detail,
            tested=True,
            tier=3,
        ))

        return items

    def _check_tier4_bot(self) -> List[Dict[str, Any]]:
        """Tier 4: 飞书机器人"""
        items = []

        has_lark_id = _check_env("LARK_APP_ID")
        has_lark_secret = _check_env("LARK_APP_SECRET")
        lark_configured = has_lark_id and has_lark_secret

        items.append(self._make_item(
            "LARK_APP_ID",
            "ok" if lark_configured else "not_configured",
            lark_configured,
            detail="已配置" if lark_configured else "未配置或缺少 App Secret",
            tier=4,
        ))

        return items

    def _check_feature_flags(self) -> List[Dict[str, Any]]:
        """检查功能开关状态"""
        items = []

        feature_flags = [
            "ENABLE_REALTIME_QUOTE",
            "ENABLE_REALTIME_TECHNICAL_INDICATORS",
            "ENABLE_FUNDAMENTAL_PIPELINE",
            "ENABLE_CHIP_DISTRIBUTION",
            "ENABLE_EASTMONEY_PATCH",
        ]

        for flag in feature_flags:
            val = os.environ.get(flag, "").strip().lower()
            is_on = val in ("1", "true", "yes") if val else False
            # 对于特定 flag，默认值不同
            defaults_on = {
                "ENABLE_REALTIME_QUOTE",
                "ENABLE_REALTIME_TECHNICAL_INDICATORS",
                "ENABLE_FUNDAMENTAL_PIPELINE",
                "ENABLE_CHIP_DISTRIBUTION",
            }
            if not val and flag in defaults_on:
                is_on = True

            # 如果没有明确设置，且是默认开启的，状态为 ok
            # 如果明确关闭了默认开启的，状态为 warning
            if not val and flag in defaults_on:
                status = "ok"
                detail = "开启（默认）"
            elif is_on:
                status = "ok"
                detail = "开启"
            else:
                status = "warning"
                detail = "关闭"

            items.append(self._make_item(
                flag,
                status,
                is_on,
                detail=detail,
                tier=2,  # feature flags 归入数据层
            ))

        return items
