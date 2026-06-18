"""
飞书 Bot 专用 HTTP 客户端 — 封装 StockFish Flask API。

所有方法都是 async，通过 aiohttp 调用 localhost:8000。
/analyze 阻塞 30-60s，/batch/analyze 异步轮询。
"""

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional

import aiohttp
from loguru import logger


class LarkClient:
    """异步 HTTP 客户端，封装 StockFish 分析 API。"""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self._base = base_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=1200)  # /analyze 可能 10min+
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── 单股分析 ──────────────────────────────────────────

    async def analyze(
        self,
        symbol: str,
        master: str = "",
        cost_price: float = 0.0,
        shares: int = 0,
        total_assets: float = 0.0,
        available_cash: float = 0.0,
    ) -> Dict[str, Any]:
        """POST /api/analyze — 同步阻塞，返回完整 AnalysisState dict。"""
        session = await self._ensure_session()
        payload: Dict[str, Any] = {"symbol": symbol}
        if cost_price:
            payload["cost_price"] = cost_price
        if shares:
            payload["shares"] = shares
        if total_assets:
            payload["total_assets"] = total_assets
        if available_cash:
            payload["available_cash"] = available_cash
        if master:
            payload["master"] = master

        logger.info(f"[LarkClient] analyze: {symbol}" + (f" master={master}" if master else ""))
        async with session.post(f"{self._base}/api/analyze", json=payload) as resp:
            data = await resp.json()
            logger.info(
                f"[LarkClient] analyze done: {symbol} status={data.get('status')}"
            )
            return data

    # ── 批量分析 ──────────────────────────────────────────

    async def batch_analyze(
        self,
        symbols: List[str],
        master: str = "",
        cost_prices: Optional[List[float]] = None,
        shares_list: Optional[List[int]] = None,
        total_assets: float = 0.0,
        available_cash: float = 0.0,
    ) -> Dict[str, Any]:
        """POST /api/batch/analyze — 异步，轮询直到完成。

        1. 启动批量任务 → 获得 task_id
        2. 每 2 秒轮询 GET /api/batch/analyze/<task_id>
        3. 完成后返回完整结果
        """
        session = await self._ensure_session()

        # 启动
        payload: Dict[str, Any] = {"symbols": "/".join(symbols)}  # API 要求 / 分隔字符串
        if master:
            payload["master"] = master
        if cost_prices:
            payload["cost_prices"] = cost_prices
        if shares_list:
            payload["shares"] = shares_list
        if total_assets:
            payload["total_assets"] = total_assets
        if available_cash:
            payload["available_cash"] = available_cash

        logger.info(
            f"[LarkClient] batch_analyze start: {len(symbols)} stocks"
            + (f" master={master}" if master else "")
        )
        async with session.post(f"{self._base}/api/batch/analyze", json=payload) as resp:
            task = await resp.json()

        task_id = task.get("task_id")
        if not task_id:
            logger.error(f"[LarkClient] batch_analyze: no task_id in response: {task}")
            return task

        # 轮询
        poll_interval = 2.0
        max_wait = 1800.0  # 最长等 30 分钟
        waited = 0.0
        while waited < max_wait:
            await asyncio.sleep(poll_interval)
            waited += poll_interval
            async with session.get(
                f"{self._base}/api/batch/analyze/{task_id}"
            ) as resp:
                status = await resp.json()
            st = status.get("status", "")
            if st == "completed":
                logger.info(
                    f"[LarkClient] batch_analyze done: "
                    f"{status.get('success_count', 0)}/{status.get('total', 0)} ok"
                )
                return status
            elif st == "error":
                logger.error(f"[LarkClient] batch_analyze error: {status.get('error')}")
                return status
            # "queued" or "running" → 继续等

        logger.warning(f"[LarkClient] batch_analyze timeout after {max_wait}s")
        return {"task_id": task_id, "status": "timeout", "error": "批量分析超时"}

    async def batch_analyze_with_progress(
        self,
        symbols: List[str],
        master: str = "",
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """POST /api/batch/analyze — 逐只返回，无超时。

        progress_callback(symbol, data, index, total) — 每完成一只调用一次
        持续轮询直到全部完成，无超时限制。
        """
        session = await self._ensure_session()

        payload: Dict[str, Any] = {"symbols": "/".join(symbols)}
        if master:
            payload["master"] = master

        logger.info(f"[LarkClient] batch_with_progress: {len(symbols)} stocks")
        async with session.post(
            f"{self._base}/api/batch/analyze", json=payload
        ) as resp:
            task = await resp.json()

        task_id = task.get("task_id")
        if not task_id:
            return task

        seen: set = set()
        poll_interval = 3.0

        while True:
            await asyncio.sleep(poll_interval)
            async with session.get(
                f"{self._base}/api/batch/analyze/{task_id}"
            ) as resp:
                status = await resp.json()

            results = status.get("results", [])
            total = status.get("total", len(symbols))

            # 逐只返回已完成的
            for r in results:
                sym = r.get("symbol", "")
                st = r.get("status", "")
                data = r.get("data") or {}
                if sym and st == "complete" and sym not in seen:
                    seen.add(sym)
                    if progress_callback:
                        try:
                            await progress_callback(sym, data, len(seen), total)
                        except Exception:
                            pass

            st = status.get("status", "")
            if st == "completed":
                logger.info(
                    f"[LarkClient] batch done: {status.get('success_count', 0)}/{total}"
                )
                return status
            elif st == "error":
                return status
            # 无超时，继续轮询

    # ── 大师列表 ──────────────────────────────────────────

    async def get_masters(self) -> List[Dict[str, str]]:
        """GET /api/masters — 返回所有可选大师。"""
        session = await self._ensure_session()
        async with session.get(f"{self._base}/api/masters") as resp:
            data = await resp.json()
            return data.get("masters", [])

    # ── Qlib 数据更新 ────────────────────────────────────

    async def qlib_data_update(self) -> Dict[str, Any]:
        """POST /api/qlib/data/update — 下载最新 qlib 数据。"""
        session = await self._ensure_session()
        logger.info("[LarkClient] qlib_data_update start")
        async with session.post(f"{self._base}/api/qlib/data/update") as resp:
            task = await resp.json()
        return await self._poll_qlib_task(task.get("task_id", ""), "data/update")

    # ── Qlib 推理 ──────────────────────────────────────────

    async def qlib_infer(self, model: str = "2026-06-12-csi300-alpha158-fintune") -> Dict[str, Any]:
        """POST /api/qlib/infer — Qlib 模型推理。"""
        session = await self._ensure_session()
        logger.info(f"[LarkClient] qlib_infer model={model}")
        async with session.post(
            f"{self._base}/api/qlib/infer", json={"model": model}
        ) as resp:
            task = await resp.json()
        return await self._poll_qlib_task(task.get("task_id", ""), "infer")

    async def _poll_qlib_task(self, task_id: str, path: str) -> Dict[str, Any]:
        """轮询 qlib SSE 流直到任务完成（最长 2 小时）。"""
        if not task_id:
            return {"status": "error", "error": "未获取到 task_id"}

        session = await self._ensure_session()
        stream_url = f"{self._base}/api/qlib/{path}/{task_id}/stream"
        poll_interval = 15  # 每 15 秒查一次
        max_wait = 7200     # 最长 2 小时
        waited = 0

        while waited < max_wait:
            await asyncio.sleep(poll_interval)
            waited += poll_interval

            try:
                # 每次请求 SSE 端点，读最新的状态
                async with session.get(
                    stream_url,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    text = await resp.text()
                    # 解析 SSE: "data: {...}\n\n"
                    last_status = None
                    for line in text.split("\n"):
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                last_status = data
                            except json.JSONDecodeError:
                                pass

                    if last_status:
                        st = last_status.get("status", "")
                        if st == "completed":
                            logger.info("[LarkClient] qlib done")
                            return {"status": "completed", **last_status}
                        elif st == "failed":
                            return {
                                "status": "error",
                                "error": last_status.get("message", "推理失败"),
                            }
                        # running → 继续等
            except asyncio.TimeoutError:
                logger.debug(f"[LarkClient] poll timeout at {waited}s, retrying...")
            except Exception as e:
                logger.warning(f"[LarkClient] poll error at {waited}s: {e}")

        return {"task_id": task_id, "status": "timeout", "error": "推理超时（超过 2 小时）"}

    # ── 健康检查 ──────────────────────────────────────────

    async def health_check(self) -> bool:
        """检查 Flask API 是否可用。"""
        try:
            session = await self._ensure_session()
            async with session.get(
                f"{self._base}/api/config", timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                return resp.status == 200
        except Exception:
            return False
