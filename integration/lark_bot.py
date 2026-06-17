"""
飞书 Bot 核心 — WebSocket 生命周期、消息路由、分析编排。

通过 WebSocket 长连接接收消息，无需公网 IP。
运行时: python integration/lark_bot.py
"""

import asyncio
import json
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# 确保 StockFish 根目录在 sys.path 中（兼容从任意目录运行）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import lark_oapi as lark
from loguru import logger

from config import settings
from integration.lark_card import CardBuilder
from integration.lark_client import LarkClient
from integration.lark_prefs import UserPrefManager

# ── 消息解析 ──────────────────────────────────────────────

# 匹配 A 股代码：6 位数字，以 000/001/002/003/300/301/600/601/603/605/688 开头
# 也支持 .BJ 后缀的北交所代码
_STOCK_PATTERN = re.compile(
    r"\b(?:00[0-3]\d{3}|30[0-1]\d{3}|60[0-5]\d{3}|68[89]\d{3})"
    r"(?:\.(?:BJ|SZ|SH))?\b"
)

# 匹配 / 分割的多只股票代码
_MULTI_STOCK_PATTERN = re.compile(
    r"^[\s]*"
    r"(?:" + _STOCK_PATTERN.pattern + r")"
    r"(?:\s*/\s*" + _STOCK_PATTERN.pattern + r")+"
    r"[\s]*$"
)

# 提取 --master <key> 参数
_MASTER_ARG = re.compile(r"--master\s+(\w+)")

# 匹配 @Bot 提及的纯文本提取
_MENTION_CLEAN = re.compile(r"@\S+\s*")


@dataclass
class ParsedMessage:
    """解析后的用户消息。"""
    raw: str
    clean_text: str              # 去掉 @Bot 后的纯文本
    is_master_command: bool = False
    master_cmd: str = ""          # "buffett" | "off" | "list"
    is_help: bool = False
    is_batch: bool = False
    symbols: List[str] = field(default_factory=list)
    inline_master: str = ""       # 消息中 --master <key> 指定的值


def parse_message(text: str) -> Optional[ParsedMessage]:
    """解析用户消息，提取意图和参数。

    优先级:
    1. /master <key|off|list>  → 大师偏好设置
    2. /help /帮助                → 帮助
    3. /update_data               → Qlib 数据更新
    4. /qlib_inference [code]     → Qlib 推理
    5. 多只股票 (/ 分割)          → 批量分析
    6. 单只股票                   → 单股分析
    7. 不匹配                     → None
    """
    # 去掉 @Bot 提及
    clean = _MENTION_CLEAN.sub("", text).strip()
    result = ParsedMessage(raw=text, clean_text=clean)

    # 提取 --master 参数
    m = _MASTER_ARG.search(clean)
    if m:
        result.inline_master = m.group(1).lower()
        clean = _MASTER_ARG.sub("", clean).strip()
        result.clean_text = clean

    # 1. /master 命令
    if clean.lower().startswith("/master"):
        parts = clean.split(None, 1)
        result.master_cmd = parts[1].strip().lower() if len(parts) > 1 else "list"
        result.is_master_command = True
        return result

    # 2. /help 命令
    if clean.lower() in ("/help", "help", "帮助", "/帮助"):
        result.is_help = True
        return result

    # 3. /update_data 命令
    if clean.lower().startswith("/update_data"):
        result.is_help = False
        result.symbols = ["__update_data__"]
        return result

    # 4. /qlib_inference 命令
    if clean.lower().startswith("/qlib_inference"):
        result.is_help = False
        symbols = _STOCK_PATTERN.findall(clean)
        result.symbols = ["__qlib_infer__"]
        return result

    # 5. 批量: 检测多只股票 / 分割
    if "/" in clean and _MULTI_STOCK_PATTERN.match(clean):
        raw_symbols = [s.strip() for s in clean.split("/") if s.strip()]
        symbols = [_normalize_symbol(s) for s in raw_symbols if _STOCK_PATTERN.match(s)]
        if len(symbols) >= 2:
            result.is_batch = True
            result.symbols = symbols
            return result

    # 6. 单股
    symbols = _STOCK_PATTERN.findall(clean)
    if symbols:
        result.symbols = [_normalize_symbol(s) for s in symbols[:1]]
        return result

    # 7. 不匹配
    return None


def _normalize_symbol(symbol: str) -> str:
    """统一股票代码格式：去掉后缀，转大写。"""
    s = symbol.strip().upper()
    # 保留 .BJ 后缀（北交所），去掉 .SZ/.SH
    if s.endswith(".SZ") or s.endswith(".SH"):
        s = s[:-3]
    return s


# ── Bot 主类 ──────────────────────────────────────────────

class StockFishBot:
    """飞书 Bot — WebSocket 长连接 + 消息路由 + 分析编排。"""

    def __init__(self):
        app_id = settings.LARK_APP_ID
        app_secret = settings.LARK_APP_SECRET
        if not app_id or not app_secret:
            raise RuntimeError(
                "LARK_APP_ID 和 LARK_APP_SECRET 未配置，请在 .env 中设置"
            )

        self._bot_name = settings.LARK_BOT_NAME or "StockFish"
        self._client = lark.Client.builder() \
            .app_id(app_id) \
            .app_secret(app_secret) \
            .log_level(lark.LogLevel.INFO) \
            .build()

        self._api = LarkClient(base_url=settings.STOCKFISH_API_URL)
        self._cards = CardBuilder()
        self._prefs = UserPrefManager()

        # 专用 asyncio 事件循环（在后台线程运行），用于异步 API 调用
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop, daemon=True, name="lark-async"
        )
        self._loop_thread.start()

    def _run_loop(self) -> None:
        """后台线程：运行 asyncio 事件循环。"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _dispatch(self, coro) -> None:
        """将异步任务调度到后台事件循环。"""
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    def start(self) -> None:
        """启动 WebSocket 连接，进入消息循环（阻塞主线程）。"""
        logger.info(f"[StockFishBot] 启动中... bot_name={self._bot_name}")

        handler = lark.EventDispatcherHandler.builder("", "") \
            .register_p2_im_message_receive_v1(self._on_message) \
            .build()

        ws_client = lark.ws.Client(
            settings.LARK_APP_ID,
            settings.LARK_APP_SECRET,
            event_handler=handler,
        )
        logger.info("[StockFishBot] WebSocket 已连接，等待消息...")
        ws_client.start()

    def stop(self) -> None:
        """停止 Bot（WebSocket 由 SDK 管理，随进程退出）。"""
        self._loop.call_soon_threadsafe(self._loop.stop)
        logger.info("[StockFishBot] 已停止")

    # ── 消息处理入口 ───────────────────────────────────────

    def _on_message(self, data: lark.im.v1.P2ImMessageReceiveV1) -> None:
        """飞书消息事件回调（在 SDK 的工作线程中调用，同步方法）。"""
        event = data.event
        message = event.message
        chat_id = message.chat_id
        open_id = event.sender.sender_id.open_id if event.sender else None

        # 只处理文本消息
        if message.message_type != "text":
            return

        text = json.loads(message.content).get("text", "").strip()
        if not text:
            return

        logger.info(f"[StockFishBot] 收到消息 chat={chat_id} text={text[:80]}")

        # 保存用户 → chat 映射（供后续推送用）
        if open_id:
            self._prefs.save_chat_map(open_id, chat_id)

        # 解析消息
        parsed = parse_message(text)
        if parsed is None:
            self._dispatch(self._send_help(chat_id, "未识别股票代码，请发送 600519 这样的代码。"))
            return

        # 分发到异步处理器
        if parsed.is_master_command:
            self._dispatch(self._handle_master_cmd(chat_id, open_id, parsed))
        elif parsed.is_help:
            self._dispatch(self._send_help(chat_id))
        elif parsed.symbols and parsed.symbols[0] == "__update_data__":
            self._dispatch(self._handle_update_data(chat_id))
        elif parsed.symbols and parsed.symbols[0] == "__qlib_infer__":
            self._dispatch(self._handle_qlib_inference(chat_id, parsed))
        elif parsed.is_batch:
            self._dispatch(self._handle_batch(chat_id, open_id, parsed))
        else:
            self._dispatch(self._handle_single(chat_id, open_id, parsed))

    # ── 大师偏好处理 ──────────────────────────────────────

    async def _handle_master_cmd(
        self, chat_id: str, open_id: Optional[str], parsed: ParsedMessage
    ) -> None:
        """处理 /master 命令。"""
        cmd = parsed.master_cmd

        if cmd == "off":
            if open_id:
                self._prefs.set_master(open_id, "")
            await self._send_text(chat_id, "✅ 大师模式已关闭，分析将不指定投资大师。")

        elif cmd == "list":
            masters = await self._api.get_masters()
            if masters:
                lines = ["🎓 **可选投资大师**\n"]
                for m in masters:
                    key = m.get("key", "")
                    name = m.get("name", "")
                    style = m.get("style", "")
                    lines.append(f"- **{key}** — {name}（{style}）")
                lines.append(f"\n使用 `/master <key>` 设置默认大师，如 `/master buffett`")
                lines.append("使用 `/master off` 关闭大师模式。")
                await self._send_text(chat_id, "\n".join(lines))
            else:
                await self._send_text(chat_id, "⚠️ 未找到可用大师列表，请检查 LLM 配置。")

        else:
            # 验证大师 key 是否存在
            masters = await self._api.get_masters()
            valid_keys = {m.get("key", "") for m in masters}
            if cmd in valid_keys:
                if open_id:
                    self._prefs.set_master(open_id, cmd)
                master_name = next(
                    (m.get("name", cmd) for m in masters if m.get("key") == cmd), cmd
                )
                await self._send_text(
                    chat_id, f"✅ 已设置默认大师为 **{master_name}**（{cmd}）。\n"
                    f"后续分析将自动使用该大师视角。使用 `--master <key>` 可单次覆盖。"
                )
            else:
                await self._send_text(
                    chat_id,
                    f"⚠️ 未知大师 `{cmd}`。请使用 `/master list` 查看可选大师。",
                )

    # ── 单股分析 ───────────────────────────────────────────

    async def _handle_single(
        self, chat_id: str, open_id: Optional[str], parsed: ParsedMessage
    ) -> None:
        """处理单股分析请求。"""
        symbol = parsed.symbols[0]
        master = parsed.inline_master or self._prefs.get_master(open_id)

        # 发送初始进度
        master_label = f"（{self._master_label(master)}视角）" if master else ""
        await self._send_text(chat_id, f"⏳ **{symbol}** 开始分析...{master_label}")

        # 后台心跳：每 12 秒汇报等待时间
        heartbeat_running = True

        async def _heartbeat():
            waited = 0
            while heartbeat_running:
                await asyncio.sleep(12)
                waited += 12
                if heartbeat_running:
                    await self._send_text(
                        chat_id,
                        f"⏳ **{symbol}** 分析中...（已等待 {waited} 秒）{master_label}",
                    )

        heartbeat_task = asyncio.create_task(_heartbeat())

        try:
            result = await self._api.analyze(symbol, master=master)
        except Exception as e:
            heartbeat_running = False
            heartbeat_task.cancel()
            logger.error(f"[StockFishBot] analyze failed: {symbol} — {e}")
            await self._send_error(chat_id, symbol, str(e))
            return

        heartbeat_running = False
        heartbeat_task.cancel()

        if result.get("status") == "error":
            await self._send_error(chat_id, symbol, result.get("error", "未知错误"))
            return

        # 发送结果卡片
        await self._send_text(chat_id, f"✅ **{symbol}** 分析完成！")
        if master:
            card = self._cards.build_master_card(result)
        else:
            card = self._cards.build_analysis_card(result)
        await self._send_card(chat_id, card)

    # ── 批量分析 ──────────────────────────────────────────

    async def _handle_batch(
        self, chat_id: str, open_id: Optional[str], parsed: ParsedMessage
    ) -> None:
        """批量分析——每完成一只立刻返回卡片，不等待全部。"""
        symbols = parsed.symbols
        master = parsed.inline_master or self._prefs.get_master(open_id)
        total = len(symbols)

        await self._send_text(
            chat_id,
            f"📊 批量分析启动：**{total}** 只股票\n"
            + "、".join(symbols[:6])
            + (f"等..." if total > 6 else "")
            + "\n\n分析好一支就立刻返回，不用等到全部完成。",
        )

        # 回调：每完成一只立刻发卡片
        async def on_progress(sym: str, data: dict, idx: int, total_count: int):
            if master:
                card = self._cards.build_master_card(data)
            else:
                card = self._cards.build_analysis_card(data)
            await self._send_text(chat_id, f"✅ **{idx}/{total_count}** — {sym} 完成")
            await self._send_card(chat_id, card)

        try:
            result = await self._api.batch_analyze_with_progress(
                symbols, master=master, progress_callback=on_progress
            )
        except Exception as e:
            logger.error(f"[StockFishBot] batch_analyze failed: {e}")
            await self._send_error(chat_id, "批量分析", str(e))
            return

        if result.get("status") == "error":
            await self._send_error(chat_id, "批量分析", result.get("error", "未知错误"))
            return

        # 全部完成，发汇总卡片
        success = result.get("success_count", 0)
        await self._send_text(chat_id, f"✅ 批量分析全部完成：{success}/{total} 成功")
        card = self._cards.build_batch_card(result)
        await self._send_card(chat_id, card)

    # ── Qlib 数据更新 ──────────────────────────────────────

    async def _handle_update_data(self, chat_id: str) -> None:
        """处理 /update_data 命令。"""
        await self._send_text(chat_id, "📦 Qlib 数据更新启动...\n下载可能需要几分钟，请耐心等待。")

        heartbeat_running = True

        async def _heartbeat():
            waited = 0
            while heartbeat_running:
                await asyncio.sleep(120)  # 2 分钟汇报一次
                waited += 2
                if heartbeat_running:
                    await self._send_text(
                        chat_id,
                        f"⏳ 数据下载中...（已等待 {waited} 分钟）",
                    )

        heartbeat_task = asyncio.create_task(_heartbeat())

        try:
            result = await self._api.qlib_data_update()
        except Exception as e:
            heartbeat_running = False
            heartbeat_task.cancel()
            logger.error(f"[StockFishBot] qlib_data_update failed: {e}")
            await self._send_text(chat_id, f"❌ 数据更新失败: {e}")
            return

        heartbeat_running = False
        heartbeat_task.cancel()

        if result.get("status") == "completed":
            message = result.get("message", "更新完成")
            await self._send_text(chat_id, f"✅ Qlib 数据更新完成！\n{message}")
        else:
            error = result.get("error", "未知错误")
            await self._send_text(chat_id, f"❌ 数据更新失败: {error}")

    # ── Qlib 推理 ──────────────────────────────────────────

    async def _handle_qlib_inference(
        self, chat_id: str, parsed: ParsedMessage
    ) -> None:
        """处理 /qlib_inference 命令。"""
        DEFAULT_MODEL = "2026-06-12-csi300-alpha158"
        await self._send_text(
            chat_id, f"🤖 Qlib 推理启动\n模型: {DEFAULT_MODEL}\n每 5 分钟汇报一次进度..."
        )

        # 后台心跳
        heartbeat_running = True

        async def _heartbeat():
            waited = 0
            while heartbeat_running:
                await asyncio.sleep(300)  # 5 分钟汇报一次
                waited += 5
                if heartbeat_running:
                    await self._send_text(
                        chat_id,
                        f"⏳ Qlib 推理中...（已等待 {waited} 分钟）",
                    )

        heartbeat_task = asyncio.create_task(_heartbeat())

        try:
            result = await self._api.qlib_infer(model=DEFAULT_MODEL)
        except Exception as e:
            heartbeat_running = False
            heartbeat_task.cancel()
            logger.error(f"[StockFishBot] qlib_infer failed: {e}")
            await self._send_text(chat_id, f"❌ Qlib 推理失败: {e}")
            return

        heartbeat_running = False
        heartbeat_task.cancel()

        if result.get("status") == "completed":
            count = result.get("count", 0)
            stocks = result.get("stocks", "")
            pred_date = result.get("pred_date", "")
            message = result.get("message", "推理完成")

            resp = [f"✅ Qlib 推理完成！", f"模型: {DEFAULT_MODEL}"]
            if pred_date:
                resp.append(f"预测日期: {pred_date}")
            if count:
                resp.append(f"选出股票: {count} 只")
            resp.append(f"\n{message}")
            if stocks:
                resp.append(f"\n**选股结果**:\n{stocks}")
            await self._send_text(chat_id, "\n".join(resp))
        else:
            error = result.get("error", "未知错误")
            await self._send_text(chat_id, f"❌ Qlib 推理失败: {error}")

    # ── 消息发送 ───────────────────────────────────────────

    async def _send_text(self, chat_id: str, content: str) -> None:
        """发送纯文本消息（通过 IM API）。"""
        try:
            req = lark.im.v1.CreateMessageRequest.builder() \
                .receive_id_type("chat_id") \
                .request_body(
                    lark.im.v1.CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("text")
                    .content(json.dumps({"text": content}))
                    .build()
                ).build()
            resp = self._client.im.v1.message.create(req)
            if not resp.success():
                logger.error(
                    f"[StockFishBot] send_text failed: "
                    f"code={resp.code} msg={resp.msg}"
                )
        except Exception as e:
            logger.error(f"[StockFishBot] send_text exception: {e}")

    async def _send_card(self, chat_id: str, card: Dict[str, Any]) -> None:
        """发送互动卡片。"""
        try:
            req = lark.im.v1.CreateMessageRequest.builder() \
                .receive_id_type("chat_id") \
                .request_body(
                    lark.im.v1.CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("interactive")
                    .content(json.dumps(card, ensure_ascii=False))
                    .build()
                ).build()
            resp = self._client.im.v1.message.create(req)
            if not resp.success():
                logger.error(
                    f"[StockFishBot] send_card failed: "
                    f"code={resp.code} msg={resp.msg}"
                )
        except Exception as e:
            logger.error(f"[StockFishBot] send_card exception: {e}")

    async def _send_help(self, chat_id: str, extra: str = "") -> None:
        """发送帮助卡片。"""
        if extra:
            await self._send_text(chat_id, extra)
        card = self._cards.build_help_card()
        await self._send_card(chat_id, card)

    async def _send_error(self, chat_id: str, symbol: str, error_msg: str) -> None:
        """发送错误提示。"""
        card = self._cards.build_error_card(symbol, error_msg)
        await self._send_card(chat_id, card)

    # ── 工具 ────────────────────────────────────────────────

    def _master_label(self, key: str) -> str:
        """大师 key → 显示名（幂等缓存）。"""
        if not key:
            return ""
        # 简单映射，不需要每次查 API
        LABELS = {
            "graham": "格雷厄姆·深度价值",
            "buffett": "巴菲特·价值质量",
            "fisher": "费雪·成长",
            "lynch": "林奇·GARP",
            "templeton": "邓普顿·逆向",
            "soros": "索罗斯·反身性",
            "dalio": "达利欧·全天候",
        }
        return LABELS.get(key, key)


# ── 入口 ──────────────────────────────────────────────────

def main():
    """独立进程入口。"""
    logger.info("StockFish Feishu Bot 启动...")
    bot = StockFishBot()
    try:
        bot.start()  # 阻塞，运行 WebSocket 消息循环
    except KeyboardInterrupt:
        logger.info("收到中断信号，停止 Bot...")
        bot.stop()


if __name__ == "__main__":
    main()
