"""
飞书消息卡片构建器 — Card JSON v2.0 格式。

三种分析卡片 + 帮助卡片 + 错误卡片。
A 股配色：红涨绿跌。
"""

from typing import Any, Dict, List, Optional, Union

# ── 信号 → 飞书 Header 颜色映射 ────────────────────────────

_SIGNAL_COLORS: Dict[str, str] = {
    "强烈看多": "red",
    "看多": "red",
    "偏多": "orange",
    "中性": "grey",
    "偏空": "turquoise",
    "看空": "green",
    "强烈看空": "green",
}

_SIGNAL_EMOJI: Dict[str, str] = {
    "强烈看多": "🟥",
    "看多": "🟧",
    "偏多": "🟨",
    "中性": "⬜",
    "偏空": "🟩",
    "看空": "🟦",
    "强烈看空": "🟩",
}


def _signal_color(label: str) -> str:
    for k, v in _SIGNAL_COLORS.items():
        if k in label:
            return v
    return "grey"


def _signal_emoji(label: str) -> str:
    for k, v in _SIGNAL_EMOJI.items():
        if k in label:
            return v
    return "⬜"


def _safe_str(val: Any, default: str = "--") -> str:
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return f"{val:.2f}"
    if isinstance(val, dict):
        return default
    return str(val)


def _safe_money(val: Any) -> str:
    """安全格式化金额，非数字返回 --。"""
    if val is None:
        return "--"
    if isinstance(val, (int, float)):
        return f"¥{val:.2f}"
    return "--"


def _safe_score(val: Any) -> str:
    """安全格式化评分。"""
    if val is None:
        return "--"
    if isinstance(val, (int, float)):
        return f"+{val:.1f}" if val > 0 else f"{val:.1f}"
    return str(val)


def _pct(val: Any) -> str:
    if val is None:
        return "--"
    if isinstance(val, (int, float)):
        sign = "+" if val > 0 else ""
        return f"{sign}{val:.2f}%"
    return "--"


def _direction_cn(d: str) -> str:
    m = {
        "up": "上涨", "down": "下跌", "neutral": "震荡",
        "bullish": "看多", "bearish": "看空",
        "上涨": "上涨", "下跌": "下跌", "震荡": "震荡",
        "看多": "看多", "看空": "看空", "中性": "震荡",
    }
    return m.get(d, d or "--")


# ── 基础卡片骨架 ──────────────────────────────────────────

def _card(
    title: str,
    subtitle: str = "",
    color: str = "blue",
    elements: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    return {
        "schema": "2.0",
        "config": {"update_multi": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "subtitle": {"tag": "plain_text", "content": subtitle},
            "template": color,
        },
        "body": {
            "direction": "vertical",
            "padding": "12px 12px 12px 12px",
            "elements": elements or [],
        },
    }


def _md(content: str) -> Dict:
    return {"tag": "markdown", "content": content}


def _hr() -> Dict:
    return {"tag": "hr"}


def _button(text: str, url: str, btn_type: str = "primary") -> Dict:
    return {
        "tag": "button",
        "text": {"tag": "plain_text", "content": text},
        "type": btn_type,
        "width": "default",
        "behaviors": [{"type": "open_url", "default_url": url}],
    }


# ── CardBuilder ─────────────────────────────────────────────

class CardBuilder:
    """构建各种飞书消息卡片。"""

    def __init__(self, report_base_url: str = ""):
        self._report_base = report_base_url  # 完整报告的基础 URL

    # ── 普通分析卡片 ──────────────────────────────────────

    def build_analysis_card(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """从 /api/analyze 结果构建普通分析卡片。"""
        symbol = result.get("symbol", "--")
        name = result.get("stock_name", symbol)
        score = result.get("score_breakdown", {}) or {}
        final_score = score.get("final", 0)
        label = score.get("label", "中性")
        quote = result.get("quote", {}) or {}
        signals = result.get("signals", {}) or {}
        overall = signals.get("overall", {}) or {}
        pred = result.get("prediction_summary", {}) or {}
        action = pred.get("suggested_action", {}) or {}

        color = _signal_color(label)
        emoji = _signal_emoji(label)

        # ── Header ──
        score_str = _safe_score(final_score)
        title = f"{emoji} {label} {score_str}  {name} ({symbol})"

        # ── 行情行 ──
        price = quote.get("price")
        change_pct_val = quote.get("change_pct")
        pe = quote.get("pe")
        pb = quote.get("pb")
        price_s = _safe_money(price)
        change_s = _pct(change_pct_val) if change_pct_val is not None else "--"
        pe_s = _safe_str(pe)
        pb_s = _safe_str(pb)

        # ── 估值 ──
        val_level = result.get("valuation_level", "--") or "--"
        val_pct = result.get("valuation_percentile")
        val_pct_s = f"{val_pct:.1f}%" if isinstance(val_pct, (int, float)) else "--"
        buy_price = result.get("suggested_buy_price", 0)
        buy_s = _safe_money(buy_price)

        # ── 多周期预测 ──
        st = pred.get("short_term", {}) or {}
        mt = pred.get("mid_term", {}) or {}
        lt = pred.get("long_term", {}) or {}

        # ── 建议操作 ──
        act_name = action.get("action", "--") or "--"
        sl = action.get("stop_loss")
        tp = action.get("take_profit")
        sl_s = _safe_money(sl)
        tp_s = _safe_money(tp)

        # ── 组装 Body ──
        elements = [
            _md(
                f"**现价** {price_s}　|　"
                f"**涨跌** {change_s}　|　"
                f"**PE** {pe_s}　|　"
                f"**PB** {pb_s}"
            ),
            _md(
                f"**估值水平**: {val_level}　(PE分位: {val_pct_s})\n"
                f"**建议买点**: {buy_s}"
            ),
            _hr(),
            _md(
                f"**多周期预测**\n"
                f"| 周期 | 方向 | 涨跌幅 | 置信度 |\n"
                f"| :-- | :-- | :-- | :-- |\n"
                f"| 短期1-2周 | {_direction_cn(st.get('direction',''))} | "
                f"{_pct(st.get('change_pct')) if st.get('change_pct') is not None else '--'} | "
                f"{st.get('confidence','--')} |\n"
                f"| 中期1-3月 | {_direction_cn(mt.get('direction',''))} | "
                f"{_pct(mt.get('change_pct')) if mt.get('change_pct') is not None else '--'} | "
                f"{mt.get('confidence','--')} |\n"
                f"| 长期6-12月 | {_direction_cn(lt.get('direction',''))} | "
                f"{_pct(lt.get('change_pct')) if lt.get('change_pct') is not None else '--'} | "
                f"{lt.get('confidence','--')} |"
            ),
            _hr(),
            _md(
                f"**建议操作**: {act_name}　|　"
                f"止损 {sl_s}　|　"
                f"止盈 {tp_s}"
            ),
        ]

        # 重要新闻（最多 2 条看多 + 2 条看空）
        bull_news = result.get("important_bullish_news", []) or []
        bear_news = result.get("important_bearish_news", []) or []
        if bull_news or bear_news:
            news_lines = ["**📰 重要消息**"]
            for n in bull_news[:2]:
                title_n = n.get("title", "")[:60]
                src = n.get("source", "")
                news_lines.append(f"🟧 [{title_n}]({n.get('url','')}) — {src}")
            for n in bear_news[:2]:
                title_n = n.get("title", "")[:60]
                src = n.get("source", "")
                news_lines.append(f"🟩 [{title_n}]({n.get('url','')}) — {src}")
            elements.append(_hr())
            elements.append(_md("\n".join(news_lines)))

        return _card(title=title, color=color, elements=elements)

    # ── 大师分析卡片 ──────────────────────────────────────

    def build_master_card(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """从 /api/analyze 结果构建大师分析卡片。"""
        pred = result.get("prediction_summary", {}) or {}
        cio = pred.get("cio_decision", {}) or {}

        if not cio or not cio.get("master_name"):
            # 降级：没有 CIO 决策时返回普通卡片
            return self.build_analysis_card(result)

        symbol = result.get("symbol", "--")
        name = result.get("stock_name", symbol)
        score = result.get("score_breakdown", {}) or {}
        final_score = score.get("final", 0)
        label = score.get("label", "中性")
        quote = result.get("quote", {}) or {}
        pred_combined = result.get("prediction_summary", {}) or {}
        action = pred_combined.get("suggested_action", {}) or {}

        color = _signal_color(label)
        emoji = _signal_emoji(label)

        score_str = _safe_score(final_score)
        master_key = cio.get("master_key", cio.get("master_name", ""))
        title = (
            f"{emoji} {label} {score_str}  {name} ({symbol})\n"
            f"🎓 {cio.get('master_name', '')}"
        )

        # ── 行情行 ──
        price = quote.get("price")
        change_pct_val = quote.get("change_pct")
        pe = quote.get("pe")
        price_s = _safe_money(price)
        change_s = _pct(change_pct_val) if change_pct_val is not None else "--"
        pe_s = _safe_str(pe)

        val_level = result.get("valuation_level", "--") or "--"
        buy_price = result.get("suggested_buy_price", 0)
        buy_s = _safe_money(buy_price)

        # ── CIO 决策 ──
        summary = cio.get("decision_summary", "")[:200]
        base = cio.get("base_case", {}) or {}
        bull = cio.get("bull_case", {}) or {}
        bear = cio.get("bear_case", {}) or {}
        order = cio.get("order", {}) or {}
        order_action = order.get("action", "--")
        order_size = order.get("position_size", "--")
        order_entry = order.get("entry_condition", "--")
        order_sl = order.get("stop_loss")
        order_tp = order.get("take_profit")
        order_sl_s = _safe_money(order_sl)
        order_tp_s = _safe_money(order_tp)

        # ── 大师多周期预测 ──
        st = cio.get("short_term", {}) or {}
        mt = cio.get("mid_term", {}) or {}
        lt = cio.get("long_term", {}) or {}

        elements = [
            _md(
                f"**现价** {price_s}　|　"
                f"**涨跌** {change_s}　|　"
                f"**PE** {pe_s}"
            ),
            _md(f"**估值**: {val_level}　|　**建议买点**: {buy_s}"),
            _hr(),
            _md(f"**🎓 {cio.get('master_name', '')} 决策**\n\n{summary}"),
            _md(
                f"**三场景分析**\n"
                f"| 场景 | 方向 | 目标价 | 概率 |\n"
                f"| :-- | :-- | :-- | :-- |\n"
                f"| 基准 | {_direction_cn(base.get('direction',''))} | "
                f"¥{_safe_str(base.get('target'))} | "
                f"{_safe_str(base.get('probability'))} |\n"
                f"| 乐观 | {_direction_cn(bull.get('direction',''))} | "
                f"¥{_safe_str(bull.get('target'))} | "
                f"{_safe_str(bull.get('probability'))} |\n"
                f"| 悲观 | {_direction_cn(bear.get('direction',''))} | "
                f"¥{_safe_str(bear.get('target'))} | "
                f"{_safe_str(bear.get('probability'))} |"
            ),
            _md(
                f"**多周期预测**\n"
                f"| 周期 | 方向 | 涨跌幅 |\n"
                f"| :-- | :-- | :-- |\n"
                f"| 短期 | {_direction_cn(st.get('direction',''))} | "
                f"{_pct(st.get('change_pct')) if st.get('change_pct') is not None else '--'} |\n"
                f"| 中期 | {_direction_cn(mt.get('direction',''))} | "
                f"{_pct(mt.get('change_pct')) if mt.get('change_pct') is not None else '--'} |\n"
                f"| 长期 | {_direction_cn(lt.get('direction',''))} | "
                f"{_pct(lt.get('change_pct')) if lt.get('change_pct') is not None else '--'} |"
            ),
            _hr(),
            _md(
                f"**操作**: {order_action}　|　"
                f"**仓位**: {order_size}\n"
                f"**入场**: {order_entry}\n"
                f"**止损**: {order_sl_s}　|　"
                f"**止盈**: {order_tp_s}"
            ),
        ]

        # 建议操作（非 CIO 的通用建议）
        act_name = action.get("action", "")
        if act_name:
            elements.append(_md(f"**通用建议**: {act_name}"))

        return _card(title=title, color=color, elements=elements)

    # ── 批量分析卡片 ──────────────────────────────────────

    def build_batch_card(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """从批量分析结果构建卡片。"""
        results = result.get("results", [])
        total = result.get("total", len(results))
        success_count = result.get("success_count", 0)
        error_count = result.get("error_count", 0)
        summary = result.get("summary") or {}
        quality_pick = result.get("quality_pick") or {}

        # ── 构建排序表 ──
        scored = []
        for r in results:
            data = r.get("data", {}) or {}
            score = data.get("score_breakdown", {}) or {}
            pred = data.get("prediction_summary", {}) or {}
            action_obj = pred.get("suggested_action", {}) or {}
            scored.append({
                "symbol": r.get("symbol", ""),
                "name": data.get("stock_name", r.get("symbol", "")),
                "score": score.get("final", 0),
                "label": score.get("label", "--"),
                "valuation": data.get("valuation_level", "--") or "--",
                "action": action_obj.get("action", "--") or "--",
            })

        # 按评分降序排列
        scored.sort(key=lambda x: x["score"], reverse=True)

        # 生成表格（最多 10 行）
        table_lines = [
            f"**📊 批量结果** ({success_count}/{total} 成功"
            + (f"，{error_count} 失败" if error_count else "")
            + ")\n\n| # | 股票 | 评分 | 估值 | 建议 |\n"
            "| :-- | :-- | :-- | :-- | :-- |"
        ]
        for i, item in enumerate(scored[:10], 1):
            s = _safe_score(item["score"])
            table_lines.append(
                f"| {i} | {item['symbol']} {item['name']} | "
                f"{s} | {item['valuation']} | {item['action']} |"
            )

        elements = [_md("\n".join(table_lines))]

        # ── 优质推荐 ──
        best = quality_pick.get("best_stock") if quality_pick else None
        if best:
            best_sym = best.get("symbol", "")
            best_name = best.get("name", best_sym)
            best_reasons = best.get("reasons", [])
            best_action = best.get("suggested_action", "--")
            best_pct = best.get("suggested_position_pct", "--")
            reasons_str = "；".join(best_reasons[:3]) if best_reasons else "--"
            elements.append(_hr())
            elements.append(_md(
                f"**🏆 最佳推荐**: {best_sym} {best_name}\n"
                f"理由: {reasons_str}\n"
                f"操作: **{best_action}**　|　建议仓位: {best_pct}"
            ))

            runner = quality_pick.get("runner_up")
            if runner:
                elements.append(_md(
                    f"**🥈 次选**: {runner.get('symbol','')} {runner.get('name','')}"
                ))

        # ── 共性主题 ──
        common = summary.get("common_themes", []) or []
        if common:
            lines = ["\n**🔍 共性主题**"]
            for t in common[:5]:
                lines.append(f"· {t}")
            elements.append(_md("\n".join(lines)))

        # ── 整体评估 ──
        overall = summary.get("overall_assessment", "") or summary.get("summary_text", "")
        if overall:
            txt = overall[:300]
            elements.append(_hr())
            elements.append(_md(f"**📝 整体评估**: {txt}"))

        title = f"📊 批量分析 ({success_count}/{total})"
        # 综合评分颜色：整体偏高用红色，否则蓝色
        avg_score = sum(x["score"] for x in scored) / max(len(scored), 1)
        color = "blue"
        if avg_score >= 1.5:
            color = "red"
        elif avg_score <= -1.0:
            color = "green"

        return _card(title=title, color=color, elements=elements)

    # ── 帮助卡片 ──────────────────────────────────────────

    def build_help_card(self) -> Dict[str, Any]:
        """构建使用帮助卡片。"""
        elements = [
            _md(
                "**📋 使用方法**\n\n"
                "**1. 单股分析** — 直接发送股票代码\n"
                "　如: `600519`\n\n"
                "**2. 批量分析** — 用 `/` 分割多只股票\n"
                "　如: `600519/000858/300750`\n\n"
                "**3. 大师分析** — 指定投资大师视角\n"
                "　设默认: `/master buffett`\n"
                "　单次用: `600519 --master graham`\n"
                "　关闭: `/master off`\n"
                "　查看: `/master list`\n\n"
                "**4. Qlib 数据更新** — `/update_data`\n"
                "　下载最新 qlib 市场数据\n\n"
                "**5. Qlib 推理** — `/qlib_inference`\n"
                "　使用默认微调模型推理选股\n\n"
                "**6. 本帮助** — `/help`"
            ),
            _hr(),
            _md(
                "**🎓 可选大师**\n"
                "`graham` 格雷厄姆 · 深度价值\n"
                "`buffett` 巴菲特 · 价值质量\n"
                "`fisher` 费雪 · 成长投资\n"
                "`lynch` 林奇 · GARP\n"
                "`templeton` 邓普顿 · 逆向全球\n"
                "`soros` 索罗斯 · 反身性宏观\n"
                "`dalio` 达利欧 · 全天候"
            ),
            _hr(),
            _md("⚡ 提示: 分析通常需要 30-60 秒，批量分析需要数分钟。"),
        ]
        return _card(
            title="📋 StockFish 使用帮助",
            subtitle="支持单股/批量分析 + 7位投资大师视角",
            color="blue",
            elements=elements,
        )

    # ── 错误卡片 ───────────────────────────────────────────

    def build_error_card(self, symbol: str, error_msg: str) -> Dict[str, Any]:
        """构建错误提示卡片。"""
        short_msg = str(error_msg)[:300]
        elements = [
            _md(f"**股票**: {symbol}\n\n{short_msg}"),
            _hr(),
            _md("💡 请检查股票代码是否正确，或稍后重试。"),
        ]
        return _card(
            title="⚠️ 分析失败",
            subtitle=f"无法分析 {symbol}",
            color="grey",
            elements=elements,
        )
