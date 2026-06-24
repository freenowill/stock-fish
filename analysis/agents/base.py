"""
Base Agent — 所有员工和 CIO 的共享基类

提供:
- LLM 客户端初始化（复用 OpenAI 兼容 API）
- 安全的 JSON 解析
- 带超时和异常保护的 LLM 调用
- 从 analysis state dict 中提取各维度数据的工具方法
"""
import json
import os
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class EmployeeReport:
    """一名员工的分析报告 — 标准化格式，所有 8 名员工输出此结构"""
    employee_id: str = ""           # e1 ~ e8
    role: str = ""                 # 角色名（中文）
    department: str = ""           # 所属部门
    outlook: str = "中性"          # 看多/看空/中性
    confidence: str = "低"         # 高/中/低
    score: float = 0.0             # -10 ~ +10
    key_points: list = field(default_factory=list)
    risks: list = field(default_factory=list)
    raw_output: str = ""
    error: Optional[str] = None    # 非空表示此员工报告生成失败


@dataclass
class CIODecision:
    """最终决策人的决策输出 — 结构化决策框架"""
    master_name: str = ""          # 大师名（中文）
    master_key: str = ""           # 大师 key (buffett/graham/...)
    decision_summary: str = ""     # 决策摘要 (120字内)
    rationale: str = ""            # 详细决策逻辑 (200-300字)

    # 证据链
    evidence_chain: list = field(default_factory=list)

    # 三情景分析
    base_case: Optional[Dict] = None
    bull_case: Optional[Dict] = None
    bear_case: Optional[Dict] = None

    # 操作指令
    order: Optional[Dict] = None

    # 多周期预测
    short_term: Optional[Dict] = None
    mid_term: Optional[Dict] = None
    long_term: Optional[Dict] = None

    # 风险监控
    risk_monitoring: list = field(default_factory=list)

    # 决策质量
    decision_quality: Optional[Dict] = None

    # 否决回应 (当风险经理行使软否决权时)
    veto_response: str = ""

    # 非经常性损益说明 (CIO 对净利润质量的推断)
    extraordinary_items_note: str = ""

    # 原始输出
    raw_llm_output: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'master_name': self.master_name,
            'master_key': self.master_key,
            'decision_summary': self.decision_summary,
            'rationale': self.rationale,
            'evidence_chain': self.evidence_chain,
            'base_case': self.base_case,
            'bull_case': self.bull_case,
            'bear_case': self.bear_case,
            'order': self.order,
            'short_term': self.short_term,
            'mid_term': self.mid_term,
            'long_term': self.long_term,
            'risk_monitoring': self.risk_monitoring,
            'decision_quality': self.decision_quality,
            'veto_response': self.veto_response,
            'extraordinary_items_note': self.extraordinary_items_note,
            'raw_llm_output': self.raw_llm_output,
            'error': self.error,
        }


class BaseAgent:
    """
    所有 Agent 的基类 — 封装 LLM 调用、JSON 解析、超时、重试
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None):
        from config import settings
        self.api_key = api_key or os.environ.get('LLM_API_KEY') or getattr(settings, 'LLM_API_KEY', None) or ''
        self.base_url = base_url or os.environ.get('LLM_BASE_URL') or getattr(settings, 'LLM_BASE_URL', None) or 'https://api.openai.com/v1'
        self.model = model or os.environ.get('LLM_MODEL_NAME') or getattr(settings, 'LLM_MODEL_NAME', None) or 'gpt-4o-mini'
        self.max_retries = 2
        self.timeout = 45  # 单次 LLM 调用超时 (秒)

    @property
    def has_llm(self) -> bool:
        return bool(self.api_key)

    # ── LLM 调用 ──

    def _call_llm(self, system_prompt: str, user_prompt: str,
                  temperature: float = 0.3, use_json_mode: bool = True) -> dict:
        """
        调用 LLM，返回解析后的 JSON dict。
        带重试和异常保护 — 永不抛异常，失败返回空 dict。
        """
        if not self.has_llm:
            logger.warning("LLM API key 未配置，跳过 LLM 调用")
            return {}

        from openai import OpenAI

        for attempt in range(self.max_retries + 1):
            try:
                client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                timeout=self.timeout, max_retries=1)
                kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                }
                if use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                    kwargs["max_tokens"] = 4096  # CIO 决策 JSON 较长，需要足够 token

                resp = client.chat.completions.create(**kwargs)
                raw = resp.choices[0].message.content or "{}"
                return self._parse_json(raw)

            except Exception as e:
                logger.warning(f"LLM 调用失败 (attempt {attempt+1}/{self.max_retries+1}): {e}")
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))  # 递增退避

        logger.error("LLM 调用全部重试失败")
        return {}

    # ── JSON 解析 ──

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """安全解析 LLM 返回的 JSON — 处理截断、markdown包裹、末尾多余文本。"""
        try:
            raw = raw.strip()
            # 1. 去掉 markdown code block
            if raw.startswith('```json'):
                raw = raw.split('```json')[1].split('```')[0]
            elif raw.startswith('```'):
                raw = raw.split('```')[1].split('```')[0]

            # 2. 找到最外层的 { ... }
            start = raw.find('{')
            end = raw.rfind('}')
            if start >= 0 and end > start:
                raw = raw[start:end + 1]

            return json.loads(raw)
        except (json.JSONDecodeError, KeyError, IndexError):
            # 3. 最终尝试: 修复常见错误 (尾部多余逗号/未闭合字符串等)
            try:
                import re
                start = raw.find('{')
                end = raw.rfind('}')
                if start >= 0 and end > start:
                    candidate = raw[start:end + 1]
                    # 移除尾部多余逗号
                    candidate = re.sub(r',\s*}', '}', candidate)
                    candidate = re.sub(r',\s*]', ']', candidate)
                    return json.loads(candidate)
            except Exception:
                pass
            logger.warning(f"JSON 解析失败 (已尝试修复): {raw[:200]}...")
            return {}

    # ── 数据提取工具 (从 state dict 中提取各维度数据) ──

    @staticmethod
    def _safe_get(d: Any, key: str, default: Any = 'N/A') -> Any:
        """安全从 dict 取值，兼容 None 和非 dict 类型"""
        if isinstance(d, dict):
            return d.get(key, default)
        return default

    @staticmethod
    def _is_financial_industry(state: dict) -> bool:
        """判断行业是否金融业（银行/保险/证券），这些行业高负债是结构性的"""
        ctx = state.get('industry_context') or {}
        names = str(ctx.get('industry_names', ''))
        stock_name = str(state.get('stock_name', ''))
        return any(k in names or k in stock_name for k in ('银行', '保险', '证券', '金融', '券商'))

    @staticmethod
    def build_tech_context(state: dict) -> str:
        """提取技术面数据 (复用自 PredictionNode._build_tech_data)"""
        ti = state.get('technical_indicators', {}) or {}
        q = state.get('quote', {}) or {}
        price = q.get('price', 0) if isinstance(q, dict) else 0
        lines = [
            f"现价: {price}  涨跌: {q.get('change_pct', '?')}%",
            f"RSI(14): {ti.get('rsi_14', '?')}  MACD柱: {ti.get('macd_hist', '?')}",
            f"KDJ: K={ti.get('kdj_k', '?')} D={ti.get('kdj_d', '?')} J={ti.get('kdj_j', '?')}",
            f"MA5: {ti.get('ma5', '?')}  MA10: {ti.get('ma10', '?')}  MA20: {ti.get('ma20', '?')}  MA60: {ti.get('ma60', '?')}",
            f"布林带: 下轨{ti.get('boll_lower', '?')}  中轨{ti.get('boll_middle', '?')}  上轨{ti.get('boll_upper', '?')}",
            f"量比: {ti.get('volume_ratio', '?')}  振幅: {ti.get('amplitude', '?')}%",
        ]
        return "\n".join(lines)

    @staticmethod
    def build_fund_context(state: dict) -> str:
        """提取基本面数据 (复用自 PredictionNode._build_fund_data)"""
        fs = state.get('financial_summary', {}) or {}
        q = state.get('quote', {}) or {}
        vp = state.get('valuation_percentile')
        vp_str = f"{vp:.1f}%" if vp is not None else "N/A"
        ft = state.get('financial_trends') or {}

        lines = [
            f"PE: {q.get('pe', '?')}  PB: {q.get('pb', '?')}  市值: {q.get('market_cap', '?')}亿",
            f"估值等级: {state.get('valuation_level', '?')} (PE分位: {vp_str})",
            f"历史PE均值: {state.get('historical_pe_avg') or 'N/A'}  "
            f"建议买入价: {state.get('suggested_buy_price', '?')}",
            f"EPS: {fs.get('eps', '?')}  ROE: {fs.get('roe', '?')}%  "
            f"经营现金流: {fs.get('operating_cash_flow', 'N/A')}亿  "
            f"自由现金流: {fs.get('free_cash_flow', 'N/A')}亿",
            f"营收: {fs.get('revenue', '?')}亿  净利: {fs.get('net_profit', '?')}亿",
            f"毛利率: {fs.get('gross_margin', '?')}%  负债率: {fs.get('debt_ratio', '?')}%  "
            f"股息率: {fs.get('dividend_yield') or q.get('dividend_yield', 'N/A')}%",
            f"营收同比: {fs.get('revenue_yoy', '?')}%  净利同比: {fs.get('net_profit_yoy', '?')}%",
        ]

        # ROIC (投入资本回报率) — 巴菲特核心指标
        roic = state.get('roic')
        if roic is not None:
            lines.append(f"ROIC: {roic:.2f}% (衡量护城河的核心指标)")

        # FCF per share
        fcf_ps = state.get('fcf_per_share')
        if fcf_ps is not None:
            lines.append(f"每股自由现金流: {fcf_ps:.2f}元")

        # 多期财务趋势
        if ft:
            roe_vals = ft.get('roe')
            if roe_vals and isinstance(roe_vals, list) and len(roe_vals) >= 3:
                lines.append(f"近5年ROE趋势: {' → '.join(f'{v:.1f}%' for v in roe_vals[-5:])}")
            eps_cagr = ft.get('eps_cagr_5y')
            if eps_cagr is not None:
                lines.append(f"EPS 5年CAGR: {eps_cagr*100:.1f}%")
            roe_stab = ft.get('roe_stability')
            if roe_stab is not None:
                lines.append(f"ROE稳定性(越低越稳): {roe_stab:.3f}")
            gm_trend = ft.get('gross_margin_trend')
            if gm_trend:
                lines.append(f"毛利率趋势: {gm_trend}")

        # 护城河评估（巴菲特）
        moat = state.get('moat_assessment')
        if moat and moat.get('moat_level'):
            sources = moat.get('moat_sources', [])
            lines.append(f"护城河判断: {moat['moat_level']}" +
                        (f" — 来源: {', '.join(sources[:3])}" if sources else ""))

        # ── 数据质量提示 ──
        data_warnings = []
        ocf = fs.get('operating_cash_flow')
        fcf_val = fs.get('free_cash_flow')
        if ocf is None or ocf == 0 or ocf == 'N/A':
            data_warnings.append("⚠️ 经营现金流数据缺失，净利润质量无法验证")
        if fcf_val is None or fcf_val == 0 or fcf_val == 'N/A':
            data_warnings.append("⚠️ 自由现金流数据缺失")
        net_profit = fs.get('net_profit')
        rev = fs.get('revenue')
        if net_profit and rev and net_profit != 'N/A' and rev != 'N/A':
            try:
                np_val = float(net_profit)
                rev_val = float(rev)
                if np_val > 0 and rev_val > 0 and np_val / rev_val > 0.3:
                    data_warnings.append("⚠️ 净利率>30%，可能存在非经常性损益")
            except (ValueError, TypeError):
                pass
        if data_warnings:
            lines.append("【数据质量】" + " | ".join(data_warnings))

        return "\n".join(lines)

    @staticmethod
    def build_valuation_context(state: dict) -> str:
        """提取估值专项数据 (比 fund_context 更聚焦于估值维度)"""
        q = state.get('quote', {}) or {}
        fs = state.get('financial_summary', {}) or {}
        vp = state.get('valuation_percentile')
        vp_str = f"{vp:.1f}%" if vp is not None else "N/A"
        lines = [
            f"当前股价: {q.get('price', '?')}元",
            f"PE: {q.get('pe', '?')}  PB: {q.get('pb', '?')}  PS: {q.get('ps', 'N/A')}",
            f"总市值: {q.get('market_cap', '?')}亿",
            f"PE 历史分位: {vp_str} (近365日)",
            f"历史 PE 均值: {state.get('historical_pe_avg') or 'N/A'}",
            f"估值等级: {state.get('valuation_level', '正常')}",
            f"建议买入价 (系统计算): {state.get('suggested_buy_price', '?')}",
            f"EPS: {fs.get('eps', '?')}  ROE: {fs.get('roe', '?')}%",
            f"股息率: {fs.get('dividend_yield') or q.get('dividend_yield', 'N/A')}%",
        ]

        # 长期PE分位
        pe_5y = state.get('valuation_percentile_5y')
        if pe_5y is not None:
            lines.append(f"5年PE分位: {pe_5y:.1f}%")
        pe_10y = state.get('valuation_percentile_10y')
        if pe_10y is not None:
            lines.append(f"10年PE分位: {pe_10y:.1f}%")

        # 股票收益率 vs 债券
        ey = state.get('earnings_yield')
        bond = state.get('bond_yield_10y')
        if ey is not None and bond is not None:
            lines.append(f"E/P收益率: {ey:.1f}% vs 10年国债: {bond:.1f}% (风险溢价: {ey-bond:.1f}%)")
        elif ey is not None:
            lines.append(f"E/P收益率: {ey:.1f}%")

        return "\n".join(lines)

    @staticmethod
    def build_sent_context(state: dict) -> str:
        """提取舆情数据 (复用自 PredictionNode._build_sent_data)"""
        sn = state.get('sentiment_news', {}) or {}
        sg = state.get('sentiment_guba', {}) or {}
        bull_news = state.get('important_bullish_news', []) or []
        bear_news = state.get('important_bearish_news', []) or []
        bull_guba = state.get('important_bullish_guba', []) or []
        bear_guba = state.get('important_bearish_guba', []) or []

        lines = [
            f"新闻: {sn.get('total_count', 0)}条  "
            f"正面{sn.get('positive_count', 0)} / 负面{sn.get('negative_count', 0)}  "
            f"平均情感分: {sn.get('avg_score', 0):.2f}",
            f"股吧: {sg.get('total_count', 0)}条  "
            f"正面{sg.get('positive_count', 0)} / 负面{sg.get('negative_count', 0)}  "
            f"平均情感分: {sg.get('avg_score', 0):.2f}",
            "",
            "近期利好:",
        ]
        for n in bull_news[:3]:
            lines.append(f"  + [{n.get('source', '')}] {n.get('title', '')}")
        for g in bull_guba[:2]:
            lines.append(f"  + [股吧] {g.get('title', '')}")
        lines.append("")
        lines.append("近期利空:")
        for n in bear_news[:3]:
            lines.append(f"  - [{n.get('source', '')}] {n.get('title', '')}")
        for g in bear_guba[:2]:
            lines.append(f"  - [股吧] {g.get('title', '')}")

        # 追加 Web 搜索结果
        lines.append("")
        lines.append(BaseAgent.build_search_context(state))

        # 情感百分位 (邓普顿: 极端情绪信号)
        sp = state.get('sentiment_percentile')
        if sp is not None:
            days = state.get('sentiment_history_days', 0)
            if sp < 5:
                lines.append(f"⚠️ 当前情感处于近{days}天最低{sp:.0f}%分位 — 极端悲观(潜在反向信号)")
            elif sp < 15:
                lines.append(f"当前情感处于近{days}天第{sp:.0f}%分位 — 偏悲观")
            elif sp > 90:
                lines.append(f"⚠️ 当前情感处于近{days}天最高{sp:.0f}%分位 — 极端乐观(需警惕)")
            else:
                lines.append(f"当前情感处于近{days}天第{sp:.0f}%分位 — 正常范围")

        # 关注热度
        ap = state.get('attention_news_percentile')
        if ap is not None and ap < 10:
            lines.append("📰 新闻关注度处于历史低位 — 该股可能被市场遗忘(邓普顿信号)")
        return "\n".join(lines)

    @staticmethod
    def build_risk_context(state: dict) -> str:
        """提取风险相关数据"""
        ti = state.get('technical_indicators', {}) or {}
        q = state.get('quote', {}) or {}
        signals = state.get('signals', {}) or {}
        score_breakdown = state.get('score_breakdown', {}) or {}
        lines = [
            f"当前股价: {q.get('price', '?')}元  涨跌幅: {q.get('change_pct', '?')}%",
            f"量比: {ti.get('volume_ratio', '?')}  振幅: {ti.get('amplitude', '?')}%",
            f"综合评分: {signals.get('score', 'N/A')} / {signals.get('label', 'N/A')}",
        ]
        # 评分置信度
        if score_breakdown:
            lines.append(f"评分置信度: {score_breakdown.get('confidence', 'N/A')}")
            lines.append(f"市场状态: {score_breakdown.get('regime', 'N/A')}")

        # VaR / 最大回撤 (新增风险量化指标)
        var_95 = state.get('var_95')
        if var_95 is not None:
            lines.append(f"VaR(95%): 日最大亏损{var_95:.1f}%")
        max_dd = state.get('max_drawdown')
        if max_dd is not None:
            lines.append(f"近1年最大回撤: {max_dd:.1f}%")
        ann_vol = state.get('annualized_volatility')
        if ann_vol is not None:
            lines.append(f"年化波动率: {ann_vol:.1f}%")

        return "\n".join(lines)

    @staticmethod
    def build_overseer_context(state: dict, other_reports: list) -> str:
        """
        为独立监察员构建上下文：所有其他员工的报告摘要 + 原始数据。
        other_reports: EmployeeReport 列表
        """
        lines = ["## 其他分析师的观点摘要\n"]
        for r in other_reports:
            if r.error:
                lines.append(f"### {r.role} ({r.department}) [报告生成失败]")
                lines.append(f"错误: {r.error}\n")
            else:
                lines.append(f"### {r.role} ({r.department})")
                lines.append(f"判断: {r.outlook}  置信度: {r.confidence}  评分: {r.score:.1f}")
                lines.append(f"关键观点:")
                for p in r.key_points:
                    lines.append(f"  - {p}")
                if r.risks:
                    lines.append(f"风险点:")
                    for risk in r.risks:
                        lines.append(f"  - {risk}")
                lines.append("")

        # 附上原始关键数据
        lines.append("## 原始市场数据\n")
        q = state.get('quote', {}) or {}
        lines.append(f"现价: {q.get('price', 'N/A')}  PE: {q.get('pe', 'N/A')}  PB: {q.get('pb', 'N/A')}")
        fs = state.get('financial_summary', {}) or {}
        lines.append(f"ROE: {fs.get('roe', 'N/A')}%  EPS: {fs.get('eps', 'N/A')}")
        pe_pct = state.get('valuation_percentile')
        pe_pct_str = f"{pe_pct:.1f}%" if pe_pct is not None else "N/A"
        lines.append(f"估值分位: {pe_pct_str}")
        return "\n".join(lines)

    @staticmethod
    def build_macro_context(state: dict) -> str:
        """提取宏观数据 (新增数据源)"""
        macro = state.get('macro_context') or {}
        lines = [
            f"SHIBOR隔夜: {macro.get('shibor', 'N/A')}%  1年期LPR: {macro.get('lpr_1y', 'N/A')}%",
            f"制造业PMI: {macro.get('pmi', 'N/A')}  CPI同比: {macro.get('cpi_yoy', 'N/A')}%",
            f"北向资金当日: {macro.get('northbound_flow', 'N/A')}亿  5日均值: {macro.get('northbound_5d_avg', 'N/A')}亿",
            f"美元/人民币: {macro.get('usd_cny', 'N/A')}",
            f"政策倾向: {macro.get('policy_tilt', 'N/A')}",
            f"大盘状态: {macro.get('market_regime', 'N/A')}",
        ]
        # 宏观经济事件
        events = macro.get('macro_events', [])
        if events:
            lines.append("近期重要经济事件:")
            for ev in events[:5]:
                stars = '★' * ev.get('importance', 1)
                lines.append(f"  [{ev.get('region', '')}] {ev.get('date', '')}: {ev.get('event', '')} {stars}")
        # 政策新闻
        headlines = macro.get('policy_headlines', [])
        if headlines:
            lines.append("政策新闻头条:")
            for h in headlines[:5]:
                lines.append(f"  - {h}")
        return "\n".join(lines)

    @staticmethod
    def build_industry_context(state: dict) -> str:
        """提取行业数据 (新增数据源)"""
        ind = state.get('industry_context') or {}
        lines = [
            f"所属行业: {ind.get('industry_name', 'N/A')}",
            f"行业PE分位: {ind.get('industry_pe_percentile', 'N/A')}%  行业动量: {ind.get('industry_momentum', 'N/A')}%",
            f"行业景气阶段: {ind.get('industry_cycle', 'N/A')}",
            f"政策影响: {ind.get('policy_impact', 'N/A')}",
            f"近期政策事件: {ind.get('policy_events', '无')}",
        ]
        return "\n".join(lines)

    @staticmethod
    def build_search_context(state: dict) -> str:
        """提取 Web 搜索结果"""
        sr = state.get('search_results') or {}
        results = sr.get('results', [])
        if not results:
            return "（无Web搜索结果）"

        lines = ["## Web 搜索结果\n"]
        dims = sr.get('dimensions', [])
        if dims:
            lines.append(f"搜索维度: {' | '.join(dims)}\n")

        for i, r in enumerate(results[:8], 1):
            title = r.get('title', '无标题')
            snippet = r.get('snippet', r.get('content', ''))
            url = r.get('url', '')
            source = r.get('source', '')
            date = r.get('date', '')
            dim = r.get('dimension', '')
            meta = f"来源: {source}" + (f" | 日期: {date}" if date else "") + (f" | 维度: {dim}" if dim else "")
            lines.append(f"{i}. **{title}**\n   {snippet}\n   {meta} | {url}")
        if sr.get('summary'):
            lines.append(f"\n搜索摘要: {sr['summary']}")
        return "\n".join(lines)
