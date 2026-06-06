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
    decision_summary: str = ""     # 决策摘要 (80字内)

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

    # 原始输出
    raw_llm_output: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'master_name': self.master_name,
            'master_key': self.master_key,
            'decision_summary': self.decision_summary,
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
        """安全解析 LLM 返回的 JSON，处理 markdown code block 包裹"""
        try:
            raw = raw.strip()
            if raw.startswith('```json'):
                raw = raw.split('```json')[1].split('```')[0]
            elif raw.startswith('```'):
                raw = raw.split('```')[1].split('```')[0]
            return json.loads(raw)
        except (json.JSONDecodeError, KeyError, IndexError):
            logger.warning(f"JSON 解析失败: {raw[:150]}")
            return {}

    # ── 数据提取工具 (从 state dict 中提取各维度数据) ──

    @staticmethod
    def _safe_get(d: Any, key: str, default: Any = 'N/A') -> Any:
        """安全从 dict 取值，兼容 None 和非 dict 类型"""
        if isinstance(d, dict):
            return d.get(key, default)
        return default

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
        lines = [
            f"PE: {q.get('pe', '?')}  PB: {q.get('pb', '?')}  市值: {q.get('market_cap', '?')}亿",
            f"估值等级: {state.get('valuation_level', '?')} (PE分位: {state.get('valuation_percentile', '?')}%)",
            f"历史PE均值: {state.get('historical_pe_avg', '?')}  建议买入价: {state.get('suggested_buy_price', '?')}",
            f"EPS: {fs.get('eps', '?')}  ROE: {fs.get('roe', '?')}%",
            f"营收: {fs.get('revenue', '?')}亿  净利: {fs.get('net_profit', '?')}亿",
            f"毛利率: {fs.get('gross_margin', '?')}%  负债率: {fs.get('debt_ratio', '?')}%",
            f"营收同比: {fs.get('revenue_yoy', '?')}%  净利同比: {fs.get('net_profit_yoy', '?')}%",
        ]
        return "\n".join(lines)

    @staticmethod
    def build_valuation_context(state: dict) -> str:
        """提取估值专项数据 (比 fund_context 更聚焦于估值维度)"""
        q = state.get('quote', {}) or {}
        fs = state.get('financial_summary', {}) or {}
        lines = [
            f"当前股价: {q.get('price', '?')}元",
            f"PE: {q.get('pe', '?')}  PB: {q.get('pb', '?')}  PS: {q.get('ps', 'N/A')}",
            f"总市值: {q.get('market_cap', '?')}亿",
            f"PE 历史分位: {state.get('valuation_percentile', '?')}% (近365日)",
            f"历史 PE 均值: {state.get('historical_pe_avg', '?')}",
            f"估值等级: {state.get('valuation_level', '正常')}",
            f"建议买入价 (系统计算): {state.get('suggested_buy_price', '?')}",
            f"EPS: {fs.get('eps', '?')}  ROE: {fs.get('roe', '?')}%",
            f"股息率: {q.get('dividend_yield', 'N/A')}%",
        ]
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
        lines.append(f"估值分位: {state.get('valuation_percentile', 'N/A')}%")
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
