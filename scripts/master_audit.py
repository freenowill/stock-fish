#!/usr/bin/env python3
"""
Master Audit — 大师分析质量审计与优化工具

随机从沪深300中挑选一支股票和一位大师，运行完整的大师分析工作流，
保存所有中间数据，进行交叉验证，最终输出审计报告和改进计划。

用法:
    python scripts/master_audit.py                          # 随机股票 + 随机大师
    python scripts/master_audit.py --symbol 600519          # 指定股票，随机大师
    python scripts/master_audit.py --master buffett         # 指定大师，随机股票
    python scripts/master_audit.py --symbol 600519 --master buffett
    python scripts/master_audit.py --seed 42                # 固定随机种子
    python scripts/master_audit.py --output-dir /tmp/audit  # 自定义输出目录
    python scripts/master_audit.py --no-llm-audit           # 跳过LLM深度审计
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# ── 确保项目根目录在 Python path 中 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

# ── CSI300 硬编码备选（当 Qlib 数据不可用时） ──
CSI300_FALLBACK = sorted([
    # 金融
    "600036", "601166", "600030", "601318", "601398", "601939",
    "601288", "601328", "600016", "000001", "600837", "601211",
    "601688", "002142",
    # 消费
    "600519", "000858", "000568", "600809", "000333", "000651",
    "600887", "002304", "600690",
    # 科技 / 新能源
    "300750", "002415", "000063", "002049", "603986", "300124",
    "603501", "002352",
    # 医药
    "600276", "300760", "000538", "300015", "002007", "000423",
    # 能源 / 材料
    "601857", "600028", "600585", "601899", "600019", "002460",
    "600309", "002709",
    # 汽车 / 工业
    "600104", "000625", "601012", "600031", "002594", "601766",
    # 地产 / 建筑
    "001979", "600048", "601668", "000002",
    # 公用 / 交通
    "600900", "601006", "600029", "601111",
    # 农业 / 其他
    "002714", "000876",
    # 军工
    "600760", "600893", "002179",
])

# ── 可用大师列表（从 cio_prompts 读取） ──
MASTER_KEYS = ["graham", "buffett", "fisher", "lynch", "templeton", "soros", "dalio"]

# ── 规则检查权重 ──
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


# ========================================================================
#  工具函数
# ========================================================================

def load_csi300_stocks() -> List[str]:
    """加载沪深300成分股列表。优先级: Qlib instruments > 硬编码备选。"""
    qlib_path = os.path.expanduser("~/.qlib/qlib_data/cn_data/instruments/csi300.txt")
    if os.path.isfile(qlib_path):
        try:
            stocks = []
            with open(qlib_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or "\t" not in line:
                        continue
                    code = line.split("\t")[0]
                    # 去除前缀 SH / SZ
                    code = code.replace("SH", "").replace("SZ", "").replace("BJ", "")
                    if code.isdigit() and len(code) == 6:
                        stocks.append(code)
            if stocks:
                logger.info(f"从 Qlib 加载沪深300成分股: {len(stocks)} 支")
                return sorted(stocks)
        except Exception as e:
            logger.warning(f"Qlib 成分股加载失败: {e}")
    logger.info(f"使用备选成分股列表: {len(CSI300_FALLBACK)} 支")
    return CSI300_FALLBACK


def get_available_masters() -> List[Dict]:
    """获取可用大师列表。"""
    try:
        from analysis.agents.cio_prompts import list_masters
        return list_masters()
    except ImportError:
        return [{"key": k, "name": k.capitalize()} for k in MASTER_KEYS]


def ensure_dir(path: str) -> str:
    """确保目录存在，返回路径。"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def safe_save_json(data: Any, filepath: str) -> bool:
    """安全地保存 JSON 文件。"""
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"保存 {filepath} 失败: {e}")
        return False


def safe_read_json(filepath: str) -> Optional[Dict]:
    """安全地读取 JSON 文件。"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def flatten_dict(d: dict, max_depth: int = 3, _depth: int = 0) -> str:
    """将嵌套 dict 展平为字符串（用于 LLM prompt）。"""
    if _depth >= max_depth:
        return str(d)[:200]
    parts = []
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict):
            parts.append(f"{k}: {flatten_dict(v, max_depth, _depth + 1)}")
        elif isinstance(v, list):
            if len(v) > 5:
                parts.append(f"{k}: [{len(v)} items, first={str(v[0])[:100]}]")
            else:
                parts.append(f"{k}: {str(v)[:300]}")
        else:
            parts.append(f"{k}: {str(v)[:200]}")
    return "\n".join(parts)


# ========================================================================
#  规则检查器 — 无 LLM 依赖
# ========================================================================

OUTLOOK_LEVELS = {
    "强烈看空": -4, "看空": -3, "偏空": -2, "中性": 0,
    "偏多": 2, "看多": 3, "强烈看多": 4,
}

ACTION_DIRECTION = {
    "买入": 2, "加仓": 1, "持有": 0, "减仓": -1, "卖出": -2, "观望": 0,
}


def _outlook_to_score(outlook: str) -> int:
    """将 outlook 文字映射为数值。"""
    return OUTLOOK_LEVELS.get(outlook, 0)


def _signals_to_score(signals: Optional[Dict]) -> Optional[int]:
    """从信号 dict 提取综合评分。"""
    if not signals:
        return None
    return signals.get("score") or signals.get("final")


def _check_statements_in_text(text: str, keywords: List[str]) -> bool:
    """检查文本中是否包含任意关键词。"""
    if not text:
        return False
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


class RuleBasedAuditor:
    """基于规则的交叉验证检查器。"""

    def __init__(self, data_dir: str, result_dir: str, master_dir: str):
        self.data_dir = data_dir
        self.result_dir = result_dir
        self.master_dir = master_dir
        self.findings: List[Dict] = []

    def add_finding(self, check: str, severity: str, passed: bool,
                    detail: str, suggestion: str = ""):
        """添加一条审计发现。"""
        if not passed:
            self.findings.append({
                "check": check,
                "severity": severity,
                "passed": False,
                "detail": detail,
                "suggestion": suggestion,
            })

    def add_pass(self, check: str, detail: str = ""):
        """添加一条通过项（不影响 findings 计数但保留记录）。"""
        pass  # 通过项仅用于报告，暂不记录

    def run_all(self):
        """运行全部规则检查。"""
        # 加载各阶段数据
        data = {
            "quote": safe_read_json(os.path.join(self.data_dir, "quote.json")) or {},
            "technical": safe_read_json(os.path.join(self.data_dir, "technical_indicators.json")) or {},
            "financial": safe_read_json(os.path.join(self.data_dir, "financial_summary.json")) or {},
        }
        result = {
            "sentiment": safe_read_json(os.path.join(self.result_dir, "sentiment.json")) or {},
            "valuation": safe_read_json(os.path.join(self.result_dir, "valuation.json")) or {},
            "scoring": safe_read_json(os.path.join(self.result_dir, "scoring.json")) or {},
            "risk": safe_read_json(os.path.join(self.result_dir, "risk_metrics.json")) or {},
        }
        master_data = {
            "cio": safe_read_json(os.path.join(self.master_dir, "cio_decision.json")) or {},
            "employees": safe_read_json(os.path.join(self.master_dir, "employee_reports.json")) or [],
            "prediction": safe_read_json(os.path.join(self.master_dir, "prediction_summary.json")) or {},
        }

        cio = master_data["cio"]
        employees = master_data["employees"]
        scoring = result["scoring"]
        valuation = result["valuation"]
        risk = result["risk"]
        quote = data["quote"]
        signals = scoring.get("signals", scoring)

        # ── 检查1: 方向一致性 ──
        cio_outlook = cio.get("decision_summary", "")
        cio_action = ""
        if cio.get("order"):
            cio_action = cio["order"].get("action", "")
        signals_label = signals.get("label", "") if signals else ""
        signals_score_val = _signals_to_score(signals)

        outlook_val = _outlook_to_score(cio_outlook)
        action_val = ACTION_DIRECTION.get(cio_action, 0)

        if signals_score_val is not None:
            score_label = "看多" if signals_score_val >= 2 else "看空" if signals_score_val <= -2 else "中性"
            if abs(action_val) >= 2 and signals_score_val * action_val < 0:
                self.add_finding(
                    "方向一致性",
                    "high",
                    passed=False,
                    detail=f"CIO 操作建议「{cio_action}」与综合评分「{signals_score_val} ({score_label})」方向矛盾",
                    suggestion="检查 CIO 是否过度依赖某一位员工观点而忽略了评分信号"
                )
            elif abs(action_val) >= 1 and abs(signals_score_val or 0) >= 4 and signals_score_val * action_val < 0:
                self.add_finding(
                    "方向一致性（轻度）",
                    "medium",
                    passed=False,
                    detail=f"CIO 操作「{cio_action}」vs 评分信号强度「{signals_score_val} ({score_label})」，存在轻度分歧",
                    suggestion="评估 CIO 权衡各维度时的权重是否合理"
                )

        # ── 检查2: PE 分位矛盾 ──
        pe_pct = valuation.get("valuation_percentile")
        if pe_pct is not None:
            if pe_pct > 80 and action_val >= 1:
                self.add_finding(
                    "高估值买入风险",
                    "high",
                    passed=False,
                    detail=f"PE 处于 {pe_pct:.0f}% 分位（高估值），但 CIO 建议「{cio_action}」",
                    suggestion="高估值时买入需有极强的成长逻辑支撑，建议检查 CIO 是否合理考虑了估值风险"
                )
            elif pe_pct < 20 and action_val <= -1:
                self.add_finding(
                    "低估值卖出风险",
                    "high",
                    passed=False,
                    detail=f"PE 处于 {pe_pct:.0f}% 分位（低估值），但 CIO 建议「{cio_action}」",
                    suggestion="低估值时卖出可能错失安全边际机会，建议检查 CIO 是否过度关注短期负面因素"
                )

        # ── 检查3: VaR 与仓位 ──
        var_95 = risk.get("var_95")
        position = (cio.get("order") or {}).get("position_size_pct", 0)
        if var_95 is not None and position > 0:
            if var_95 > 20 and position > 50:
                self.add_finding(
                    "VaR 仓位预警",
                    "medium",
                    passed=False,
                    detail=f"VaR(95%)={var_95:.1f}%（高风险），但建议仓位 {position}%",
                    suggestion=f"VaR={var_95:.1f}% 时仓位建议不超过 30%"
                )

        # ── 检查4: 风险经理否决检查 ──
        risk_report = None
        for emp in employees:
            if emp.get("employee_id") == "risk" or emp.get("role", "").find("风险") >= 0:
                risk_report = emp
                break
        if risk_report and risk_report.get("score") is not None:
            risk_score = risk_report["score"]
            if risk_score <= -7 and action_val >= 1:
                self.add_finding(
                    "风险经理否决忽略",
                    "high",
                    passed=False,
                    detail=f"风险经理评分 {risk_score}（≤-7，软否决阈值），但 CIO 仍建议「{cio_action}」",
                    suggestion="CIO 应详细说明为何否决风险经理的意见，evidence_chain 中需有明确回应"
                )
            elif risk_score <= -5 and action_val >= 2:
                self.add_finding(
                    "风险经理预警忽略",
                    "medium",
                    passed=False,
                    detail=f"风险经理评分 {risk_score}（≤-5），但 CIO 强烈看多/买入",
                    suggestion="CIO 应在 rationale 中详细回应风险经理的担忧"
                )

        # ── 检查5: 止损合理性 ──
        stop_loss = (cio.get("order") or {}).get("stop_loss", {})
        take_profit = (cio.get("order") or {}).get("take_profit", {})
        if isinstance(stop_loss, dict) and stop_loss.get("level"):
            sl_level = float(stop_loss["level"])
            price = quote.get("price", 0)
            if price > 0:
                sl_pct = abs(sl_level - price) / price * 100
                if var_95 is not None and sl_pct < var_95:
                    self.add_finding(
                        "止损过窄",
                        "medium",
                        passed=False,
                        detail=f"止损幅度 {sl_pct:.1f}% 小于 VaR(95%)={var_95:.1f}%，易被波动触发",
                        suggestion=f"建议将止损设置在 VaR 的 1.5-2 倍之外（{var_95 * 1.5:.1f}%）"
                    )

        # ── 检查6: 多空矛盾（技术面 vs 评分技术维度） ──
        tech_indicators = data.get("technical", {})
        rsi_14 = tech_indicators.get("rsi_14")
        if rsi_14 is not None:
            if rsi_14 > 70 and action_val >= 1:
                self.add_finding(
                    "RSI超买区买入",
                    "medium",
                    passed=False,
                    detail=f"RSI(14)={rsi_14:.1f}（超买区），但 CIO 建议「{cio_action}」",
                    suggestion="超买区买入需有充分理由（如强趋势延续），否则建议等待回调"
                )
            elif rsi_14 < 30 and action_val <= -1:
                self.add_finding(
                    "RSI超卖区卖出",
                    "medium",
                    passed=False,
                    detail=f"RSI(14)={rsi_14:.1f}（超卖区），但 CIO 建议「{cio_action}」",
                    suggestion="超卖区卖出可能割肉在地板上，建议确认无更大基本面风险后再决策"
                )

        # ── 检查7: 止盈 vs 止损不对称 ──
        if isinstance(take_profit, dict) and take_profit.get("level_1"):
            tp_level = float(take_profit["level_1"])
            price = quote.get("price", 0)
            if price > 0 and isinstance(stop_loss, dict) and stop_loss.get("level"):
                sl_level = float(stop_loss["level"])
                tp_pct = (tp_level - price) / price * 100
                sl_pct = abs(sl_level - price) / price * 100
                # 只检查买入情况
                if action_val >= 0 and tp_pct > 0 and sl_pct < tp_pct * 0.3:
                    self.add_finding(
                        "盈亏比偏低",
                        "low",
                        passed=False,
                        detail=f"止损 {sl_pct:.1f}% / 止盈 {tp_pct:.1f}% = {sl_pct / tp_pct:.2f}（建议 < 0.5）",
                        suggestion="盈亏比至少 1:2，当前止损幅度过大"
                    )

        # ── 检查8: 数据完整性 ──
        required_files = [
            ("quote.json", "行情数据"),
            ("technical_indicators.json", "技术指标"),
            ("financial_summary.json", "财务摘要"),
            ("sentiment.json", "情感分析"),
            ("valuation.json", "估值分析"),
            ("scoring.json", "综合评分"),
            ("risk_metrics.json", "风险指标"),
        ]
        missing = []
        for fname, label in required_files:
            base_dir = self.data_dir if fname in ["quote.json", "technical_indicators.json", "financial_summary.json"] else self.result_dir
            if not os.path.isfile(os.path.join(base_dir, fname)):
                missing.append(label)
        if missing:
            self.add_finding(
                "数据完整性",
                "low",
                passed=False,
                detail=f"缺失数据: {', '.join(missing)}",
                suggestion="检查数据采集阶段是否有异常"
            )

        # ── 检查9: 证据链覆盖 ──
        evidence = cio.get("evidence_chain", [])
        if evidence and isinstance(evidence, list):
            employees_found = set()
            for item in evidence:
                text = str(item).lower()
                for emp in employees:
                    eid = emp.get("employee_id", "")
                    role = emp.get("role", "")
                    if eid and _check_statements_in_text(text, [eid, role[:2]]):
                        employees_found.add(eid)
            expected_ids = {e.get("employee_id") for e in employees if e.get("employee_id")}
            # Overseer 可能没有 employee_id
            expected_ids.discard(None)
            uncovered = expected_ids - employees_found
            if uncovered:
                self.add_finding(
                    "证据链覆盖不全",
                    "low",
                    passed=False,
                    detail=f"CIO 证据链未引用员工: {', '.join(uncovered)}",
                    suggestion="所有 7 位员工（不含 Overseer）的报告都应在 evidence_chain 中被引用"
                )

        # ── 检查10: 场景概率和 ──
        base_prob = (cio.get("base_case") or {}).get("probability", 0) or 0
        bull_prob = (cio.get("bull_case") or {}).get("probability", 0) or 0
        bear_prob = (cio.get("bear_case") or {}).get("probability", 0) or 0
        prob_sum = base_prob + bull_prob + bear_prob
        if prob_sum > 0 and abs(prob_sum - 1.0) > 0.05:
            self.add_finding(
                "场景概率和不等于1",
                "low",
                passed=False,
                detail=f"三种场景概率之和 = {prob_sum:.0%}（base={base_prob:.0%}, bull={bull_prob:.0%}, bear={bear_prob:.0%}）",
                suggestion="三种场景概率之和应为 100%"
            )

        # ── 检查11: CIO 决策是否清晰 ──
        if not cio.get("rationale") or len(cio.get("rationale", "")) < 50:
            self.add_finding(
                "决策逻辑过简",
                "medium",
                passed=False,
                detail="CIO 的 rationale 不足 50 字，缺少详细的决策推演",
                suggestion="CIO 应逐条列出采纳/否决各员工观点的理由"
            )

        return self.findings


# ========================================================================
#  LLM 深度审计
# ========================================================================

class LLMAuditor:
    """使用独立的 LLM 调用进行深度审计。"""

    AUDIT_SYSTEM_PROMPT = """你是一位专业的投资分析质量审计师（Quality Auditor）。你的任务是对一份「大师投资分析决策」进行全面审计。

审计重点:
1. 数据支撑度 — CIO 的每个核心论据是否有原始数据支持？有无凭空断言？
2. 逻辑一致性 — CIO 的推理链是否自洽？有无逻辑跳跃或矛盾？
3. 哲学一致性 — 决策是否与大师本人的投资哲学一致？（如：巴菲特不应追高热门股）
4. 盲点识别 — CIO 是否忽略了某些关键风险或信号？Overseer 是否尽责？
5. 信息利用度 — 是否充分利用了所有可用的数据维度？

请逐条列出你的审计发现，对每条发现标注严重度（high/medium/low）。"""

    def __init__(self):
        self.client = None
        self.model = None
        self._init_llm()

    def _init_llm(self):
        """初始化 LLM 客户端（复用项目配置）。"""
        try:
            from openai import OpenAI
            from config import settings
            if settings.LLM_API_KEY:
                self.client = OpenAI(
                    api_key=settings.LLM_API_KEY,
                    base_url=settings.LLM_BASE_URL,
                )
                self.model = settings.LLM_MODEL_NAME
        except Exception as e:
            logger.warning(f"LLM 审计客户端初始化失败: {e}")

    @property
    def available(self) -> bool:
        return self.client is not None

    def audit(self, data_dir: str, result_dir: str, master_dir: str) -> List[Dict]:
        """执行 LLM 深度审计。"""
        if not self.available:
            logger.warning("LLM 审计不可用（无 API Key），跳过")
            return []

        # 构建审计上下文
        context = self._build_audit_context(data_dir, result_dir, master_dir)
        if not context.strip():
            return []

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.AUDIT_SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            if not raw:
                return []
            result = json.loads(raw)
            findings = result.get("findings", result.get("audit_findings", []))
            if isinstance(findings, list):
                return findings
            # 尝试将整个输出作为单个 finding
            return [{
                "check": "LLM 深度审计",
                "severity": "medium",
                "detail": str(result).strip()[:500],
                "source": "llm_audit",
            }]
        except Exception as e:
            logger.error(f"LLM 审计调用失败: {e}")
            return [{
                "check": "LLM 深度审计",
                "severity": "low",
                "detail": f"LLM 审计调用异常: {e}",
                "source": "llm_audit_error",
            }]

    def _build_audit_context(self, data_dir: str, result_dir: str, master_dir: str) -> str:
        """构建审计上下文（包含数据 + 结果 + 大师决策）。"""
        sections = []

        # 加载大师决策
        cio = safe_read_json(os.path.join(master_dir, "cio_decision.json")) or {}
        employees = safe_read_json(os.path.join(master_dir, "employee_reports.json")) or []
        prediction = safe_read_json(os.path.join(master_dir, "prediction_summary.json")) or {}

        # CIO决策摘要
        sections.append("=== CIO 决策 ===")
        sections.append(json.dumps({
            "decision_summary": cio.get("decision_summary", ""),
            "rationale": cio.get("rationale", ""),
            "evidence_chain": cio.get("evidence_chain", []),
            "order": cio.get("order", {}),
            "base_case": cio.get("base_case", {}),
            "bull_case": cio.get("bull_case", {}),
            "bear_case": cio.get("bear_case", {}),
        }, ensure_ascii=False, indent=2)[:3000])

        # 员工报告摘要
        sections.append("\n=== 员工报告摘要 ===")
        for emp in employees[:8]:
            sections.append(json.dumps({
                "id": emp.get("employee_id", ""),
                "role": emp.get("role", ""),
                "outlook": emp.get("outlook", ""),
                "confidence": emp.get("confidence", ""),
                "score": emp.get("score", ""),
                "key_points": emp.get("key_points", [])[:5],
                "risks": emp.get("risks", [])[:3],
            }, ensure_ascii=False, indent=2)[:800])

        # 关键原始数据
        sections.append("\n=== 关键原始数据 ===")
        for fname, label in [
            ("quote.json", "行情"),
            ("financial_summary.json", "财务摘要"),
        ]:
            data = safe_read_json(os.path.join(data_dir, fname))
            if data:
                sections.append(f"--- {label} ---")
                sections.append(json.dumps(data, ensure_ascii=False, indent=2)[:1000])

        for fname, label in [
            ("sentiment.json", "情感分析"),
            ("valuation.json", "估值"),
            ("scoring.json", "评分"),
            ("risk_metrics.json", "风险指标"),
        ]:
            data = safe_read_json(os.path.join(result_dir, fname))
            if data:
                sections.append(f"--- {label} ---")
                sections.append(json.dumps(data, ensure_ascii=False, indent=2)[:800])

        # 审计指令
        sections.append("""
=== 审计指令 ===
请以 JSON 格式输出审计结果，格式如下:
{
    "audit_findings": [
        {
            "check": "审计项名称",
            "severity": "high/medium/low",
            "detail": "具体发现描述，引用数据",
            "suggestion": "改进建议"
        }
    ],
    "summary": "总体评价（100字内）"
}
""")
        return "\n".join(sections)


# ========================================================================
#  报告生成
# ========================================================================

class ReportGenerator:
    """生成审计报告和改进计划。"""

    def __init__(self, output_dir: str, symbol: str, master_name: str,
                 master_key: str, findings: List[Dict],
                 llm_findings: List[Dict], data_completeness: Dict):
        self.output_dir = output_dir
        self.symbol = symbol
        self.master_name = master_name
        self.master_key = master_key
        self.findings = findings
        self.llm_findings = llm_findings
        self.data_completeness = data_completeness

    def generate_report(self) -> str:
        """生成审计报告 report.txt。"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        all_findings = self.findings + self.llm_findings
        sorted_findings = sorted(all_findings, key=lambda f: SEVERITY_ORDER.get(f.get("severity", "low"), 99))
        n_high = sum(1 for f in sorted_findings if f.get("severity") == "high")

        lines = [
            "=" * 60,
            "  大师分析质量审计报告",
            "=" * 60,
            f"  股票:  {self.symbol}",
            f"  大师:  {self.master_name} ({self.master_key})",
            f"  时间:  {now}",
            f"  发现:  {len(sorted_findings)} 项（高风险 {n_high} 项）",
            "=" * 60,
            "",
        ]

        # 数据完整性
        lines.append("─" * 40)
        lines.append("  数据完整性")
        lines.append("─" * 40)
        dc = self.data_completeness
        total = dc.get("total", 0)
        ok = dc.get("ok", 0)
        pct = (ok / total * 100) if total > 0 else 0
        lines.append(f"  完成度: {ok}/{total} ({pct:.0f}%)")
        for item in dc.get("items", []):
            icon = "✅" if item.get("ok") else "❌"
            lines.append(f"    {icon} {item['label']}")
        lines.append("")

        # 规则检查结果
        if self.findings:
            lines.append("─" * 40)
            lines.append("  规则检查结果")
            lines.append("─" * 40)
            for f in sorted(self.findings, key=lambda x: SEVERITY_ORDER.get(x.get("severity", "low"), 99)):
                sev = f.get("severity", "low")
                sev_label = {"high": "🔴 HIGH", "medium": "🟡 MED", "low": "🟢 LOW"}.get(sev, sev)
                lines.append(f"  [{sev_label}] {f['check']}")
                lines.append(f"          {f['detail']}")
                if f.get("suggestion"):
                    lines.append(f"          → {f['suggestion']}")
                lines.append("")
        else:
            lines.append("  规则检查全部通过 ✅")
            lines.append("")

        # LLM 审计发现
        if self.llm_findings:
            lines.append("─" * 40)
            lines.append("  LLM 深度审计发现")
            lines.append("─" * 40)
            for f in self.llm_findings:
                sev = f.get("severity", "medium")
                sev_label = {"high": "🔴 HIGH", "medium": "🟡 MED", "low": "🟢 LOW"}.get(sev, sev)
                lines.append(f"  [{sev_label}] {f.get('check', '审计发现')}")
                lines.append(f"          {f.get('detail', '')[:500]}")
                if f.get("suggestion"):
                    lines.append(f"          → {f['suggestion'][:300]}")
                lines.append("")

        # 综合评级
        lines.append("─" * 40)
        lines.append("  综合可信度评级")
        lines.append("─" * 40)
        if n_high >= 3:
            rating = "低 ⚠️"
        elif n_high >= 1:
            rating = "中 🔶"
        else:
            rating = "高 ✅"
        lines.append(f"  评级: {rating}")
        lines.append(f"  高风险: {n_high} 项")
        lines.append(f"  中风险: {sum(1 for f in sorted_findings if f.get('severity') == 'medium')} 项")
        lines.append(f"  低风险: {sum(1 for f in sorted_findings if f.get('severity') == 'low')} 项")
        lines.append("")

        report = "\n".join(lines)
        report_path = os.path.join(self.output_dir, "report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"审计报告已保存: {report_path}")
        return report_path

    def generate_plan(self) -> str:
        """生成改进计划 plan.txt。"""
        all_findings = self.findings + self.llm_findings
        high_priority = [f for f in all_findings if f.get("severity") == "high"]
        medium_priority = [f for f in all_findings if f.get("severity") == "medium"]
        low_priority = [f for f in all_findings if f.get("severity") == "low"]

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "=" * 60,
            "  大师分析改进计划",
            "=" * 60,
            f"  股票:  {self.symbol}",
            f"  大师:  {self.master_name} ({self.master_key})",
            f"  时间:  {now}",
            "=" * 60,
            "",
        ]

        # P1 问题
        lines.append("─" * 40)
        lines.append("  🔴 P1 — 需立即修复")
        lines.append("─" * 40)
        if high_priority:
            for i, f in enumerate(high_priority, 1):
                lines.append(f"  {i}. {f['check']}")
                lines.append(f"     问题: {f['detail']}")
                lines.append(f"     建议: {f.get('suggestion', '需人工复核')}")
                lines.append("")
        else:
            lines.append("  无 P1 问题 ✅")
            lines.append("")

        # P2 问题
        lines.append("─" * 40)
        lines.append("  🟡 P2 — 可改进")
        lines.append("─" * 40)
        if medium_priority:
            for i, f in enumerate(medium_priority, 1):
                lines.append(f"  {i}. {f['check']}")
                lines.append(f"     问题: {f['detail']}")
                lines.append(f"     建议: {f.get('suggestion', '下次迭代优化')}")
                lines.append("")
        else:
            lines.append("  无 P2 问题 ✅")
            lines.append("")

        # P3 建议
        lines.append("─" * 40)
        lines.append("  🟢 P3 — 优化建议")
        lines.append("─" * 40)
        if low_priority:
            for i, f in enumerate(low_priority, 1):
                lines.append(f"  {i}. {f['check']}")
                lines.append(f"     问题: {f['detail']}")
                lines.append(f"     建议: {f.get('suggestion', '长期优化方向')}")
                lines.append("")
        else:
            lines.append("  无 P3 问题 ✅")
            lines.append("")

        # 架构建议
        lines.append("─" * 40)
        lines.append("  架构改进方向")
        lines.append("─" * 40)
        lines.append("  1. 如反复出现数据缺失，检查 provider 数据采集完整性")
        lines.append("  2. 如 CIO 频繁忽略风险否决，考虑在 BaseAgent 层增加强制回应机制")
        lines.append("  3. 如证据链覆盖不全，优化 CIO prompt 要求逐条引用")
        lines.append("  4. 考虑在 analysis/agents/cio_prompts.py 中增加大师风格检查约束")
        lines.append("")

        plan = "\n".join(lines)
        plan_path = os.path.join(self.output_dir, "plan.txt")
        with open(plan_path, "w", encoding="utf-8") as f:
            f.write(plan)
        logger.info(f"改进计划已保存: {plan_path}")
        return plan_path


# ========================================================================
#  主流程
# ========================================================================

class StockMasterAuditor:
    """大师分析审计总控。"""

    def __init__(self, output_dir: str = "master-audit", seed: Optional[int] = None,
                 no_llm_audit: bool = False):
        self.output_dir = Path(output_dir)
        self.seed = seed if seed is not None else random.randint(0, 999999)
        self.no_llm_audit = no_llm_audit
        self.symbol = None
        self.master_key = None
        self.master_name = None
        self._setup_logging()

    def _setup_logging(self):
        """配置日志。"""
        logger.remove()
        logger.add(sys.stderr, format="<level>{level: <8}</level> | {message}", level="INFO")

    def select(self, symbol: Optional[str] = None, master_key: Optional[str] = None):
        """选择股票和大师。"""
        random.seed(self.seed)
        logger.info(f"随机种子: {self.seed}")

        # 选择股票
        if symbol:
            self.symbol = symbol
            logger.info(f"指定股票: {self.symbol}")
        else:
            stocks = load_csi300_stocks()
            self.symbol = random.choice(stocks)
            logger.info(f"随机选取股票: {self.symbol}")

        # 选择大师
        if master_key:
            self.master_key = master_key.lower()
            logger.info(f"指定大师: {master_key}")
        else:
            masters = get_available_masters()
            chosen = random.choice(masters)
            self.master_key = chosen["key"]
            logger.info(f"随机选取大师: {chosen.get('name', self.master_key)} ({self.master_key})")

        # 获取大师展示名称
        for m in get_available_masters():
            if m["key"] == self.master_key:
                self.master_name = m.get("name", self.master_key.capitalize())
                break
        if not self.master_name:
            self.master_name = self.master_key.capitalize()

    def run(self) -> int:
        """执行完整的审计流程。返回退出码。"""
        if not self.symbol or not self.master_key:
            logger.error("未选择股票或大师")
            return 1

        logger.info("")
        logger.info("=" * 50)
        logger.info(f"  大师分析审计开始")
        logger.info(f"  股票: {self.symbol}  大师: {self.master_name} ({self.master_key})")
        logger.info("=" * 50)

        # 自动生成日期序号目录: master-audit/YYYY-MM-DD-NN/
        today = datetime.now().strftime("%Y-%m-%d")
        max_seq = 0
        if self.output_dir.exists():
            for d in self.output_dir.iterdir():
                if d.is_dir() and d.name.startswith(today):
                    try:
                        seq = int(d.name.split("-")[-1])
                        max_seq = max(max_seq, seq)
                    except ValueError:
                        pass
        seq_dir_name = f"{today}-{max_seq + 1:02d}"
        self.output_dir = self.output_dir / seq_dir_name
        logger.info(f"  输出目录: {self.output_dir}")

        # 创建输出目录
        safe = f"{self.symbol}_{self.master_key}".replace(".", "_")
        data_dir = ensure_dir(str(self.output_dir / "data" / safe))
        result_dir = ensure_dir(str(self.output_dir / "result" / safe))
        master_dir = ensure_dir(str(self.output_dir / "master" / safe))

        # ── 第1步: 运行大师分析 ──
        logger.info("")
        logger.info("[Step 1/4] 运行大师分析工作流...")
        result = self._run_analysis()
        if result is None:
            logger.error("分析失败，无法继续审计")
            return 1

        # ── 第2步: 保存中间数据 ──
        logger.info("")
        logger.info("[Step 2/4] 保存中间数据...")
        self._save_data(result, data_dir, result_dir, master_dir)

        # ── 第3步: 交叉验证 ──
        logger.info("")
        logger.info("[Step 3/4] 交叉验证...")
        findings, llm_findings = self._cross_validate(data_dir, result_dir, master_dir)

        # ── 第4步: 生成报告 ──
        logger.info("")
        logger.info("[Step 4/4] 生成报告...")
        completeness = self._calc_data_completeness(data_dir, result_dir, master_dir)
        gen = ReportGenerator(
            output_dir=str(self.output_dir),
            symbol=self.symbol,
            master_name=self.master_name,
            master_key=self.master_key,
            findings=findings,
            llm_findings=llm_findings,
            data_completeness=completeness,
        )
        report_path = gen.generate_report()
        plan_path = gen.generate_plan()

        # ── 输出摘要 ──
        logger.info("")
        logger.info("=" * 50)
        logger.info("  审计完成")
        logger.info("=" * 50)
        logger.info(f"  数据:    {data_dir}")
        logger.info(f"  结果:    {result_dir}")
        logger.info(f"  大师:    {master_dir}")
        logger.info(f"  报告:    {report_path}")
        logger.info(f"  计划:    {plan_path}")
        n_total = len(findings) + len(llm_findings)
        n_high = sum(1 for f in findings + llm_findings if f.get("severity") == "high")
        if n_total > 0:
            logger.info(f"  发现:    {n_total} 项（高风险 {n_high} 项）")
        else:
            logger.info(f"  发现:    全部通过 ✅")
        logger.info("")

        return 0

    def _run_analysis(self) -> Optional[Dict]:
        """运行 StockAnalysisAgent.analyze()。

        Note: Monkey-patch _extract_detailed_financials 使其在无网络环境不阻塞。
        akshare 的 stock_financial_abstract 不受 mock 后端控制且无内置超时。
        """
        # ── Monkey-patch: 替换 akshare 直接调用为安全版本 ──
        self._patch_agent_for_offline()

        def _do_analyze() -> Optional[Dict]:
            """实际分析调用。"""
            from analysis.agent import StockAnalysisAgent
            from config import settings
            _ = settings

            agent = StockAnalysisAgent()
            return agent.analyze(
                symbol=self.symbol,
                cost_price=0,
                master=self.master_key,
            )

        try:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
            TIMEOUT = 300  # 总超时 5min（足够 LLM 调用 + 数据采集）

            logger.info(f"  调用 analyze(symbol={self.symbol}, master={self.master_key})...")
            logger.info(f"  ⏱ 总超时: {TIMEOUT}s")

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_analyze)
                result = future.result(timeout=TIMEOUT)

            if result and result.get("status") == "error":
                logger.error(f"  分析返回错误: {result.get('error', '未知错误')}")
                return None

            outlook = result.get('prediction_summary', {}).get('outlook', 'N/A')
            logger.info(f"  分析完成: status={result.get('status')}, outlook={outlook}")
            return result

        except FutureTimeout:
            logger.error(f"  分析总超时（>{TIMEOUT}s）")
            return None
        except Exception as e:
            logger.error(f"  分析异常: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _patch_agent_for_offline(self):
        """全面禁用所有可能阻塞的网络调用。

        在 mock 或无网络环境下，以下代码会挂起：
        - akshare.stock_financial_abstract （_extract_detailed_financials）
        - akshare 其他 API 调用（_fetch_macro_context、_search_management_quality 等）
        - _safe_ak_call 方法

        通过在 import 层面打补丁让 akshare 所有函数返回空 DataFrame。
        """
        import sys
        import pandas as pd

        class _MockAkShare:
            """替换 akshare 模块所有方法为空操作。"""
            def __getattr__(self, name):
                return lambda *a, **kw: pd.DataFrame()

        sys.modules['akshare'] = _MockAkShare()

        # 注意：在 mock 模式下 _fetch_macro_context 会在 import analysis.agent 时
        # 立即执行，补丁必须在 import 之前。但这里是为了线程安全
        logger.debug("  全局禁用: akshare → _MockAkShare (所有调用返回空DataFrame)")

    def _save_data(self, result: Dict, data_dir: str, result_dir: str, master_dir: str):
        """从分析结果中拆分并保存数据到3个目录。"""
        ps = result.get("prediction_summary") or {}
        signals = result.get("signals") or result.get("score_breakdown") or {}

        # ── analysis/data/ — 原始采集数据 ──
        data_files = {
            "quote.json": result.get("quote"),
            "technical_indicators.json": result.get("technical_indicators"),
            "financial_summary.json": result.get("financial_summary"),
            "news.json": result.get("news", [])[:20],
            "guba_posts.json": result.get("guba_posts", [])[:20],
            "macro_context.json": result.get("macro_context"),
            "industry_context.json": result.get("industry_context"),
            "search_results.json": result.get("search_results"),
        }
        for fname, data in data_files.items():
            if data is not None and data:
                safe_save_json(data, os.path.join(data_dir, fname))

        # ── analysis/result/ — 各环节处理结果 ──
        result_files = {
            "sentiment.json": {
                "news": result.get("sentiment_news"),
                "guba": result.get("sentiment_guba"),
                "important_bullish_news": result.get("important_bullish_news", [])[:3],
                "important_bearish_news": result.get("important_bearish_news", [])[:3],
                "sentiment_percentile": result.get("sentiment_percentile"),
            },
            "valuation.json": {
                "valuation_level": result.get("valuation_level"),
                "valuation_percentile": result.get("valuation_percentile"),
                "valuation_percentile_5y": result.get("valuation_percentile_5y"),
                "valuation_percentile_10y": result.get("valuation_percentile_10y"),
                "suggested_buy_price": result.get("suggested_buy_price"),
                "historical_pe_avg": result.get("historical_pe_avg"),
                "pe_avg_5y": result.get("pe_avg_5y"),
                "pe_avg_10y": result.get("pe_avg_10y"),
            },
            "scoring.json": {
                "signals": signals,
            },
            "risk_metrics.json": {
                "var_95": result.get("var_95"),
                "max_drawdown": result.get("max_drawdown"),
                "beta": result.get("beta"),
                "annualized_volatility": result.get("annualized_volatility"),
                "earnings_yield": result.get("earnings_yield"),
                "equity_risk_premium": result.get("equity_risk_premium"),
            },
            "financial_depth.json": {
                "roic": result.get("roic"),
                "fcf_per_share": result.get("fcf_per_share"),
                "operating_cash_flow_per_share": result.get("operating_cash_flow_per_share"),
                "owner_earnings_per_share": result.get("owner_earnings_per_share"),
                "financial_trends": result.get("financial_trends"),
            },
            "peer_valuation.json": result.get("peer_valuation"),
            "moat_assessment.json": result.get("moat_assessment"),
        }
        for fname, data in result_files.items():
            if data is not None:
                safe_save_json(data, os.path.join(result_dir, fname))

        # ── analysis/master/ — 大师分析结果 ──
        cio_decision = ps.get("cio_decision")
        employee_reports = ps.get("employee_reports", [])
        master_files = {
            "cio_decision.json": cio_decision,
            "employee_reports.json": employee_reports,
            "prediction_summary.json": {
                "outlook": ps.get("outlook"),
                "confidence": ps.get("confidence"),
                "price_target_current": ps.get("price_target_current"),
                "price_target_low": ps.get("price_target_low"),
                "price_target_high": ps.get("price_target_high"),
                "reason": ps.get("reason"),
                "short_term": ps.get("short_term"),
                "mid_term": ps.get("mid_term"),
                "long_term": ps.get("long_term"),
                "suggested_action": ps.get("suggested_action"),
                "risk_factors": result.get("risk_factors"),
            },
        }
        for fname, data in master_files.items():
            if data is not None:
                safe_save_json(data, os.path.join(master_dir, fname))

        saved = sum(
            1 for files in [data_files, result_files, master_files]
            for fname, data in (files.items() if isinstance(files, dict) else files.items() if isinstance(files, dict) else [])
        )
        # 修复: 计算实际保存的文件数
        data_count = sum(1 for v in data_files.values() if v is not None)
        result_count = sum(1 for v in result_files.values() if v is not None)
        master_count = sum(1 for v in master_files.values() if v is not None)
        logger.info(f"  已保存: data/{data_count} + result/{result_count} + master/{master_count} 文件")

    def _cross_validate(self, data_dir: str, result_dir: str, master_dir: str):
        """运行交叉验证。"""
        # 规则检查
        logger.info("  运行规则检查...")
        auditor = RuleBasedAuditor(data_dir, result_dir, master_dir)
        auditor.run_all()
        findings = auditor.findings
        logger.info(f"  规则检查完成: {len(findings)} 项发现")

        # LLM 深度审计
        llm_findings = []
        if not self.no_llm_audit:
            logger.info("  LLM 深度审计...")
            llm_auditor = LLMAuditor()
            if llm_auditor.available:
                llm_findings = llm_auditor.audit(data_dir, result_dir, master_dir)
                logger.info(f"  LLM 审计完成: {len(llm_findings)} 项发现")
            else:
                logger.info("  LLM 审计不可用（跳过）")
        else:
            logger.info("  LLM 审计已禁用（--no-llm-audit）")

        return findings, llm_findings

    def _calc_data_completeness(self, data_dir: str, result_dir: str,
                                master_dir: str) -> Dict:
        """计算数据完整性。"""
        checks = [
            (os.path.join(data_dir, "quote.json"), "行情数据"),
            (os.path.join(data_dir, "technical_indicators.json"), "技术指标"),
            (os.path.join(data_dir, "financial_summary.json"), "财务摘要"),
            (os.path.join(data_dir, "news.json"), "新闻"),
            (os.path.join(data_dir, "guba_posts.json"), "股吧"),
            (os.path.join(data_dir, "macro_context.json"), "宏观数据"),
            (os.path.join(data_dir, "industry_context.json"), "行业数据"),
            (os.path.join(result_dir, "sentiment.json"), "情感分析"),
            (os.path.join(result_dir, "valuation.json"), "估值分析"),
            (os.path.join(result_dir, "scoring.json"), "综合评分"),
            (os.path.join(result_dir, "risk_metrics.json"), "风险指标"),
            (os.path.join(result_dir, "financial_depth.json"), "财务深度"),
            (os.path.join(master_dir, "cio_decision.json"), "CIO决策"),
            (os.path.join(master_dir, "employee_reports.json"), "员工报告"),
            (os.path.join(master_dir, "prediction_summary.json"), "预测摘要"),
        ]
        items = []
        ok_count = 0
        for path, label in checks:
            exists = os.path.isfile(path)
            items.append({"path": path, "label": label, "ok": exists})
            if exists:
                ok_count += 1
        return {"total": len(checks), "ok": ok_count, "items": items}


# ========================================================================
#  CLI 入口
# ========================================================================

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="大师分析质量审计与优化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/master_audit.py
  python scripts/master_audit.py --symbol 600519 --master buffett
  python scripts/master_audit.py --seed 42
  python scripts/master_audit.py --no-llm-audit
        """,
    )
    parser.add_argument("--symbol", type=str, default=None,
                        help="股票代码（如 600519），不指定则随机")
    parser.add_argument("--master", type=str, default=None,
                        help="大师 key（如 buffett），不指定则随机")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（默认按日期）")
    parser.add_argument("--output-dir", type=str, default="master-audit",
                        help="输出目录（默认 master-audit，自动按日期+序号建子目录）")
    parser.add_argument("--no-llm-audit", action="store_true",
                        help="跳过 LLM 深度审计")
    return parser.parse_args()


def main():
    """主入口。"""
    args = parse_args()

    auditor = StockMasterAuditor(
        output_dir=args.output_dir,
        seed=args.seed,
        no_llm_audit=args.no_llm_audit,
    )
    auditor.select(symbol=args.symbol, master_key=args.master)
    exit_code = auditor.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
