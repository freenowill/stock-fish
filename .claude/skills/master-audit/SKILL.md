---
name: master-audit
description: 大师分析质量审计与优化 — 随机抽检沪深300+大师，运行全流程分析，保存中间数据，交叉验证，生成改进报告和计划
dependencies: python>=3.10
---

# 大师分析质量审计

随机从沪深300挑选一支股票和一位大师，运行完整的大师分析工作流，保存所有中间数据，进行交叉验证，最终输出审计报告和改进计划。

## 调用方式

### 方式一：Claude Code 技能（当前环境）

```
/master-audit                          # 随机股票 + 随机大师
/master-audit --symbol 600519           # 指定股票，随机大师
/master-audit --master buffett          # 指定大师，随机股票
/master-audit --symbol 600519 --master buffett  # 全部指定
/master-audit --seed 42                 # 固定种子复现
```

### 方式二：直接运行脚本（任何终端）

```bash
# 在项目根目录下执行
python scripts/master_audit.py                                           # 随机股票+大师
python scripts/master_audit.py --symbol 600519                            # 指定股票
python scripts/master_audit.py --master buffett                           # 指定大师
python scripts/master_audit.py --symbol 600519 --master buffett           # 全部指定
python scripts/master_audit.py --seed 42                                  # 固定种子
python scripts/master_audit.py --output-dir /tmp/audit                    # 自定义输出目录
python scripts/master_audit.py --no-llm-audit                             # 跳过 LLM 深度审计（仅规则审计）
python scripts/master_audit.py --help                                     # 查看全部参数
```

## 相关脚本和模块

| 文件 | 角色 | 说明 |
|---|---|---|
| `scripts/master_audit.py` | **主脚本** | 审计入口，组合选股→分析→审计→报告全流程 |
| `analysis/agent.py` | 分析引擎 | `StockAnalysisAgent.analyze()` 驱动 5 步分析管道 |
| `analysis/agents/base.py` | 基类定义 | `EmployeeReport` / `CIODecision` / `BaseAgent` 基础类 |
| `analysis/agents/cio.py` | 最终决策 | CIO 大师决策逻辑 |
| `analysis/agents/cio_prompts.py` | 大师定义 | 7 位大师 system prompt + 输出 schema (`list_masters()`) |
| `analysis/agents/overseer.py` | 独立监察 | 跨员工报告核查，挑战假设和盲点 |
| `analysis/agents/macro_agent.py` | 宏观分析员 | SHIBOR/PMI/CPI 等宏观因子分析 |
| `analysis/agents/policy_agent.py` | 政策分析员 | 行业周期和政策影响评估 |
| `analysis/agents/valuation_agent.py` | 估值分析员 | PE分位/DCF/安全边际分析 |
| `analysis/agents/fundamental_agent.py` | 基本面分析员 | ROE/EPS/护城河/成长性分析 |
| `analysis/agents/technical_agent.py` | 技术分析员 | RSI/MACD/MA/布林带分析 |
| `analysis/agents/sentiment_agent.py` | 舆情分析员 | 新闻/股吧情感分析 |
| `analysis/agents/risk_manager.py` | 风险经理 | VaR/回撤/下行风险评估 |
| `analysis/scoring.py` | 评分引擎 | 三层加权评分系统（技术/基本面/情绪） |
| `analysis/nodes/prediction_node.py` | LLM 预测 | 3 种模式预测（大师/多智能体/规则） |
| `market_data/a_stock_provider.py` | 数据后端 | A 股数据源自动选择 |

## 执行步骤

1. **激活环境**（如需要）：`source venv/bin/activate`
2. **运行脚本**：`python scripts/master_audit.py [参数]`
3. **读取结果**：取 `master-audit/{date-seq}/report.txt`（审计报告）和 `plan.txt`（改进计划）
4. **清理**：确认后清理数据或保留

## 输出目录结构

```
master-audit/{YYYY-MM-DD-NN}/
├── data/{symbol}_{master}/          # 原始采集数据（行情/技术指标/财报/新闻/股吧/宏观/行业）
├── result/{symbol}_{master}/        # 处理结果（情感分析/估值/评分/风险/财务深度）
├── master/{symbol}_{master}/        # 大师分析结果（CIO 决策 / 8 份员工报告 / 预测摘要）
├── report.txt                       # 审计报告（按严重度：HIGH > MED > LOW）
└── plan.txt                         # 改进计划
```

## 约束

- 始终在项目根目录下执行
- 每次运行自动创建 `master-audit/YYYY-MM-DD-NN/` 目录
- 审计发现按严重度排序展示：**HIGH** > **MED** > **LOW**
- `--symbol` 参数接收沪深 300 成分股代码（如 `600519`、`000858`、`300750`）
- `--master` 参数接收大师 key：`graham`、`buffett`、`fisher`、`lynch`、`templeton`、`soros`、`dalio`
- `--seed` 不指定时按日期生成种子，相同种子可复现同一组股票+大师选择
- `--no-llm-audit` 模式下仅执行规则层面的数据完整性/一致性审计，不发起 LLM 深度评估
