---
name: master-audit
description: 大师分析质量审计与优化 — 随机抽检沪深300+大师，运行全流程分析，保存中间数据，交叉验证，生成改进报告和计划
dependencies: python>=3.10
---

# 大师分析质量审计

随机从沪深300挑选一支股票和一位大师，运行完整的大师分析工作流，保存所有中间数据，进行交叉验证，最终输出审计报告和改进计划。

## 触发条件

当用户提及以下内容时自动触发：
- "审计大师分析"、"质量审计"、"优化大师"
- "/master-audit"
- "随机抽检"、"交叉验证"

## 用法

```
/master-audit                          # 随机股票 + 随机大师
/master-audit --symbol 600519           # 指定股票，随机大师
/master-audit --master buffett          # 指定大师，随机股票
/master-audit --symbol 600519 --master buffett  # 全部指定
/master-audit --seed 42                 # 指定随机种子
```

## 执行步骤

1. **激活环境**: `source venv/bin/activate`（必要时）
2. **运行脚本**: `python scripts/master_audit.py [参数]`
3. **读取结果**: 读取 `analysis/report.txt` 和 `analysis/plan.txt`，呈现摘要
4. **清理**: 询问用户是否要查看 JSON 明细或清理数据

## 输出文件

| 路径 | 内容 |
|---|---|
| `master-audit/{date-seq}/data/{symbol}_{master}/` | 原始采集数据（行情/技术/财务/新闻/股吧/宏观/行业） |
| `master-audit/{date-seq}/result/{symbol}_{master}/` | 各环节处理结果（情感/估值/评分/风险/财务深度） |
| `master-audit/{date-seq}/master/{symbol}_{master}/` | 大师分析结果（CIO决策/8份员工报告/预测摘要） |
| `master-audit/{date-seq}/report.txt` | 审计报告 |
| `master-audit/{date-seq}/plan.txt` | 改进计划 |

## 约束

- 始终在项目根目录下执行
- 每次运行自动创建 `master-audit/YYYY-MM-DD-NN/` 目录
- 审计发现按严重度排序展示：HIGH > MED > LOW
