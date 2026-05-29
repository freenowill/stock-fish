# StockFish — A 股智能分析 + 股价推演系统

> **完整决策链路**：[qlib-zh](https://github.com/freenowill/qlib-zh) 因子选股模型 → 选出 Top-K 候选股票 → StockFish 多因子深度分析 + MiroFish OASIS 群体智能推演 → 辅助投资决策

A 股多因子分析引擎：多源行情采集 → 情感计算 → 估值评估 → 信号生成 → LLM 预测，支持 MiroFish OASIS 群体智能模拟推演。

## 架构

```
POST /api/analyze
  │
  ├─ Step 1: 数据采集 (多源策略，6 Fetcher 自动切换 + 熔断保护)
  │   ├─ Tushare Pro (sxsc)   → 行情 / 历史K线 / 历史PE / 基本面
  │   ├─ 东方财富 (efinance)  → 实时行情 / 板块排名
  │   ├─ 腾讯财经/新浪/东财  → 实时行情多源优先 (akshare)
  │   ├─ 通达信 (pytdx)       → 免费行情兜底
  │   ├─ 证券宝 (baostock)    → 学术级免费数据
  │   └─ 雅虎财经 (yfinance)  → 全球行情 + A股/港股/美股
  │
  ├─ Step 1.5: 新闻搜索 (7 引擎搜索 API，优先 Tavily)
  │   ├─ Tavily / Bocha / Brave / SerpAPI / MiniMax / SearXNG
  │   └─ 兜底: 新浪财经 + 财联社/雪球/华尔街见闻
  │
  ├─ Step 2: 情感分析
  │   └─ HuggingFace 多语言情感模型 → 5级分类 + 关键词规则降级
  │
  ├─ Step 3: 估值 + 基本面 + 信号生成
  │   ├─ PE 3年历史分位 → 很低/偏低/正常/偏高/很高
  │   ├─ 基本面聚合: 增长率/收益/机构持股/资金流向/龙虎榜
  │   ├─ 建议买入价 ← PE均值回归 + 布林下轨支撑
  │   └─ 加权评分 ← RSI/MACD/KDJ/均线/布林/估值/舆情
  │
  ├─ Step 4: LLM 综合预测
  │   └─ deepseek-v4-flash → 3 Agent 并行辩论 + Moderatord 综合裁决
  │
  └─ Step 5: 前端渲染
      └─ 6卡片 + 技术/基本面 + 新闻/股吧摘要 (A股红涨绿跌)
```

### 预测推演链路

```
/api/predict (异步)
  │
  ├─ 分析 (同上) → 种子文档生成
  ├─ MiroFish OASIS 模拟 (可选, HTTP桥接)
  ├─ 社交情感 [可选] Reddit / X / Polymarket 美股热度
  └─ HTML 预测报告输出 → reports/
```

## 快速开始

### 1. 配置

```bash
cp .env.example .env
# 编辑 .env，填入 API Key
```

必需：

| 变量 | 说明 |
|------|------|
| `LLM_API_KEY` | DeepSeek / OpenAI 兼容 API Key |
| `LLM_BASE_URL` | API 地址 (如 `https://api.deepseek.com`) |
| `LLM_MODEL_NAME` | 模型名 (如 `deepseek-v4-flash`) |
| `STOCK_BACKEND` | 数据后端：`advanced` / `tushare` / `akshare` / `mock` |

数据源（`STOCK_BACKEND=advanced` 时自动启用）：

| 变量 | 说明 |
|------|------|
| `TUSHARE_TOKEN` | Tushare Pro 数据令牌 (sxsc_tushare 代理) |
| `TAVILY_API_KEY` | Tavily 搜索引擎 Key (新闻搜索) |
| `BOCHA_API_KEY` | 博查搜索 Key (中文新闻，推荐) |
| `BRAVE_API_KEY` | Brave Search Key |
| `SERPAPI_API_KEY` | SerpAPI (Google) Key |
| `FINNHUB_API_KEY` | Finnhub 美股数据 |
| `LONGBRIDGE_APP_KEY` 等 | 长桥 OpenAPI 港美股 |
| `SOCIAL_SENTIMENT_API_KEY` | 美股社交情感 (Reddit/X/Polymarket) |

可选：

| 变量 | 说明 |
|------|------|
| `ZEP_API_KEY` | MiroFish 图记忆 (Zep Cloud) |
| `MIROFISH_HOST` | MiroFish 服务地址 |
| `EFINANCE_CALL_TIMEOUT` | 东方财富超时秒数 (默认30，境外建议5) |
| `SEARXNG_PUBLIC_INSTANCES_ENABLED` | SearXNG 公共实例 (境外建议 false) |

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动

```bash
# 本地模式
bash run.sh --local

# Docker 模式
bash run.sh

# Docker (跳过 MiroFish)
bash run.sh --no-mirofish
```

访问 `http://localhost:8000`

### 4. API 示例

```bash
# 多因子分析
curl -X POST http://localhost:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"600519"}'

# 股价推演
curl -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"600519","scenario":"base"}'
```

## 数据源

### 行情数据（6 Fetcher 自动切换 + 熔断保护）

| Fetcher | 优先级 | 数据来源 | 覆盖市场 | 需要 Key |
|---------|--------|----------|----------|----------|
| TushareFetcher | -1 (有Token时) | Tushare Pro (sxsc 代理) | A股+港股 | TUSHARE_TOKEN |
| EfinanceFetcher | 0 | 东方财富 | A股 | 否 |
| AkshareFetcher | 1 | 东方财富/新浪/腾讯 | A股+港股 | 否 |
| PytdxFetcher | 2 | 通达信行情服务器 | A股 | 否 |
| BaostockFetcher | 3 | 证券宝 (学术级) | A股 | 否 |
| YfinanceFetcher | 4 | 雅虎财经 + Stooq 兜底 | 全球 | 否 |
| FinnhubFetcher | - | Finnhub API | 美股 | FINNHUB_API_KEY |
| AlphaVantageFetcher | - | AlphaVantage | 美股 | ALPHAVANTAGE_API_KEY |
| LongbridgeFetcher | 5 | 长桥 OpenAPI | 港美股 | 长桥凭证 |
| TickFlowFetcher | 99 | TickFlow SDK | A股指数 | TICKFLOW_API_KEY |

### 实时行情（独立优先级链）

默认: `腾讯财经 → 新浪财经 → 东方财富(efinance) → 东方财富(akshare_em) → Tushare`

可通过 `REALTIME_SOURCE_PRIORITY` 环境变量自定义顺序。

### 新闻搜索（7 引擎，按优先级自动切换）

| 引擎 | 说明 | 需要 Key |
|------|------|----------|
| Tavily | AI 搜索，英语新闻为主 | TAVILY_API_KEY |
| Bocha (博查) | 中文新闻最优 | BOCHA_API_KEY |
| Brave Search | 隐私优先搜索 | BRAVE_API_KEY |
| SerpAPI | Google 搜索 | SERPAPI_API_KEY |
| Anspire | AI 聚合搜索 | ANSPIRE_API_KEYS |
| MiniMax | 中文搜索 | MINIMAX_API_KEYS |
| SearXNG | 自建/公共元搜索 | 否 (可选自建实例) |
| 兜底 | 新浪财经 + 财联社/雪球/华尔街见闻 | 否 |

### 社交情感（可选）

| 平台 | 数据 | 需要 Key |
|------|------|----------|
| Reddit | 个股热度/情感/热门帖 | SOCIAL_SENTIMENT_API_KEY |
| X (Twitter) | 趋势股票热度 | 同上 |
| Polymarket | 预测市场趋势 | 同上 |

> 仅美股支持。API: api.adanos.org

## 界面展示

<p align="center">
  <img src="document/1.png" width="49%" alt="-5~+5 综合评分 + 技术指标卡片">
  <img src="document/2.png" width="49%" alt="基本面数据 + 评分逐因子明细">
</p>

<p align="center">
  <img src="document/3.png" width="49%" alt="重要新闻摘要 + 多 Agent 辩论">
  <img src="document/4.png" width="49%" alt="多周期预测 + 操作建议 + 风险提示">
</p>

## 评分算法

三层加权评分体系，总分 **-5 ~ +5**（负数为偏负面，正数为偏正面）。

```
最终分 = 技术面权重 × 技术分 + 基本面权重 × 基本面分 + 舆情面权重 × 舆情分
```

### 自适应权重

| 市场状态 | 判断依据 | 技术面 | 基本面 | 舆情面 |
|----------|----------|--------|--------|--------|
| 趋势上涨 | MA5 > MA20 > MA60 且发散 | 55% ↑ | 25% ↓ | 20% |
| 趋势下跌 | MA5 < MA20 < MA60 且发散 | 55% ↑ | 25% ↓ | 20% |
| 震荡盘整 | 均线交叉缠绕 | 45% ↓ | 35% ↑ | 20% |

> 趋势市中趋势指标（均线排列、动量）权重放大；震荡市中均值回归指标（RSI、布林带）权重放大。
> 数据缺失时自动重新分配权重。

### 一、技术面（6 因子）

| 因子 | 贡献范围 | 算法 |
|------|----------|------|
| **RSI(14)** | [-2, +2] | 非线性加速映射：35~45 中性平缓，<25 极度超卖加速到 +2，>75 极度超买加速到 -2 |
| **MACD** | [-1.5, +1.5] | `MACD_hist ÷ ATR(14) × 0.3`，波动率归一化消除高价股/低价股差异 |
| **均线排列** | [-1.5, +1.5] | 严格多头排列(price>MA5>MA10>MA20): +1.5；严格空头排列: -1.5；部分对齐按比例给分 |
| **布林带 %B** | [-1, +1] | `%B = (price−lower)/(upper−lower)`，%B<0.1 触下轨反弹 +1，%B>0.9 触上轨回调 -1 |
| **量价关系** | [-1, +1] | 低位放量大涨: 突破 +1；高位放量大跌: 出逃 -1；缩量: -0.3 |
| **20日动量** | [-1.5, +1.5] | `(price−MA20) ÷ MA20 × 100%`，涨幅超 10% 封顶 +1.5 |

### 二、基本面（4 因子）

| 因子 | 贡献范围 | 算法 |
|------|----------|------|
| **PE 估值分位** | [-3, +3] | PE <5%分位: +3；5~15%: +2；15~30%: +1；30~70%: 0；70~85%: -1；85~95%: -2；>95%: -3。**盈利趋势修正**：利润正增长加 0.2~0.5，负增长减 0.5，防止价值陷阱 |
| **ROE** | [-1.5, +1.5] | ROE>25%: +1.5；15~25%: +1.0；8~15%: +0.5；<0: -1.5。**负债扣分**：资产负债率 > 70% 扣 0.3 |
| **利润增长** | [-0.5, +1] | 净利润同比增速 × 0.025 + 营收增速 × 0.01，利润增速 > 30% 封顶 +1 |
| **分红潜力** | [-0.3, +0.5] | 高 ROE + 低负债 → 推断高分红意愿，A 股红利策略加分 |

### 三、舆情面

| 因子 | 贡献范围 | 算法 |
|------|----------|------|
| **新闻情感** | [-2.5, +2.5] | HuggingFace 多语言情感模型 → 5 级分类（非常负面/负面/中性/正面/非常正面），`avg_score × 2.5/0.9` 映射。模型不可达时降级为关键词规则引擎 |
| **股吧情感** | [-2.5, +2.5] | 同上，覆盖散户情绪维度 |
| **一致性调整** | ±0.5 | 新闻与股吧同向且强信号: +0.5（共振强化）；反向: -0.5（分歧削弱） |

### 评分解读

| 分值 | 标签 | 含义 |
|------|------|------|
| +4.0 ~ +5.0 | 强烈看多 | 技术+基本面+情绪共振向上 |
| +2.0 ~ +3.9 | 看多 | 多数指标积极 |
| +0.5 ~ +1.9 | 偏多 | 略积极，可关注 |
| -0.4 ~ +0.4 | 中性 | 多空平衡，观望 |
| -1.9 ~ -0.5 | 偏空 | 略消极，谨慎 |
| -3.9 ~ -2.0 | 看空 | 多数指标消极 |
| -5.0 ~ -4.0 | 强烈看空 | 技术+基本面+情绪共振向下 |

> **置信度判断**：|总分| > 3.5 且技术面和基本面同向 → 高置信度；|总分| > 1.5 → 中置信度；否则低置信度。

配色采用 A 股传统：**红涨绿跌**。

## 目录结构

```
StockFish/
├── app.py                     # Flask 主入口 / API 路由 / SSE
├── config.py                  # pydantic-settings 全局配置
├── run.sh                     # 一键部署 (Docker/本地)
├── docker-compose.yml         # StockFish + MiroFish 编排
├── requirements.txt
├── static/index.html          # 单页前端
│
├── market_data/               # 数据层
│   ├── a_stock_provider.py    # 主入口 + Mock/AkShare/BaoStock/Tushare/Advanced 后端
│   ├── tushare_provider.py    # Tushare Pro 后端 (sxsc_tushare SDK)
│   ├── provider_adapter.py    # AdvancedBackend: DataFetcherManager → BaseStockBackend
│   ├── compat.py              # 配置兼容适配器
│   ├── news_sources.py        # 传统新闻源 (新浪/财联社/雪球/东方财富股吧)
│   ├── sentiment_collector.py # 情感分析器 (HuggingFace 模型 + 规则降级)
│   │
│   ├── data_fetchers/         # 多源数据 Fetcher (from daily_stock_analysis)
│   │   ├── base.py            # BaseFetcher + DataFetcherManager (策略模式)
│   │   ├── realtime_types.py  # UnifiedRealtimeQuote + ChipDistribution + CircuitBreaker
│   │   ├── us_index_mapping.py
│   │   ├── efinance_fetcher.py
│   │   ├── akshare_fetcher.py
│   │   ├── tushare_fetcher.py
│   │   ├── pytdx_fetcher.py
│   │   ├── baostock_fetcher.py
│   │   ├── yfinance_fetcher.py
│   │   ├── finnhub_fetcher.py
│   │   ├── alphavantage_fetcher.py
│   │   ├── longbridge_fetcher.py
│   │   ├── fundamental_adapter.py
│   │   └── yfinance_fundamental_adapter.py
│   │
│   ├── stock_index/           # 股票名称索引
│   │   ├── stock_mapping.py
│   │   ├── stock_index_loader.py
│   │   └── stock_index_remote_service.py
│   │
│   ├── search/                # 7引擎搜索服务
│   │   └── search_service.py
│   │
│   ├── social_sentiment/      # 社交情感 (Reddit/X/Polymarket)
│   │   └── social_sentiment_service.py
│   │
│   └── patches/               # 反爬补丁
│       └── eastmoney_patch.py
│
├── analysis/                  # 分析引擎
│   ├── agent.py               # 5步管线主控
│   ├── state/state.py         # 分析状态定义
│   └── nodes/prediction_node.py # LLM/规则预测节点 (3 Agent 并行 + Moderatord)
│
├── simulation_bridge/         # MiroFish 桥接层
│   ├── orchestrator.py        # 模拟编排器
│   └── seed_builder.py        # 种子文档构建
│
├── prediction_report/         # 报告生成
│   └── report_generator.py    # HTML/JSON 报告
│
├── sxsc_tushare/              # 山西证券 Tushare 封装
└── reports/                   # 输出: 预测报告
```

## 引用

- [Tushare Pro](https://tushare.pro/) — A 股数据接口 (sxsc_tushare 代理)
- [DeepSeek](https://api.deepseek.com) — LLM 推理
- [Tavily](https://tavily.com) — AI 搜索 API
- [BettaFish](https://github.com/freenowill/BettaFish) — 多智能体舆情分析
- [MiroFish](https://github.com/freenowill/MiroFish) — OASIS 群体智能模拟引擎
- [AkShare](https://github.com/akfamily/akshare) — 金融数据接口
- [qlib-zh](https://github.com/freenowill/qlib-zh) — A 股因子选股模型
