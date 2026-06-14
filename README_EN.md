<p align="center">
  <img src="document/label.png" alt="StockFish Logo" width="320">
</p>

# StockFish — A-Share Intelligent Analysis & Price Simulation System

### Full Decision Pipeline

<p align="center">
  <img src="document/pipeline_flow.png" alt="Full Decision Pipeline" width="800">
</p>

### StockFish Multi-Factor Analysis Engine

<p align="center">
  <img src="document/stockfish_flow.png" alt="StockFish Multi-Factor Analysis Engine" width="960">
</p>

## 📋 Todo List

**🚀 Core Architecture** (done)
- [x] Multi-source data providers
- [x] LLM-powered analysis on sentiment, fundamentals, financials, and stock scoring
- [x] Multi-agent debate system with analysis output
- [x] MiroFish intelligent simulation integration

**🧠 Master Decision Architecture** (completed)
7 investment masters to choose from, 8 employees analyze → CIO final verdict
- [x] 8 Agent + CIO decision framework
- [x] Front-end master selector + multi-master comparison
- [x] Macro/industry analyst integration

**🤖 Qlib Inference** (in progress)
- [x] One-click data update
- [ ] One-click training & fine-tuning (in progress)
- [x] Inference-only mode

**📊 Batch Stock Analysis** (completed)
- [x] `POST /api/batch/analyze` sequential multi-symbol analysis
- [x] Real-time SSE per-stock result + sequential front-end rendering
- [x] Batch summary (ranking/common themes/divergences) + quality pick
- [x] Per-stock result cached in batch_results/

**⏪ Backtesting System** (scoring-engine based, no LLM)
- [ ] Historical window traversal + look-ahead bias free
- [ ] IC / IR / Win Rate / Cumulative Return
- [ ] Adaptive weight module standalone evaluation

**💬 Feishu Integration**
- [ ] `/analyze` `/batch` `/backtest` `/watch` commands
- [ ] Message cards (normal / master / batch / backtest)
- [ ] Watchlist + market open/close brief push

---

## Master Decision Mode

### Value Investing Style
- **Benjamin Graham**: Known as the "Father of Value Investing" and "Dean of Wall Street"
- **Warren Buffett**: Graham's most distinguished student, who elevated value investing to global prominence and is recognized worldwide as the preeminent investment master

### Growth Investing Style
- **Philip Fisher**: Pioneered growth stock investing, hailed as the "Pioneer of Growth Stock Investing"
- **Peter Lynch**: Legendary fund manager who mastered the growth investing strategy

### Contrarian Investing Style
- **John Templeton**: Renowned for contrarian investing, his creed: "Buy when others are selling in despair, sell when others are buying in enthusiasm"

### Macro Investing Style
- **George Soros**: Leading figure in global macro strategy, famous for shorting the British pound and attacking the Thai baht. His philosophy, deeply influenced by his teacher Karl Popper, emphasizes cognitive biases of market participants and the reflexivity effect on markets
- **Ray Dalio**: Founder of Bridgewater Associates. Unlike Soros's view of the economy as a machine, Dalio's investment system emphasizes historical cycles and logical analysis to predict asset performance under different economic environments

---

## 🎬 Demo

<table>
<tr>
<td align="center" width="50%">
  <a href="https://b23.tv/aL2XhZG">
    <img src="document/demo_local_thumb.png" alt="🧠 Master Decision Mode" width="100%">
  </a>
  <br><b>🧠 Master Decision Mode</b>
  <br>StockFish multi-factor analysis + Master CIO decision (8 agents + CIO verdict)
</td>
<td align="center" width="50%">
  <a href="https://b23.tv/AJChmUb">
    <img src="document/demo_mirofish_thumb.png" alt="🧠 Smart Simulation Mode" width="100%">
  </a>
  <br><b>🧠 Smart Simulation Mode</b>
  <br>StockFish + MiroFish OASIS swarm intelligence simulation
</td>
</tr>
</table>

### 🧠 Simulation Knowledge Graph

<p align="center">
  <img src="document/knowledge_graph.png" alt="MiroFish OASIS Swarm Intelligence Knowledge Graph" width="640">
</p>

> Knowledge graph built by the MiroFish OASIS engine: 7 agent roles (Buffett/Munger/Valuation/Sentiment/Fundamental/Technical/Risk Manager) interact in a simulated social network, with Zep GraphRAG extracting entity relationships to form multi-dimensional price prediction reasoning networks.

---

## 📦 Docker Deployment (Recommended)

Uses pre-built Docker images — no manual dependency installation required.

### Prerequisites

- Docker & Docker Compose
- `.env` config file (see below)

### One-Click Start

```bash
bash run.sh
```

The script automatically:
1. Pulls `zhuhai123/stockfish-stockfish:latest` + `zhuhai123/stockfish-mirofish:latest`
2. Mounts project code into containers, starts both services
3. Waits for services to be ready

> **Images only provide the runtime environment** (Python packages, Node.js, camel-ai, etc.). Application code comes from the project file mount.
> After modifying code, simply run `docker compose up -d` to rebuild containers — no need to rebuild the image.

### Other Modes

```bash
bash run.sh --local          # Run directly with local Python (no Docker)
bash run.sh --no-mirofish    # Docker deploy, StockFish only
```

### Access Services

| Service | URL |
|---------|-----|
| StockFish Frontend | http://localhost:8000 |
| MiroFish Frontend | http://localhost:3000 |
| MiroFish API | http://localhost:5001 |

### Management Commands

```bash
docker compose logs -f stockfish    # View logs
docker compose logs -f mirofish     # View MiroFish logs
docker compose down                 # Stop services
docker compose pull                 # Update to latest images
```

---

## 📦 Qlib Pre-trained Model Download

### CSI300 Alpha158

Pre-trained LightGBM model, walk-forward trained (8 folds, covering 2016~2023), designed for `--predict-only` incremental inference.

**Download:**
```bash
wget https://github.com/freenowill/stock-fish/releases/latest/download/csi300-alpha158.tar.gz
tar -xzf csi300-alpha158.tar.gz -C qlib-zh/models/
```

**Release page:** [github.com/freenowill/stock-fish/releases](https://github.com/freenowill/stock-fish/releases)

Directory structure after extraction:
```
qlib-zh/models/2026-05-27-csi300-alpha158/model_predict/walk_forward/
├── 2016-05-26/          # fold 1 checkpoint
├── ...
├── 2023-05-26/          # fold 8 checkpoint
└── segments/            # fold index (used by inference to locate the latest fold)
```

---

## 🚀 Quick Start

### 1. Configure `.env`

```bash
cp .env.example .env
# Edit .env, fill in API keys
```

**Required variables:**

| Variable | Description |
|----------|-------------|
| `LLM_API_KEY` | DeepSeek / OpenAI compatible API Key |
| `LLM_BASE_URL` | API endpoint (e.g. `https://api.deepseek.com`) |
| `LLM_MODEL_NAME` | Model name (e.g. `deepseek-v4-flash`) |
| `STOCK_BACKEND` | Data backend: `advanced` / `tushare` / `akshare` / `mock` |

**Data sources (auto-enabled when `STOCK_BACKEND=advanced`, one or more search engines supported, configure any):**

| Variable | Description |
|----------|-------------|
| `TUSHARE_TOKEN` | Tushare Pro data token (sxsc_tushare proxy) |
| `TAVILY_API_KEY` | Tavily search engine Key (news search) |
| `BOCHA_API_KEY` | Bocha search Key (Chinese news, recommended) |
| `BRAVE_API_KEY` | Brave Search Key |
| `SERPAPI_API_KEY` | SerpAPI (Google) Key |
| `FINNHUB_API_KEY` | Finnhub US stock data |
| `LONGBRIDGE_APP_KEY` etc. | Longbridge OpenAPI HK/US stocks |
| `SOCIAL_SENTIMENT_API_KEY` | US stock social sentiment (Reddit/X/Polymarket) |

**Optional:**

| Variable | Description |
|----------|-------------|
| `ZEP_API_KEY` | MiroFish graph memory (Zep Cloud) |
| `MIROFISH_HOST` | MiroFish service address (default `mirofish` in Docker) |
| `EFINANCE_CALL_TIMEOUT` | Eastmoney timeout in seconds (default 30, recommend 5 outside China) |
| `SEARXNG_PUBLIC_INSTANCES_ENABLED` | SearXNG public instances (recommend `false` outside China) |
| `YFINANCE_PRIORITY` | yfinance priority (recommend `0` outside China, fastest) |

### 2. Install Dependencies (Local Mode)

```bash
pip install -r requirements.txt
```

### 3. Start

```bash
# Docker mode (default)
bash run.sh

# Local mode
bash run.sh --local

# Docker (skip MiroFish)
bash run.sh --no-mirofish
```

### 4. API Examples

```bash
# Multi-factor analysis
curl -X POST http://localhost:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"600519"}'

# Price simulation (async, ~15 min)
curl -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"600519","scenario":"base"}'
```

---

## 🏗 Architecture

```
POST /api/analyze
  │
  ├─ Step 1: Data Collection (multi-source strategy, 6 fetchers auto-failover + circuit breaker)
  │   ├─ Tushare Pro (sxsc)   → Quote / Historical K-line / Historical PE / Fundamentals
  │   ├─ Eastmoney (efinance)  → Real-time quote / Sector rankings
  │   ├─ Tencent/Sina/Eastmoney → Multi-source real-time quote priority (akshare)
  │   ├─ TDX (pytdx)           → Free quote fallback
  │   ├─ Baostock              → Academic-grade free data
  │   ├─ Yahoo Finance (yfinance) → Global quotes + A-shares/HK/US stocks
  │   └─ Longbridge            → HK/US stocks OpenAPI
  │
  ├─ Step 1.5: News Search (7 search engine APIs)
  │   ├─ Tavily / Bocha / Brave / SerpAPI / MiniMax / SearXNG
  │   └─ Fallback: NewsNow aggregator (cls/xueqiu/wallstreetcn)
  │
  ├─ Step 2: Sentiment Analysis
  │   └─ HuggingFace multilingual sentiment model → 5-level classification + keyword rule fallback
  │
  ├─ Step 3: Valuation + Fundamentals + Signal Generation
  │   ├─ 3-year PE historical percentile → very low / low / normal / high / very high
  │   ├─ Fundamentals aggregate: growth rate / earnings / institutional holdings / capital flow / dragon-tiger
  │   ├─ Suggested buy price ← PE mean reversion + Bollinger lower band support
  │   └─ Weighted score ← RSI/MACD/KDJ/MA/Bollinger/Valuation/Sentiment
  │
  ├─ Step 4: LLM Prediction
  │   └─ 3 parallel Agent debate + Moderator synthesis → JSON prediction
  │
  └─ Step 5: Frontend Rendering
      └─ 6 cards + technical/fundamental + news/guba summary (A-stock: red=up, green=down)
```

### Prediction Simulation Pipeline

```
/api/predict (async)
  │
  ├─ Analysis (same as above) → Seed document generation
  ├─ MiroFish OASIS simulation (HTTP bridge)
  │   ├─ Zep GraphRAG knowledge graph
  │   ├─ Agent profile generation (investor/analyst/media/company/regulator)
  │   └─ Multi-round social simulation (5 rounds, OASIS engine)
  ├─ Report Agent generates simulation report
  └─ HTML prediction report output → reports/
```

---

## 📊 Data Sources

### Market Data

| Fetcher | Priority | Source | Coverage |
|---------|----------|--------|----------|
| YfinanceFetcher | 0 | Yahoo Finance + Stooq fallback | Global |
| TushareFetcher | -1 (with Token) | Tushare Pro (sxsc proxy) | A-shares + HK |
| PytdxFetcher | 2 | TDX quote servers | A-shares |
| BaostockFetcher | 3 | Baostock (academic-grade) | A-shares |
| AkshareFetcher | 4 | Eastmoney/Sina/Tencent | A-shares + HK |
| EfinanceFetcher | 5 | Eastmoney | A-shares |
| FinnhubFetcher | - | Finnhub API | US stocks |
| LongbridgeFetcher | - | Longbridge OpenAPI | HK/US stocks |

### Real-Time Quotes (Independent Priority Chain)

Default: `Tencent → Tushare → Eastmoney(efinance) → Eastmoney(akshare_em)`

Customizable via `REALTIME_SOURCE_PRIORITY` environment variable.

### News Search

| Engine | Description | Requires Key |
|--------|-------------|--------------|
| Tavily | AI search, English-focused | `TAVILY_API_KEY` |
| Bocha | Best for Chinese news | `BOCHA_API_KEY` |
| Brave Search | Privacy-first search | `BRAVE_API_KEY` |
| SerpAPI | Google Search | `SERPAPI_API_KEY` |
| MiniMax | Chinese search | `MINIMAX_API_KEYS` |
| SearXNG | Self-hosted/public metasearch | Optional |
| NewsNow | cls + xueqiu + wallstreetcn aggregate | No |

### Social Sentiment (Optional, US Stocks Only)

| Platform | Data | API |
|----------|------|-----|
| Reddit | Stock mentions, sentiment, trending posts | api.adanos.org |
| X (Twitter) | Trending stock mentions | api.adanos.org |
| Polymarket | Prediction market trends | api.adanos.org |

---

## 📈 Scoring Algorithm

Three-layer weighted scoring system, total score **-5 ~ +5**.

```
Final = Technical Weight × Technical Score + Fundamental Weight × Fundamental Score + Sentiment Weight × Sentiment Score
```

### Adaptive Weights

| Market Regime | Criteria | Technical | Fundamental | Sentiment |
|---------------|----------|-----------|-------------|-----------|
| Trending Up | MA5 > MA20 > MA60, diverging | 55% ↑ | 25% ↓ | 20% |
| Trending Down | MA5 < MA20 < MA60, diverging | 55% ↑ | 25% ↓ | 20% |
| Ranging | Moving averages intertwined | 45% ↓ | 35% ↑ | 20% |

### Technical (6 Factors)

| Factor | Range | Algorithm |
|--------|-------|-----------|
| **RSI(14)** | [-2, +2] | Non-linear acceleration mapping, <25 oversold→+2, >75 overbought→-2 |
| **MACD** | [-1.5, +1.5] | `MACD_hist ÷ ATR(14) × 0.3`, volatility normalized |
| **MA Alignment** | [-1.5, +1.5] | Strict bullish alignment→+1.5, strict bearish→-1.5 |
| **Bollinger %B** | [-1, +1] | %B<0.1 lower band bounce→+1, %B>0.9 upper band pullback→-1 |
| **Volume-Price** | [-1, +1] | Low-volume surge→+1, high-volume dump→-1 |
| **20-Day Momentum** | [-1.5, +1.5] | Capped at +1.5 for gains >10% |

### Fundamentals (4 Factors)

| Factor | Range | Algorithm |
|--------|-------|-----------|
| **PE Valuation Percentile** | [-3, +3] | PE <5%→+3, 5~15%→+2...>95%→-3. **Earnings trend adjusted** |
| **ROE** | [-1.5, +1.5] | >25%→+1.5, <0→-1.5. **Debt penalty**: >70% deduct 0.3 |
| **Profit Growth** | [-0.5, +1] | Net profit YoY×0.025 + revenue×0.01, capped at +1 for >30% |
| **Dividend Potential** | [-0.3, +0.5] | High ROE + low debt → high dividend willingness |

### Sentiment

| Factor | Range | Algorithm |
|--------|-------|-----------|
| **News Sentiment** | [-2.5, +2.5] | 5-level classification mapping, rule fallback |
| **Guba Sentiment** | [-2.5, +2.5] | Same as above, covers retail investor sentiment |
| **Consistency Adjustment** | ±0.5 | News & guba aligned with strong signal→+0.5, opposite→-0.5 |

### Score Interpretation

| Score | Label | Meaning |
|-------|-------|---------|
| +4.0 ~ +5.0 | Strongly Bullish | Technical + fundamental + sentiment resonance upward |
| +2.0 ~ +3.9 | Bullish | Majority of indicators positive |
| +0.5 ~ +1.9 | Slightly Bullish | Mildly positive, worth watching |
| -0.4 ~ +0.4 | Neutral | Balanced, wait and see |
| -1.9 ~ -0.5 | Slightly Bearish | Mildly negative, exercise caution |
| -3.9 ~ -2.0 | Bearish | Majority of indicators negative |
| -5.0 ~ -4.0 | Strongly Bearish | Technical + fundamental + sentiment resonance downward |

> **Confidence**: \|Total\| > 3.5 and technical + fundamental aligned → High; \|Total\| > 1.5 → Medium; otherwise Low.

Color scheme follows A-share convention: **Red = Up, Green = Down**.

---

## 📁 Directory Structure

```
StockFish/
├── app.py                     # Flask entry / API routes / SSE
├── config.py                  # pydantic-settings global config
├── run.sh                     # One-click deploy (Docker/local)
├── docker-compose.yml         # StockFish + MiroFish orchestration
├── requirements.txt
├── static/index.html          # Single-page frontend
│
├── market_data/               # Data layer
│   ├── a_stock_provider.py    # Main entry, multi-backend auto-switch
│   ├── tushare_provider.py    # Tushare Pro backend
│   ├── provider_adapter.py    # AdvancedBackend adapter
│   ├── news_sources.py        # Plugin-based news sources
│   ├── sentiment_collector.py # Sentiment analyzer
│   ├── data_fetchers/         # 11 multi-source data fetchers
│   ├── search/                # 7 search engine services
│   ├── social_sentiment/      # Social sentiment (Reddit/X/Polymarket)
│   └── stock_index/           # Stock name index
│
├── analysis/                  # Analysis engine
│   ├── agent.py               # 4-step pipeline controller
│   ├── scoring.py             # -5~+5 scoring engine
│   ├── state/state.py         # Analysis state definition
│   └── nodes/prediction_node.py # LLM multi-agent debate prediction
│
├── simulation_bridge/         # MiroFish bridge layer
│   ├── orchestrator.py        # Simulation orchestrator
│   └── seed_builder.py        # Seed document builder
│
├── prediction_report/         # Report generation
│   └── report_generator.py    # HTML/JSON report
│
├── reports/                   # Prediction report output
└── sxsc_tushare/              # Shanxi Securities Tushare wrapper
```

---

## 🔗 References

- [DeepSeek](https://platform.deepseek.com/usage) — LLM inference
- [Tushare Pro](https://tushare.pro/) — A-share data API
- [Tavily](https://tavily.com) — AI search API
- [AkShare](https://github.com/akfamily/akshare) — Financial data API
- [BettaFish](https://github.com/666ghj/BettaFish) — Multi-agent sentiment analysis
- [MiroFish](https://github.com/666ghj/MiroFish) — OASIS swarm intelligence simulation engine
- [qlib-zh](https://github.com/freenowill/qlib-zh) — A-share factor stock selection model
