# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start server (port 8000)
python app.py

# One-click deploy (Docker or local)
bash run.sh              # Docker: pulls zhuhai123/stockfish-* images, starts StockFish + MiroFish
bash run.sh --local      # Local Python (no Docker, port 8000)
bash run.sh --no-mirofish # Docker: StockFish only, skip MiroFish

# Install dependencies
pip install -r requirements.txt

# API: multi-factor analysis (~2 min)
curl -X POST http://localhost:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"600519","cost_price":150}'

# API: analysis with master decision mode (add "master" param)
curl -X POST http://localhost:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"600519","cost_price":150,"master":"buffett"}'

# API: prediction pipeline (async, ~15 min, SSE progress stream)
curl -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"600519","scenario":"base"}'

# API: prediction with master
curl -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"600519","scenario":"base","master":"graham"}'

# API: batch analysis (async, SSE)
curl -X POST http://localhost:8000/api/batch/analyze \
  -H 'Content-Type: application/json' \
  -d '{"symbols":["600519","000858","300750"],"cost_prices":{"600519":150}}'

# API: Qlib model inference
curl -X POST http://localhost:8000/api/qlib/infer \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"600519","model":"alpha360"}'

# SSE progress stream
curl -N http://localhost:8000/api/predict/<task_id>/stream

# Check prediction status
curl http://localhost:8000/api/predict/<task_id>

# Download prediction report (HTML)
curl http://localhost:8000/api/predict/<task_id>/report

# List available masters
curl http://localhost:8000/api/masters

# List all predictions
curl http://localhost:8000/api/predictions

# List Qlib models
curl http://localhost:8000/api/qlib/models

# System config
curl http://localhost:8000/api/config
```

No test framework exists (`tests/__init__.py` is empty). No linting/formatting config.

## Architecture

**5-step analysis pipeline** (`POST /api/analyze`):

1. **Data Collection** (`analysis/agent.py:StockAnalysisAgent.analyze`) — fetches quote, technical indicators, financials, news, guba posts via `AStockProvider` (auto-selects backend). Supports A-shares, BSE stocks (`.BJ` suffix auto-converted for Tushare), HK/US stocks.
2. **Sentiment** (`market_data/sentiment_collector.py`) — HuggingFace multilingual model → 5-class sentiment, with keyword-rule fallback.
3. **Valuation + Signal Generation** (`analysis/scoring.py:ScoringEngine`) — PE percentile → valuation level (很低~很高), suggested buy price (PE mean-reversion + Bollinger lower support). -5~+5 composite score: technical (RSI/MACD/MA/Bollinger/volume/momentum) + fundamental (PE/ROE/growth/dividend) + sentiment (news/guba). Adaptive weights based on market regime.
4. **LLM Prediction** (`analysis/nodes/prediction_node.py`) — Three modes:
   - **Multi-agent (master) mode** (when `master` param set): 8 employee agents → CIO master decision (see Multi-Agent System below)
   - **Legacy multi-agent mode** (no master, `LLM_API_KEY` set): 3 parallel agents (tech/fundamental/sentiment) → Moderator synthesis
   - **Rule mode** (no `LLM_API_KEY`): threshold-based scoring with price range
5. **Frontend rendering** — `static/index.html`: dark-theme single-page SPA (~1665 lines, no build step).

### Multi-Agent Investment Decision System (`analysis/agents/`)

When a `master` parameter is passed to `POST /api/analyze`, the system activates the full investment committee pipeline:

**8 Employees (5 Departments)** — all extend `BaseAgent`, output `EmployeeReport`:

| Employee | ID | Department | Role |
|---|---|---|---|
| MacroAgent | `macro` | 宏观部 | 宏观分析师 — SHIBOR/PMI/CPI/北向资金/汇率/政策倾向 |
| PolicyAgent | `policy` | 宏观部 | 行业政策分析师 — 行业周期/政策影响/景气阶段 |
| ValuationAgent | `valuation` | 研究部 | 估值分析师 — PE分位/DCF/清算价值/安全边际 |
| FundamentalAgent | `fundamental` | 研究部 | 基本面分析师 — ROE/EPS/护城河/管理层/成长性 |
| TechnicalAgent | `technical` | 研究部 | 技术分析师 — RSI/MACD/MA/布林带/量价/形态 |
| SentimentAgent | `sentiment` | 交易部 | 舆情分析师 — 新闻/股吧情感/极端情绪信号 |
| RiskManager | `risk` | 风控部 | 风险经理 — VaR/最大回撤/下行风险 (≥7分行使软否决权) |
| Overseer | `overseer` | 监察部 | 独立监察员 — 读取所有其他报告后挑战假设、找盲点 |

**Execution flow**: 7 employees run in parallel (`ThreadPoolExecutor(max_workers=7)`), then Overseer reads all reports and runs analysis, then CIO makes final decision.

**7 Master Investors (CIO)** — selectable via `master` param, each with distinct investment philosophy:

| Key | Name | Style | Era |
|---|---|---|---|
| `graham` | 本杰明·格雷厄姆 | 深度价值投资 | 1930s-1970s |
| `buffett` | 沃伦·巴菲特 | 价值+质量投资 | 1960s-至今 |
| `fisher` | 菲利普·费雪 | 成长投资 | 1950s-1990s |
| `lynch` | 彼得·林奇 | GARP (合理价格增长) | 1977-1990 |
| `templeton` | 约翰·邓普顿 | 逆向全球投资 | 1940s-2000s |
| `soros` | 乔治·索罗斯 | 反身性宏观交易 | 1970s-2010s |
| `dalio` | 瑞·达利欧 | 风险平价/全天候策略 | 1975-至今 |

Each master has a distinct `system_prompt` encoding 6 investment principles + per-employee weight guidance. All use the same `CIO_OUTPUT_SCHEMA` (JSON with multi-scenario analysis, multi-cycle prediction, order instructions, risk monitoring, decision quality self-assessment). Defined in `analysis/agents/cio_prompts.py`.

**CIO decision output** (`CIODecision` dataclass in `analysis/agents/base.py`): decision_summary, evidence_chain (per-employee), 3-scenario analysis (base/bull/bear), order (action/position_size/stop_loss/take_profit), multi-cycle predictions, risk_monitoring triggers, decision_quality self-assessment.

**API**: `GET /api/masters` returns available masters for the frontend dropdown.

### Batch Analysis (`POST /api/batch/analyze`)

Serial multi-stock analysis with SSE progress streaming. Accepts `symbols` list + optional `cost_prices` dict. Each stock runs through the full 5-step pipeline. Results cached in `batch_results/` directory. Frontend supports comma/newline-separated symbol input (max 20).

### Qlib Model Inference (`POST /api/qlib/infer`, `qlib-zh/`)

Microsoft Qlib integration for deep learning price prediction. Docker-based: `app.py` shells out to `docker run` with the Qlib image, mounting `qlib-zh/DATA/` for data and `qlib-zh/models/` for pre-trained models. `infer_runner.py` is loaded dynamically via importlib inside the container.

Key files:
- `qlib-zh/infer_runner.py` — Qlib inference script (Alpha360/LSTM/GRU/Transformer models)
- `qlib-zh/DATA/` — Qlib-format market data (gitignored)
- `qlib-zh/models/` — Pre-trained model checkpoints (gitignored)
- `GET /api/qlib/models` — Lists available pre-trained models
- `GET /api/qlib/index-stocks` — Lists index constituent stocks for batch Qlib inference

### Simulation Bridge (`POST /api/predict`, background thread)

- `analysis/agent.py` → `simulation_bridge/seed_builder.py` → `simulation_bridge/orchestrator.py` → MiroFish HTTP API
- Seed document embeds **7 agent roles** (Buffett/Munger/Valuation/Sentiment/Fundamental/Technical/RiskManager) + explicit entity-relationship statements for Zep GraphRAG extraction
- Pipeline: ontology generation → graph build → simulation create → agent profile prep → OASIS run → report generation
- Each stage has polling with configurable timeouts; falls back to standalone mode on any MiroFish failure
- SSE progress stream with real-time log messages (EventSource in frontend, polling fallback)

### Prediction Report (`prediction_report/report_generator.py`)

`PredictionReportGenerator` merges analysis results + simulation output into dark-theme HTML reports. Saved to `reports/` directory.

## Data Backends (`market_data/a_stock_provider.py`)

`AStockProvider` auto-selects (5 backends): `advanced` → `tushare` → `akshare` → `baostock` → `mock`. Config via `STOCK_BACKEND` env var (default: `mock`).

- **MockBackend** — Random data, zero network, for dev/demo
- **AkShareBackend** — EastMoney data via akshare (needs mainland China network)
- **BaoStockBackend** — Free, no token, academic-grade fallback
- **TushareBackend** (`tushare_provider.py`) — Tushare Pro via vendored `sxsc_tushare` SDK (山西证券 proxy). Supports `.BJ` suffix for BSE stocks.
- **AdvancedBackend** (`provider_adapter.py`) — DataFetcherManager wrapping 11 fetchers (efinance/akshare/tushare/pytdx/baostock/yfinance/finnhub/alphavantage/longbridge) with circuit-breaker failover, 7 search engines (Tavily/Bocha/Brave/SerpAPI/Anspire/MiniMax/SearXNG), social sentiment (Reddit/X/Polymarket, US stocks only), fundamental pipeline (growth/earnings/institutional/capital flow/dragon-tiger boards).

All backends implement `BaseStockBackend`: `get_quote`, `get_historical`, `get_financials`, `get_news`, `get_guba`, `get_historical_pe`.

Real-time quote priority chain (configurable via `REALTIME_SOURCE_PRIORITY`): Tencent → Sina → Eastmoney → Tushare.

## News Sources (`market_data/news_sources.py`)

Plugin architecture: each source extends `BaseNewsSource` or `BaseGubaSource`, registered in `NEWS_SOURCES`/`GUBA_SOURCES` lists. Current: SinaNews, NewsNow (cls+xueqiu+wallstreetcn aggregate), YahooFinance, XueqiuPopularity, CLSNews (disabled), EastMoneyGuba.

## Frontend (`static/index.html`)

Single-file dark-theme SPA (~1665 lines, no build step). Features:
- Stock symbol + cost price input, optional "智能推演" checkbox (toggles between `/api/analyze` and `/api/predict`)
- **Master selection dropdown** — choose from 7 investment masters (格雷厄姆/巴菲特/费雪/林奇/邓普顿/索罗斯/达利欧), populates via `GET /api/masters`. When selected, activates the multi-agent CIO pipeline; when unselected, falls back to legacy 3-agent debate
- **Batch mode toggle** — multi-symbol input (comma/newline-separated, max 20), per-stock progress bars
- **Qlib inference panel** — model selection dropdown, inference with SSE progress
- SSE real-time log streaming during simulation, polling fallback on SSE error
- Analysis view: signal header, 6-card grid (price/dividend/cost/buy price/valuation/sentiment), multi-cycle LLM predictions, suggested action with stop-loss/take-profit, technical + fundamental detail grids, score breakdown, news/guba summaries, CIO master decision card (avatar + decision_summary + 3-scenario analysis + order), multi-agent debate panel (hidden when master mode active)
- Report view: download HTML button, print/PDF export button
- A-stock color scheme: red = up, green = down

## Scoring Engine (`analysis/scoring.py`)

Three-layer weighted system (-5 to +5). See README.md for full factor tables.

Key dataclass: `ScoreResult` with `final`, `label`, `technical`, `fundamental`, `sentiment`, `regime`, `confidence`, `weights`, `breakdown: List[FactorDetail]`.

Adaptive weights: trending → technical +5%; ranging → fundamental +5%. Missing data redistributed proportionally.

## LLM Prediction (`analysis/nodes/prediction_node.py`)

Three modes controlled by `LLM_API_KEY` and `master` param:
- **Master mode** (`master` param set): 8 employee agents → Overseer review → CIO master decision with structured `CIODecision` output
- **Multi-agent mode** (`LLM_API_KEY` set, no master): 3 agents debate in parallel via `ThreadPoolExecutor(max_workers=3)`, then Moderator synthesizes → structured JSON with `short_term`/`mid_term`/`long_term` predictions (direction + change_pct + confidence + reason) and `suggested_action` (action + reason + stop_loss + take_profit). Uses OpenAI-compatible API (`response_format: json_object`). Moderator failure → majority-vote fallback.
- **Rule mode** (no key): score-threshold based, outputs signal label + price range ±5-10%.

Dataclasses: `AgentView` (per-agent), `PredictionResult` (final output, all fields).

## Seed Document 7 Agent Roles (`simulation_bridge/seed_builder.py`)

The seed builder embeds 7 distinct agent personas for MiroFish OASIS simulation:
1. **BuffettProxy** — value investing, economic moat assessment
2. **MungerProxy** — long-term competitive landscape, management quality
3. **ValuationAgent** — DCF, PE percentile, intrinsic value
4. **SentimentAgent** — NLP-driven market sentiment analysis
5. **FundamentalAnalyst** — financial health, growth sustainability
6. **TechnicalAnalyst** — chart patterns, momentum, support/resistance
7. **RiskManager** — VaR, max drawdown, position limits

Plus explicit `[Entity]` relationship statements at the end of the seed document for Zep GraphRAG extraction.

## MiroFish Backend (`MiroFish/backend/`)

Flask app (`:5001`) that StockFish's simulation bridge calls via HTTP. Key structure:

```
MiroFish/backend/
├── run.py                          # Entry point, port 5001
├── app/
│   ├── __init__.py                 # create_app() factory, registers blueprints
│   ├── config.py                   # Plain Config class (python-dotenv)
│   ├── api/                        # Blueprint endpoints
│   │   ├── graph.py               # Knowledge graph: project mgmt, ontology, graph building
│   │   ├── simulation.py          # OASIS simulation lifecycle management
│   │   └── report.py              # Report generation (async), status, download
│   ├── models/                     # TaskManager, Project (status lifecycle)
│   ├── services/                   # 14+ services: ontology, graph, simulation, report_agent, zep_*
│   └── utils/                      # LLM client, file parser, locale, retry, logger
└── scripts/                        # OASIS simulation scripts (run_twitter, run_reddit, run_parallel)
```

Uses `uv` package manager with `pyproject.toml`. Dependencies: flask, flask-cors, openai, zep-cloud, camel-oasis, camel-ai, PyMuPDF, pydantic.

## Configuration (`config.py`)

pydantic-settings `Settings` loaded from `.env`. Adds MiroFish to Python path for cross-project imports. Copy `.env.example` to `.env` to get started.

Key env vars:
- `LLM_API_KEY/BASE_URL/MODEL_NAME` — LLM config (OpenAI-compatible)
- `STOCK_BACKEND` — data source: `mock|akshare|tushare|advanced|auto`
- `TUSHARE_TOKEN` — Tushare Pro (via sxsc_tushare SDK)
- Search API keys: `TAVILY_API_KEY`, `BOCHA_API_KEY`, `BRAVE_API_KEY`, `SERPAPI_API_KEY`, `ANSPIRE_API_KEY`, `MINIMAX_API_KEYS`, `SEARXNG_BASE_URL`
- `SOCIAL_SENTIMENT_API_KEY` — Reddit/X/Polymarket (US stocks only)
- `MIROFISH_HOST/PORT` — MiroFish address (Docker: `mirofish:5001`, local: `localhost:5001`)
- `REALTIME_SOURCE_PRIORITY` — quote source order (default: `tencent,akshare_sina,efinance,akshare_em`)
- `EFINANCE_CALL_TIMEOUT` — Eastmoney timeout (default 30s, set 5s outside China)
- Feature flags: `ENABLE_REALTIME_QUOTE`, `ENABLE_FUNDAMENTAL_PIPELINE`, `ENABLE_CHIP_DISTRIBUTION`, `ENABLE_EASTMONEY_PATCH`, `ENABLE_LONGBRIDGE`, `ENABLE_MACRO`, `ENABLE_US_STOCKS`, `ENABLE_HK_STOCKS`
- Multi-source fetcher priorities: `EFINANCE_PRIORITY`, `YFINANCE_PRIORITY`, `PYTDX_PRIORITY`, `AKSHARE_PRIORITY`, `BAOSTOCK_PRIORITY`
- Circuit breaker: `CIRCUIT_BREAKER_COOLDOWN` (default 300s), rate limits for each fetcher
- `OASIS_DEFAULT_MAX_ROUNDS` (20), `OASIS_SIMULATION_AGENT_COUNT` (15)
- `ZEP_API_KEY` — Zep Cloud for MiroFish graph memory (optional)
- `QLIB_ENABLED` — Qlib model inference (default: false)
- `BATCH_MAX_CONCURRENT` — Batch analysis concurrency limit (default: 5)

## Key Conventions

- A-stock color scheme: **red = up (gain)**, green = down (loss) — opposite of Western markets
- `Quote`, `FinancialSummary`, `TechnicalIndicators`, `NewsItem`, `GubaPost` — dataclasses with `to_dict()`, used across all backends
- `AnalysisState` (`analysis/state/state.py`) — dataclass carrying full analysis state through all pipeline steps
- `ScoreResult` (`analysis/scoring.py`) — composite score with breakdown, label, weights, and confidence
- `PredictionResult`, `AgentView` (`analysis/nodes/prediction_node.py`) — LLM prediction output dataclasses
- `EmployeeReport`, `CIODecision` (`analysis/agents/base.py`) — Multi-agent decision system dataclasses
- `loguru` for logging consistently across all modules
- `config:settings` singleton — imported at module level in `app.py`, lazy-imported elsewhere
- `sxsc_tushare/` — vendored Tushare Pro SDK (山西证券 proxy), imported directly
- BSE stocks: use `.BJ` suffix (e.g., `830799.BJ`), Tushare backend auto-converts format
- `.env` must exist in project root (checked by `run.sh`); copy from `.env.example`
- `STOCK_BACKEND=mock` is default for zero-config dev; `advanced` for production
- All timeouts in the orchestrator are configurable (graph: 300s, simulation: 900s, report: 600s)
- Flask app global state in thread-safe dicts with locks: `predictions`, `batch_tasks`, `qlib_tasks`

## Directory Notes

- `analysis/agents/` — Multi-agent investment committee: `base.py` (BaseAgent/EmployeeReport/CIODecision), 8 employee agents, `cio.py` (CIOAgent), `cio_prompts.py` (7 master definitions + output schema), `overseer.py`
- `analysis/nodes/prediction_node.py` — LLM prediction: legacy 3-agent debate + master-mode CIO pipeline
- `analysis/tools/` — Agent utility tools
- `market_data/data_fetchers/` — 14 fetchers (efinance/akshare/tushare/pytdx/baostock/yfinance/finnhub/alphavantage/longbridge/fundamental/realtime)
- `market_data/search/search_service.py` — 7-engine search service (Tavily/Bocha/Brave/SerpAPI/Anspire/MiniMax/SearXNG)
- `market_data/social_sentiment/` — Reddit/X/Polymarket sentiment (US stocks only)
- `market_data/stock_index/` — Index loading, remote service, stock-to-index mapping
- `market_data/patches/` — Runtime monkey-patches (e.g., Eastmoney API fixes)
- `market_data/compat.py` — Backward-compatibility shims for data fetchers
- `simulation_bridge/` — MiroFish bridge: `orchestrator.py` (pipeline + HTTP client), `seed_builder.py` (7 agent roles seed doc)
- `prediction_report/report_generator.py` — Merges analysis + simulation into HTML reports
- `qlib-zh/` — Qlib inference: `infer_runner.py` (dynamic importlib load), `DATA/`, `models/`, `scripts/`
- `sxsc_tushare/` — Vendored Tushare Pro SDK (dataapi.py, upass.py)
- `MiroFish/backend/` — MiroFish Flask app (see MiroFish Backend section above)
- `static/index.html` — Single-file SPA frontend (~1665 lines)
- `backend/` — Empty reserved directory (no active code)
- `document/` — Documentation screenshots (1-4.png, label.png)
- `reports/` — Generated prediction report HTML output (gitignored)
- `simulation_output/` — Seed documents and scenario JSONs saved per analysis run (gitignored)
- `batch_results/` — Batch analysis result cache (gitignored)
- `TODO.md` — Full project roadmap with P0-P3 priorities, tech debt, and work estimates

## Slash Commands

### `/release-model`
Compress a Qlib model directory into `csi300-alpha158.tar.gz` and push it to `github.com/freenowill/stock-fish/releases`.

**Usage:** `/release-model <模型目录路径>`

**Example:**
```
/release-model qlib-zh/models/2026-06-07-csi300-alpha158
```

**Steps:**
1. Check existing release — if `csi300-alpha158.tar.gz` asset exists, skip entirely
2. Delete only releases **missing** the tar.gz asset (empty drafts)
3. `tar -czf /tmp/csi300-alpha158-model.tar.gz` the model directory
4. `gh release create csi300-alpha158` with the tar.gz to `freenowill/stock-fish`
5. Verify the uploaded asset

**Note:** Always uses `TMPDIR=/Users/apple/github/stock_predict/StockFish` for `gh` commands to avoid `/private/tmp` disk-full issues.
