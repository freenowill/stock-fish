# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start server (port 8000)
python app.py

# Start Feishu/Lark bot (WebSocket, no public IP needed)
python integration/lark_bot.py

# One-click deploy (Docker or local)
bash run.sh                       # Docker: pulls zhuhai123/stockfish-* images, starts StockFish + MiroFish
bash run.sh --local               # Local Python (no Docker, port 8000)
bash run.sh --no-mirofish         # Docker: StockFish only, skip MiroFish
bash run.sh --bot                 # Start both StockFish and Lark Bot (combine with --local or Docker)
bash run.sh --debug               # Debug mode: OASIS_DEBUG=true (2 agents, 2 rounds)

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

# API: Qlib training (walk-forward, Docker-in-Docker, ~2hr)
curl -X POST http://localhost:8000/api/qlib/train \
  -H 'Content-Type: application/json' \
  -d '{"target":"csi300-alpha158"}'

# API: Qlib finetuning (existing model + recent data)
curl -X POST http://localhost:8000/api/qlib/finetune \
  -H 'Content-Type: application/json' \
  -d '{"model":"2026-06-12-csi300-alpha158"}'

# API: Qlib data update (download qlib_bin.tar.gz)
curl -X POST http://localhost:8000/api/qlib/data/update

# API: List Qlib training targets
curl http://localhost:8000/api/qlib/train-targets

# API: List Qlib index stocks
curl "http://localhost:8000/api/qlib/index-stocks?index=csi300&exclude_star=true"

# API: Batch analysis status + stream
curl http://localhost:8000/api/batch/analyze/<task_id>
curl -N http://localhost:8000/api/batch/analyze/<task_id>/stream

# API: Download report HTML (POST with analysis results JSON)
curl -X POST http://localhost:8000/api/report/download \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"600519","results":{...}}'
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
5. **Frontend rendering** — `static/index.html`: dark-theme single-page SPA (~2100 lines, no build step).

### Multi-Agent Investment Decision System (`analysis/agents/`)

All agents extend `BaseAgent` (`analysis/agents/base.py`), which provides:
- OpenAI-compatible LLM client with `_call_llm(system_prompt, user_prompt, temperature, use_json_mode)` — retry (max 2), timeout (45s), exception-safe
- `_parse_json(raw)` — handles truncated JSON, markdown-wrapped blocks, trailing commas
- Static data extractors (`build_tech_context`, `build_fund_context`, `build_valuation_context`, `build_sent_context`, `build_risk_context`, `build_macro_context`, `build_industry_context`, `build_search_context`, `build_overseer_context`) — each pulls relevant fields from an AnalysisState dict and formats them as readable prompts
- `_safe_get()` and `_is_financial_industry()` utilities

**`EmployeeReport`** dataclass — standardized output for all 8 employees: `employee_id`, `role`, `department`, `outlook`, `confidence`, `score` (-10~+10), `key_points`, `risks`, `raw_output`, `error`.

**`CIODecision`** dataclass — structured CIO output: `master_name`, `master_key`, `decision_summary`, `rationale`, `evidence_chain`, 3-scenario analysis (base/bull/bear), `order` (action/position_size/stop_loss/take_profit), multi-cycle predictions, `risk_monitoring`, `decision_quality`, `veto_response`.

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

Each master has a distinct `system_prompt` encoding 6 investment principles + per-employee weight guidance. All use the same `CIO_OUTPUT_SCHEMA` (JSON with multi-scenario analysis, multi-cycle prediction, order instructions, risk monitoring, decision quality self-assessment). A separate `PORTFOLIO_OUTPUT_SCHEMA` is used for batch analysis cross-stock portfolio summaries. Defined in `analysis/agents/cio_prompts.py`.

**CIO decision output** (`CIODecision` dataclass in `analysis/agents/base.py`): decision_summary, evidence_chain (per-employee), 3-scenario analysis (base/bull/bear), order (action/position_size/stop_loss/take_profit), multi-cycle predictions, risk_monitoring triggers, decision_quality self-assessment.

**API**: `GET /api/masters` returns available masters for the frontend dropdown.

### Batch Analysis (`POST /api/batch/analyze`)

Serial multi-stock analysis with SSE progress streaming. Accepts `symbols` list + optional `cost_prices`, `shares`, `total_assets`, `available_cash` for portfolio context. Each stock runs through the full 5-step pipeline.

After all stocks complete, the system generates:
- **Portfolio Summary** — LLM-driven cross-stock analysis using `PORTFOLIO_OUTPUT_SCHEMA` (from `analysis/agents/cio_prompts.py`), ranked by investment value, with capital allocation suggestions and common themes
- **Quality Picks** — best buy recommendations guided by master investment philosophy, with rule-based fallback (composite score + valuation percentile)

Results cached in `batch_results/` directory. SSE stream (`GET /api/batch/analyze/<task_id>/stream`) pushes incremental `stock_result`, `progress`, and `batch_summary` events. Frontend supports comma/newline-separated symbol input (max 20).

### Qlib Model Training, Finetuning & Inference (`qlib-zh/`)

Docker-in-Docker Microsoft Qlib integration. StockFish containers shell out to `docker run` with the `zhuhai123/qlib-rdagent:v1` image, mounting `/var/run/docker.sock`, `~/.qlib` (Qlib data), and `~/github/qlib-zh/mlruns` (MLflow models). `QLIB_HOST_PROJECT_ROOT`, `QLIB_HOST_DATA_DIR`, `QLIB_HOST_MLRUNS_DIR` env vars control mount paths.

**Inference** (`POST /api/qlib/infer`):
- `qlib-zh/infer_runner.py` — supports Alpha360/LSTM/GRU/Transformer models
- Models loaded from `qlib-zh/DATA/analysis_outputs/<model_name>/`
- `GET /api/qlib/models` — lists available pre-trained models
- `GET /api/qlib/index-stocks?index=csi300&exclude_star=true` — lists index constituent stocks (csi300/csi500/csi1000)

**Training** (`POST /api/qlib/train`):
- `qlib-zh/train_runner.py` — walk-forward training, 3-fold (5yr train / 1yr val / 2yr test)
- Supports csi300 and csi1000 indices with LightGBM/XGBoost
- Warm-start checkpoint chain: each fold loads `params.pkl` from previous fold
- `GET /api/qlib/train-targets` — lists available training configs
- `GET /api/qlib/train/<task_id>/stream` — SSE progress stream (2hr timeout)

**Finetuning** (`POST /api/qlib/finetune`):
- `qlib-zh/finetune_alpha158.py` — single-fold finetuning on existing model checkpoints
- Uses recent data to continue training from the last fold's checkpoint
- YAML config adds `predict_disable_shape_check` for feature dimension compatibility
- `GET /api/qlib/finetune/<task_id>/stream` — SSE progress stream

**Data Update** (`POST /api/qlib/data/update`):
- `qlib-zh/data_runner.py` — downloads `qlib_bin.tar.gz` from GitHub releases (`chenditc/investment_data`)
- Extracts to `~/.qlib/qlib_data/cn_data`
- Supports SOCKS5 proxy via `QLIB_DATA_PROXY` env var
- `GET /api/qlib/data/update/<task_id>/stream` — SSE progress stream

**Practice Pipeline** (`qlib-zh/scripts/practice/`):
6-stage stock selection pipeline: `stage1_data_health.py` → `stage2_master_strategy.py` → `stage3_first_screen.py` → `stage4_risk_eval.py` → `stage5_second_screen.py` → `stage6_final_result.py`. Also includes `gen_deepseek_selections.py` (top-N + rebalancing), `time_decay_reweighter.py`, `gen_practice_yaml.py` (YAML config generation with warm-start params), and `rerun_full_backtest.py`.

**Key files:**
- `qlib-zh/infer_runner.py` — inference entry point
- `qlib-zh/train_runner.py` — training entry point (Docker-in-Docker)
- `qlib-zh/finetune_alpha158.py` — finetuning entry point
- `qlib-zh/data_runner.py` — data download entry point
- `qlib-zh/DATA/` — Qlib-format market data + trained models in `DATA/analysis_outputs/` (gitignored)
- `qlib-zh/scripts/practice/run_stage2_walk_forward.py` — core training engine (165KB)
- `qlib-zh/scripts/small/` — CSI1000-specific variants + cached_handler
- `qlib-zh/strategy_b_analyze.py` — CLI tool: applies 4 master-veto rules to Qlib top-20 predictions, outputs filtered buy list + rebalancing advice
  - Usage: `python strategy_b_analyze.py --stocks SH600000,SZ000001,... --scores 0.14,0.12,... --date 2026-06-20 [--holdings ...]`

### Simulation Bridge (`POST /api/predict`, background thread)

- `analysis/agent.py` → `simulation_bridge/seed_builder.py` → `simulation_bridge/orchestrator.py` → MiroFish HTTP API
- Seed document embeds **7 agent roles** (Buffett/Munger/Valuation/Sentiment/Fundamental/Technical/RiskManager) + explicit entity-relationship statements for Zep GraphRAG extraction
- Pipeline: ontology generation → graph build → simulation create → agent profile prep → OASIS run → report generation
- Each stage has polling with configurable timeouts; falls back to standalone mode on any MiroFish failure
- SSE progress stream with real-time log messages (EventSource in frontend, polling fallback)

### Feishu/Lark Bot Integration (`integration/`)

WebSocket-based Feishu bot (`lark_bot.py`, 36KB) enabling stock analysis from mobile Feishu, no public IP needed.

**Components:**
- `lark_bot.py` — async WebSocket handler, stock code regex parsing, message routing (`--master` arg extraction, multi-stock `/` delimiter), UserPrefManager integration (persistent master per user)
- `lark_card.py` (21KB) — Card JSON v2.0 builder, 3 analysis card types (normal/master/batch) + help card, A-stock color scheme (red=bullish, green=bearish)
- `lark_client.py` (12KB) — aiohttp-based async client wrapping StockFish Flask API (`/api/analyze`, `/api/batch/analyze`)
- `lark_prefs.py` (4KB) — JSON-file-backed user preference manager (default master, notification pref)

**Run**: `python integration/lark_bot.py` or `bash run.sh --bot` (with/without `--local`).
**Config**: Requires `LARK_APP_ID`, `LARK_APP_SECRET` in `.env`.

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

### Data Fetcher Architecture (`market_data/data_fetchers/`)

All fetchers extend a shared base (`base.py`, 130KB) providing: circuit-breaker failover, rate limiting, retry with exponential backoff, uniform `Quote`/`FinancialSummary`/`TechnicalIndicators` output.

| Fetcher | File | Focus |
|---|---|---|
| efinance | `efinance_fetcher.py` (50KB) | EastMoney via efinance, primary CN source |
| akshare | `akshare_fetcher.py` (92KB) | EastMoney via akshare, broadest coverage |
| tushare | `tushare_fetcher.py` (51KB) | Tushare Pro via vendored SDK |
| pytdx | `pytdx_fetcher.py` (18KB) | TDX protocol, low-level quote/歷史 |
| baostock | `baostock_fetcher.py` (14KB) | Free academic-grade fallback |
| yfinance | `yfinance_fetcher.py` (30KB) | Yahoo Finance, HK/US stocks |
| longbridge | `longbridge_fetcher.py` (28KB) | Longbridge OpenAPI, HK/US/CN |
| finnhub | `finnhub_fetcher.py` (6KB) | Finnhub.io, US stocks |
| alphavantage | `alphavantage_fetcher.py` (7KB) | Alpha Vantage, global |
| fundamental | `fundamental_adapter.py` (24KB) | Growth/earnings/institutional pipeline |
| realtime | `realtime_types.py` (17KB) | Quote data models & merge logic |

### Market Data Sub-modules

- `search/search_service.py` (150KB) — 7-engine search service (Tavily/Bocha/Brave/SerpAPI/Anspire/MiniMax/SearXNG) with auto health-check, multi-key round-robin, and a `SearchResult` dataclass
- `social_sentiment/social_sentiment_service.py` (14KB) — Reddit/X/Polymarket sentiment for US stocks via Adanos API
- `stock_index/` — Index constituent loading (`stock_index_loader.py`), remote service (`stock_index_remote_service.py`), stock-to-index mapping (`stock_mapping.py`)
- `patches/eastmoney_patch.py` (6KB) — Runtime monkey-patches for EastMoney API compatibility

## News Sources (`market_data/news_sources.py`)

Plugin architecture: each source extends `BaseNewsSource` or `BaseGubaSource`, registered in `NEWS_SOURCES`/`GUBA_SOURCES` lists. Current: SinaNews, NewsNow (cls+xueqiu+wallstreetcn aggregate), YahooFinance, XueqiuPopularity, CLSNews (disabled), EastMoneyGuba.

## Frontend (`static/index.html`)

Single-file dark-theme SPA (~2100 lines, no build step). Features:
- Stock symbol + cost price input, optional "智能推演" checkbox (toggles between `/api/analyze` and `/api/predict`)
- **Master selection dropdown** — choose from 7 investment masters (格雷厄姆/巴菲特/费雪/林奇/邓普顿/索罗斯/达利欧), populates via `GET /api/masters`. When selected, activates the multi-agent CIO pipeline; when unselected, falls back to legacy 3-agent debate
- **Batch mode toggle** — multi-symbol input (comma/newline-separated, max 20), per-stock progress bars
- **Qlib inference panel** — model selection dropdown, inference with SSE progress
- SSE real-time log streaming during simulation, polling fallback on SSE error
- Analysis view: signal header, 6-card grid (price/dividend/cost/buy price/valuation/sentiment), multi-cycle LLM predictions, suggested action with stop-loss/take-profit, technical + fundamental detail grids, score breakdown, news/guba summaries, CIO master decision card (avatar + decision_summary + 3-scenario analysis + order), multi-agent debate panel (hidden when master mode active)
- Report view: download HTML button, print/PDF export button
- A-stock color scheme: red = up, green = down

## Scoring Engine (`analysis/scoring.py`)

Three-layer weighted system (-5 to +5):

| Layer | Default Weight | Factors |
|---|---|---|
| Technical | 50% | RSI(14), MACD histogram, MA alignment (MA5/10/20/60), Bollinger position, volume-price, momentum |
| Fundamental | 30% | PE percentile (industry-corrected), ROE (cash-flow validated), profit growth, dividend yield |
| Sentiment | 20% | News avg score, guba avg score, emotional consistency |

**Adaptive weights**: trending market → technical +5%; ranging → fundamental +5%. Missing data redistributed proportionally.

Key dataclass: `ScoreResult` with `final`, `label`, `technical`, `fundamental`, `sentiment`, `regime`, `confidence`, `weights`, `breakdown: List[FactorDetail]`.

Signal labels map: `>=4.0`→强烈看多, `>=2.0`→看多, `>=0.5`→偏多, `>-0.5`→中性, `>=-2.0`→偏空, `>=-4.0`→看空, else→强烈看空.

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

pydantic-settings `Settings` loaded from `.env`. See `.env.example` for a complete template. Adds MiroFish and BettaFish directories to Python path on import.

### Feature Flags

| Flag | Default | Effect |
|---|---|---|
| `ENABLE_REALTIME_QUOTE` | true | Live quote fetching |
| `ENABLE_REALTIME_TECHNICAL_INDICATORS` | true | Real-time RSI/MACD/MA/Bollinger |
| `ENABLE_CHIP_DISTRIBUTION` | true | Chip/position distribution data |
| `ENABLE_EASTMONEY_PATCH` | false | Runtime monkey-patches for EastMoney |
| `ENABLE_FUNDAMENTAL_PIPELINE` | true | Growth/earnings/institutional pipeline |
| `ENABLE_LONGBRIDGE` | false | Longbridge HK/US OpenAPI |
| `ENABLE_MACRO` | false | Macro-economic data collection |
| `ENABLE_US_STOCKS` / `ENABLE_HK_STOCKS` | false | US/HK stock support |
| `PREFETCH_REALTIME_QUOTES` | true | Pre-fetch quotes on startup |
| `STOCK_INDEX_REMOTE_UPDATE_ENABLED` | true | Remote index constituent updates |

Key env vars:
- `LLM_API_KEY/BASE_URL/MODEL_NAME` — LLM config (OpenAI-compatible)
- `STOCK_BACKEND` — data source: `mock|akshare|tushare|advanced|auto` (default: `mock`)
- `TUSHARE_TOKEN` — Tushare Pro (via sxsc_tushare SDK)
- Search API keys: `TAVILY_API_KEY`, `BOCHA_API_KEY`, `BRAVE_API_KEY`, `SERPAPI_API_KEY`, `ANSPIRE_API_KEY`, `MINIMAX_API_KEYS`, `SEARXNG_BASE_URL` + comma-separated multi-key variants (`TAVILY_API_KEYS`, etc.)
- `SOCIAL_SENTIMENT_API_KEY`/`SOCIAL_SENTIMENT_API_URL` — Reddit/X/Polymarket (US stocks only)
- `MIROFISH_HOST/PORT` — MiroFish address (Docker: `mirofish:5001`, local: `localhost:5001`)
- `REALTIME_SOURCE_PRIORITY` — quote source order (default: `tencent,tushare,efinance,akshare_em`)
- `EFINANCE_CALL_TIMEOUT` — Eastmoney timeout (default 30s, set 5s outside China)
- Feature flags: see table above
- Multi-source fetcher priorities: `EFINANCE_PRIORITY`, `YFINANCE_PRIORITY`, `PYTDX_PRIORITY`, `AKSHARE_PRIORITY`, `BAOSTOCK_PRIORITY`
- Circuit breaker: `CIRCUIT_BREAKER_COOLDOWN` (default 300s), rate limits for each fetcher
- Rate limits/timeouts: `AKSHARE_SLEEP_MIN`/`MAX`, `TUSHARE_RATE_LIMIT_PER_MINUTE` (80), `MAX_RETRIES` (3), `RETRY_BASE_DELAY`/`MAX_DELAY`
- Fundamental pipeline: `FUNDAMENTAL_STAGE_TIMEOUT_SECONDS` (8s), `FUNDAMENTAL_FETCH_TIMEOUT_SECONDS` (3s), `FUNDAMENTAL_RETRY_MAX` (1), `FUNDAMENTAL_CACHE_TTL_SECONDS` (120s), `FUNDAMENTAL_CACHE_MAX_ENTRIES` (256)
- News: `NEWS_MAX_AGE_DAYS` (3), `NEWS_STRATEGY_PROFILE` (short), `BIAS_THRESHOLD` (5.0)
- Cache: `REALTIME_CACHE_TTL` (600s), `CACHE_TTL_SECONDS` (60s)
- New data source keys: `FINNHUB_API_KEY`, `ALPHAVANTAGE_API_KEY`, `LONGBRIDGE_APP_KEY/APP_SECRET/ACCESS_TOKEN/REGION`, `TICKFLOW_API_KEY`, `AKSHARE_PROXY`
- `OASIS_DEFAULT_MAX_ROUNDS` (20), `OASIS_SIMULATION_AGENT_COUNT` (15), `OASIS_DEBUG` (2 agents/2 rounds)
- `ZEP_API_KEY` — Zep Cloud for MiroFish graph memory (optional)
- `QLIB_ENABLED` (false), `QLIB_DATA_PROXY` (SOCKS5 for GitHub acceleration)
- `BATCH_MAX_CONCURRENT` — Batch analysis concurrency limit (default: 5)

## AnalysisState (`analysis/state/state.py`)

Central dataclass carrying all analysis pipeline state through 5 steps (122 lines). Key field groups:

| Group | Fields |
|---|---|
| **User input** | `symbol`, `cost_price`, `shares`, `total_assets`, `available_cash` |
| **Market data** | `quote`, `technical_indicators`, `financial_summary`, `news`, `guba_posts` |
| **Sentiment** | `sentiment_news`, `sentiment_guba` (each: `SentimentSummary` with avg_score, positive/negative/neutral counts, per-item scores) |
| **Valuation** | `valuation_level` (很低~很高), `valuation_percentile` (%, 365-day), `suggested_buy_price`, `historical_pe_avg`, `valuation_percentile_5y/10y`, `pe_avg_5y/10y` |
| **Financial depth** | `roic`, `fcf_per_share`, `operating_cash_flow_per_share`, `owner_earnings_per_share`, `financial_trends` (ROE 5yr, EPS CAGR 5yr, ROE stability, gross margin trend) |
| **Risk** | `var_95`, `max_drawdown`, `beta`, `annualized_volatility`, `earnings_yield`, `bond_yield_10y`, `equity_risk_premium` |
| **Context** | `macro_context` (SHIBOR/PMI/CPI/northbound flow/USD-CNY/policy), `industry_context` (industry PE/momentum/cycle/policy), `search_results` (multi-dimension web search) |
| **Peer/Moat** | `peer_valuation`, `moat_assessment`, `management_quality` |
| **LLM output** | `llm_analysis`, `prediction_summary`, `short_term_pred`, `mid_term_pred`, `long_term_pred`, `suggested_action`, `price_target`, `risk_factors` |
| **Scoring** | `score_breakdown` (ScoreResult dict), `signals` |
| **News summaries** | `important_bullish_news`, `important_bearish_news`, `important_bullish_guba`, `important_bearish_guba` |

All backends produce data that maps into these fields. Sentiment history is persisted per-symbol in `data/sentiment_history/` as JSON (survives restarts).

- A-stock color scheme: **red = up (gain)**, green = down (loss) — opposite of Western markets
- `loguru` for logging consistently across all modules
- `config:settings` singleton — imported at module level in `app.py`, lazy-imported elsewhere
- `sxsc_tushare/` — vendored Tushare Pro SDK (山西证券 proxy), imported directly
- BSE stocks: use `.BJ` suffix (e.g., `830799.BJ`), Tushare backend auto-converts format
- `.env` must exist in project root (checked by `run.sh`); copy from `.env.example`
- `STOCK_BACKEND=mock` is default for zero-config dev; `advanced` for production
- All timeouts in the orchestrator are configurable (graph: 300s, simulation: 900s, report: 600s)
- Flask app global state in 6 thread-safe dicts with locks: `predictions`, `batch_tasks`, `qlib_tasks`, `qlib_data_tasks`, `qlib_train_tasks`, `qlib_finetune_tasks`

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
- `qlib-zh/` — Qlib inference (`infer_runner.py`), training (`train_runner.py`), finetuning (`finetune_alpha158.py`), data download (`data_runner.py`), models in `DATA/analysis_outputs/`, 6-stage practice pipeline in `scripts/practice/`
- `sxsc_tushare/` — Vendored Tushare Pro SDK (dataapi.py, upass.py)
- `MiroFish/backend/` — MiroFish Flask app (see MiroFish Backend section above)
- `static/index.html` — Single-file SPA frontend (~2100 lines)
- `backend/` — Empty reserved directory (no active code)
- `document/` — Documentation screenshots (1-4.png, label.png)
- `reports/` — Generated prediction report HTML output (gitignored)
- `simulation_output/` — Seed documents and scenario JSONs saved per analysis run (gitignored)
- `batch_results/` — Batch analysis result cache (gitignored)
- `data/sentiment_history/` — Per-symbol sentiment analysis JSON cache (persisted across runs)
- `TODO.md` — Full project roadmap with P0-P3 priorities, tech debt, and work estimates

## Slash Commands

### `/master-audit`
大师分析质量审计 — 随机从沪深300挑选一支股票和一位大师，运行完整的 8 员工 + CIO 分析工作流，交叉验证数据完整性/逻辑一致性/哲学匹配度，输出审计报告和改进计划。

**Usage:** `/master-audit [--symbol <股票>] [--master <大师key>] [--seed <随机种子>]`

**Examples:**
```
/master-audit                                   # 随机股票 + 随机大师
/master-audit --symbol 600519 --master buffett  # 指定
/master-audit --seed 42                         # 固定种子复现
```

**Output:** `master-audit/{date-seq}/` 目录保存完整中间数据、审计报告和改进计划。

### `/release-model`
Compress a Qlib model directory into `csi300-alpha158.tar.gz` and push it to `github.com/freenowill/stock-fish/releases`.

**Usage:** `/release-model <模型目录路径>`

**Example:**
```
/release-model qlib-zh/DATA/analysis_outputs/2026-06-07-csi300-alpha158
```

**Steps:**
1. Check existing release — if `csi300-alpha158.tar.gz` asset exists, skip entirely
2. Delete only releases **missing** the tar.gz asset (empty drafts)
3. `tar -czf /tmp/csi300-alpha158-model.tar.gz` the model directory
4. `gh release create csi300-alpha158` with the tar.gz to `freenowill/stock-fish`
5. Verify the uploaded asset

**Note:** Always uses `TMPDIR=/Users/apple/github/stock_predict/StockFish` for `gh` commands to avoid `/private/tmp` disk-full issues.
