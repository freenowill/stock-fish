# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start server (port 8000)
python app.py

# One-click deploy (Docker or local)
bash run.sh          # Docker mode
bash run.sh --local  # Local mode

# API: multi-factor analysis
curl -X POST http://localhost:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"600519","cost_price":150}'

# API: prediction pipeline (background + SSE)
curl -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"600519","scenario":"base"}'

# SSE progress stream
curl -N http://localhost:8000/api/predict/<task_id>/stream

# Install dependencies
pip install -r requirements.txt
```

No test framework or test files exist yet (`tests/__init__.py` is empty). No linting/formatting config.

## Architecture

**4-step analysis pipeline** (`POST /api/analyze`):

1. **Data Collection** (`analysis/agent.py:StockAnalysisAgent.analyze`) — fetches quote, technical indicators, financials, news, guba posts via `AStockProvider` (auto-selects best backend)
2. **Sentiment** (`market_data/sentiment_collector.py`) — HuggingFace multilingual model → 5-class sentiment, with keyword-rule fallback. Reuses BettaFish's model if available.
3. **Signal Generation** (`analysis/scoring.py:ScoringEngine`) — -5~+5 composite score from technical (RSI/MACD/MA/Bollinger/volume/momentum) + fundamental (PE-percentile/ROE/growth/dividend) + sentiment (news/guba). Adaptive weights based on market regime (trending up/down vs ranging).
4. **LLM Prediction** (`analysis/nodes/prediction_node.py`) — 3 parallel agents (tech/fundamental/sentiment) each analyze their domain independently, then a Moderator agent reads all views and produces final prediction with multi-cycle outlook (short/mid/long term) and suggested action.

**Simulation bridge** (`POST /api/predict`):
- `analysis/agent.py` → `simulation_bridge/seed_builder.py` → `simulation_bridge/orchestrator.py` → MiroFish HTTP API (Zep GraphRAG ontology → OASIS agent simulation → report)
- Falls back to standalone mode if MiroFish unreachable
- Generates HTML report via `prediction_report/report_generator.py`

## Data Backends (`market_data/a_stock_provider.py`)

`AStockProvider` auto-selects: `advanced` → `tushare` → `akshare` → `baostock` → `mock`. Config via `STOCK_BACKEND` env var.

- **MockBackend** — Random data, zero network, for dev/demo
- **AkShareBackend** — EastMoney data via akshare (needs mainland China network)
- **BaoStockBackend** — Free, no token, fallback
- **TushareBackend** (`tushare_provider.py`) — Tushare Pro via sxsc_tushare SDK
- **AdvancedBackend** (`provider_adapter.py`) — DataFetcherManager wrapping 11 fetchers (efinance/akshare/tushare/pytdx/baostock/yfinance/finnhub/alphavantage/longbridge) with circuit-breaker failover + 7 search engines + social sentiment. Activated by `STOCK_BACKEND=advanced`.

All backends implement `BaseStockBackend` interface with: `get_quote`, `get_historical`, `get_financials`, `get_news`, `get_guba`, `get_historical_pe`.

## News Sources (`market_data/news_sources.py`)

Plugin architecture: each source is a class extending `BaseNewsSource` or `BaseGubaSource`, registered in `NEWS_SOURCES`/`GUBA_SOURCES` lists. Current: SinaNews, NewsNow (cls+xueqiu+wallstreetcn aggregate), YahooFinance, XueqiuPopularity, CLSNews (disabled), EastMoneyGuba.

## Scoring Engine (`analysis/scoring.py`)

Three-layer weighted system (-5 to +5):

| Layer | Weight | Factors |
|-------|--------|---------|
| Technical | 50% (adjustable) | RSI[-2,+2], MACD[-1.5,+1.5], MA alignment[-1.5,+1.5], Bollinger %B[-1,+1], Volume-price[-1,+1], 20d momentum[-1.5,+1.5] |
| Fundamental | 30% | PE percentile[-3,+3] with earnings-trend adjustment, ROE[-1.5,+1.5] with debt penalty, Growth[-0.5,+1], Dividend[-0.3,+0.5] |
| Sentiment | 20% | News[-2.5,+2.5], Guba[-2.5,+2.5], consistency adjustment±0.5 |

Adaptive weights: trending market → technical gets +5%, ranging → fundamental gets +5%. Missing data redistributed.

Scoring dataclass: `ScoreResult` with `final`, `label`, breakdown list of `FactorDetail` objects.

## LLM Prediction (`analysis/nodes/prediction_node.py`)

Two modes controlled by `LLM_API_KEY`:
- **Multi-agent mode** (key set): 3 agents debate in parallel via `ThreadPoolExecutor`, then a Moderator synthesizes → structured JSON with multi-cycle outlook + suggested action + price targets. Uses OpenAI-compatible API (`response_format: json_object`).
- **Rule mode** (no key): score-threshold based, outputs signal label + price range ±5-10%.

## Configuration (`config.py`)

pydantic-settings `Settings` loaded from `.env`. Adds BettaFish/MiroFish to Python path for cross-project imports. Key vars: `LLM_API_KEY/BASE_URL/MODEL_NAME`, `STOCK_BACKEND`, `TUSHARE_TOKEN`, search API keys (Tavily/Bocha/Brave/SerpAPI/Anspire/MiniMax/SearXNG), `SOCIAL_SENTIMENT_API_KEY`, circuit breaker cooldowns, rate limits.

## Key Conventions

- A-stock color scheme: red = up (gain), green = down (loss)
- `Quote`, `FinancialSummary`, `TechnicalIndicators`, `NewsItem`, `GubaPost` — dataclasses with `to_dict()`, used across all backends
- `AnalysisState` (`analysis/state/state.py`) — dataclass carrying full analysis state through all pipeline steps
- `loguru` for logging consistently across all modules
- `config:settings` singleton imported at module level in `app.py`, lazy in other modules
- `sxsc_tushare/` — vendored Tushare Pro SDK (山西证券 proxy), imported directly as `sxsc_tushare`
- `.env` must exist in project root (checked by `run.sh`); `STOCK_BACKEND=mock` is default for zero-config dev
