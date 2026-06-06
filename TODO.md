# StockFish TODO

> 项目任务规划 — 按优先级从高到低排列
> Project task roadmap, ordered by priority (P0 = highest).

---

## P0 — 核心功能增强 (Core Enhancement)

### □ Qlib 模型推理 (Qlib Model Inference)

集成 [Qlib](https://github.com/microsoft/qlib) 作为 AI 预测引擎，与现有技术/基本面/情感分析并行输出预测，形成模型集成。

- **Qlib Inference Engine** (`analysis/qlib_engine.py`)
  - 提供一个 `QlibEngine` 封装 Qlib 的 `ModelManager.inference()`，支持 Alpha360/LSTM/GRU/Transformer 等主流模型
  - 输入：从 AStockProvider 获取的历史 K 线（~3 年日频） + Alpha360/158 因子
  - 输出：每日收益率预测 (`pd.Series`, index=date, values=pred_return) → 聚合为 short/mid/long 价格方向与置信度
  - 返回值与现有 `PredictionResult` 兼容（`direction`, `change_pct`, `confidence`, `reason`）
- **工作模式**
  - **模型文件模式** — 本地已有 `.pkl` / `.pth` 模型文件时直接加载推理（`qlib.model.ModelManager`）
  - **API 模式** — 远程 Qlib 推理服务（通过 `ModelManager.inference()` 或自定义 API 调用）
  - **无模型模式** — 静默跳过，不影响现有流程
- **配置** — `config.py` 新增：
  - `QLIB_ENABLED` (默认 `false`)
  - `QLIB_MODEL_DIR` (模型文件目录)
  - `QLIB_MODEL_URI` (远程 URI)
  - `QLIB_DAYS` (训练/推理用历史天数, 默认 720)
- **集成进 AnalysisPipeline** (`analysis/agent.py`)
  - Step 3（Signal Generation）之后，Step 4（LLM Prediction）之前，插入同步/异步 Qlib 推理节点
  - Qlib 输出作为第 4 路信号传入 LLM Multi-Agent Debate 的 context（现有 3 agents + Qlib 因子）
- **模型下载工具** (`scripts/download_qlib_models.sh`) — 从 Qlib 官方 / 阿里云 OSS 下载预训练模型
- **评估** — 回测模块（见下方 Backtesting），对比加入 Qlib 前后的 IC / 累计收益

### □ 批量股票智能推演 (Batch Stock Simulation)

支持对多只股票并发执行分析/推演，适用于组合监控、行业扫描等场景。

- **后端**
  - `POST /api/batch/analyze` — `{"symbols": ["600519", "000858", "300750"], "cost_prices": {"600519": 150}}`
    - 每个 symbol 提交后台任务（现有 `@background` 机制）
    - 返回 `{tasks: {symbol: task_id, ...}}`
  - `GET /api/batch/status?tasks=id1,id2,id3` — 批量查询任务状态
  - `GET /api/batch/results?tasks=id1,id2,id3` — 批量拉取结果
  - 并发控制：`BATCH_MAX_CONCURRENT` (默认 5)，超出的排队等待
  - 结果聚合：生成对比表格（涨跌方向、置信度、PE 百分位、信号分并排）
- **前端** (`static/index.html`)
  - 输入区域支持多 symbol 输入（逗号/换行分隔, 最多 20 只）
  - 进度面板：每只股票一个进度条，整体进度条
  - 结果面板：对比表格 + 可点击展开单只详情
- **定时批量分析** — 配合后端 `schedule` 支持（见 P1 Scheduled Tasks）

### □ 飞书集成 (Feishu/Lark Integration)

通过飞书机器人发送股票代码、接收分析结果，实现移动端交互。

- **飞书 Bot 服务** (`integration/lark_bot.py`)
  - 使用飞书开放平台 [Card & Message API](https://open.feishu.cn/document/uAjLw4CM/ukzMukzMukzM/feishu-cards/card-components)
  - 事件订阅：`im.message.receive_v1` (含 text/mention)
  - **命令菜单**：
    - `/analyze 600519` — 单股分析
    - `/predict 600519` — 推演分析（含模拟）
    - `/batch 600519,000858,300750` — 批量分析
    - `/watch add 600519` / `/watch list` / `/watch remove 600519` — 自选股管理
  - **消息卡片**：分析结果以飞书 Interactive Card 呈现
    - 封面：信号标签（大涨/小涨/震荡/小跌/大跌）+ 评分条
    - 详情：当前价/PE/估值水平/建议买点/风险标签
    - 多周期预测：Short / Mid / Long 三列
    - 操作按钮：查看完整 HTML 报告链接、添加到自选、分享
  - **交互回调**：卡片按钮 `action` 处理（如切换周期、展开详情）
- **自选股管理** (`data/watchlist.json`)
  - 用户维度的自选股存储（简单 JSON 文件 / 进阶 SQLite）
  - 定时推送：每天 9:00 发送开盘简报（订阅自选股的上日信号变化）
- **配置** — `config.py` 新增：
  - `LARK_APP_ID`, `LARK_APP_SECRET`
  - `LARK_BOT_NAME` (机器人名称，用于 @ 触发)
  - `LARK_PUSH_ENABLED` (默认 `false`)
- **Docker 部署** — Bot 服务伴随 StockFish 进程启动（或独立容器）

---

## P1 — 工程化与质量 (Engineering & Quality)

### □ 单元测试与集成测试 (Test Suite)

- 单元测试覆盖核心模块：
  - `analysis/scoring.py` — 各因子计算、权重分配、缺失值处理、边界值 (-5, +5)
  - `analysis/agent.py` — DataCollection / Sentiment / Signal / LLM 各步骤
  - `analysis/nodes/prediction_node.py` — 3-agent 并行、moderator 合成、majority-vote 降级
  - `market_data/sentiment_collector.py` — 模型推理 + 规则 fallback
  - `market_data/a_stock_provider.py` — 所有后端 (`MockBackend`, `AkShareBackend`, `BaoStockBackend`, `TushareBackend`, `AdvancedBackend`)
  - `simulation_bridge/seed_builder.py` — 种子文档结构和 7 agent roles
  - `simulation_bridge/orchestrator.py` — 状态机转换、超时处理、降级路径
- Mock / Fixture：
  - `tests/data/` — mock 行情、财务、新闻 JSON 文件
  - `conftest.py` — pytest fixtures for Flask app client, MockBackend, LLM API mock
- 集成测试：`POST /api/analyze` + `POST /api/predict` 端到端（MockBackend + mock LLM）
- CI 集成（见 P2 CI/CD）

### □ 回测系统 (Backtesting Framework)

选择股票池、历史区间，对比预测信号与实际涨跌幅，评估模型表现。

- `analysis/backtest/backtest_engine.py`
  - 输入：股票池（list of symbols）、时间区间、调仓周期
  - 在每个时间窗口运行分析管道（或仅信号生成，跳过硬耗的 LLM 调用）
  - 记录每次 signal/score/prediction 与后续 N 日收益
- 评估指标
  - **IC (Information Coefficient)** — Spearman rank correlation（信号 vs 实际收益）
  - **IR (Information Ratio)** — IC 均值 / IC 标准差
  - **胜率 (Win Rate)** — 方向正确比例
  - **累计收益 vs 基准**（沪深 300 / 中证 500）
- 报告输出：HTML 回测报告 + `backtest_results/` CSV 明细

### □ 结果缓存 (Result Caching)

避免同一只股票在短时间内重复分析，减轻下游数据源压力。

- `analysis/cache.py`
  - Key: `analyze:{symbol}:{hash_of_params}`, TTL: 300s
  - 缓存层级：L1 = Python dict (内存), L2 = Redis / SQLite (可选)
  - `CACHE_TTL_ANALYSIS` 配置项，默认 300s
- `predict` 任务也缓存种子文档和场景 JSON（`simulation_output/` 已具备部分功能）
- 自动失效：在 `config.py` 切换后端、强制刷新时主动清除缓存

### □ Scheduled Tasks (定时任务)

- 每日收盘后自动分析自选股
- 使用 `APScheduler`（已在依赖中）或 `schedule` 库
- 任务定义：`scheduler/tasks.py`
  - `daily_analysis` — 每天 15:30 运行（A 股 15:00 收盘），分析`[watchlist]`内所有股票
  - `weekly_report` — 每周五 16:00 生成周报摘要（持股组合表现 + 信号变化）
  - `health_check` — 每 5 分钟检查 MiroFish / 数据源健康状态
- 配置：`SCHEDULER_ENABLED` (默认 `false`), `SCHEDULER_HOUR`, `SCHEDULER_MINUTE`

### □ Docker / 部署优化 (Deployment Improvements)

- Docker 镜像瘦身（当前基于 Python 基础镜像 >1GB）
  - 多阶段构建，分离依赖安装与应用代码
  - 使用 `slim` 或 `alpine` 变体（注意 C 扩展兼容性：numpy/pandas/WeasyPrint）
  - `.dockerignore` 排除无用目录（`tests/`, `document/`, `simulation_output/`, `reports/`）
- 健康检查端点：`GET /api/health` 返回各数据源状态 + 磁盘 + 内存
- docker-compose 增加飞书 Bot 服务容器

---

## P2 — 优化与扩展 (Optimization & Extension)

### □ CI/CD 流水线 (CI/CD Pipeline)

- GitHub Actions: `.github/workflows/`
  - `test.yml` — `pip install` → `pytest` (MockBackend + mock LLM)
  - `lint.yml` — 使用 `ruff` 做代码风格检查（替代 flake8, 更快）
  - `docker-build.yml` — push 到 ghcr.io / Docker Hub
  - `release.yml` — tag 触发发布
- Docker BuildKit 缓存加速 CI

### □ 性能优化 (Performance Tuning)

- **LLM 推理性能**
  - LLM 调用加入超时 + 重试 + 替补模型
  - 3 个 agent 并行调用（已有 `ThreadPoolExecutor(max_workers=3)`）→ 验证异步框架 `asyncio` + `aiohttp` 是否更优
  - `max_tokens` 限制（当前可能无限制）
- **数据获取性能**
  - `a_stock_provider.py` 的 `get_quote` + `get_historical` + `get_financials` 并行化（当前 `get_all()` 内已并行）→ 确认 exceptions 处理
  - 实时行情缓存（已有 `REALTIME_CACHE_TTL`, `cachetools.TTLCache`）→ 验证缓存有效性
  - 新闻/股吧获取加入超时（`NEWS_CRAWL_TIMEOUT`）
- **前端性能**
  - 大型报告渲染优化（当前全量 HTML 内嵌）
  - 懒加载 agent debate 面板（搜索结果量大时）

### □ 更多数据接入 (Data Source Expansion)

- **Level 2 行情** — 接入东方财富/万得的 Level 2 逐笔成交数据，用于更精细的资金流向分析和主买/主卖识别
- **另类数据**
  - 电商数据（淘宝/京东销量趋势）→ 消费类股先行指标
  - 招聘数据（BOSS 直聘/猎聘岗位数）→ 行业景气度
  - 专利数据 → 科技股研发指标
- **国际指数** — 增加美股/港股覆盖深度（已有 yfinance/finnhub 但有限）

### □ 模型集成扩展 (Model Ensemble)

- 在 Qlib 之外集成更多模型：
  - **XGBoost / LightGBM** — 基于因子表的传统 ML 预测
  - **MLP / LSTM** — 接入 `keras` / `pytorch` 自定义模型
  - **Ensemble** — 加权投票 / stacking 合成各模型输出
- 模型评估持久化：入库各模型的历史预测记录，定期计算 IC 并调整权重

### □ 用户系统与鉴权 (User System & Auth)

- 简单 token 认证：`AUTH_TOKENS` 环境变量（逗号分隔）
- API 端点增加 `Authorization: Bearer <token>` 校验
- 飞书用户与系统内 userId 映射
- 每个用户的独立 watchlist

### □ 自动化 Agent 操作 (Automated Trading Signals)

- **信号等级 + 自动触发**
  - 当 signal score >= +4 → 推送"强烈买入提醒"到飞书
  - 当 signal score <= -3 → 推送"风险预警"
  - 允许用户飞书回复 `/ack` 确认收到
- 可导出格式：东方财富 / 同花顺 / 雪球组合导入 CSV

---

## P3 — 长期规划 (Long-term)

### □ WebSocket 取代 SSE (WebSocket Upgrade)

- 当前 SSE 在部分代理（Nginx）下不稳定
- Flask-SocketIO / FastAPI WebSocket 替代
- 双层 fallback：SSE → WebSocket → Polling

### □ FastAPI 迁移 (FastAPI Migration)

- 将 Flask 逐步迁移到 FastAPI（获得原生异步、OpenAPI 文档、类型校验）
- 阶段 1：新端点（`/api/batch/*`, `/api/health`, `/api/cache/*`）用 FastAPI
- 阶段 2：存量端点并行运行
- 阶段 3：全部切到 FastAPI

### □ 数据库持久化 (Database)

当前状态全部在内存（`app.py:analysis_cache`, `predict_tasks` dict）。接入 SQLite / PostgreSQL：
- 任务记录：用户每次分析请求的 symbol、时间、结果状态、耗时
- 信号记录：历史信号变化、胜率跟踪
- 自选股：用户维度的 watchlist
- 飞书消息：已推送的消息记录，避免重复

### □ 多语言 / 国际化 (i18n)

- 前端文案中英文切换
- 报告输出中英文双版
- 飞书消息根据用户飞书语言设置切换

---

## 技术债务 & 代码质量 (Tech Debt)

- [ ] `config.py` 使用 lazy import — 确认 @lru_cache 注解，避免循环引用
- [ ] `simulation_bridge/orchestrator.py` 中的硬编码参数（5 rounds, 7 entity types）改为 config 驱动
- [ ] `market_data/news_sources.py` 中的 disabled sources (CLSNews) 清理或修复
- [ ] `app.py` 的 `predict_tasks` 全局 dict → 增加大小限制 + TTL 清理
- [ ] 异常处理统一化：用 `loguru` 记录所有异常，前端以结构化 JSON 返回错误
- [ ] 前端 `static/index.html` 超 700 行 — 拆分为多个模块（建议保留单文件但引入模板片段）
- [ ] `requirements.txt` 版本锁定 (pip freeze > requirements.txt)
- [ ] `README.md` 的英文错别字修正 (`Brill Band` → `Bollinger Band`, `SUGGESTED BUY PRICE` → `SUGGESTED`)

---

## 我们已做的 3 项 (Your 3 Items) — 实现路径

| 项目 | 预计工作量 | 关键文件 |
|------|-----------|---------|
| **Qlib 模型推理** | ~3-5 天 | `analysis/qlib_engine.py` (新建), `analysis/agent.py` (改造), `config.py` (新增配置), `scripts/*.sh` (下载脚本) |
| **批量股票智能推演** | ~3-5 天 | `app.py` (新增 API), `static/index.html` (批量 UI), `orchestrator.py` (并发控制), `scheduler/*` (定时) |
| **飞书集成** | ~5-7 天 | `integration/lark_bot.py` (新建), `integration/lark_card.py` (卡片模板), `app.py` (lifespan), `config.py`, `Dockerfile.lark` |

> **Legend:** □ = 待开始 ◐ = 进行中 ✓ = 已完成

*Last updated: 2026-06-06*
