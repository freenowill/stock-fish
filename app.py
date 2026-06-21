"""
StockFish - A 股实时分析 + 股价推演系统 (v2)

API:
  POST /api/analyze     完整多因子分析（技术+基本面+舆情+预测）
  POST /api/predict     启动股价推演（支持 base/bull/bear 场景）
  GET  /api/predict/<id>  推演状态
  GET  /api/predict/<id>/stream  SSE 推演进度流
  GET  /api/predict/<id>/report  推演报告 HTML
  GET  /api/config      系统配置
"""
import importlib.util
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context, send_file
from flask_cors import CORS
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings
from analysis.agent import StockAnalysisAgent
from analysis.batch_analyzer import BatchAnalyzer
from simulation_bridge.orchestrator import SimulationOrchestrator
from prediction_report.report_generator import PredictionReportGenerator

# ---- 解析 qlib-zh 目录（兼容 git worktree，models/DATA 在主仓库中） ----
_qlib_zh_dir = Path(__file__).resolve().parent / "qlib-zh"

def _resolve_qlib_dir() -> Path:
    """返回 qlib-zh 的 DATA 目录所在位置（git worktree 时回退到主仓库）"""
    data_dir = _qlib_zh_dir / "DATA"
    # 如果 DATA 目录存在，直接使用
    if data_dir.exists():
        return _qlib_zh_dir
    # worktree: 解析主仓库路径
    gitfile = _qlib_zh_dir.parent / ".git"
    if gitfile.is_file():
        content = gitfile.read_text().strip()
        if content.startswith("gitdir:"):
            # gitdir: /path/to/main/.git/worktrees/name
            git_dir = Path(content.split(":", 1)[1].strip())
            # .git/worktrees/name → parent 3× 回到主仓库根目录
            main_repo = git_dir.parent.parent.parent if "worktrees" in str(git_dir) else git_dir.parent
            candidate = main_repo / "qlib-zh"
            if candidate.exists():
                return candidate
    return _qlib_zh_dir

_qlib_base_dir = _resolve_qlib_dir()
logger.info(f"Qlib 基础目录: {_qlib_base_dir}")

# ---- Qlib 推理模块 (目录名含连字符，使用 importlib 加载) ----
_spec = importlib.util.spec_from_file_location("infer_runner", _qlib_zh_dir / "infer_runner.py")
_infer_runner_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_infer_runner_mod)
run_qlib_inference = _infer_runner_mod.run_inference

# ---- Qlib 数据更新模块 ----
_spec_data = importlib.util.spec_from_file_location("data_runner", _qlib_zh_dir / "data_runner.py")
_data_runner_mod = importlib.util.module_from_spec(_spec_data)
_spec_data.loader.exec_module(_data_runner_mod)
run_qlib_data_update = _data_runner_mod.run_data_update

# ---- Qlib 训练模块 ----
_spec_train = importlib.util.spec_from_file_location("train_runner", _qlib_zh_dir / "train_runner.py")
_train_runner_mod = importlib.util.module_from_spec(_spec_train)
_spec_train.loader.exec_module(_train_runner_mod)
run_qlib_training = _train_runner_mod.run_training

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# ===== 全局状态 =====
agent = StockAnalysisAgent()
batch_analyzer = BatchAnalyzer()
orchestrator = SimulationOrchestrator()
report_gen = PredictionReportGenerator()
predictions = {}
_predictions_lock = threading.Lock()
batch_tasks = {}
_batch_lock = threading.Lock()
qlib_tasks = {}
_qlib_lock = threading.Lock()
qlib_data_tasks = {}
_qlib_data_lock = threading.Lock()
qlib_train_tasks = {}
_qlib_train_lock = threading.Lock()
qlib_finetune_tasks = {}
_qlib_finetune_lock = threading.Lock()


# ==========================================
#  API: 分析 (Phase 2 - StockEngine Agent)
# ==========================================

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """完整多因子分析"""
    data = request.get_json(silent=True) or {}
    symbol = data.get('symbol', '').strip().upper()
    cost_price = data.get('cost_price', 0)
    master = data.get('master', '').strip().lower()
    shares = data.get('shares', 0) or 0
    total_assets = data.get('total_assets', 0) or 0.0
    available_cash = data.get('available_cash', 0) or 0.0
    if not symbol:
        return jsonify({'error': '请提供股票代码'}), 400

    logger.info(f"开始深度分析 [{symbol}] 成本价={cost_price} 持仓={shares}股 总资产={total_assets} 可用={available_cash} master={master or 'off'}")
    result = agent.analyze(symbol, cost_price=float(cost_price) if cost_price else 0.0,
                           master=master, shares=int(shares),
                           total_assets=float(total_assets), available_cash=float(available_cash))
    logger.info(f"[{symbol}] 分析完成 (状态: {result.get('status')})")
    return jsonify(result)


# ==========================================
#  API: 推演 (Phase 3+4 - Bridge + Report)
# ==========================================

@app.route('/api/predict', methods=['POST'])
def predict():
    """启动股价推演"""
    data = request.get_json(silent=True) or {}
    symbol = data.get('symbol', '').strip().upper()
    scenario = data.get('scenario', 'base')
    cost_price = data.get('cost_price', 0)
    master = data.get('master', '').strip().lower()
    shares = data.get('shares', 0) or 0
    total_assets = data.get('total_assets', 0) or 0.0
    available_cash = data.get('available_cash', 0) or 0.0

    if not symbol:
        return jsonify({'error': '请提供股票代码'}), 400

    task_id = f"pred_{uuid.uuid4().hex[:12]}"

    pred_data = {
        'task_id': task_id,
        'symbol': symbol,
        'scenario': scenario,
        'cost_price': float(cost_price) if cost_price else 0.0,
        'master': master,
        'shares': int(shares),
        'total_assets': float(total_assets),
        'available_cash': float(available_cash),
        'status': 'pending',
        'progress': 0.0,
        'message': '',
        'analysis': None,
        'simulation': None,
        'report': None,
        'report_html_path': None,
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
    }

    with _predictions_lock:
        predictions[task_id] = pred_data

    logger.info(f"[{symbol}] 启动推演 task_id={task_id}, scenario={scenario}")

    # 后台执行完整流水线
    def _run_pipeline():
        try:
            # Step 1: 分析
            _update_prediction(task_id, 0.1, 'analyzing', '正在进行多因子分析...')
            result = agent.analyze(symbol, cost_price=pred_data.get('cost_price', 0),
                                   master=pred_data.get('master', ''),
                                   shares=pred_data.get('shares', 0),
                                   total_assets=pred_data.get('total_assets', 0),
                                   available_cash=pred_data.get('available_cash', 0))
            if result.get('status') == 'error':
                _update_prediction(task_id, 1.0, 'failed', f"分析失败: {result.get('error')}")
                return
            _update_prediction(task_id, 0.4, 'analyzing', '分析完成', analysis=result)

            # Step 2: 模拟推演
            def _sim_progress(p, msg):
                progress = 0.4 + p * 0.4
                _update_prediction(task_id, progress, 'simulating', msg)

            _update_prediction(task_id, 0.4, 'simulating', '启动模拟推演引擎...')
            sim_result = orchestrator.orchestrate(result, scenario=scenario, progress_callback=_sim_progress)

            # 检查推演是否失败（不再降级，失败即报错）
            if sim_result.get('status') == 'failed':
                err_msg = sim_result.get('error', 'MiroFish 推演失败')
                raise RuntimeError(f"MiroFish 推演失败: {err_msg}")

            _update_prediction(task_id, 0.8, 'simulating', '模拟推演完成', simulation=sim_result)

            # Step 3: 生成报告
            _update_prediction(task_id, 0.9, 'generating_report', '生成预测报告...')
            report = report_gen.generate(result, sim_result)
            html_path = report_gen.save(report)

            _update_prediction(task_id, 1.0, 'completed', '推演完成', report=report, report_html_path=html_path)

        except Exception as e:
            import traceback
            logger.error(f"[{symbol}] 推演失败: {e}\n{traceback.format_exc()}")
            _update_prediction(task_id, 1.0, 'failed', f"推演异常: {str(e)}")

    thread = threading.Thread(target=_run_pipeline, daemon=True)
    thread.start()

    return jsonify({
        'task_id': task_id,
        'symbol': symbol,
        'scenario': scenario,
        'status': 'queued',
    })


@app.route('/api/predict/<task_id>', methods=['GET'])
def predict_status(task_id: str):
    with _predictions_lock:
        pred = predictions.get(task_id)
    if not pred:
        return jsonify({'error': '任务不存在'}), 404
    return jsonify({
        'task_id': pred['task_id'],
        'symbol': pred['symbol'],
        'scenario': pred['scenario'],
        'status': pred['status'],
        'progress': pred['progress'],
        'message': pred['message'],
        'created_at': pred['created_at'],
        'completed_at': pred['completed_at'],
    })


@app.route('/api/predict/<task_id>/stream', methods=['GET'])
def predict_stream(task_id: str):
    """SSE 推演进度流"""
    def generate():
        last_progress = -1
        last_yield_time = time.time()
        while True:
            with _predictions_lock:
                pred = predictions.get(task_id)
            if not pred:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break

            data = {
                'status': pred['status'],
                'progress': pred['progress'],
                'message': pred['message'],
            }
            current_progress = pred['progress']

            if pred['status'] in ('completed', 'failed'):
                data['report'] = pred.get('report')
                data['report_html_path'] = pred.get('report_html_path')
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break

            if current_progress != last_progress:
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                last_progress = current_progress
                last_yield_time = time.time()
            elif time.time() - last_yield_time > 15:
                yield ": heartbeat\n\n"
                last_yield_time = time.time()

            time.sleep(1)

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/predict/<task_id>/report', methods=['GET'])
def predict_report(task_id: str):
    with _predictions_lock:
        pred = predictions.get(task_id)
    if not pred:
        return jsonify({'error': '任务不存在'}), 404
    if pred['status'] != 'completed':
        return jsonify({'error': '任务尚未完成', 'status': pred['status'], 'progress': pred['progress']}), 200

    # 返回 HTML 报告
    html_path = pred.get('report_html_path')
    if html_path and os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()
        return Response(html, mimetype='text/html')
    return jsonify(pred.get('report', {}))


# ==========================================
#  API: 批量分析 (Batch Analysis)
# ==========================================

@app.route('/api/batch/analyze', methods=['POST'])
def batch_analyze():
    """启动批量股票分析"""
    data = request.get_json(silent=True) or {}
    symbols_raw = data.get('symbols', '').strip()
    cost_prices_raw = data.get('cost_prices', '').strip()
    shares_raw = data.get('shares', '').strip()
    master = data.get('master', '').strip().lower()
    total_assets = data.get('total_assets', 0) or 0.0
    available_cash = data.get('available_cash', 0) or 0.0

    if not symbols_raw:
        return jsonify({'error': '请提供至少一只股票代码（多只以 / 分隔）'}), 400

    # 解析 / 分隔的输入
    symbols = [s.strip().upper() for s in symbols_raw.split('/') if s.strip()]
    cost_prices = [float(c.strip()) if c.strip() else 0.0 for c in cost_prices_raw.split('/')] if cost_prices_raw else []
    shares_list = [int(s.strip()) if s.strip() else 0 for s in shares_raw.split('/')] if shares_raw else []

    # 校验
    if cost_prices and len(cost_prices) != len(symbols):
        return jsonify({'error': f'成本价数量({len(cost_prices)})与股票数量({len(symbols)})不一致'}), 400
    if shares_list and len(shares_list) != len(symbols):
        return jsonify({'error': f'数量({len(shares_list)})与股票数量({len(symbols)})不一致'}), 400

    # 补齐缺失
    while len(cost_prices) < len(symbols):
        cost_prices.append(0.0)
    while len(shares_list) < len(symbols):
        shares_list.append(0)

    task_id = f"batch_{uuid.uuid4().hex[:12]}"

    task_data = {
        'task_id': task_id,
        'symbols': symbols,
        'cost_prices': cost_prices,
        'shares_list': shares_list,
        'total_assets': float(total_assets),
        'available_cash': float(available_cash),
        'master': master,
        'status': 'pending',
        'message': '',
        'progress': 0.0,
        'results': [],
        'summary': None,
        'quality_pick': None,
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
    }

    with _batch_lock:
        batch_tasks[task_id] = task_data

    logger.info(f"批量分析启动 task_id={task_id}, symbols={symbols}, master={master or 'off'}")

    def _run_batch():
        try:
            def _progress(event_type, event_data):
                if event_type == 'progress':
                    current = event_data.get('current', 0)
                    total = event_data.get('total', 1)
                    _update_batch(task_id, current / total,
                                 'running', event_data.get('message', ''),
                                 current_stock=event_data.get('symbol', ''))
                elif event_type == 'stock_result':
                    _update_batch(task_id, None, 'running',
                                 event_data.get('message', ''),
                                 add_result={
                                     'symbol': event_data.get('symbol', ''),
                                     'data': event_data.get('data', {}),
                                 })
                elif event_type == 'batch_summary':
                    _update_batch(task_id, 0.9, 'summarizing', '批量总结完成',
                                 summary=event_data.get('summary'),
                                 quality_pick=event_data.get('quality_pick'))
                elif event_type == 'completed':
                    _update_batch(task_id, 1.0, 'completed', event_data.get('message', ''))

            result = batch_analyzer.run_batch(
                symbols=symbols,
                cost_prices=cost_prices,
                shares_list=shares_list,
                total_assets=float(total_assets),
                available_cash=float(available_cash),
                master=master,
                progress_callback=_progress,
            )

            # 如果 completed 事件没发出来（降级路径）
            with _batch_lock:
                bt = batch_tasks.get(task_id)
                if bt and bt['status'] not in ('completed', 'failed'):
                    bt['status'] = 'completed'
                    bt['progress'] = 1.0
                    bt['message'] = '批量分析完成'
                    bt['summary'] = result.get('summary')
                    bt['quality_pick'] = result.get('quality_pick')
                    bt['completed_at'] = datetime.now().isoformat()

        except Exception as e:
            import traceback
            logger.error(f"批量分析失败: {e}\n{traceback.format_exc()}")
            _update_batch(task_id, 1.0, 'failed', f"批量分析异常: {str(e)}")

    thread = threading.Thread(target=_run_batch, daemon=True)
    thread.start()

    return jsonify({
        'task_id': task_id,
        'symbols': symbols,
        'status': 'queued',
    })


@app.route('/api/batch/analyze/<task_id>', methods=['GET'])
def batch_status(task_id: str):
    """查询批量分析任务状态"""
    with _batch_lock:
        bt = batch_tasks.get(task_id)
    if not bt:
        return jsonify({'error': '任务不存在'}), 404
    return jsonify({
        'task_id': bt['task_id'],
        'symbols': bt['symbols'],
        'status': bt['status'],
        'progress': bt['progress'],
        'message': bt['message'],
        'results': bt.get('results', []),
        'results_count': len(bt.get('results', [])),
        'total': len(bt.get('symbols', [])),
        'success_count': sum(1 for r in bt.get('results', []) if r.get('status') == 'complete'),
        'summary': bt.get('summary'),
        'quality_pick': bt.get('quality_pick'),
        'created_at': bt['created_at'],
        'completed_at': bt['completed_at'],
    })


@app.route('/api/batch/analyze/<task_id>/stream', methods=['GET'])
def batch_stream(task_id: str):
    """SSE 批量分析进度流"""
    def generate():
        last_progress = -1
        last_result_count = 0
        last_yield_time = time.time()
        while True:
            with _batch_lock:
                bt = batch_tasks.get(task_id)
            if not bt:
                yield f"data: {json.dumps({'type': 'error', 'message': '任务不存在'})}\n\n"
                break

            current_progress = bt.get('progress', 0)
            results = bt.get('results', [])
            current_result_count = len(results)

            yielded = False

            # 推送新完成的 stock_result
            if current_result_count > last_result_count:
                for r in results[last_result_count:]:
                    yield f"data: {json.dumps({'type': 'stock_result', 'symbol': r['symbol'], 'data': r['data']}, ensure_ascii=False)}\n\n"
                last_result_count = current_result_count
                yielded = True

            # 推送 progress
            if current_progress != last_progress:
                msg = bt.get('message', '')
                current_stock = bt.get('current_stock', '')
                yield f"data: {json.dumps({'type': 'progress', 'progress': current_progress, 'message': msg, 'current_stock': current_stock}, ensure_ascii=False)}\n\n"
                last_progress = current_progress
                yielded = True

            # 终端状态推送 summary + quality_pick
            if bt['status'] in ('completed', 'failed'):
                if bt['status'] == 'completed':
                    summary = bt.get('summary')
                    quality_pick = bt.get('quality_pick')
                    if summary or quality_pick:
                        yield f"data: {json.dumps({'type': 'batch_summary', 'summary': summary, 'quality_pick': quality_pick}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': bt['status'], 'message': bt.get('message', '')})}\n\n"
                break

            if yielded:
                last_yield_time = time.time()
            elif time.time() - last_yield_time > 15:
                yield ": heartbeat\n\n"
                last_yield_time = time.time()

            time.sleep(1)

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


def _update_batch(task_id, progress, status, message, **kwargs):
    """更新批量分析任务状态（线程安全）"""
    with _batch_lock:
        bt = batch_tasks.get(task_id)
        if not bt:
            return
        if progress is not None:
            bt['progress'] = progress
        bt['status'] = status
        bt['message'] = message
        # 追加结果
        add_result = kwargs.pop('add_result', None)
        if add_result:
            bt.setdefault('results', []).append(add_result)
        for k, v in kwargs.items():
            bt[k] = v
        if status in ('completed', 'failed'):
            bt['completed_at'] = datetime.now().isoformat()


def _update_qlib(task_id, progress, status, message, **kwargs):
    """更新 qlib 推理任务状态（线程安全）"""
    with _qlib_lock:
        qt = qlib_tasks.get(task_id)
        if not qt:
            return
        if progress is not None:
            qt['progress'] = progress
        qt['status'] = status
        qt['message'] = message
        for k, v in kwargs.items():
            qt[k] = v
        if status in ('completed', 'failed'):
            qt['completed_at'] = datetime.now().isoformat()


# ==========================================
#  API: Qlib 推理
# ==========================================

@app.route('/api/qlib/models', methods=['GET'])
def qlib_models():
    """列出所有可用模型（扫描 DATA/analysis_outputs/）"""
    scan_dir = _qlib_base_dir / "DATA" / "analysis_outputs"
    models = []

    if scan_dir.exists():
        for d in sorted(scan_dir.iterdir()):
            if not d.is_dir():
                continue
            name = d.name

            market = "unknown"
            if "csi300" in name.lower():
                market = "csi300"
            elif "csi1000" in name.lower():
                market = "csi1000"

            date_part = name[:10] if len(name) >= 10 and name[4] == "-" else ""
            has_scores = (d / "model_predict" / "scores.csv").exists()
            is_finetune = "fintune" in name.lower()

            models.append({
                "name": name,
                "market": market,
                "date": date_part,
                "has_scores": has_scores,
                "is_finetune": is_finetune,
                "in_analysis_outputs": True,  # 始终为 True
            })

    return jsonify(models)


@app.route('/api/qlib/train-targets', methods=['GET'])
def qlib_train_targets():
    """返回可用的训练目标配置（用于训练面板的模型选择器）"""
    targets = [
        {
            "value": "csi300-alpha158",
            "label": "沪深300 Alpha158",
            "market": "csi300",
            "benchmark": "SH000300",
            "description": "沪深300成分股 + Alpha158因子 + LightGBM walk-forward全量训练（约20-40分钟）"
        },
    ]
    return jsonify(targets)


@app.route('/api/qlib/infer', methods=['POST'])
def qlib_infer():
    """启动 qlib 推理任务"""
    data = request.get_json(silent=True) or {}
    model = data.get('model', '').strip()
    holdings = data.get('holdings', '').strip()

    if not model:
        return jsonify({'error': '请选择模型'}), 400

    models_dir = _qlib_base_dir / "DATA" / "analysis_outputs"
    if not (models_dir / model).exists():
        return jsonify({'error': f'模型不存在: {model}'}), 400

    task_id = f"qlib_{uuid.uuid4().hex[:12]}"

    task_data = {
        'task_id': task_id,
        'model': model,
        'status': 'pending',
        'message': '',
        'progress': 0.0,
        'stocks': '',
        'count': 0,
        'scores': [],
        'pred_date': '',
        'error': '',
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
    }

    with _qlib_lock:
        qlib_tasks[task_id] = task_data

    logger.info(f"Qlib 推理启动 task_id={task_id}, model={model}")

    def _run():
        try:
            def _progress(event_data):
                status = event_data.get('status', 'running')
                message = event_data.get('message', '')
                if status == 'completed':
                    _update_qlib(task_id, 1.0, 'completed', message,
                                 stocks=event_data.get('stocks', ''),
                                 count=event_data.get('count', 0),
                                 scores=event_data.get('scores', []),
                                 pred_date=event_data.get('pred_date', ''),
                                 strategy_b=event_data.get('strategy_b', {}))
                else:
                    progress = 0.5 if '推理' in message else 0.1
                    _update_qlib(task_id, progress, 'running', message)

            result = run_qlib_inference(model, top_n=20, progress_callback=_progress, holdings=holdings)

            # 确保完成状态
            with _qlib_lock:
                qt = qlib_tasks.get(task_id)
                if qt and qt['status'] not in ('completed', 'failed'):
                    qt['status'] = 'completed'
                    qt['progress'] = 1.0
                    qt['stocks'] = result.get('stocks', '')
                    qt['count'] = result.get('count', 0)
                    qt['scores'] = result.get('scores', [])
                    qt['pred_date'] = result.get('pred_date', '')
                    qt['message'] = f"完成 — 已选出 {result.get('count', 0)} 只股票"

        except Exception as e:
            logger.error(f"Qlib 推理失败 task_id={task_id}: {e}")
            _update_qlib(task_id, 0.0, 'failed', str(e), error=str(e))

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return jsonify({'task_id': task_id, 'status': 'pending'})


@app.route('/api/qlib/infer/<task_id>/stream', methods=['GET'])
def qlib_infer_stream(task_id):
    """SSE 流 — qlib 推理进度"""
    def generate():
        last_progress = -1
        last_message = ''
        last_yield_time = time.time()
        while True:
            with _qlib_lock:
                qt = qlib_tasks.get(task_id)
            if not qt:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break

            data = {
                'status': qt.get('status'),
                'progress': qt.get('progress', 0),
                'message': qt.get('message', ''),
            }
            current_progress = qt.get('progress', 0)

            if qt.get('status') == 'completed':
                data['stocks'] = qt.get('stocks', '')
                data['count'] = qt.get('count', 0)
                data['pred_date'] = qt.get('pred_date', '')
                data['strategy_b'] = qt.get('strategy_b', {})
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break

            if qt.get('status') == 'failed':
                data['error'] = qt.get('error', '未知错误')
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break

            current_message = qt.get('message', '')
            if current_progress != last_progress or current_message != last_message:
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                last_progress = current_progress
                last_message = current_message
                last_yield_time = time.time()
            elif time.time() - last_yield_time > 15:
                yield ": heartbeat\n\n"
                last_yield_time = time.time()

            time.sleep(1)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


# ---- 指数成分股缓存 ----
_index_stocks_cache = {}
_index_stocks_cache_time = {}


@app.route('/api/qlib/index-stocks', methods=['GET'])
def qlib_index_stocks():
    """返回指数成分股列表。从 ~/.qlib/qlib_data/cn_data/instruments/ 读取"""
    index_name = request.args.get('index', 'csi300').strip().lower()
    exclude_star = request.args.get('exclude_star', 'false').strip().lower() == 'true'

    if index_name not in ('csi300', 'csi500', 'csi1000'):
        return jsonify({'error': f'不支持的指数: {index_name}，支持 csi300/csi500/csi1000'}), 400

    # 缓存 1 小时
    cache_key = f"{index_name}_{exclude_star}"
    now = time.time()
    if cache_key in _index_stocks_cache and (now - _index_stocks_cache_time.get(cache_key, 0)) < 3600:
        return jsonify(_index_stocks_cache[cache_key])

    # 读取 qlib 数据中的成分股文件
    inst_file = Path.home() / ".qlib" / "qlib_data" / "cn_data" / "instruments" / f"{index_name}.txt"
    if not inst_file.exists():
        return jsonify({'error': f'成分股文件不存在: {inst_file}'}), 404

    # 解析：instrument start_date end_date
    date_groups = {}
    for line in inst_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) < 3:
            continue
        inst, start, end = parts[0], parts[1], parts[2]
        date_groups.setdefault(end, []).append(inst)

    # 取最大 end_date 作为当前成分股
    if not date_groups:
        return jsonify({'stocks': '', 'count': 0})

    max_date = max(date_groups.keys())
    stocks = sorted(set(date_groups[max_date]))

    # 转换 instrument 为纯代码: SZ000001 → 000001
    codes = []
    for inst in stocks:
        code = inst[2:] if inst.startswith(('SZ', 'SH', 'BJ')) else inst
        # 剔除科创板 (688xxx)
        if exclude_star and code.startswith('688'):
            continue
        codes.append(code)

    result = {
        'stocks': '/'.join(codes),
        'count': len(codes),
        'index': index_name,
        'date': max_date,
        'exclude_star': exclude_star,
    }

    _index_stocks_cache[cache_key] = result
    _index_stocks_cache_time[cache_key] = now

    return jsonify(result)


# ==========================================
#  API: Qlib 数据更新
# ==========================================

@app.route('/api/qlib/data/update', methods=['POST'])
def qlib_data_update():
    """启动 qlib 数据下载更新任务"""
    task_id = f"qdata_{uuid.uuid4().hex[:12]}"
    task_data = {
        'task_id': task_id,
        'status': 'pending',
        'message': '',
        'progress': 0.0,
        'error': '',
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
    }
    with _qlib_data_lock:
        qlib_data_tasks[task_id] = task_data

    logger.info(f"Qlib 数据更新启动 task_id={task_id}")

    def _run():
        try:
            def _progress(event_data):
                with _qlib_data_lock:
                    qt = qlib_data_tasks.get(task_id)
                    if not qt:
                        return
                    status = event_data.get('status', 'running')
                    msg = event_data.get('message', '')
                    progress = event_data.get('progress')
                    if progress is not None:
                        qt['progress'] = progress
                    qt['status'] = status
                    qt['message'] = msg
                    if status in ('completed', 'failed'):
                        qt['completed_at'] = datetime.now().isoformat()

            result = run_qlib_data_update(progress_callback=_progress)

            with _qlib_data_lock:
                qt = qlib_data_tasks.get(task_id)
                if qt and qt['status'] not in ('completed', 'failed'):
                    qt['status'] = 'completed'
                    qt['progress'] = 1.0
                    qt['message'] = result.get('message', '数据更新完成')

        except Exception as e:
            logger.error(f"Qlib 数据更新失败 task_id={task_id}: {e}")
            with _qlib_data_lock:
                qt = qlib_data_tasks.get(task_id)
                if qt:
                    qt['status'] = 'failed'
                    qt['error'] = str(e)
                    qt['message'] = f'数据更新失败: {e}'

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({'task_id': task_id, 'status': 'pending'})


@app.route('/api/qlib/data/update/<task_id>/stream', methods=['GET'])
def qlib_data_update_stream(task_id):
    """SSE 流 — qlib 数据更新进度"""
    def generate():
        last_progress = -1
        last_message = ''
        last_yield_time = time.time()
        while True:
            with _qlib_data_lock:
                qt = qlib_data_tasks.get(task_id)
            if not qt:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break

            data = {
                'status': qt.get('status'),
                'progress': qt.get('progress', 0),
                'message': qt.get('message', ''),
            }
            current_progress = qt.get('progress', 0)
            current_message = qt.get('message', '')

            if qt.get('status') == 'completed':
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break

            if qt.get('status') == 'failed':
                data['error'] = qt.get('error', '未知错误')
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break

            if current_progress != last_progress or current_message != last_message:
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                last_progress = current_progress
                last_message = current_message
                last_yield_time = time.time()
            elif time.time() - last_yield_time > 15:
                yield ": heartbeat\n\n"
                last_yield_time = time.time()

            time.sleep(1)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


# ==========================================
#  API: Qlib 模型训练
# ==========================================

@app.route('/api/qlib/train', methods=['POST'])
def qlib_train():
    """启动 qlib 模型训练任务"""
    data = request.get_json(silent=True) or {}
    market = data.get('market', 'csi300')
    target = data.get('target', '').strip()
    # 如果前端传了 target，从 target 解析 market（如 csi300-alpha158 → csi300）
    if target:
        target_lower = target.lower()
        if 'csi300' in target_lower:
            market = 'csi300'
        elif 'csi1000' in target_lower:
            market = 'csi1000'
    model_mode = data.get('model_mode', 'robust')
    hold_num = int(data.get('hold_num', 20))

    if market not in ('csi300', 'csi1000'):
        return jsonify({'error': f'不支持的市场: {market}'}), 400
    if model_mode not in ('default', 'robust'):
        return jsonify({'error': f'不支持的模型模式: {model_mode}'}), 400

    task_id = f"qtrain_{uuid.uuid4().hex[:12]}"
    task_data = {
        'task_id': task_id,
        'status': 'pending',
        'message': '',
        'progress': 0.0,
        'model_name': '',
        'error': '',
        'backtest_metrics': {},
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
    }
    with _qlib_train_lock:
        qlib_train_tasks[task_id] = task_data

    logger.info(f"Qlib 训练启动 task_id={task_id}, market={market}")

    def _run():
        try:
            def _progress(event_data):
                with _qlib_train_lock:
                    qt = qlib_train_tasks.get(task_id)
                    if not qt:
                        return
                    status = event_data.get('status', 'running')
                    msg = event_data.get('message', '')
                    progress = event_data.get('progress')
                    if progress is not None:
                        qt['progress'] = progress
                    qt['status'] = status
                    qt['message'] = msg
                    if status in ('completed', 'failed'):
                        qt['completed_at'] = datetime.now().isoformat()

            result = run_qlib_training(
                market=market,
                model_mode=model_mode,
                hold_num=hold_num,
                lightgbm_only=True,
                progress_callback=_progress,
            )

            with _qlib_train_lock:
                qt = qlib_train_tasks.get(task_id)
                if qt and qt['status'] not in ('completed', 'failed'):
                    qt['status'] = 'completed'
                    qt['progress'] = 1.0
                    qt['model_name'] = result.get('model_name', '')
                    qt['message'] = result.get('message', '训练完成')
                    qt['backtest_metrics'] = result.get('backtest_metrics', {})

        except Exception as e:
            logger.error(f"Qlib 训练失败 task_id={task_id}: {e}")
            with _qlib_train_lock:
                qt = qlib_train_tasks.get(task_id)
                if qt:
                    qt['status'] = 'failed'
                    qt['error'] = str(e)
                    qt['message'] = f'训练失败: {e}'

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return jsonify({'task_id': task_id, 'status': 'pending'})


@app.route('/api/qlib/train/<task_id>/stream', methods=['GET'])
def qlib_train_stream(task_id):
    """SSE 流 — qlib 训练进度"""
    def generate():
        last_progress = -1
        last_message = ''
        last_yield_time = time.time()
        while True:
            with _qlib_train_lock:
                qt = qlib_train_tasks.get(task_id)
            if not qt:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break

            data = {
                'status': qt.get('status'),
                'progress': qt.get('progress', 0),
                'message': qt.get('message', ''),
            }
            current_progress = qt.get('progress', 0)
            current_message = qt.get('message', '')

            if qt.get('status') == 'completed':
                data['model_name'] = qt.get('model_name', '')
                data['backtest_metrics'] = qt.get('backtest_metrics', {})
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break

            if qt.get('status') == 'failed':
                data['error'] = qt.get('error', '未知错误')
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break

            if current_progress != last_progress or current_message != last_message:
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                last_progress = current_progress
                last_message = current_message
                last_yield_time = time.time()
            elif time.time() - last_yield_time > 15:
                yield ": heartbeat\n\n"
                last_yield_time = time.time()

            time.sleep(1)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


# ==========================================
#  API: Qlib 模型微调
# ==========================================

@app.route('/api/qlib/finetune', methods=['POST'])
def qlib_finetune():
    """启动 qlib 模型微调任务"""
    data = request.get_json(silent=True) or {}
    base_model = data.get('base_model', '').strip()

    if not base_model:
        return jsonify({'error': '请选择基础模型'}), 400

    # 验证基础模型存在
    base_model_dir = _qlib_base_dir / "DATA" / "analysis_outputs" / base_model
    if not base_model_dir.exists():
        base_model_dir = None

    if not base_model_dir:
        return jsonify({'error': f'基础模型不存在: {base_model}'}), 400

    model_name = f"{base_model}-fintune"
    task_id = f"qft_{uuid.uuid4().hex[:12]}"
    task_data = {
        'task_id': task_id,
        'status': 'pending',
        'message': '',
        'progress': 0.0,
        'model_name': model_name,
        'error': '',
        'backtest_metrics': {},
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
    }
    with _qlib_finetune_lock:
        qlib_finetune_tasks[task_id] = task_data

    logger.info(f"Qlib 微调启动 task_id={task_id}, base_model={base_model}")

    def _run():
        try:
            docker_base_dir = f"/work/DATA/analysis_outputs/{base_model}"
            cfg = _resolve_model_config(base_model)
            template_docker = cfg["template"]
            benchmark = cfg["benchmark"]

            def _log(msg: str, **extra):
                with _qlib_finetune_lock:
                    qt = qlib_finetune_tasks.get(task_id)
                    if not qt:
                        return
                    status = extra.get('status', 'running')
                    progress = extra.get('progress')
                    if progress is not None:
                        qt['progress'] = progress
                    qt['status'] = status
                    qt['message'] = msg

            _log(f"基础模型: {base_model}")
            _log(f"输出名称: {model_name}")
            _log(f"开始 Docker 微调...")

            # 构建 Docker 命令
            # 使用 docker-compose 传递的宿主机路径环境变量（兼容 Docker 内运行）
            _stockfish_root = Path(__file__).resolve().parent
            host_project_root = os.environ.get("QLIB_HOST_PROJECT_ROOT", str(_stockfish_root / "qlib-zh"))
            host_qlib_data = os.environ.get("QLIB_HOST_DATA_DIR", str(Path.home() / ".qlib"))
            host_mlruns = os.environ.get("QLIB_HOST_MLRUNS_DIR", str(Path.home() / "github" / "qlib-zh" / "mlruns"))
            docker_image = "zhuhai123/qlib-rdagent:v1"
            workdir = "/work"

            output_root_ctr = f"{workdir}/DATA/analysis_outputs/{model_name}"
            predict_out_ctr = f"{output_root_ctr}/model_predict"

            cmd = [
                "docker", "run", "--rm",
                "--memory", "12g",
                "--memory-swap", "12g",
                "-e", "QLIB_DATA_DIR=/root/.qlib/qlib_data/cn_data",
                "-e", f"TARGET_MARKET={cfg['market']}",
                "-e", f"TARGET_BENCHMARK={benchmark}",
                "-e", "CASH_TOTAL=100000",
                "-e", "PYTHONUNBUFFERED=1",
                "-e", "OMP_NUM_THREADS=8",
                "-v", f"{host_project_root}:{workdir}",
                "-v", f"{host_qlib_data}:/root/.qlib",
                "-v", f"{host_mlruns}:{workdir}/mlruns",
                "-w", workdir,
                docker_image,
                "python3", "scripts/finetune_alpha158.py",
                "--base-model-dir", docker_base_dir,
                "--output-name", model_name,
                "--template", template_docker,
                "--output-root", output_root_ctr,
                "--experiment-name", model_name,
                "--train-years", "5",
                "--valid-years", "1",
                "--hold-num", "20",
                "--model-mode", "robust",
            ]

            import subprocess as _subprocess
            process = _subprocess.Popen(
                cmd,
                stdout=_subprocess.PIPE,
                stderr=_subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    _log(f"[Docker] {line[:300]}")

            process.wait(timeout=3600)  # 1小时超时
            if process.returncode != 0:
                raise RuntimeError(f"Docker 退出码: {process.returncode}")

            # 提取回测指标
            summary_path = _qlib_base_dir / "DATA" / "analysis_outputs" / model_name / "finetune_summary.json"
            bt_metrics = {}
            if summary_path.exists():
                try:
                    import json as _json
                    summary = _json.loads(summary_path.read_text(encoding="utf-8"))
                    bt_metrics = summary.get("backtest", {})
                except Exception:
                    pass

            msg = f"微调完成: {model_name}"
            if bt_metrics.get("sharpe_ratio") is not None:
                msg += f" | 夏普比: {bt_metrics['sharpe_ratio']}"
            _log(msg, progress=1.0, status="completed")

            with _qlib_finetune_lock:
                qt = qlib_finetune_tasks.get(task_id)
                if qt and qt['status'] not in ('completed', 'failed'):
                    qt['status'] = 'completed'
                    qt['progress'] = 1.0
                    qt['model_name'] = model_name
                    qt['message'] = msg
                    qt['backtest_metrics'] = bt_metrics

        except Exception as e:
            logger.error(f"Qlib 微调失败 task_id={task_id}: {e}")
            with _qlib_finetune_lock:
                qt = qlib_finetune_tasks.get(task_id)
                if qt:
                    qt['status'] = 'failed'
                    qt['error'] = str(e)
                    qt['message'] = f'微调失败: {e}'

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return jsonify({'task_id': task_id, 'status': 'pending', 'model_name': model_name})


@app.route('/api/qlib/finetune/<task_id>/stream', methods=['GET'])
def qlib_finetune_stream(task_id):
    """SSE 流 — qlib 微调进度"""
    def generate():
        last_progress = -1
        last_message = ''
        last_yield_time = time.time()
        while True:
            with _qlib_finetune_lock:
                qt = qlib_finetune_tasks.get(task_id)
            if not qt:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break

            data = {
                'status': qt.get('status'),
                'progress': qt.get('progress', 0),
                'message': qt.get('message', ''),
            }
            current_progress = qt.get('progress', 0)
            current_message = qt.get('message', '')

            if qt.get('status') == 'completed':
                data['model_name'] = qt.get('model_name', '')
                data['backtest_metrics'] = qt.get('backtest_metrics', {})
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break

            if qt.get('status') == 'failed':
                data['error'] = qt.get('error', '未知错误')
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                break

            if current_progress != last_progress or current_message != last_message:
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                last_progress = current_progress
                last_message = current_message
                last_yield_time = time.time()
            elif time.time() - last_yield_time > 15:
                yield ": heartbeat\n\n"
                last_yield_time = time.time()

            time.sleep(1)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


def _resolve_model_config(model_name: str) -> dict:
    """解析模型名称对应的市场配置（用于微调）"""
    name_lower = model_name.lower()
    if "csi1000" in name_lower:
        return {
            "market": "csi1000",
            "benchmark": "SH000852",
            "template": "/work/scripts/small/templates/workflow_config_lightgbm_Alpha158_csi1000.yaml",
        }
    else:
        return {
            "market": "csi300",
            "benchmark": "SH000300",
            "template": "/work/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml",
        }


# ==========================================
#  API: 分析报告下载
# ==========================================

@app.route('/api/report/download', methods=['POST'])
def download_analysis_report():
    """接收分析结果 JSON，生成 HTML 报告并返回下载"""
    data = request.get_json(silent=True) or {}
    if not data or not data.get('symbol'):
        return jsonify({'error': '请提供分析结果数据'}), 400

    symbol = data.get('symbol', 'unknown')
    try:
        report = report_gen.generate(data, simulation_result=None)
        html = report_gen.to_html(report)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        response = Response(html, mimetype='text/html')
        response.headers['Content-Disposition'] = (
            f'attachment; filename="{symbol}_analysis_{timestamp}.html"'
        )
        return response
    except Exception as e:
        logger.error(f"生成分析报告失败 [{symbol}]: {e}")
        return jsonify({'error': f'报告生成失败: {str(e)}'}), 500


# ==========================================
#  API: 系统
# ==========================================

@app.route('/api/config', methods=['GET', 'POST'])
def config():
    if request.method == 'GET':
        return jsonify({
            'backend': agent.provider.backend_name,
            'tushare_token_configured': bool(os.environ.get('TUSHARE_TOKEN')),
            'llm_configured': bool(agent.prediction_node.api_key),
            'llm_model': agent.prediction_node.model,
            'mirofish_available': orchestrator.client.health_check(),
            'mirofish_url': orchestrator.client.base_url,
        })
    data = request.get_json(silent=True) or {}
    if 'backend' in data:
        new_backend = data['backend']
        agent.provider._backend = None
        agent.provider._requested_backend = new_backend
    return jsonify({'status': 'ok'})


@app.route('/api/masters', methods=['GET'])
def list_masters():
    """返回可用的大师决策者列表"""
    from analysis.agents.cio_prompts import list_masters as get_masters
    return jsonify({'masters': get_masters()})


@app.route('/api/predictions', methods=['GET'])
def list_predictions():
    with _predictions_lock:
        items = [
            {
                'task_id': p['task_id'],
                'symbol': p['symbol'],
                'scenario': p['scenario'],
                'status': p['status'],
                'progress': p['progress'],
                'created_at': p['created_at'],
            }
            for p in predictions.values()
        ]
    return jsonify(items)


@app.route('/')
def index():
    with open(os.path.join(app.static_folder, 'index.html'), encoding='utf-8') as f:
        return f.read()


# ==========================================
#  工具
# ==========================================

def _update_prediction(task_id, progress, status, message, **kwargs):
    with _predictions_lock:
        pred = predictions.get(task_id)
        if not pred:
            return
        pred['progress'] = progress
        pred['status'] = status
        pred['message'] = message
        for k, v in kwargs.items():
            pred[k] = v
        if status in ('completed', 'failed'):
            pred['completed_at'] = datetime.now().isoformat()


# ==========================================
#  启动
# ==========================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="StockFish v2")
    parser.add_argument(
        "--with-bot",
        action="store_true",
        help="在后台线程启动飞书 Bot（需要 LARK_APP_ID/LARK_APP_SECRET）",
    )
    args = parser.parse_args()

    if args.with_bot:
        import threading

        from integration.lark_bot import StockFishBot

        def _run_bot():
            bot = StockFishBot()
            bot.start()  # 阻塞，在后台线程运行 WebSocket

        t = threading.Thread(target=_run_bot, daemon=True, name="lark-bot")
        t.start()
        logger.info("飞书 Bot 已启动（后台线程）")

    logger.info(f"StockFish v2 启动: http://{settings.HOST}:{settings.PORT}")
    app.run(host=settings.HOST, port=settings.PORT, debug=settings.DEBUG, threaded=True)
