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
from simulation_bridge.orchestrator import SimulationOrchestrator
from prediction_report.report_generator import PredictionReportGenerator

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# ===== 全局状态 =====
agent = StockAnalysisAgent()  # 从 STOCK_BACKEND env 自动读取
# 初始化后端（触发自动检测）
_ = agent.provider.backend
orchestrator = SimulationOrchestrator()
report_gen = PredictionReportGenerator()
predictions = {}
_predictions_lock = threading.Lock()


# ==========================================
#  API: 分析 (Phase 2 - StockEngine Agent)
# ==========================================

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """完整多因子分析"""
    data = request.get_json(silent=True) or {}
    symbol = data.get('symbol', '').strip().upper()
    cost_price = data.get('cost_price', 0)
    if not symbol:
        return jsonify({'error': '请提供股票代码'}), 400

    logger.info(f"开始深度分析 [{symbol}] 成本价={cost_price}")
    result = agent.analyze(symbol, cost_price=float(cost_price) if cost_price else 0.0)
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

    if not symbol:
        return jsonify({'error': '请提供股票代码'}), 400

    task_id = f"pred_{uuid.uuid4().hex[:12]}"

    pred_data = {
        'task_id': task_id,
        'symbol': symbol,
        'scenario': scenario,
        'cost_price': float(cost_price) if cost_price else 0.0,
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
            result = agent.analyze(symbol, cost_price=pred_data.get('cost_price', 0))
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
            _update_prediction(task_id, 0.8, 'simulating', '模拟推演完成', simulation=sim_result)

            # Step 3: 生成报告
            _update_prediction(task_id, 0.9, 'generating_report', '生成预测报告...')
            report = report_gen.generate(result, sim_result)
            html_path = report_gen.save(report)

            _update_prediction(task_id, 1.0, 'completed', '推演完成', report=report, report_html_path=html_path)

        except Exception as e:
            logger.error(f"[{symbol}] 推演失败: {e}")
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
    logger.info(f"StockFish v2 启动: http://{settings.HOST}:{settings.PORT}")
    app.run(host=settings.HOST, port=settings.PORT, debug=settings.DEBUG, threaded=True)
