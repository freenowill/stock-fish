"""
Simulation Bridge 编排器

将 StockEngine 分析结果桥接到 MiroFish OASIS 模拟。
通过 HTTP API 调用 MiroFish（支持 Docker 跨容器通信）。
"""
import json
import os
import sys
import time
import uuid
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable

import requests
from loguru import logger

from config import settings
from simulation_bridge.seed_builder import SeedDocumentBuilder


class MiroFishHTTPClient:
    """MiroFish HTTP API 客户端"""

    def __init__(self, host: str = "localhost", port: int = 5001):
        self.base_url = f"http://{host}:{port}/api"
        self.session = requests.Session()
        self.session.timeout = (10, 120)  # (connect, read)

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _get(self, path: str, **kwargs):
        return self.session.get(self._url(path), **kwargs)

    def _post(self, path: str, **kwargs):
        return self.session.post(self._url(path), **kwargs)

    def _safe_data(self, result, endpoint: str = "") -> dict:
        """安全提取 API 响应中的 data 字段，自动处理非预期格式"""
        if isinstance(result, str):
            raise RuntimeError(f"{endpoint} 返回非预期格式(字符串): {result[:200]}")
        if not isinstance(result, dict):
            raise RuntimeError(f"{endpoint} 返回非预期类型: {type(result).__name__}")
        if not result.get('success'):
            raise RuntimeError(result.get('error', f'{endpoint} 返回失败'))
        data = result.get('data', {})
        if isinstance(data, str):
            raise RuntimeError(f"{endpoint} data 字段为非预期格式(字符串): {data[:200]}")
        if not isinstance(data, dict):
            raise RuntimeError(f"{endpoint} data 字段为非预期类型: {type(data).__name__}")
        return data

    def health_check(self) -> bool:
        try:
            r = self._get("/graph/project/list", timeout=5)
            return r.status_code < 500
        except Exception as e:
            logger.debug(f"MiroFish 不可达: {e}")
            return False

    # ---------- 图谱 ----------

    def generate_ontology(self, seed_text: str, simulation_requirement: str,
                          project_name: str = "StockFish Prediction") -> Dict:
        """上传种子文档 → 生成本体"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         encoding='utf-8', delete=False) as f:
            f.write(seed_text)
            tmp_path = f.name

        try:
            with open(tmp_path, 'rb') as f:
                files = {'files': (f'seed_{uuid.uuid4().hex[:8]}.txt', f, 'text/plain')}
                data = {
                    'simulation_requirement': simulation_requirement,
                    'project_name': project_name,
                }
                r = self._post("/graph/ontology/generate", files=files, data=data)
            return self._safe_data(r.json(), "本体生成")
        finally:
            os.unlink(tmp_path)

    def build_graph(self, project_id: str) -> Dict:
        """构建知识图谱（异步）"""
        r = self._post("/graph/build", json={"project_id": project_id})
        return self._safe_data(r.json(), "图谱构建")

    def poll_graph_task(self, task_id: str, timeout: int) -> Dict:
        """轮询图谱构建任务"""
        start = time.time()
        while time.time() - start < timeout:
            r = self._get(f"/graph/task/{task_id}")
            data = self._safe_data(r.json(), "图谱任务查询")
            status = data.get('status', '')
            if status == 'completed':
                return data
            elif status == 'failed':
                raise RuntimeError(data.get('error', '图谱构建失败'))
            time.sleep(3)
        raise TimeoutError("图谱构建超时")

    # ---------- 模拟 ----------

    def create_simulation(self, project_id: str, graph_id: str) -> Dict:
        """创建模拟"""
        r = self._post("/simulation/create", json={
            "project_id": project_id,
            "graph_id": graph_id,
            "enable_twitter": True,
            "enable_reddit": True,
        })
        return self._safe_data(r.json(), "模拟创建")

    def prepare_simulation(self, simulation_id: str, entity_types: list = None) -> Dict:
        """准备模拟（实体→配置→Agent 画像）"""
        if entity_types is None:
            entity_types = ["Investor", "Analyst", "Media", "Company", "Regulator"]
        r = self._post("/simulation/prepare", json={
            "simulation_id": simulation_id,
            "entity_types": entity_types,
            "use_llm_for_profiles": True,
            "parallel_profile_count": 5,
        })
        return self._safe_data(r.json(), "模拟准备")

    def poll_prepare_status(self, simulation_id: str, timeout: int) -> Dict:
        """轮询模拟准备状态"""
        start = time.time()
        while time.time() - start < timeout:
            r = self._post("/simulation/prepare/status", json={
                "simulation_id": simulation_id
            })
            data = self._safe_data(r.json(), "模拟准备状态")
            status = data.get('status', '')
            if status in ('completed', 'ready'):
                return data
            elif status == 'failed':
                raise RuntimeError(data.get('error', '模拟准备失败'))
            time.sleep(5)
        raise TimeoutError("模拟准备超时")

    def start_simulation(self, simulation_id: str, max_rounds: int = 20) -> Dict:
        """启动 OASIS 模拟"""
        r = self._post("/simulation/start", json={
            "simulation_id": simulation_id,
            "platform": "parallel",
            "max_rounds": max_rounds,
        })
        return self._safe_data(r.json(), "模拟启动")

    def poll_simulation_status(self, simulation_id: str, timeout: int) -> Dict:
        """轮询模拟运行状态"""
        start = time.time()
        while time.time() - start < timeout:
            r = self._get(f"/simulation/{simulation_id}/run-status")
            data = self._safe_data(r.json(), "模拟运行状态")
            status = data.get('runner_status', data.get('status', ''))
            progress = data.get('progress', 0)
            if status == 'completed':
                return data
            elif status in ('failed', 'stopped'):
                raise RuntimeError(f"模拟终止: {data.get('error', '')}")
            time.sleep(5)
        raise TimeoutError("模拟运行超时")

    # ---------- 报告 ----------

    def generate_report(self, simulation_id: str) -> Dict:
        """生成预测报告"""
        r = self._post("/report/generate", json={
            "simulation_id": simulation_id,
        })
        return self._safe_data(r.json(), "报告生成")

    def poll_report_status(self, task_id: str, timeout: Optional[int] = None) -> Dict:
        """轮询报告生成状态（timeout=None 表示无超时限制）"""
        start = time.time()
        while True:
            if timeout and time.time() - start > timeout:
                raise TimeoutError("报告生成超时")
            r = self._post("/report/generate/status", json={"task_id": task_id})
            data = self._safe_data(r.json(), "报告状态轮询")
            status = data.get('status', '')
            if status == 'completed':
                return data
            elif status == 'failed':
                raise RuntimeError(data.get('error', '报告生成失败'))
            time.sleep(3)

    def get_report(self, report_id: str) -> Dict:
        """获取报告内容"""
        r = self._get(f"/report/{report_id}")
        return self._safe_data(r.json(), "获取报告")


class SimulationOrchestrator:
    """
    模拟编排器：StockFish 分析 → MiroFish OASIS 推演 → 预测报告
    """

    def __init__(self):
        host = os.environ.get('MIROFISH_HOST') or getattr(settings, 'MIROFISH_HOST', 'localhost')
        port = int(os.environ.get('MIROFISH_PORT') or getattr(settings, 'MIROFISH_PORT', 5001))
        self.client = MiroFishHTTPClient(host=host, port=port)
        self.output_dir = os.path.join(
            os.path.dirname(__file__), '..', 'simulation_output'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        # debug 模式：2 Agent / 2轮
        self.debug = bool(os.environ.get('OASIS_DEBUG', '').lower() in ('true', '1', 'yes')) or getattr(settings, 'OASIS_DEBUG', False)

    def orchestrate(self, analysis_result: Dict[str, Any],
                    scenario: str = "base",
                    progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        执行一次完整的模拟推演编排

        返回:
            {
                'status': 'standalone' | 'simulated' | 'failed',
                'seed_text': str,
                'scenarios': [...],
                'simulation_id': str | None,
                'report': dict | None,
            }
        """
        symbol = analysis_result.get('symbol', '')
        name = analysis_result.get('stock_name', '')
        result = {
            'symbol': symbol, 'name': name, 'scenario': scenario,
            'status': 'pending', 'seed_text': '', 'scenarios': [],
            'simulation_id': None, 'report': None,
        }

        # Step 1: 构建种子文档
        seed_text = SeedDocumentBuilder.build(analysis_result)
        result['seed_text'] = seed_text
        scenarios = SeedDocumentBuilder.build_scenario_scenarios(analysis_result, debug=self.debug)
        result['scenarios'] = scenarios

        # 保存到文件
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(self.output_dir, f"{symbol}_seed_{ts}.txt"),
                  'w', encoding='utf-8') as f:
            f.write(seed_text)
        with open(os.path.join(self.output_dir, f"{symbol}_scenarios_{ts}.json"),
                  'w', encoding='utf-8') as f:
            json.dump(scenarios, f, ensure_ascii=False, indent=2)

        # Step 2: 检查 MiroFish 是否可达
        if not self.client.health_check():
            logger.info(f"MiroFish 不可达（{self.client.base_url}），使用 Standalone 模式")
            result['status'] = 'standalone'
            return result

        if progress_callback:
            progress_callback(0.2, "连接 MiroFish 成功，上传种子文档...")

        try:
            # Step 3: 上传种子文档 → 生成本体
            sim_req = f"Predict the stock price movement of {name}({symbol}) using multi-agent simulation. Scenario: {scenario}"
            project_name = f"StockFish-{symbol}-{ts}"
            ont_data = self.client.generate_ontology(seed_text, sim_req, project_name)
            project_id = ont_data['project_id']
            logger.info(f"项目创建: {project_id}")

            if progress_callback:
                progress_callback(0.3, "本体生成完成，构建知识图谱...")

            # Step 4: 构建图谱
            graph_data = self.client.build_graph(project_id)
            task_id = graph_data['task_id']
            graph_result = self.client.poll_graph_task(task_id, timeout=900)
            graph_id = (graph_result.get('result') or {}).get('graph_id') or ont_data.get('graph_id', '')
            logger.info(f"图谱构建完成: {graph_id}")

            if progress_callback:
                progress_callback(0.5, "图谱构建完成，创建模拟...")

            # Step 5: 创建模拟
            sim_data = self.client.create_simulation(project_id, graph_id)
            simulation_id = sim_data['simulation_id']
            result['simulation_id'] = simulation_id

            if progress_callback:
                progress_callback(0.6, "模拟创建完成，准备 Agent 配置...")

            # Step 6: 准备模拟（MiroFish 侧 LLM 生成 Agent 画像）
            # entity_types 必须匹配 MiroFish 本体论中定义的类型
            # 本体类型: Agent, Company, Regulator, MediaOutlet, RetailInvestor, FundManager, ...
            # 7 个 Agent 角色全部映射到 "Agent" 类型
            prepare_timeout = 600 if self.debug else 600
            entity_types = ["Agent", "Company", "Regulator", "MediaOutlet",
                           "RetailInvestor", "FundManager", "InstitutionalInvestor"]
            self.client.prepare_simulation(simulation_id, entity_types=entity_types)
            self.client.poll_prepare_status(simulation_id, timeout=prepare_timeout)

            if progress_callback:
                if self.debug:
                    msg = "Agent 配置完成，启动 OASIS 模拟 (debug: 7Agent/1轮)..."
                else:
                    msg = "Agent 配置完成，启动 OASIS 模拟 (7Agent/5轮)..."
                progress_callback(0.7, msg)

            # Step 7: 启动模拟
            max_rounds = 1 if self.debug else 5
            self.client.start_simulation(simulation_id, max_rounds=max_rounds)

            if progress_callback:
                msg = "模拟运行中..." if self.debug else "模拟运行中（约 5-10 分钟）..."
                progress_callback(0.8, msg)

            # Step 8: 等待模拟完成
            sim_timeout = 600 if self.debug else 900
            sim_result = self.client.poll_simulation_status(simulation_id, timeout=sim_timeout)
            logger.info(f"模拟完成: {simulation_id}")

            if progress_callback:
                progress_callback(0.9, "模拟完成，生成预测报告...")

            # Step 9: 生成报告（MiroFish ReACT Agent 逐章节生成，较慢）
            report_data = self.client.generate_report(simulation_id)

            # 处理两种返回情况：
            # A) 新生成: {simulation_id, report_id, task_id, status: "generating"}
            # B) 已有报告: {simulation_id, report_id, status: "completed", already_generated: true}
            if report_data.get('already_generated') or report_data.get('status') == 'completed':
                # 报告已存在，直接获取
                report_id = report_data.get('report_id', '')
                if not report_id:
                    raise RuntimeError("报告已存在但未获取到 report_id")
                logger.info(f"报告已存在，直接获取: {report_id}")
            else:
                # 新生成，轮询等待完成（无超时限制）
                report_task_id = report_data.get('task_id', '')
                if not report_task_id:
                    raise RuntimeError("报告生成未返回 task_id")
                report_result = self.client.poll_report_status(report_task_id)
                # 安全提取 report_id: result 字段可能是嵌套 dict 或字符串
                raw_result = report_result.get('result', {})
                if isinstance(raw_result, str):
                    raw_result = {}
                report_id = raw_result.get('report_id', '') if isinstance(raw_result, dict) else ''
                if not report_id:
                    raise RuntimeError("模拟完成但未获取到 report_id，报告生成失败")

            report = self.client.get_report(report_id)
            logger.info(f"MiroFish 报告字段: {list(report.keys())}, markdown长度: {len(report.get('markdown_content', '') or '')}")
            result['report'] = report
            result['status'] = 'simulated'
            logger.info(f"StockFish→MiroFish 推演完成: {symbol}")

        except Exception as e:
            logger.error(f"MiroFish 推演失败: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)

        if progress_callback:
            progress_callback(1.0, "完成")

        return result
