"""
Batch Stock Analyzer — 批量股票分析编排器

支持多只股票串行分析、结果缓存、批量总结 + 优质股票推荐。
与 app.py 中的 predict 模式一致：后台线程 + 回调更新进度。

用法:
    analyzer = BatchAnalyzer()
    analyzer.run_batch(
        symbols=["600519", "000858"],
        cost_prices=[150.0, 0.0],
        shares_list=[100, 0],
        total_assets=500000,
        available_cash=100000,
        master="buffett",
        progress_callback=callback,
    )
"""
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Callable, Any
from loguru import logger

from analysis.agent import StockAnalysisAgent


class BatchAnalyzer:
    """批量股票分析编排器"""

    def __init__(self):
        self._agent = StockAnalysisAgent()
        self._cache_dir = Path(__file__).resolve().parent.parent / 'batch_results'

    # ── 主入口 ──

    def run_batch(
        self,
        symbols: List[str],
        cost_prices: List[float],
        shares_list: List[int],
        total_assets: float,
        available_cash: float,
        master: str = '',
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        """
        串行执行多只股票分析，每完成一只触发回调。

        Args:
            symbols: 股票代码列表
            cost_prices: 对应的成本价列表
            shares_list: 对应的持仓数量列表
            total_assets: 总资产（所有股票共用）
            available_cash: 可用资金（所有股票共用）
            master: 大师 key（所有股票共用）
            progress_callback: 回调 fn(event_type, data)

        Returns:
            { task_id, symbols, status, results: [...], summary: {...}, quality_pick: {...} }
        """
        task_id = f"batch_{uuid.uuid4().hex[:12]}"
        total = len(symbols)

        if progress_callback:
            progress_callback('progress', {
                'current': 0, 'total': total,
                'message': f'批量分析启动，共 {total} 只股票',
            })

        # 创建缓存目录
        task_cache_dir = self._cache_dir / task_id
        task_cache_dir.mkdir(parents=True, exist_ok=True)

        all_results = []
        for i, symbol in enumerate(symbols):
            # 通知开始
            if progress_callback:
                progress_callback('progress', {
                    'current': i + 1, 'total': total, 'symbol': symbol,
                    'message': f'正在分析 [{symbol}] ({i+1}/{total})...',
                })

            # 执行分析
            try:
                cost = float(cost_prices[i]) if i < len(cost_prices) else 0.0
                shares = int(shares_list[i]) if i < len(shares_list) else 0

                result = self._agent.analyze(
                    symbol, cost_price=cost, master=master,
                    shares=shares, total_assets=total_assets,
                    available_cash=available_cash,
                )

                # 缓存结果
                cache_path = task_cache_dir / f'{symbol}.json'
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                all_results.append({
                    'symbol': symbol,
                    'status': result.get('status', 'complete'),
                    'data': result,
                })

                if progress_callback:
                    status = result.get('status', 'complete')
                    progress_callback('stock_result', {
                        'current': i + 1, 'total': total,
                        'symbol': symbol, 'data': result,
                        'message': f'[{symbol}] 分析完成 ({status})',
                    })

            except Exception as e:
                logger.error(f"[{symbol}] 批量分析失败: {e}")
                error_result = {
                    'symbol': symbol, 'status': 'error',
                    'error': str(e),
                }
                all_results.append(error_result)

                if progress_callback:
                    progress_callback('stock_result', {
                        'current': i + 1, 'total': total,
                        'symbol': symbol, 'data': {'status': 'error', 'error': str(e)},
                        'message': f'[{symbol}] 分析失败: {e}',
                    })

        # 批量总结 + 优质推荐
        success_results = [r for r in all_results if r['status'] == 'complete']

        batch_summary = None
        quality_pick = None

        if len(success_results) >= 2:
            if progress_callback:
                progress_callback('progress', {
                    'current': total, 'total': total,
                    'message': '正在生成批量分析总结...',
                })

            batch_summary = self._summarize(success_results, master, total_assets, available_cash)

            if progress_callback:
                progress_callback('progress', {
                    'current': total, 'total': total,
                    'message': '正在筛选优质股票...',
                })

            quality_pick = self._pick_best(success_results, master, total_assets, available_cash)

        elif len(success_results) == 1:
            # 只有一只成功，无需批量总结
            batch_summary = {
                'summary_text': f"仅成功分析 {success_results[0]['symbol']}，无足够数据做批量对比。",
                'ranking': [],
                'overall_assessment': '只有一只股票完成分析，建议返回单股模式查看更多。',
            }
            quality_pick = {
                'best_stock': {
                    'symbol': success_results[0]['symbol'],
                    'name': success_results[0]['data'].get('stock_name', ''),
                    'reasons': ['唯一成功分析的股票'],
                    'suggested_action': '参考该股单独分析结果',
                },
                'runner_up': None,
            }

        if progress_callback:
            progress_callback('batch_summary', {
                'summary': batch_summary,
                'quality_pick': quality_pick,
            })

        summary = {
            'task_id': task_id,
            'symbols': symbols,
            'total': total,
            'success_count': len(success_results),
            'error_count': len(all_results) - len(success_results),
            'results': all_results,
            'summary': batch_summary,
            'quality_pick': quality_pick,
            'cached_in': str(task_cache_dir),
        }

        if progress_callback:
            progress_callback('completed', {
                'message': f'批量分析完成 ({len(success_results)}/{total} 成功)',
            })

        return summary

    # ── 批量总结 ──

    def _summarize(
        self,
        results: List[Dict],
        master: str,
        total_assets: float,
        available_cash: float,
    ) -> Optional[Dict]:
        """调用 LLM 拉通所有股票结果做总结"""
        from analysis.agents.cio_prompts import get_master_info
        from openai import OpenAI

        master_info = get_master_info(master) if master else None
        master_name = master_info['name'] if master_info else '综合'

        # 构建每只股票的摘要
        stocks_text_parts = []
        for i, r in enumerate(results):
            d = r['data']
            pred = d.get('prediction_summary', {}) or {}
            q = d.get('quote', {}) or {}
            fs = d.get('financial_summary', {}) or {}
            score_bd = d.get('score_breakdown', {}) or {}

            # CIO / legacy outlook
            cio = pred.get('cio_decision', {}) or {}
            order = cio.get('order', {}) or {}
            suggested = pred.get('suggested_action', {}) or {}

            parts = [
                f"### {i+1}. {d.get('stock_name', '')}({r['symbol']})",
                f"- 现价: {q.get('price', 'N/A')}元  PE: {q.get('pe', 'N/A')}  PB: {q.get('pb', 'N/A')}",
                f"- 估值等级: {d.get('valuation_level', 'N/A')} (PE分位: {d.get('valuation_percentile', 'N/A')}%)",
                f"- ROE: {fs.get('roe', 'N/A')}%  EPS: {fs.get('eps', 'N/A')}",
                f"- 综合评分: {score_bd.get('final', 'N/A')} / {score_bd.get('label', 'N/A')}",
            ]

            if cio.get('decision_summary'):
                parts.append(f"- CIO决策: {cio['decision_summary'][:120]}")
            elif pred.get('outlook'):
                parts.append(f"- 预测方向: {pred.get('outlook', 'N/A')} (置信度: {pred.get('confidence', 'N/A')})")

            if order.get('action'):
                parts.append(f"- 操作建议: {order.get('action')} (建议仓位: {order.get('position_size_pct', 'N/A')}%)")
            elif suggested.get('action'):
                parts.append(f"- 操作建议: {suggested.get('action')}")

            st = pred.get('short_term') or {}
            mt = pred.get('mid_term') or {}
            if st.get('direction'):
                parts.append(f"- 短期(1-2周): {st.get('direction')} {st.get('change_pct', 0):+}%")
            if mt.get('direction'):
                parts.append(f"- 中期(1-3月): {mt.get('direction')} {mt.get('change_pct', 0):+}%")

            parts.append("")
            stocks_text_parts.append("\n".join(parts))

        stocks_text = "\n".join(stocks_text_parts)

        system_prompt = f"""你是一位资深投资组合经理。你面前是{len(results)}只股票各自的分析报告（含{'CIO大师' if master else ''}决策），请你做跨股票对比分析。

你的任务:
1. 横向对比所有股票，找出共性趋势和分歧点
2. 按投资价值排序（综合考虑安全边际、成长性、风险、当前估值分位）
3. 在{total_assets:.0f}元总资产、{available_cash:.0f}元可用资金的约束下，给出整体配置建议

输出严格JSON:
{{
  "summary_text": "跨股票对比总结 (150-200字)",
  "ranking": [
    {{"symbol": "600519", "rank": 1, "name": "股票名", "score": 8.5, "reason": "入选理由(30字)"}}
  ],
  "common_themes": ["共性1", "共性2"],
  "key_divergences": ["分歧1"],
  "overall_assessment": "在给定资金约束下的整体配置建议 (100字内)"
}}

注意: ranking 必须包含所有成功分析的股票。"""

        user_prompt = f"""## 资金约束
- 总资产: {total_assets:.0f}元
- 可用资金: {available_cash:.0f}元
- 投资风格: {master_name}

## 各股票分析摘要

{stocks_text}

请给出跨股票对比分析和排序。"""

        try:
            result = self._call_llm(system_prompt, user_prompt, temperature=0.3)
            return result
        except Exception as e:
            logger.error(f"批量总结 LLM 调用失败: {e}")
            return self._fallback_summary(results)

    # ── 优质股票推荐 ──

    def _pick_best(
        self,
        results: List[Dict],
        master: str,
        total_assets: float,
        available_cash: float,
    ) -> Optional[Dict]:
        """调用 LLM 从多只股票中选出最值得买入的"""
        from analysis.agents.cio_prompts import get_master_info

        master_info = get_master_info(master) if master else None
        master_name = master_info['name'] if master_info else '综合'

        stocks_text_parts = []
        for r in results:
            d = r['data']
            pred = d.get('prediction_summary', {}) or {}
            q = d.get('quote', {}) or {}
            fs = d.get('financial_summary', {}) or {}
            score_bd = d.get('score_breakdown', {}) or {}

            cio = pred.get('cio_decision', {}) or {}
            order = cio.get('order', {}) or {}
            suggested = pred.get('suggested_action', {}) or {}

            parts = [
                f"## {d.get('stock_name', '')}({r['symbol']})",
                f"- 现价: {q.get('price', 'N/A')}元  估值: {d.get('valuation_level', 'N/A')}  综合评分: {score_bd.get('final', 'N/A')}",
                f"- ROE: {fs.get('roe', 'N/A')}%  EPS: {fs.get('eps', 'N/A')}  市值: {q.get('market_cap', 'N/A')}亿",
            ]
            if cio.get('decision_summary'):
                parts.append(f"- CIO决策摘要: {cio['decision_summary'][:150]}")
            if order.get('action'):
                parts.append(f"- 操作建议: {order.get('action')}  目标仓位: {order.get('position_size_pct', 'N/A')}%")
            bc = cio.get('base_case') or {}
            if bc.get('target'):
                parts.append(f"- 基准目标价: {bc.get('target')}元 (概率: {bc.get('probability', 'N/A')})")

            st = pred.get('short_term') or {}
            mt = pred.get('mid_term') or {}
            if st.get('direction'):
                parts.append(f"- 短期: {st.get('direction')} {st.get('change_pct', 0):+}%")
            if mt.get('direction'):
                parts.append(f"- 中期: {mt.get('direction')} {mt.get('change_pct', 0):+}%")
            parts.append("")
            stocks_text_parts.append("\n".join(parts))

        stocks_text = "\n".join(stocks_text_parts)

        system_prompt = f"""你是一位资深基金经理，投资风格: {master_name}。你有{total_assets:.0f}元总资产和{available_cash:.0f}元可用资金。

请根据以下{len(results)}只股票的分析结果，选出**当前最值得买入的一只股票**。

选择标准:
1. 安全边际（估值分位越低越好）
2. 成长性（ROE/盈利增长）
3. 风险收益比（上行空间 vs 下行风险）
4. 与投资风格的匹配度
5. 当前价格是否有吸引力

输出严格JSON:
{{
  "best_stock": {{
    "symbol": "600519",
    "name": "股票名",
    "reasons": ["理由1: 具体数据支撑(20字)", "理由2", "理由3"],
    "suggested_action": "买入/加仓/观望",
    "suggested_position_pct": 15,
    "position_note": "在{total_assets:.0f}元总资产、{available_cash:.0f}元可用资金约束下的仓位建议说明(50字)",
    "target_price": 0,
    "risk_note": "主要风险提示(30字)"
  }},
  "runner_up": {{
    "symbol": "000858",
    "name": "股票名",
    "reasons": ["次选理由(20字)"]
  }},
  "selection_rationale": "为什么选择该股而非其他候选的综合判断(100字)"
}}"""

        user_prompt = f"""## 候选股票\n{stocks_text}\n请选出当前最值得买入的股票。"""

        try:
            result = self._call_llm(system_prompt, user_prompt, temperature=0.3)
            return result
        except Exception as e:
            logger.error(f"优质股票推荐 LLM 调用失败: {e}")
            return self._fallback_pick(results)

    # ── LLM 调用 ──

    def _call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> dict:
        """调用 LLM（复用配置）"""
        from config import settings

        api_key = os.environ.get('LLM_API_KEY') or getattr(settings, 'LLM_API_KEY', '')
        base_url = os.environ.get('LLM_BASE_URL') or getattr(settings, 'LLM_BASE_URL', 'https://api.openai.com/v1')
        model = os.environ.get('LLM_MODEL_NAME') or getattr(settings, 'LLM_MODEL_NAME', 'gpt-4o-mini')

        if not api_key:
            logger.warning("LLM API key 未配置")
            return {}

        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=60, max_retries=1)

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
            max_tokens=4096,
        )

        raw = resp.choices[0].message.content or "{}"
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """安全解析 LLM 返回的 JSON"""
        try:
            raw = raw.strip()
            if raw.startswith('```json'):
                raw = raw.split('```json')[1].split('```')[0]
            elif raw.startswith('```'):
                raw = raw.split('```')[1].split('```')[0]
            start = raw.find('{')
            end = raw.rfind('}')
            if start >= 0 and end > start:
                raw = raw[start:end + 1]
            return json.loads(raw)
        except (json.JSONDecodeError, KeyError, IndexError):
            import re
            try:
                start = raw.find('{')
                end = raw.rfind('}')
                if start >= 0 and end > start:
                    candidate = raw[start:end + 1]
                    candidate = re.sub(r',\s*}', '}', candidate)
                    candidate = re.sub(r',\s*]', ']', candidate)
                    return json.loads(candidate)
            except Exception:
                pass
            logger.warning(f"JSON 解析失败: {raw[:200]}...")
            return {}

    # ── 降级: 规则总结 ──

    def _fallback_summary(self, results: List[Dict]) -> Dict:
        """LLM 不可用时的规则降级总结"""
        ranking = []
        for i, r in enumerate(results):
            d = r['data']
            score_bd = d.get('score_breakdown', {}) or {}
            final_score = score_bd.get('final', 0) or 0
            ranking.append({
                'symbol': r['symbol'],
                'rank': i + 1,
                'name': d.get('stock_name', ''),
                'score': final_score,
                'reason': f"综合评分 {final_score:+.1f}，估值分位 {d.get('valuation_percentile', 'N/A')}%",
            })

        # 按评分排序
        ranking.sort(key=lambda x: x['score'], reverse=True)
        for i, item in enumerate(ranking):
            item['rank'] = i + 1

        return {
            'summary_text': f"(规则降级) 基于综合评分的简单排序。共{len(results)}只股票，最高评分: {ranking[0]['score'] if ranking else 'N/A'}。",
            'ranking': ranking,
            'common_themes': [],
            'key_divergences': [],
            'overall_assessment': '建议启用 LLM 获得深度跨股票分析和配置建议。',
        }

    def _fallback_pick(self, results: List[Dict]) -> Dict:
        """LLM 不可用时的规则降级选股"""
        if not results:
            return {'best_stock': None, 'runner_up': None, 'selection_rationale': '无可用数据'}

        # 按综合评分 + PE 分位加权排序
        scored = []
        for r in results:
            d = r['data']
            score_bd = d.get('score_breakdown', {}) or {}
            val_pct = d.get('valuation_percentile', 50) or 50
            final_score = score_bd.get('final', 0) or 0
            # 低估值 + 高评分的组合最优
            composite = final_score * 1.5 - (val_pct / 100) * 2
            scored.append((composite, r))

        scored.sort(key=lambda x: x[0], reverse=True)

        best = scored[0][1]
        best_d = best['data']
        runner = scored[1][1] if len(scored) > 1 else None

        return {
            'best_stock': {
                'symbol': best['symbol'],
                'name': best_d.get('stock_name', ''),
                'reasons': [
                    f"估值分位: {best_d.get('valuation_percentile', 'N/A')}%",
                    f"综合评分: {best_d.get('score_breakdown', {}).get('final', 'N/A')}",
                ],
                'suggested_action': '买入',
                'suggested_position_pct': 15,
                'position_note': '规则降级推荐，建议启用LLM获得更全面的分析',
                'target_price': 0,
                'risk_note': '数据不完整，请谨慎参考',
            },
            'runner_up': {
                'symbol': runner['symbol'],
                'name': runner_d.get('stock_name', '') if runner and (runner_d := runner['data']) else '',
                'reasons': ['次优评分'] if runner else [],
            } if runner else None,
            'selection_rationale': f'(规则降级) 基于综合评分和估值分位的简单排序选出 {best["symbol"]}。建议启用 LLM 获得深度分析。',
        }
