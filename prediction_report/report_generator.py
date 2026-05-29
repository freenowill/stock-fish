"""
预测报告生成器

将 StockEngine 分析 + 模拟推演结果合并为最终预测报告。
支持 HTML 和 JSON 格式输出。
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class PredictionReportGenerator:
    """
    预测报告生成器

    用法:
        gen = PredictionReportGenerator()
        report = gen.generate(analysis_result, simulation_result)
        html = gen.to_html(report)
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or str(
            Path(__file__).resolve().parent.parent / "reports"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self, analysis_result: Dict[str, Any],
                 simulation_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """生成完整的预测报告"""
        symbol = analysis_result.get('symbol', '')
        name = analysis_result.get('stock_name', analysis_result.get('name', ''))
        signals = analysis_result.get('signals', {}) or {}
        ps = analysis_result.get('prediction_summary', {}) or {}
        risk = analysis_result.get('risk_factors', []) or []
        quote = analysis_result.get('quote', {}) or {}
        ti = analysis_result.get('technical_indicators', {}) or {}
        fs = analysis_result.get('financial_summary', {}) or {}
        sn = analysis_result.get('sentiment_news', {}) or {}
        sg = analysis_result.get('sentiment_guba', {}) or {}

        # 计算综合置信度
        confidence_score = self._calc_confidence(analysis_result, simulation_result)

        report = {
            'report_id': f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'symbol': symbol,
            'name': name,
            'title': f"{name}({symbol}) 股价预测报告",

            'summary': {
                'outlook': ps.get('outlook', '中性'),
                'confidence': ps.get('confidence', '低'),
                'confidence_score': confidence_score,
                'current_price': quote.get('price', 0),
                'price_target_low': ps.get('price_target_low'),
                'price_target_high': ps.get('price_target_high'),
                'signal_score': signals.get('score', 0),
                'overall_signal': signals.get('overall', 'neutral'),
            },

            'market_data': {
                'quote': {
                    'price': quote.get('price'),
                    'change_pct': quote.get('change_pct'),
                    'pe': quote.get('pe'),
                    'pb': quote.get('pb'),
                    'market_cap': quote.get('market_cap'),
                    'turnover_rate': quote.get('turnover_rate'),
                },
                'technical': {
                    'rsi_14': ti.get('rsi_14'),
                    'macd': ti.get('macd_hist'),
                    'kdj': f"{ti.get('kdj_k')}/{ti.get('kdj_d')}/{ti.get('kdj_j')}",
                    'ma5': ti.get('ma5'),
                    'ma10': ti.get('ma10'),
                    'ma20': ti.get('ma20'),
                    'bollinger': f"{ti.get('boll_lower')}~{ti.get('boll_middle')}~{ti.get('boll_upper')}",
                },
                'financial': {
                    'revenue': fs.get('revenue'),
                    'net_profit': fs.get('net_profit'),
                    'eps': fs.get('eps'),
                    'roe': fs.get('roe'),
                },
            },

            'sentiment': {
                'news_score': sn.get('avg_score', 0),
                'guba_score': sg.get('avg_score', 0),
                'news_count': sn.get('total_count', 0),
                'guba_count': sg.get('total_count', 0),
            },

            'risk_analysis': {
                'risk_factors': [r.get('factor', '') if isinstance(r, dict) else r for r in risk],
                'positive_factors': ps.get('reason', '').split(',') if ps.get('reason') else [],
            },

            'analysis_text': analysis_result.get('llm_analysis', ''),

            'signals': signals.get('details', []),

            'simulation': None,
        }

        if simulation_result:
            mreport = simulation_result.get('report') or {}
            report['simulation'] = {
                'status': simulation_result.get('status'),
                'scenario': simulation_result.get('scenario'),
                'scenarios': simulation_result.get('scenarios', []),
                'seed_text': simulation_result.get('seed_text', '')[:500],
                'mirofish_report': {
                    'markdown': (mreport.get('markdown_content') or '')[:3000],
                    'sections': (mreport.get('sections') or [])[:5],
                    'simulation_rounds': mreport.get('simulation_rounds', 0),
                    'agent_count': mreport.get('agent_count', 0),
                } if mreport.get('markdown_content') or mreport.get('sections') else None,
            }

        return report

    def to_html(self, report: Dict[str, Any]) -> str:
        """将报告渲染为 HTML"""
        s = report['summary']
        md = report['market_data']
        sn = report['sentiment']
        ri = report['risk_analysis']

        signal_color = {'bullish': '#00d4aa', 'bearish': '#ff4757', 'neutral': '#ffa502'}
        color = signal_color.get(s['overall_signal'], '#888')

        outlook_icon = {'看多': '📈', '看空': '📉', '中性': '➡️'}

        sim = report.get('simulation') or {}
        mr = sim.get('mirofish_report') or {}
        if mr.get('markdown'):
            sim_section = f'''  <div class="section">
    <div class="section-title">MiroFish 群体智能推演报告</div>
    <div class="analysis">{mr['markdown']}</div>
  </div>
'''
        else:
            sim_section = ''

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8">
<title>{report['title']}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          background: #0f1923; color: #e0e6ed; max-width: 900px; margin: 0 auto; padding: 20px; }}
  h1 {{ font-size: 22px; border-bottom: 2px solid {color}; padding-bottom: 8px; }}
  .summary {{ background: #152436; border: 1px solid {color}; border-radius: 12px; padding: 20px; margin: 16px 0; }}
  .price-target {{ font-size: 32px; font-weight: 600; color: {color}; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 12px 0; }}
  .card {{ background: #152436; border-radius: 8px; padding: 12px; border: 1px solid #1e3a5f; }}
  .label {{ color: #6b8db5; font-size: 11px; text-transform: uppercase; }}
  .value {{ font-size: 18px; font-weight: 500; margin-top: 4px; }}
  .section {{ margin: 20px 0; }}
  .section-title {{ font-size: 16px; font-weight: 600; color: #8ab4f8; margin-bottom: 8px; }}
  .analysis {{ background: #152436; border-radius: 8px; padding: 16px; white-space: pre-wrap; line-height: 1.6; }}
  .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin: 2px; }}
  .tag-positive {{ background: rgba(0,212,170,.15); color: #00d4aa; }}
  .tag-negative {{ background: rgba(255,71,87,.15); color: #ff4757; }}
  .footer {{ text-align: center; color: #6b8db5; font-size: 12px; margin-top: 40px; }}
</style></head>
<body>
  <h1>{report['title']}</h1>
  <div class="summary">
    <div style="font-size: 48px; text-align: center;">{outlook_icon.get(s['outlook'], '➡️')}</div>
    <div style="text-align: center; font-size: 14px; color: #6b8db5; margin-top: 8px;">
      信号: <span style="color:{color};font-weight:600;">{s['overall_signal']}</span>
      | 置信度: {s['confidence']} ({s['confidence_score']}%)
      | 生成: {report['generated_at'][:16]}
    </div>
  </div>

  <div class="section">
    <div class="section-title">价格预测</div>
    <div style="text-align:center;padding:16px;">
      <span style="color:#6b8db5;">目标区间</span><br/>
      <span class="price-target">{s['price_target_low']}</span>
      <span style="font-size:20px;color:#6b8db5;"> ~ </span>
      <span class="price-target">{s['price_target_high']}</span>
      <br/><span style="color:#6b8db5;font-size:14px;">当前: {s['current_price']}</span>
    </div>
  </div>

  <div class="grid">
    <div class="card"><div class="label">PE / PB</div><div class="value">{md['quote']['pe']} / {md['quote']['pb']}</div></div>
    <div class="card"><div class="label">市值</div><div class="value">{md['quote']['market_cap']}亿</div></div>
    <div class="card"><div class="label">RSI / MACD</div><div class="value">{md['technical']['rsi_14']} / {md['technical']['macd']}</div></div>
    <div class="card"><div class="label">舆情</div><div class="value">新闻{sn['news_score']} 股吧{sn['guba_score']}</div></div>
  </div>

  <div class="section">
    <div class="section-title">风险因素</div>
    {''.join(f'<span class="tag tag-negative">{f}</span> ' for f in ri['risk_factors']) or '暂无'}
  </div>

  <div class="section">
    <div class="section-title">分析详情</div>
    <div class="analysis">{report['analysis_text']}</div>
  </div>

  <div class="section">
    <div class="section-title">信号明细</div>
    {''.join(
      f'<div style="padding:4px 0;border-bottom:1px solid #1e3a5f;font-size:13px;">'
      f'<span class="{"positive" if sig.get("impact")=="positive" else "negative"}">'
      f'{"▲" if sig.get("impact")=="positive" else "▼"}</span> '
      f'{sig.get("factor","")} '
      f'<span style="color:#6b8db5;float:right;">权重{sig.get("weight","")}</span></div>'
      for sig in report['signals']
    )}
  </div>

  {sim_section}

  <div class="footer">
    StockFish AI Prediction · {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>
    <span style="font-size:11px;">本报告仅供参考，不构成投资建议</span>
  </div>
</body></html>"""

    def save(self, report: Dict[str, Any]) -> str:
        """保存报告到文件"""
        html = self.to_html(report)
        name = report['symbol']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_path = os.path.join(self.output_dir, f"{name}_prediction_{timestamp}.html")
        json_path = os.path.join(self.output_dir, f"{name}_prediction_{timestamp}.json")

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return html_path

    @staticmethod
    def _calc_confidence(analysis: dict, simulation: Optional[dict] = None) -> float:
        """计算综合置信度 0-100"""
        score = 50  # base

        signals = analysis.get('signals', {}) or {}
        ps = analysis.get('prediction_summary', {}) or {}

        # 信号强度加分
        abs_score = abs(signals.get('score', 0) or 0)
        score += min(abs_score * 5, 20)

        # 舆情数据量加分
        sn = analysis.get('sentiment_news', {}) or {}
        sg = analysis.get('sentiment_guba', {}) or {}
        total = (sn.get('total_count', 0) or 0) + (sg.get('total_count', 0) or 0)
        score += min(total * 2, 15)

        # 有模拟结果加分
        if simulation and simulation.get('status') == 'simulated':
            score += 15

        return min(max(score, 0), 100)
