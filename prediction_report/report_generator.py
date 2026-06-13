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
        sb = analysis_result.get('score_breakdown', {}) or {}
        cio = ps.get('cio_decision', {}) or {}

        # 计算综合置信度
        confidence_score = self._calc_confidence(analysis_result, simulation_result)

        report = {
            'report_id': f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'symbol': symbol,
            'name': name,
            'title': f"{name}({symbol}) 股价分析报告",

            'summary': {
                'outlook': ps.get('outlook', '中性'),
                'confidence': ps.get('confidence', '低'),
                'confidence_score': confidence_score,
                'current_price': quote.get('price', 0),
                'price_target_low': ps.get('price_target_low'),
                'price_target_high': ps.get('price_target_high'),
                'signal_score': signals.get('score', 0),
                'overall_signal': signals.get('overall', 'neutral'),
                'valuation_level': analysis_result.get('valuation_level', ''),
                'valuation_percentile': analysis_result.get('valuation_percentile'),
                'suggested_buy_price': analysis_result.get('suggested_buy_price'),
                'cost_price': analysis_result.get('cost_price', 0),
                'dividend': quote.get('dividend'),
            },

            'market_data': {
                'quote': {
                    'price': quote.get('price'),
                    'change_pct': quote.get('change_pct'),
                    'pe': quote.get('pe'),
                    'pb': quote.get('pb'),
                    'market_cap': quote.get('market_cap'),
                    'turnover_rate': quote.get('turnover_rate'),
                    'dividend': quote.get('dividend'),
                },
                'technical': {
                    'rsi_14': ti.get('rsi_14'),
                    'macd': ti.get('macd_hist'),
                    'kdj': f"{ti.get('kdj_k')}/{ti.get('kdj_d')}/{ti.get('kdj_j')}",
                    'ma5': ti.get('ma5'),
                    'ma10': ti.get('ma10'),
                    'ma20': ti.get('ma20'),
                    'bollinger': f"{ti.get('boll_lower')}~{ti.get('boll_middle')}~{ti.get('boll_upper')}",
                    'volume_ratio': ti.get('volume_ratio'),
                },
                'financial': {
                    'revenue': fs.get('revenue'),
                    'net_profit': fs.get('net_profit'),
                    'eps': fs.get('eps'),
                    'roe': fs.get('roe'),
                    'gross_margin': fs.get('gross_margin'),
                    'debt_ratio': fs.get('debt_ratio'),
                },
            },

            'sentiment': {
                'news_score': sn.get('avg_score', 0),
                'guba_score': sg.get('avg_score', 0),
                'news_count': sn.get('total_count', 0),
                'guba_count': sg.get('total_count', 0),
                'news_positive': sn.get('positive_count', 0),
                'news_negative': sn.get('negative_count', 0),
                'guba_positive': sg.get('positive_count', 0),
                'guba_negative': sg.get('negative_count', 0),
            },

            'risk_analysis': {
                'risk_factors': [r.get('factor', '') if isinstance(r, dict) else r for r in risk],
                'positive_factors': ps.get('reason', '').split(',') if ps.get('reason') else [],
            },

            'analysis_text': analysis_result.get('llm_analysis', ''),

            'signals': signals.get('details', []),

            'score_breakdown': {
                'final': sb.get('final'),
                'technical': sb.get('technical'),
                'fundamental': sb.get('fundamental'),
                'sentiment': sb.get('sentiment'),
                'regime': sb.get('regime'),
                'confidence': sb.get('confidence'),
                'breakdown': sb.get('breakdown', []),
            },

            'cio_decision': {
                'master_name': cio.get('master_name', ''),
                'decision_summary': cio.get('decision_summary', ''),
                'rationale': cio.get('rationale', ''),
                'order': cio.get('order', {}),
                'base_case': cio.get('base_case', {}),
                'bull_case': cio.get('bull_case', {}),
                'bear_case': cio.get('bear_case', {}),
                'evidence_chain': cio.get('evidence_chain', []),
                'risk_monitoring': cio.get('risk_monitoring', []),
                'decision_quality': cio.get('decision_quality', {}),
                'veto_response': cio.get('veto_response', ''),
            } if cio.get('master_name') else None,

            'employee_reports': ps.get('employee_reports') or [],

            'important_bullish_news': analysis_result.get('important_bullish_news') or [],
            'important_bearish_news': analysis_result.get('important_bearish_news') or [],

            'multi_cycle': {
                'short_term': ps.get('short_term') or {},
                'mid_term': ps.get('mid_term') or {},
                'long_term': ps.get('long_term') or {},
            },

            'suggested_action': analysis_result.get('suggested_action') or {},

            'simulation': None,
        }

        if simulation_result:
            mreport = simulation_result.get('report') or {}
            sim_note = simulation_result.get('simulation_note', '')
            report['simulation'] = {
                'status': simulation_result.get('status'),
                'scenario': simulation_result.get('scenario'),
                'scenarios': simulation_result.get('scenarios', []),
                'seed_text': simulation_result.get('seed_text', '')[:500],
                'simulation_note': sim_note,
                'mirofish_report': {
                    'markdown': mreport.get('markdown_content') or '',
                    'sections': (mreport.get('sections') or [])[:5],
                    'simulation_rounds': mreport.get('simulation_rounds', 0),
                    'agent_count': mreport.get('agent_count', 0),
                } if mreport.get('markdown_content') or mreport.get('sections') else None,
            }

        return report

    def to_html(self, report: Dict[str, Any]) -> str:
        """将报告渲染为 HTML"""
        s = report.get('summary', {})
        md = report.get('market_data', {})
        mq = md.get('quote', {}) if isinstance(md, dict) else {}
        mtech = md.get('technical', {}) if isinstance(md, dict) else {}
        mfin = md.get('financial', {}) if isinstance(md, dict) else {}
        sn = report.get('sentiment', {})
        ri = report.get('risk_analysis', {})
        sb = report.get('score_breakdown', {})
        cio = report.get('cio_decision') or {}
        emp_reports = report.get('employee_reports', [])
        bull_news = report.get('important_bullish_news', [])
        bear_news = report.get('important_bearish_news', [])
        mc = report.get('multi_cycle', {})
        action = report.get('suggested_action') or {}

        signal_color = {'bullish': '#ff4757', 'bearish': '#00d4aa', 'neutral': '#ffa502'}
        color = signal_color.get(s.get('overall_signal', 'neutral'), '#888')

        outlook_icon = {'看多': '📈', '看空': '📉', '中性': '➡️'}

        # --- Simulation section ---
        sim = report.get('simulation') or {}
        mr = sim.get('mirofish_report') or {}
        if mr and mr.get('markdown'):
            md_text = mr['markdown']
            try:
                from markdown_it import MarkdownIt
                md_parser = MarkdownIt()
                rendered = md_parser.render(md_text)
            except ImportError:
                rendered = md_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                rendered = f'<p>{rendered}</p>'
                rendered = rendered.replace('\n\n', '</p><p>').replace('\n', '<br/>')
            sim_section = f'''  <div class="section">
    <div class="section-title">🐟 MiroFish 群体智能推演报告</div>
    <div class="analysis mirofish-report">{rendered}</div>
  </div>
'''
        else:
            sim_section = ''

        # --- CIO Decision section ---
        cio_section = ''
        if cio.get('master_name'):
            cio_section += f'''  <div class="section">
    <div class="section-title">🧠 大师决策 · {cio.get('master_name', '')}</div>
    <div class="cio-card" style="background:#152436;border:2px solid #ffa502;border-radius:14px;padding:22px;margin:12px 0;">
'''

            if cio.get('veto_response') and cio['veto_response'] != '无':
                cio_section += f'''      <div style="background:rgba(255,71,87,.1);border:1px solid #ff4757;border-radius:8px;padding:10px 14px;margin-bottom:12px;color:#ff6b7a;font-size:13px;">⚠️ 风险经理否决回应: {cio['veto_response']}</div>
'''

            cio_section += f'''      <div style="font-size:15px;line-height:1.7;color:#e0e6ed;padding:12px;background:rgba(255,255,255,.03);border-radius:8px;margin-bottom:14px;">{cio.get('decision_summary', '-')}</div>
'''

            if cio.get('rationale'):
                cio_section += f'''      <div style="margin-bottom:14px;">
        <div style="font-size:13px;font-weight:600;color:#8ab4f8;margin-bottom:6px;">📝 决策逻辑详解</div>
        <div style="font-size:14px;color:#ccd6f6;line-height:1.7;padding:8px 12px;background:rgba(255,255,255,.03);border-radius:6px;">{cio['rationale']}</div>
      </div>
'''

            if cio.get('evidence_chain'):
                cio_section += '''      <div style="margin-bottom:14px;">
        <div style="font-size:13px;font-weight:600;color:#8ab4f8;margin-bottom:6px;">📋 关键证据链</div>
'''
                for e in cio['evidence_chain']:
                    cio_section += f'        <div style="font-size:12px;padding:4px 10px;background:rgba(255,255,255,.04);border-radius:6px;color:#c0c8d0;margin-bottom:4px;">📋 {e}</div>\n'
                cio_section += '      </div>\n'

            # Three scenarios
            base = cio.get('base_case', {})
            bull = cio.get('bull_case', {})
            bear = cio.get('bear_case', {})
            if base.get('direction') or bull.get('direction') or bear.get('direction'):
                cio_section += '''      <div style="margin-bottom:14px;">
        <div style="font-size:13px;font-weight:600;color:#8ab4f8;margin-bottom:6px;">📊 三情景分析</div>
        <div class="grid" style="grid-template-columns:repeat(3,1fr);">
'''
                dir_cls = lambda d: 'color:#ff4757;' if (d and '涨' in str(d)) else ('color:#00d4aa;' if (d and '跌' in str(d)) else 'color:#ffa502;')
                for label, sc in [('📉 悲观', bear), ('➡️ 基准', base), ('📈 乐观', bull)]:
                    prob = f"{int(sc.get('probability', 0) * 100)}%" if sc.get('probability') is not None else '?'
                    cio_section += f'''          <div style="background:rgba(0,0,0,.2);border-radius:10px;padding:12px;text-align:center;">
            <div style="font-size:11px;color:#6b8db5;">{label} ({prob})</div>
            <div style="font-size:16px;font-weight:700;margin:4px 0;{dir_cls(sc.get('direction'))}">{sc.get('direction', '-')}</div>
            <div style="font-size:11px;color:#6b8db5;">目标: ¥{sc.get('target', '-')}</div>
          </div>
'''
                cio_section += '        </div>\n      </div>\n'

            # Order
            order = cio.get('order', {})
            if order.get('action'):
                act = order.get('action', '')
                act_cls = 'color:#ff4757;' if ('买' in str(act) or '加' in str(act)) else ('color:#00d4aa;' if ('卖' in str(act) or '减' in str(act)) else 'color:#ffa502;')
                cio_section += f'''      <div style="margin-bottom:14px;">
        <div style="font-size:13px;font-weight:600;color:#8ab4f8;margin-bottom:6px;">📈 操作指令</div>
        <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(120px,1fr));">
          <div style="background:rgba(0,0,0,.2);border-radius:8px;padding:10px;text-align:center;">
            <div class="label">操作</div>
            <div style="font-size:20px;font-weight:700;{act_cls}">{act}</div>
          </div>
          <div style="background:rgba(0,0,0,.2);border-radius:8px;padding:10px;text-align:center;">
            <div class="label">目标仓位</div>
            <div style="font-size:18px;font-weight:600;color:#ffa502;">{order.get('position_size_pct', 0)}%</div>
          </div>
          <div style="background:rgba(0,0,0,.2);border-radius:8px;padding:10px;text-align:center;">
            <div class="label">止损</div>
            <div style="font-size:18px;font-weight:600;color:#00d4aa;">¥{(order.get('stop_loss') or {}).get('level') if isinstance(order.get('stop_loss'), dict) else (order.get('stop_loss') or '-')}</div>
          </div>
          <div style="background:rgba(0,0,0,.2);border-radius:8px;padding:10px;text-align:center;">
            <div class="label">止盈</div>
            <div style="font-size:18px;font-weight:600;color:#ff4757;">¥{(lambda tp: (tp.get('level_1') or tp.get('level_2') or '-') if isinstance(tp, dict) else (tp or '-'))(order.get('take_profit'))}</div>
          </div>
        </div>
'''
                if order.get('order_rationale'):
                    cio_section += f'        <div style="font-size:14px;color:#ccd6f6;margin-top:8px;padding:6px 10px;background:rgba(255,255,255,.05);border-radius:6px;line-height:1.6;">💡 操作理由: {order["order_rationale"]}</div>\n'
                if order.get('position_note'):
                    cio_section += f'        <div style="font-size:13px;color:#ffa502;margin-top:6px;padding:6px 10px;background:rgba(255,165,2,.08);border-radius:6px;">📊 仓位分析: {order["position_note"]}</div>\n'
                cio_section += '      </div>\n'

            # Risk monitoring
            if cio.get('risk_monitoring'):
                cio_section += '''      <div style="margin-bottom:14px;">
        <div style="font-size:13px;font-weight:600;color:#8ab4f8;margin-bottom:6px;">⚡ 风险监控</div>
'''
                for rm in cio['risk_monitoring']:
                    cio_section += f'        <div style="padding:6px 10px;background:rgba(255,71,87,.06);border-left:3px solid #ff4757;border-radius:4px;margin-bottom:4px;font-size:12px;color:#c0c8d0;">⚡ {rm.get("trigger", "")} → {rm.get("action", "")}</div>\n'
                cio_section += '      </div>\n'

            # Decision quality
            dq = cio.get('decision_quality', {})
            cio_section += f'''      <div style="display:flex;gap:16px;align-items:center;font-size:12px;color:#6b8db5;padding-top:12px;border-top:1px solid rgba(255,255,255,.06);">
        <span>置信度: {dq.get('confidence', '中')}</span>
        <span>⚠ {', '.join(dq.get('key_uncertainties', ['无']))}</span>
        <span>下次回顾: {dq.get('next_review', '3个交易日后')}</span>
      </div>
'''
            cio_section += '    </div>\n  </div>\n'

        # --- Score breakdown section ---
        score_section = ''
        if sb.get('breakdown'):
            score_section += '''  <div class="section">
    <div class="section-title">📊 评分明细</div>
'''
            for d in sb['breakdown']:
                arrow = '▲' if d.get('impact') == 'positive' else ('▼' if d.get('impact') == 'negative' else '◆')
                cls = 'positive' if d.get('impact') == 'positive' else ('negative' if d.get('impact') == 'negative' else '')
                contrib = d.get('contribution', 0)
                contrib_str = f"{'+' if contrib > 0 else ''}{contrib:.1f}" if contrib else ''
                score_section += f'''    <div style="padding:5px 0;font-size:13px;border-bottom:1px solid #1a2d42;">
      <span style="color:{'#ff4757' if cls == 'positive' else ('#00d4aa' if cls == 'negative' else '#ffa502')};">{arrow}</span> {d.get('factor', '')}
      <span style="color:#6b8db5;float:right;">{contrib_str} {d.get('description', '')}</span>
    </div>
'''
            score_section += '  </div>\n'

        # --- Employee reports section ---
        emp_section = ''
        if emp_reports:
            emp_section += f'''  <div class="section">
    <div class="section-title">👥 部门分析报告 ({len(emp_reports)}份)</div>
'''
            dept_map = {'macro': '#8ab4f8', 'policy': '#8ab4f8', 'valuation': '#ffa502', 'fundamental': '#ffa502', 'technical': '#ffa502', 'sentiment': '#00d4aa', 'risk': '#ff4757', 'overseer': '#ff6348'}
            for r in emp_reports:
                border_color = dept_map.get(r.get('employee_id', ''), '#1e3a5f')
                out_cls = 'positive' if r.get('outlook') == '看多' else ('negative' if r.get('outlook') == '看空' else 'neutral')
                out_color = '#ff4757' if out_cls == 'positive' else ('#00d4aa' if out_cls == 'negative' else '#ffa502')
                emp_section += f'''    <div style="background:rgba(0,0,0,.15);border-radius:8px;padding:10px 14px;margin-bottom:6px;font-size:12px;border-left:3px solid {border_color};">
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;">
        <span style="font-weight:600;">{r.get('role', '?')}</span>
        <span style="font-size:10px;color:#6b8db5;padding:1px 6px;background:rgba(255,255,255,.05);border-radius:3px;">{r.get('department', '')}</span>
'''
                if r.get('error'):
                    emp_section += f'        <span style="color:#ff6b7a;font-style:italic;">⚠ {r["error"]}</span>\n'
                else:
                    emp_section += f'        <span style="padding:1px 8px;border-radius:3px;font-size:11px;background:rgba({ "255,71,87" if out_cls == "positive" else ("0,212,170" if out_cls == "negative" else "255,165,2") },.15);color:{out_color};">{r.get("outlook", "-")} ({r.get("score", "-")})</span>\n'
                emp_section += '        <span style="font-size:10px;color:#6b8db5;">置信度: ' + str(r.get('confidence', '-')) + '</span>\n      </div>\n'
                if r.get('key_points'):
                    for p in r['key_points']:
                        emp_section += f'      <div style="color:#c0c8d0;">+ {p}</div>\n'
                if r.get('risks'):
                    for rk in r['risks']:
                        emp_section += f'      <div style="color:#ff6b7a;font-size:11px;">- {rk}</div>\n'
                emp_section += '    </div>\n'
            emp_section += '  </div>\n'

        # --- News section ---
        news_section = ''
        if bull_news or bear_news:
            news_section += '''  <div class="section">
    <div class="section-title">📰 重要新闻摘要 <span style="font-weight:400;font-size:11px;">(近3天 · LLM 筛选)</span></div>
'''
            for n in bull_news:
                src = n.get('source', '') or ''
                time_str = (n.get('publish_time', '') or '')[:10]
                news_section += f'    <div style="padding:10px 14px;border-radius:8px;font-size:13px;border-left:3px solid #ff4757;background:rgba(255,71,87,.08);margin-bottom:6px;line-height:1.5;"><span style="color:#ff4757;font-weight:600;">利好</span> {n.get("title", "")} <span style="color:#6b8db5;font-size:11px;">[{src} {time_str}]</span></div>\n'
            for n in bear_news:
                src = n.get('source', '') or ''
                time_str = (n.get('publish_time', '') or '')[:10]
                news_section += f'    <div style="padding:10px 14px;border-radius:8px;font-size:13px;border-left:3px solid #00d4aa;background:rgba(0,212,170,.08);margin-bottom:6px;line-height:1.5;"><span style="color:#00d4aa;font-weight:600;">利空</span> {n.get("title", "")} <span style="color:#6b8db5;font-size:11px;">[{src} {time_str}]</span></div>\n'
            news_section += '  </div>\n'

        # --- Multi-cycle predictions section ---
        mc_section = ''
        st_p = mc.get('short_term', {})
        mt_p = mc.get('mid_term', {})
        lt_p = mc.get('long_term', {})
        if st_p.get('direction') or mt_p.get('direction') or lt_p.get('direction'):
            mc_section += '''  <div class="section">
    <div class="section-title">📅 多周期预测</div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
'''
            pct_fmt = lambda p: f"{'+' if (p or 0) > 0 else ''}{(p or 0):.1f}%" if p is not None else '-'
            dir_cls = lambda d: 'color:#ff4757;' if (d and '涨' in str(d)) else ('color:#00d4aa;' if (d and '跌' in str(d)) else 'color:#ffa502;')
            for label, p in [('短期 (1~2周)', st_p), ('中期 (1~3月)', mt_p), ('长期 (6~12月)', lt_p)]:
                mc_section += f'''      <div style="background:#152436;border:1px solid #1e3a5f;border-radius:10px;padding:16px;text-align:center;">
        <div style="font-size:11px;color:#6b8db5;">{label}</div>
        <div style="font-size:20px;font-weight:600;margin:4px 0;{dir_cls(p.get('direction'))}">{p.get('direction', '-')}</div>
        <div style="font-size:14px;font-weight:500;{dir_cls(p.get('direction'))}">{pct_fmt(p.get('change_pct'))}</div>
        <div style="font-size:11px;color:#6b8db5;margin-top:4px;">{p.get('reason', '')}</div>
      </div>
'''
            mc_section += '    </div>\n  </div>\n'

        # --- Action section ---
        action_section = ''
        if action.get('action'):
            act = action.get('action', '')
            act_icon = '▲' if ('买' in str(act) or '加' in str(act)) else ('▼' if ('卖' in str(act) or '减' in str(act)) else '◆')
            act_cls = 'color:#ff4757;' if ('买' in str(act) or '加' in str(act)) else ('color:#00d4aa;' if ('卖' in str(act) or '减' in str(act)) else 'color:#ffa502;')
            action_section += f'''  <div class="section">
    <div style="background:#152436;border:1px solid #ffa502;border-radius:12px;padding:18px;display:flex;gap:16px;align-items:center;">
      <div style="font-size:32px;{act_cls}">{act_icon}</div>
      <div style="flex:1;">
        <div style="font-size:18px;font-weight:600;{act_cls}">建议操作: {act}</div>
        <div style="font-size:13px;color:#8ab4f8;margin-top:4px;line-height:1.5;">{action.get('reason', '')}</div>
        <div style="font-size:12px;color:#6b8db5;margin-top:6px;">止损位: ¥{action.get('stop_loss', '-')} | 止盈位: ¥{action.get('take_profit', '-')}</div>
      </div>
    </div>
  </div>
'''

        # --- Valuation info ---
        val_level = s.get('valuation_level', '')
        val_pct = s.get('valuation_percentile')
        buy_price = s.get('suggested_buy_price')
        val_section = ''
        if val_level or buy_price is not None:
            val_section += '''  <div class="section">
    <div class="section-title">💰 估值与建议买点</div>
    <div class="grid">
'''
            if val_level:
                val_clr = '#ff4757' if val_level in ('很低', '偏低') else ('#00d4aa' if val_level in ('偏高', '很高') else '#ffa502')
                val_section += f'''      <div class="card">
        <div class="label">估值水平</div>
        <div class="value" style="color:{val_clr};">{val_level}</div>
        <div class="sub">PE分位: {val_pct if val_pct is not None else '-'}%</div>
      </div>
'''
            if buy_price is not None:
                cur_price = s.get('current_price', 0)
                gap = f"{(buy_price / cur_price - 1) * 100:.1f}%" if cur_price else ''
                val_section += f'''      <div class="card">
        <div class="label">建议买入价</div>
        <div class="value" style="color:#ff6348;">¥{buy_price:.2f}</div>
        <div class="sub">距现价 {gap}</div>
      </div>
'''
            cost = s.get('cost_price', 0)
            if cost > 0:
                val_section += f'''      <div class="card">
        <div class="label">成本价格</div>
        <div class="value" style="color:#ffa502;">¥{cost:.2f}</div>
        <div class="sub">输入成本</div>
      </div>
'''
            val_section += '    </div>\n  </div>\n'

        # --- Main HTML ---
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
  .analysis {{ background: #152436; border-radius: 8px; padding: 16px; line-height: 1.6; }}
  .mirofish-report h1 {{ font-size: 22px; color: #ffa502; border-bottom: 1px solid #2a3f5a; padding-bottom: 8px; margin: 24px 0 14px; }}
  .mirofish-report h2 {{ font-size: 19px; color: #ffa502; margin: 20px 0 12px; }}
  .mirofish-report h3 {{ font-size: 17px; color: #8ab4f8; margin: 18px 0 10px; }}
  .mirofish-report h4, .mirofish-report h5, .mirofish-report h6 {{ font-size: 15px; color: #8ab4f8; margin: 14px 0 8px; }}
  .mirofish-report p {{ margin: 8px 0; }}
  .mirofish-report ul, .mirofish-report ol {{ margin: 8px 0; padding-left: 24px; }}
  .mirofish-report li {{ margin: 4px 0; }}
  .mirofish-report table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }}
  .mirofish-report th {{ background: #1a2d42; color: #8ab4f8; padding: 8px 12px; text-align: left; border: 1px solid #1e3a5f; font-weight: 600; }}
  .mirofish-report td {{ padding: 8px 12px; border: 1px solid #1e3a5f; }}
  .mirofish-report tr:nth-child(even) {{ background: rgba(26,45,66,.5); }}
  .mirofish-report blockquote {{ border-left: 3px solid #ffa502; margin: 12px 0; padding: 8px 16px; background: rgba(255,165,2,.08); color: #c0c8d0; }}
  .mirofish-report code {{ background: #0f1923; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 13px; color: #ffa502; }}
  .mirofish-report pre {{ background: #0f1923; padding: 12px; border-radius: 8px; overflow-x: auto; margin: 12px 0; }}
  .mirofish-report pre code {{ background: none; padding: 0; color: #e0e6ed; }}
  .mirofish-report a {{ color: #ff6348; text-decoration: underline; }}
  .mirofish-report hr {{ border: none; border-top: 1px solid #1e3a5f; margin: 20px 0; }}
  .mirofish-report strong {{ color: #fff; }}
  .mirofish-report em {{ color: #c0c8d0; }}
  .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin: 2px; }}
  .tag-positive {{ background: rgba(255,71,87,.15); color: #ff4757; }}
  .tag-negative {{ background: rgba(0,212,170,.15); color: #00d4aa; }}
  .footer {{ text-align: center; color: #6b8db5; font-size: 12px; margin-top: 40px; }}
</style></head>
<body>
  <h1>{report['title']}</h1>
  <div class="summary">
    <div style="font-size: 48px; text-align: center;">{outlook_icon.get(s.get('outlook', '中性'), '➡️')}</div>
    <div style="text-align: center; font-size: 14px; color: #6b8db5; margin-top: 8px;">
      信号: <span style="color:{color};font-weight:600;">{s.get('overall_signal', 'neutral')}</span>
      | 综合评分: {s.get('signal_score', '-')}
      | 置信度: {s.get('confidence', '-')} ({s.get('confidence_score', 0)}%)
      | 生成: {report['generated_at'][:16]}
    </div>
  </div>

{val_section}
  <div class="section">
    <div class="section-title">价格预测</div>
    <div style="text-align:center;padding:16px;">
      <span style="color:#6b8db5;">目标区间</span><br/>
      <span class="price-target">{s.get('price_target_low', '-')}</span>
      <span style="font-size:20px;color:#6b8db5;"> ~ </span>
      <span class="price-target">{s.get('price_target_high', '-')}</span>
      <br/><span style="color:#6b8db5;font-size:14px;">当前: {s.get('current_price', '-')}</span>
    </div>
  </div>

  <div class="grid">
    <div class="card"><div class="label">PE / PB</div><div class="value">{mq.get('pe', '-')} / {mq.get('pb', '-')}</div></div>
    <div class="card"><div class="label">市值</div><div class="value">{mq.get('market_cap', '-')}亿</div></div>
    <div class="card"><div class="label">EPS / ROE</div><div class="value">{mfin.get('eps', '-')} / {mfin.get('roe', '-')}%</div></div>
    <div class="card"><div class="label">RSI / MACD</div><div class="value">{mtech.get('rsi_14', '-')} / {mtech.get('macd', '-')}</div></div>
    <div class="card"><div class="label">舆情</div><div class="value">新闻{sn.get('news_score', '-')} 股吧{sn.get('guba_score', '-')}</div></div>
    <div class="card"><div class="label">换手率</div><div class="value">{mq.get('turnover_rate', '-')}%</div></div>
  </div>

{mc_section}
{action_section}
{cio_section}
{emp_section}
{score_section}
{news_section}
  <div class="section">
    <div class="section-title">风险因素</div>
    {''.join(f'<span class="tag tag-negative">{f}</span> ' for f in ri.get('risk_factors', [])) or '暂无'}
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
    StockFish AI Analysis · {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>
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
