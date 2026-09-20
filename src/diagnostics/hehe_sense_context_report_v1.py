"""Offline viewer generation and post-release CPU analysis. No GPU imports."""
from __future__ import annotations

import base64
import csv
import io
from pathlib import Path

import numpy as np

from diagnostics import hehe_sense_context_inputs_v1 as c
from diagnostics.case_attention_capture_v1 import aggregate


def component_requests(prepared, requests):
    """Add reviewed semantic subdivisions for analysis only; native inputs unchanged."""
    from copy import deepcopy
    records = {r['request_id']: r for r in c.read(Path(prepared) / 'definition-components.json')['records']}
    display = deepcopy(requests)
    for req in display:
        item = records[req['request_id']]
        for part in item['components']:
            req['spans'].append({'id': 'lex-0419:component:' + part['component'],
                'kind': 'lexicon_definition_component', 'parent': item['parent_span_id'],
                'label': '嘿嘿·' + {'original': '原释义', 'ordinary': '普通义'}[part['component']],
                'char_start': part['char_start'], 'char_end': part['char_end'],
                'token_positions': part['owned_token_positions']})
        if item['separator_or_cross_boundary_tokens']:
            req['spans'].append({'id': 'lex-0419:component:boundary',
                'kind': 'lexicon_definition_component', 'parent': item['parent_span_id'],
                'label': '嘿嘿·分段边界', 'char_start': None, 'char_end': None,
                'token_positions': item['separator_or_cross_boundary_tokens']})
    return display


def viewer_data(requests, status, scores=None):
    return {'schema_version': 'case-attention-view/v1', 'status': status, 'roles': dict(zip(c.ROLES, c.ROLE_LABELS)), 'role_order': list(c.ROLES),
            'layers': 36, 'heads': 32, 'scores': scores or {}, 'requests': requests,
            'query_labels': {'Q01':'Q01 · #3169 普通聊天','Q02':'Q02 · #3660 攻击','Q03':'Q03 · 引用并反对'},
            'notes': ['三条依赖查询、九个输入；探索性案例，非独立确认集。',
                      '双义版本更长且原义在前、普通义在后；长度、顺序和词形重复同时变化。',
                      'Q02另含性经历贬损；Q03为已采用的AI反驳改写。',
                      '灰色表示未提供或不可见；无示例时示例结束为结构性NA。',
                      '嘿嘿完整释义与两段子片段重叠，不能相加；段间边界另列。',
                      '注意力没有正负标签意义，也不证明因果贡献。']}


def write_viewer(path, data):
    template = (c.ROOT / 'tools/hehe_sense_context_viewer_v1/viewer.html').read_text(encoding='utf-8')
    # Prevent a real source string from terminating the non-executable JSON tag.
    payload = c.canonical(data).decode('utf-8').replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    c.require(template.count('__DATA_JSON__') == 1, 'Viewer template placeholder differs')
    Path(path).write_text(template.replace('__DATA_JSON__', payload), encoding='utf-8')


def preview(prepared):
    prepared = Path(prepared)
    path = prepared / 'viewer.html'
    c.require(not path.exists(), 'Do not overwrite a delivered viewer')
    write_viewer(path, viewer_data(component_requests(prepared, c.lines(prepared / 'inputs.jsonl')), 'GPU未运行：这里只展示真实材料和已冻结位置，不含注意力测量'))
    return c.info(path)


def analyze(prepared, run, output):
    from diagnostics.hehe_sense_context_runtime_v1 import check_run, load_record
    from diagnostics.cross_model_applicability_execution_v1 import readout
    prepared, run, out = map(Path, (prepared, run, output))
    checked = check_run(prepared, run)
    c.require(checked['status'] == 'complete', 'Full run must be normally released before analysis')
    c.require(not out.exists(), 'New analysis directory required')
    _, profile, requests = c.validate(prepared)
    requests = component_requests(prepared, requests)
    refs = {r['query_id']: r for r in c.read(prepared / 'analysis-references.json')['references']}
    q = checked['qualification']
    scores, summaries, compact = {}, [], []
    out.mkdir(parents=True)
    for req in requests:
        rec, vector, attention = load_record(run, 'production', req, c.sha(run / 'binding.json'), profile)
        score = readout(vector, profile['candidate_tokens'], q['margin_error_bound'])
        score.update(request_id=req['request_id'], query_id=req['query_id'], condition=req['condition'],
                     reference=refs[req['query_id']]['reference'], raw_reference_correct=score['raw_prediction'] == refs[req['query_id']]['reference'])
        score['dictionary_id'] = req['dictionary_id']
        score['reference_aligned_margin'] = score['m'] if score['reference'] == '无' else -score['m']
        score['resolved_reference_correct'] = score['raw_reference_correct'] if score['resolution'] != 'numerical_unresolved' else None
        scores[req['request_id']] = score
        aggregated = aggregate(attention, req)
        for row in aggregated:
            row['mass_error_bound'] = q['attention_mass_bound'] if row['visible'] else None
            row['density_error_bound'] = q['attention_mass_bound'] / row['mean_visible_tokens'] if row['visible'] else None
        c.write(out / (req['request_id'] + '-aggregates.json'), {'request_id': req['request_id'], 'aggregates': aggregated,
                 'attention_mass_bound': q['attention_mass_bound'], 'source': rec['attention']})
        # A visualization copy only. Native maps and FP64 aggregation remain the
        # authoritative numeric artifacts; float32 display is explicitly labeled.
        little = np.asarray(attention, dtype='<f4')
        display = {'request_id': req['request_id'], 'prompt_sha256': req['prompt_sha256'],
                   'shape': list(little.shape), 'encoding': 'little-endian-float32-base64-visualization-only',
                   'data': base64.b64encode(little.tobytes()).decode('ascii'), 'score': score,
                   'source': rec['attention'], 'aggregates': aggregated}
        c.write(out / (req['request_id'] + '.view.json'), display)
        for a in aggregated:
            if a['visible']:
                mass=np.asarray(a['mass'],dtype=np.float64);density=np.asarray(a['density'],dtype=np.float64)
                compact.append({'request_id':req['request_id'],'role':a['role'],'span_id':a['span_id'],
                                'all_layer_head_mean_mass':float(mass.mean()),'all_layer_head_mean_density':float(density.mean()),
                                'late_18_35_mean_mass':float(mass[18:36].mean()) if mass.shape[0]>=36 else None,
                                'late_18_35_mean_density':float(density[18:36].mean()) if density.shape[0]>=36 else None,
                                'mean_visible_tokens':a['mean_visible_tokens']})
                for layer in range(profile['layers']):
                    summaries.append({'request_id': req['request_id'], 'role': a['role'], 'span_id': a['span_id'], 'layer': layer,
                                      'mean_head_mass': float(np.mean(a['mass'][layer])),
                                      'mean_head_density': float(np.mean(a['density'][layer])),
                                      'mean_visible_tokens': a['mean_visible_tokens']})
    comparisons = []
    for expression in c.read(prepared/'comparisons.json')['comparisons']:
        coefficients={term['request_id']:term['coefficient'] for term in expression['terms']}
        value=sum(coefficient*scores[rid]['m'] for rid,coefficient in coefficients.items())
        bound=sum(abs(coefficient)*scores[rid]['margin_error_bound'] for rid,coefficient in coefficients.items())
        transition=None
        if len(coefficients)==2 and sorted(coefficients.values())==[-1,1]:
            left=scores[next(k for k,v in coefficients.items() if v==1)]
            right=scores[next(k for k,v in coefficients.items() if v==-1)]
            transition='unresolved' if 'numerical_unresolved' in (left['resolution'],right['resolution']) else (
                'repair' if left['raw_reference_correct'] and not right['raw_reference_correct'] else
                'damage' if right['raw_reference_correct'] and not left['raw_reference_correct'] else 'unchanged')
        reference = left['reference']
        comparisons.append(dict(expression,reference_aligned_delta=value if reference == '无' else -value,delta_m=value,bound=bound,direction_resolved=abs(value)>bound,
                                direction='positive' if value>bound else 'negative' if value < -bound else 'unresolved',
                                transition=transition,value=value,numerical_bound=bound))
    c.write(out/'attention-summary.json',compact)
    c.write(out / 'scores.json', list(scores.values()))
    c.write(out / 'comparisons.json', comparisons)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), delimiter='\t')
    writer.writeheader()
    writer.writerows(summaries)
    (out / 'layer-summary.tsv').write_text(stream.getvalue(), encoding='utf-8')
    write_viewer(out / 'index.html', viewer_data(requests, 'GPU原始结果已封存，工作进程已正常退出；本页为CPU分析', scores))
    c.write(out / 'audit.json', {'CPU_reconstruction_passed': True, 'query_references_joined_after_release': True,
        'release': c.read(run / 'state.json')['resource_release'], 'raw_seal': c.info(run / 'raw-seal.json'),
        'prepared_manifest': c.info(prepared / 'manifest.json'), 'independent_cases_claimed': False, 'query_count': 3, 'construction_dependency': 'Q03 derives from Q02; all exposed exploration',
        'attention_is_causal_explanation': False, 'display_copy_precision': 'float32', 'analysis_precision': 'float64'})
    c.write(out / 'manifest.json', {'artifacts': [c.info(p) for p in sorted(out.iterdir()) if p.is_file()]})
    return {'output': str(out), 'conditions': len(scores), 'comparisons': len(comparisons)}
