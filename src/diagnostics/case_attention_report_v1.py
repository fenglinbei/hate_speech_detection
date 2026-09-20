"""Offline viewer generation and post-release CPU analysis. No GPU imports."""
from __future__ import annotations

import base64
import csv
import io
from pathlib import Path

import numpy as np

from diagnostics import case_attention_inputs_v1 as c
from diagnostics.case_attention_capture_v1 import aggregate


def viewer_data(requests, status, scores=None):
    return {'schema_version': 'case-attention-view/v1', 'status': status, 'roles': dict(zip(c.ROLES, c.ROLE_LABELS)), 'role_order': list(c.ROLES),
            'layers': 36, 'heads': 32, 'scores': scores or {}, 'requests': requests,
            'notes': ['两个已暴露案例；各条件不是独立样本。', '灰色表示未提供或尚不可见，不能解释为模型忽略材料。',
                      '跨边界 token 单列；父片段与子片段有重叠，不能相加。',
                      '词典结束→示例结束的变化同时改变目标位置和可见前缀，仅作描述。',
                      '注意力大小没有正负标签含义，也不等于因果贡献。']}


def write_viewer(path, data):
    template = (c.ROOT / 'tools/case_attention_viewer_v1/viewer.html').read_text(encoding='utf-8')
    # Prevent a real source string from terminating the non-executable JSON tag.
    payload = c.canonical(data).decode('utf-8').replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    c.require(template.count('__DATA_JSON__') == 1, 'Viewer template placeholder differs')
    Path(path).write_text(template.replace('__DATA_JSON__', payload), encoding='utf-8')


def preview(prepared):
    prepared = Path(prepared)
    path = prepared / 'viewer.html'
    c.require(not path.exists(), 'Do not overwrite a delivered viewer')
    write_viewer(path, viewer_data(c.lines(prepared / 'inputs.jsonl'), 'GPU未运行：这里只展示真实材料和已冻结位置，不含注意力测量'))
    return c.info(path)


def analyze(prepared, run, output):
    from diagnostics.case_attention_runtime_v1 import check_run, load_record
    from diagnostics.cross_model_applicability_execution_v1 import readout
    prepared, run, out = map(Path, (prepared, run, output))
    checked = check_run(prepared, run)
    c.require(checked['status'] == 'complete', 'Full run must be normally released before analysis')
    c.require(not out.exists(), 'New analysis directory required')
    _, profile, requests = c.validate(prepared)
    refs = {r['query_id']: r for r in c.read(prepared / 'analysis-references.json')['references']}
    q = checked['qualification']
    scores, summaries = {}, []
    out.mkdir(parents=True)
    for req in requests:
        rec, vector, attention = load_record(run, 'production', req, c.sha(run / 'binding.json'), profile)
        score = readout(vector, profile['candidate_tokens'], q['margin_error_bound'])
        score.update(request_id=req['request_id'], query_id=req['query_id'], condition=req['condition'],
                     reference=refs[req['query_id']]['reference'], raw_reference_correct=score['raw_prediction'] == refs[req['query_id']]['reference'])
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
                for layer in range(profile['layers']):
                    summaries.append({'request_id': req['request_id'], 'role': a['role'], 'span_id': a['span_id'], 'layer': layer,
                                      'mean_head_mass': float(np.mean(a['mass'][layer])),
                                      'mean_head_density': float(np.mean(a['density'][layer])),
                                      'mean_visible_tokens': a['mean_visible_tokens']})
    comparisons = []
    for qid in ('541', '3169'):
        for left, right, name in [('D', 'C0', 'add_D'), ('L', 'C0', 'add_L'), ('LD', 'D', 'add_L_given_D'),
                                   ('LD', 'L', 'add_D_given_L'), ('LC', 'L', 'category_without_D'), ('LDC', 'LD', 'category_with_D')]:
            a, b = scores[f'case-{qid}-{left}'], scores[f'case-{qid}-{right}']
            value = a['m'] - b['m']
            bound = a['margin_error_bound'] + b['margin_error_bound']
            transition = 'unresolved' if 'numerical_unresolved' in (a['resolution'], b['resolution']) else (
                'repair' if a['raw_reference_correct'] and not b['raw_reference_correct'] else
                'damage' if b['raw_reference_correct'] and not a['raw_reference_correct'] else 'unchanged')
            comparisons.append({'query_id': qid, 'name': name, 'left': a['request_id'], 'right': b['request_id'],
                                'delta_m': value, 'bound': bound, 'direction_resolved': abs(value) > bound, 'transition': transition})
        a = {k: scores[f'case-{qid}-{k}'] for k in ('C0', 'D', 'L', 'LD')}
        comparisons.append({'query_id': qid, 'name': 'L_D_interaction',
            'delta_m': a['LD']['m'] - a['D']['m'] - a['L']['m'] + a['C0']['m'],
            'bound': sum(v['margin_error_bound'] for v in a.values()), 'scope': 'nonadditivity in this score scale only'})
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
        'prepared_manifest': c.info(prepared / 'manifest.json'), 'independent_cases': 2,
        'attention_is_causal_explanation': False, 'display_copy_precision': 'float32', 'analysis_precision': 'float64'})
    c.write(out / 'manifest.json', {'artifacts': [c.info(p) for p in sorted(out.iterdir()) if p.is_file()]})
    return {'output': str(out), 'conditions': len(scores), 'comparisons': len(comparisons)}
