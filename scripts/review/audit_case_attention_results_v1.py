#!/usr/bin/env python3
"""Independent CPU audit and static figures for a normally released result set."""
from __future__ import annotations

import argparse
import base64
import csv
from decimal import Decimal, getcontext
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROLES = ('pre_answer', 'lexicon_end', 'demos_end', 'query_end', 'query_all', 'query_focal')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def info(p):
    p = Path(p)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return {'path': str(p.resolve()), 'bytes': p.stat().st_size, 'sha256': h.hexdigest()}


def verify(item):
    assert info(item['path']) == item, item['path']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=ROOT / 'reviews/case-attention-v1/prepared-01')
    parser.add_argument('--run', type=Path, default=ROOT / 'reviews/case-attention-v1/run-01')
    parser.add_argument('--results', type=Path, default=ROOT / 'reviews/case-attention-v1/results-01')
    parser.add_argument('--output', type=Path, default=ROOT / 'reviews/case-attention-v1/report-01')
    args = parser.parse_args()
    assert not args.output.exists(), 'Never overwrite a delivered audit/report'
    state = read(args.run / 'state.json')
    assert state['status'] == 'complete' and state['worker_exit_code'] == 0 and state['owned_worker_absent']
    verify(state['resource_release'])
    for invocation in state['invocations']:
        assert not Path('/proc', str(invocation['worker_pid'])).exists()
        assert not Path('/proc', str(invocation['controller_pid'])).exists()
    manifest = read(args.results / 'manifest.json')
    for artifact in manifest['artifacts']:
        verify(artifact)
    audit = read(args.results / 'audit.json')
    assert audit['query_references_joined_after_release']
    for key in ('raw_seal', 'prepared_manifest', 'release'):
        verify(audit[key])
    reqs = [json.loads(x) for x in (args.prepared / 'inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    profile = read(args.prepared / 'model-profile.json')
    qualification = read(args.run / 'qualification.json')
    scores = {x['request_id']: x for x in read(args.results / 'scores.json')}
    getcontext().prec = 60
    D = Decimal.from_float
    exact_scores, aggregation_samples, stage_rows, measured = {}, 0, [], {}
    maxima = {'row_sum_error': 0., 'aggregation_error': 0., 'decimal_aggregation_error': 0., 'score_error': 0., 'comparison_error': 0.}
    audited_elements = 0
    for req in reqs:
        rid = req['request_id']
        record = read(args.run / 'records/production' / (rid + '.json'))
        verify(record['vector']); verify(record['attention'])
        v = np.load(record['vector']['path'], allow_pickle=False)
        a = np.load(record['attention']['path'], allow_pickle=False)
        assert a.shape == (36, 32, 6, req['prompt_tokens']) and a.dtype == np.float64
        assert np.isfinite(a).all() and np.min(a) >= 0
        m = D(float(v[profile['candidate_tokens']['无']])) - D(float(v[profile['candidate_tokens']['有']]))
        exact_scores[rid] = m
        maxima['score_error'] = max(maxima['score_error'], float(abs(D(scores[rid]['m']) - m)))
        assert scores[rid]['raw_prediction'] == ('无' if m > 0 else '有')
        assert scores[rid]['margin_error_bound'] == qualification['margin_error_bound']
        stored = read(args.results / (rid + '-aggregates.json'))
        rows = {(x['role'], x['span_id']): x for x in stored['aggregates']}
        measured[rid] = rows
        display = read(args.results / (rid + '.view.json'))
        assert display['prompt_sha256'] == req['prompt_sha256'] and display['shape'] == list(a.shape)
        encoded = np.frombuffer(base64.b64decode(display['data']), dtype='<f4').reshape(a.shape)
        assert np.array_equal(encoded, a.astype('<f4'))
        for ri, role in enumerate(ROLES):
            positions = req['roles'][role]
            if positions:
                error = float(np.max(np.abs(a[:, :, ri].sum(-1) - 1)))
                maxima['row_sum_error'] = max(maxima['row_sum_error'], error)
                assert np.count_nonzero(a[:, :, ri, max(positions) + 1:]) == 0
            else:
                assert np.count_nonzero(a[:, :, ri]) == 0
            for span in req['spans']:
                keys = span['token_positions']
                available = math.fsum(sum(k <= q for k in keys) for q in positions) / len(positions) if positions else 0
                row = rows[(role, span['id'])]
                assert row['visible'] == bool(available) and row['mean_visible_tokens'] == available
                if not available:
                    assert row['mass'] is None and row['density'] is None
                    continue
                # Independent, ordered key accumulation for every layer and head.
                mass = np.zeros((36, 32), dtype=np.float64)
                for key in keys:
                    mass += a[:, :, ri, key]
                error = max(float(np.max(np.abs(mass - row['mass']))), float(np.max(np.abs(mass / available - row['density']))))
                maxima['aggregation_error'] = max(maxima['aggregation_error'], error)
                audited_elements += mass.size * 2
                assert row['mass_error_bound'] == qualification['attention_mass_bound']
                assert row['density_error_bound'] == qualification['attention_mass_bound'] / available
                # Decimal endpoints, centre and largest head cover all visible spans/roles.
                indices = {(0, 0), (17, 15), (35, 31), tuple(np.unravel_index(np.argmax(mass), mass.shape))}
                for layer, head in indices:
                    exact = sum((D(float(a[layer, head, ri, k])) for k in keys), Decimal(0))
                    err = max(abs(D(row['mass'][layer][head]) - exact),
                              abs(D(row['density'][layer][head]) - exact / D(available)))
                    maxima['decimal_aggregation_error'] = max(maxima['decimal_aggregation_error'], float(err))
                    aggregation_samples += 2
                if span['kind'] in ('demo', 'lexicon'):
                    stage_rows.append({'request_id': rid, 'role': role, 'span_id': span['id'], 'kind': span['kind'],
                                       'label': span['label'], 'mean_all_layers_heads_mass': float(mass.mean()),
                                       'mean_all_layers_heads_density': float((mass / available).mean())})
        del display, encoded, a
    for row in read(args.results / 'comparisons.json'):
        if row['name'] == 'L_D_interaction':
            m = {k: exact_scores[f'case-{row["query_id"]}-{k}'] for k in ('LD', 'D', 'L', 'C0')}
            expected = m['LD'] - m['D'] - m['L'] + m['C0']
            bound = 4 * D(qualification['margin_error_bound'])
        else:
            expected = exact_scores[row['left']] - exact_scores[row['right']]
            bound = 2 * D(qualification['margin_error_bound'])
            assert row['direction_resolved'] == (abs(expected) > bound)
            left, right = scores[row['left']], scores[row['right']]
            transition = 'repair' if left['raw_reference_correct'] and not right['raw_reference_correct'] else 'damage' if right['raw_reference_correct'] and not left['raw_reference_correct'] else 'unchanged'
            assert row['transition'] == transition
        assert D(row['bound']) == bound
        maxima['comparison_error'] = max(maxima['comparison_error'], float(abs(D(row['delta_m']) - expected)))
    assert maxima['score_error'] == maxima['comparison_error'] == 0
    assert maxima['row_sum_error'] <= 2e-6
    assert maxima['aggregation_error'] < 1e-13 and maxima['decimal_aggregation_error'] < 1e-13
    args.output.mkdir()
    with (args.output / 'stage-material-summary.tsv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(stage_rows[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(stage_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    for query in ('541', '3169'):
        for metric in ('mass', 'density'):
            fig, axes = plt.subplots(2, 3, figsize=(17, 13), constrained_layout=True)
            all_images = []
            for i, condition in enumerate(('LD', 'LDC')):
                req = next(r for r in reqs if r['query_id'] == query and r['condition'] == condition)
                spans = [s for s in req['spans'] if s['kind'] in ('lexicon', 'demo')]
                for j, role in enumerate(('lexicon_end', 'demos_end', 'pre_answer')):
                    matrix = np.full((36, len(spans)), np.nan)
                    for k, span in enumerate(spans):
                        row = measured[req['request_id']][(role, span['id'])]
                        if row['visible']:
                            matrix[:, k] = np.asarray(row[metric]).mean(axis=1)
                    all_images.append((axes[i, j], matrix, spans, condition, role))
            vmax = max(float(np.nanmax(x[1])) for x in all_images)
            cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#dddddd')
            for ax, matrix, spans, condition, role in all_images:
                im = ax.imshow(matrix, origin='upper', aspect='auto', vmin=0, vmax=vmax, cmap=cmap)
                ax.set_title(condition + ' · ' + {'lexicon_end':'词典结束', 'demos_end':'完整示例结束／查询前', 'pre_answer':'回答前'}[role], fontproperties=font)
                ax.set_xticks(range(len(spans)), [s['label'] for s in spans], rotation=90, fontproperties=font, fontsize=8)
                ax.set_yticks([0, 5, 11, 17, 23, 29, 35], [1, 6, 12, 18, 24, 30, 36]); ax.set_ylabel('Layer (1-based)')
            fig.colorbar(im, ax=axes.ravel().tolist(), label='Mean of 32 heads; original attention denominator', shrink=.7)
            title = '片段总注意力质量' if metric == 'mass' else '每可见 token 注意力'
            fig.suptitle(f'#{query} · {title}\nLD=词条释义＋示例；LDC=另含原始类别；灰色=尚不可见；阶段变化仅作描述', fontproperties=font, fontsize=15)
            for ext in ('png', 'pdf'):
                fig.savefig(args.output / f'case-{query}-{metric}.{ext}', dpi=160)
            plt.close(fig)
    receipt = {'status':'pass', 'CPU_only':True, 'source':info(Path(__file__)), 'results_manifest':info(args.results / 'manifest.json'),
               'checked_at_unix':time.time(), 'conditions':12, 'comparisons':14, 'decimal_precision':60,
               'all_aggregate_elements_checked':audited_elements, 'decimal_aggregate_values_checked':aggregation_samples,
               'all_display_arrays_equal_float32_copy':True, 'maxima':maxima,
               'figure_scope':'All LD/LDC dictionaries and demos; 36 layers and mean of all 32 heads; three fixed stages; shared color scale within each figure. Gray is structural NA. Exploratory, not causal.',
               'artifacts':[info(p) for p in sorted(args.output.iterdir())]}
    (args.output / 'independent-audit.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
