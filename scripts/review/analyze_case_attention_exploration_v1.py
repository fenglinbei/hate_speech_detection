#!/usr/bin/env python3
"""CPU-only, post-hoc description of sealed case attention; no new inference."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'reviews/case-attention-v1'
OUT = WORK / 'interpretation-01'
ROLES = ('pre_answer', 'lexicon_end', 'demos_end', 'query_end', 'query_all', 'query_focal')
TARGETS = {'541': 'lex-0215', '3169': 'lex-0419'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def info(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4194304), b''):
            h.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def main():
    assert not OUT.exists(), 'Preserve prior interpretation snapshots'
    state = read(WORK / 'run-01/state.json')
    assert state['status'] == 'complete' and state['owned_worker_absent'] and state['worker_exit_code'] == 0
    prior = read(ROOT / 'docs/research/experiment-plans/case-attention-v1/results-current.json')
    assert info(prior['closeout_manifest']['path']) == prior['closeout_manifest']
    for item in read(WORK / 'results-01/manifest.json')['artifacts']:
        assert info(item['path']) == item
    reqs = [json.loads(x) for x in (WORK / 'prepared-01/inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    scores = read(WORK / 'results-01/scores.json')
    summaries, raw_tokens, heads, sources = [], [], {}, []
    for req in reqs:
        rid = req['request_id']
        path = WORK / 'results-01' / (rid + '-aggregates.json')
        sources.append(info(path))
        aggregates = read(path)['aggregates']
        for role in ROLES:
            rows = [a for a in aggregates if a['role'] == role and a['visible']]
            data = {'request_id': rid, 'query_id': req['query_id'], 'condition': req['condition'], 'role': role,
                    'role_available': bool(req['roles'][role]), 'groups': {}, 'target': None, 'ranked_demos': {}}
            for kind in sorted(set(a['kind'] for a in rows)):
                group = [a for a in rows if a['kind'] == kind]
                mass = sum(float(np.mean(a['mass'])) for a in group)
                tokens = sum(a['mean_visible_tokens'] for a in group)
                data['groups'][kind] = {'mean_mass': mass, 'available_tokens': tokens, 'pooled_density': mass / tokens}
            lex = [a for a in rows if a['kind'] == 'lexicon']
            target = next((a for a in lex if a['span_id'] == TARGETS[req['query_id']]), None)
            if target:
                mass = np.asarray(target['mass'])
                peak = np.unravel_index(np.argmax(mass), mass.shape)
                data['target'] = {'span_id': target['span_id'], 'label': target['label'], 'mean_mass': float(mass.mean()),
                    'mean_density': float(np.mean(target['density'])), 'tokens': target['mean_visible_tokens'],
                    'rank_mass': int(1 + sum(np.mean(a['mass']) > mass.mean() for a in lex)),
                    'rank_density': int(1 + sum(np.mean(a['density']) > np.mean(target['density']) for a in lex)),
                    'lexicon_count': len(lex), 'layer_means': mass.mean(1).tolist(),
                    'posthoc_peak_head': {'layer_zero_based': int(peak[0]), 'head_zero_based': int(peak[1]), 'mass': float(mass[peak])},
                    'components': [{'label': a['label'], 'kind': a['kind'], 'mean_mass': float(np.mean(a['mass'])),
                                    'mean_density': float(np.mean(a['density']))} for a in rows if a['span_id'].startswith(target['span_id'] + ':')]}
            demos = [a for a in rows if a['kind'] == 'demo']
            for metric in ('mass', 'density'):
                data['ranked_demos'][metric] = [{'span_id': a['span_id'], 'label': a['label'], 'mean_mass': float(np.mean(a['mass'])),
                    'mean_density': float(np.mean(a['density'])), 'tokens': a['mean_visible_tokens']}
                    for a in sorted(demos, key=lambda a: np.mean(a[metric]), reverse=True)]
            summaries.append(data)
        if req['condition'] not in ('LD', 'LDC'):
            continue
        record = read(WORK / 'run-01/records/production' / (rid + '.json'))
        assert info(record['attention']['path']) == record['attention']
        sources.append(record['attention'])
        raw = np.load(record['attention']['path'], allow_pickle=False)
        target = next(s for s in req['spans'] if s['id'] == TARGETS[req['query_id']])
        for role in ('query_focal', 'pre_answer'):
            index = ROLES.index(role)
            head = raw[:, :, index, target['token_positions']].sum(axis=-1)
            frozen = next(a for a in aggregates if a['role'] == role and a['span_id'] == target['id'])
            assert np.array_equal(head, np.asarray(frozen['mass']))
            if req['condition'] == 'LD':
                heads[(req['query_id'], role)] = head
            weights = raw[:, :, index].mean(axis=(0, 1))
            positions = list(map(int, np.argsort(-weights)[:20]))
            raw_tokens.append({'request_id': rid, 'role': role, 'top20_posthoc': [
                {'position': k, 'text': req['token_text'][k], 'owner': req['token_owners'][k], 'mean_mass': float(weights[k])} for k in positions]})
    OUT.mkdir()
    metrics = {'scope': 'Post-hoc descriptive CPU analysis; two exposed cases; peaks/ranks are not confirmed causal units.',
               'averaging': 'Equal average over all 36 layers and 32 heads, then sum disjoint spans within each kind.',
               'group_warning': 'Whole demo/lexicon containers overlap components and some structure tokens; never sum all kinds together.',
               'source_closeout': prior['closeout_manifest'], 'sources': sources, 'summaries': summaries, 'token_rankings': raw_tokens}
    (OUT / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/case-attention-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    conditions = ('C0', 'D', 'L', 'LD', 'LC', 'LDC')
    for ax, qid in zip(axes, ('541', '3169')):
        values = [next(s['m'] for s in scores if s['query_id'] == qid and s['condition'] == c) for c in conditions]
        bars = ax.bar(conditions, values, color=['#19857b' if x > 0 else '#be5939' for x in values])
        ax.axhline(0, color='#555', lw=1)
        ax.set_ylim(-27, 34); ax.set_title(f'#{qid} · 参考为“无”', fontproperties=font)
        ax.set_ylabel('m = z(无) − z(有)', fontproperties=font)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + (1 if value >= 0 else -1), f'{value:+.2f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=9)
    fig.suptitle('加入资料后的行为变化：正值偏“无”，负值偏“有”', fontproperties=font)
    fig.savefig(OUT / 'behavior.png', dpi=160); fig.savefig(OUT / 'behavior.pdf'); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for i, qid in enumerate(('541', '3169')):
        for j, role in enumerate(('query_focal', 'pre_answer')):
            ax = axes[i, j]
            im = ax.imshow(heads[(qid, role)], aspect='auto', vmin=0, vmax=1, cmap='viridis')
            ax.set_title(f'#{qid} · ' + ('查询焦点词位置' if role == 'query_focal' else '回答前'), fontproperties=font)
            ax.set_xlabel('Head (0-based)'); ax.set_ylabel('Layer (0-based)')
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Attention mass to matching dictionary entry', shrink=.7)
    fig.suptitle('LD · 对“男同／嘿嘿”整条词条的注意力：全部层与头\n仅作描述；焦点词读数为指定查询 token 的平均', fontproperties=font)
    fig.savefig(OUT / 'matched-entry-heads.png', dpi=160); fig.savefig(OUT / 'matched-entry-heads.pdf'); plt.close(fig)
    receipt = {'status': 'pass', 'CPU_only': True, 'source': info(Path(__file__)), 'source_closeout': prior['closeout_manifest'],
               'requests': 12, 'role_summaries': len(summaries), 'raw_vs_aggregate_maps_exact': 8,
               'all_source_hashes_verified': True, 'artifacts': [info(p) for p in sorted(OUT.iterdir())]}
    (OUT / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('source', 'artifacts')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
