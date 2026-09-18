#!/usr/bin/env python3
"""Descriptive cross-model tables, preserving every registered comparison.

Requires released complete runs and their final independent audit receipts.
No cross-model raw-margin averages or inferential thresholds are introduced.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics import cross_model_applicability_execution_v1 as c

KEYS = ['qwen3-8b', 'qwen3-14b', 'glm4-9b-chat']


def agreement(signs):
    if 'positive' in signs and 'negative' in signs:
        return 'resolved_direction_conflict'
    if set(signs) == {'numerical_unresolved'}:
        return 'all_numerical_unresolved'
    if 'numerical_unresolved' in signs:
        return 'includes_numerical_unresolved'
    assert len(set(signs)) == 1
    return 'same_resolved_direction'


def tsv(path, rows):
    assert rows
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(rows)


def build(output):
    assert not output.exists(), 'New result directory required'
    frames = {'new': {}, 'legacy': {}}
    sources = []
    for key in KEYS:
        version = '03' if key.startswith('glm') else '01'
        run = c.WORK / f'run-{key}-{version}'
        state = c.read(run / 'state.json')
        assert state['status'] == 'complete' and state['worker_exit_code'] == 0 and state['owned_worker_absent']
        audit_path = c.WORK / 'audits' / f'{key}-results-01.json'
        audit = c.read(audit_path)
        assert audit['status'] == 'pass' and audit['run_status'] == 'complete' and audit['model_key'] == key
        assert audit['binding'] == c.info(run / 'binding.json')
        for a in audit['analyses']:
            c.verify(a['source'])
            frame = 'new' if Path(a['source']['path']).name.startswith('new-') else 'legacy'
            assert key not in frames[frame]
            frames[frame][key] = c.read(a['source']['path'])
            sources.append(a['source'])
        sources.extend([c.info(audit_path), c.info(run / 'state.json'), state['resource_release']])
    old_path = ROOT / 'reviews/cross-term-behavior-execution-v1/results-01/results.json'
    n_path = ROOT / 'reviews/cross-term-next-token-v1/results-01/results.json'
    for path in [old_path, n_path]:
        manifest_path = path.parent / 'manifest.json'
        manifest = c.read(manifest_path)
        found = [r for r in manifest['artifacts'] if Path(r['path']).name == 'results.json']
        assert len(found) == 1
        c.verify(found[0]); assert found[0]['sha256'] == c.sha(path)
        sources.extend([c.info(path), c.info(manifest_path)])
    old, original = c.read(old_path), c.read(n_path)
    ns = [r for r in original['scores'] if r['condition_id'] in old['historical_N']]
    assert len(old['scores']) == 120 and len(ns) == 36
    frames['legacy']['qwen3-8b'] = {'scores': old['scores'] + ns, 'comparisons': old['expressions']}
    tables, summaries = {}, {}
    for frame, models in frames.items():
        assert set(models) == set(KEYS)
        scores = {key: {r['condition_id']: r for r in data['scores']} for key, data in models.items()}
        comps = {key: {r['comparison_id']: r for r in data['comparisons']} for key, data in models.items()}
        expected_scores, expected_comps = (384, 1152) if frame == 'new' else (156, 168)
        assert all(set(scores[key]) == set(scores[KEYS[0]]) and len(scores[key]) == expected_scores for key in KEYS)
        assert all(set(comps[key]) == set(comps[KEYS[0]]) and len(comps[key]) == expected_comps for key in KEYS)
        comparison_rows, score_rows, query_rows = [], [], []
        buckets = defaultdict(Counter); kind_buckets = defaultdict(Counter)
        for cid, base in sorted(comps[KEYS[0]].items()):
            role = base.get('analysis_role', 'legacy_exposed_development')
            row = {k: base[k] for k in ['comparison_id', 'query_id', 'kind', 'priority']}
            row['analysis_role'] = role
            signs = []
            for key in KEYS:
                r = comps[key][cid]
                assert r['terms'] == base['terms'] and r['query_id'] == base['query_id'] and r['kind'] == base['kind']
                for k in ['value', 'bound', 'resolution']:
                    row[f'{key}_{k}'] = r['effect'][k]
                row[f'{key}_reference_aligned_change'] = r['reference_aligned_change']
                row[f'{key}_classification_transition'] = r['verified_classification_transition']
                signs.append(r['effect']['resolution'])
            row['descriptive_direction_agreement'] = agreement(signs)
            comparison_rows.append(row)
            buckets[role, base['priority']][row['descriptive_direction_agreement']] += 1
            kind_buckets[role, base['priority'], base['kind']][row['descriptive_direction_agreement']] += 1
        qgroups = defaultdict(list)
        for cid, base in sorted(scores[KEYS[0]].items()):
            role = base.get('analysis_role', 'legacy_exposed_development')
            subframe = base.get('source_frame', 'legacy_N_diagnostic' if cid in old['historical_N'] else 'legacy_core')
            row = {'condition_id': cid, 'query_id': base['query_id'], 'family_id': base['family_id'],
                   'analysis_role': role, 'source_frame': subframe, 'adopted_reference': base['adopted_reference']}
            for key in KEYS:
                r = scores[key][cid]
                assert r['query_id'] == base['query_id'] and r['adopted_reference'] == base['adopted_reference']
                for k in ['m', 'margin_error_bound', 'raw_prediction', 'raw_correct', 'conservative_correct', 'legal_mass']:
                    row[f'{key}_{k}'] = r[k]
                row[f'{key}_physical_score_id'] = r['physical_score_id']
                row[f'{key}_origin'] = 'historical_science_reused' if frame == 'legacy' and key == KEYS[0] else 'new_sealed_production'
            score_rows.append(row); qgroups[base['query_id'], subframe].append(row)
        for (qid, subframe), group in sorted(qgroups.items()):
            row = {'query_id': qid, 'source_frame': subframe, 'analysis_role': group[0]['analysis_role'],
                   'adopted_reference': group[0]['adopted_reference'], 'correlated_conditions': len(group)}
            for key in KEYS:
                row[f'{key}_raw_correct'] = sum(r[f'{key}_raw_correct'] for r in group)
                row[f'{key}_conservative_correct'] = sum(r[f'{key}_conservative_correct'] for r in group)
            query_rows.append(row)
        tables[frame] = {'comparisons': comparison_rows, 'scores': score_rows, 'queries': query_rows}
        summaries[frame] = {'scores_per_model': expected_scores, 'comparisons_per_model': expected_comps,
            'groups': [{'analysis_role': role, 'priority': priority, 'counts': dict(counts), 'total': sum(counts.values())}
                       for (role, priority), counts in sorted(buckets.items())],
            'by_kind': [{'analysis_role': role, 'priority': priority, 'kind': kind, 'counts': dict(counts), 'total': sum(counts.values())}
                        for (role, priority, kind), counts in sorted(kind_buckets.items())]}
    output.mkdir()
    for frame, tableset in tables.items():
        for name, rows in tableset.items():
            tsv(output / f'{frame}-{name}.tsv', rows)
    report = {'schema_version': 'cross-model-applicability-descriptive-comparison/v1', 'created_at': c.now(),
              'models': KEYS, 'summaries': summaries,
              'scope': 'Two naturalistic primary development families and two explanatory supplementary families, kept separate. One conservative dependency cluster; no confirmation claim or new statistical/equivalence threshold.',
              'cross_model_raw_margin_averages': False,
              'direction_agreement_is_descriptive_not_independent_replication': True,
              'old_8b_scientific_endpoints_and_bounds_preserved': True,
              'GLM_native_content_boundary_amendment': c.info(c.WORK / 'prepared-glm-content-boundary-01/amendment.json'),
              'all_registered_comparisons_preserved': True, 'sources': sources}
    c.atomic(output / 'comparison-summary.json', report)
    c.atomic(output / 'manifest.json', {'schema_version': 'cross-model-applicability-comparison-artifacts/v1',
             'created_at': c.now(), 'artifacts': [c.info(p) for p in sorted(output.iterdir()) if p.is_file()],
             'sources': [c.info(Path(__file__).resolve()), *sources]})
    return {frame: data['groups'] for frame, data in summaries.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=c.WORK / 'comparison-01')
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), ensure_ascii=False, indent=2))
