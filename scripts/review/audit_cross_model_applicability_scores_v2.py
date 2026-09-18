#!/usr/bin/env python3
"""Read-only, independent Decimal audit after qualification and worker release.

This adds arithmetic evidence; it does not alter the frozen scorer or its gates.
Query references are read only when completed production results are supplied.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal, localcontext
import json
from pathlib import Path
import runpy
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics import cross_model_applicability_execution_v1 as c


def dec(value):
    return Decimal.from_float(float(value))


def sign(value, bound):
    return 'positive' if value > bound else 'negative' if value < -bound else 'numerical_unresolved'


def audit(run, results=None):
    binding = c.read(run / 'binding.json')
    entry = ROOT / 'scripts/review/run_cross_model_applicability_v1.py'
    if 'runtime_amendment' in binding:
        c.verify(binding['runtime_amendment'])
        amendment = c.read(binding['runtime_amendment']['path'])
        c.verify(amendment['entry_point'])
        entry = Path(amendment['entry_point']['path'])
        assert entry == ROOT / 'scripts/review/run_cross_model_applicability_utf8_v1.py'
    runner = runpy.run_path(str(entry))
    c.verify(binding['prepared_manifest'])
    prepared = Path(binding['prepared_manifest']['path']).parent
    verified = runner['check_run'](prepared, run)
    _, profiles, requests = c.check_prepared(prepared)
    profile = next(p for p in profiles if p['model_key'] == verified['model_key'])
    q = verified['qualification']
    passes = ['reference', 'repeat', 'left_padding', 'right_padding', 'reverse_order']
    expected_count = 2700
    if verified['status'] == 'complete':
        passes.append('production')
        expected_count += 384 if verified['model_key'] == 'qwen3-8b' else 540
    margins, receipts, counts = {}, {}, Counter()
    max_atom_error = Decimal(0)
    for stage in passes:
        margins[stage] = {}
        selected_ids = (profile['production_ids'] if stage == 'production' else
                        [r['request_id'] for r in requests[verified['model_key']]])
        for rid in sorted(selected_ids):
            path, _ = runner['paths'](run, stage, rid)
            r = c.read(path)
            vector = np.load(r['raw_logits']['path'], allow_pickle=False)
            yes = dec(vector[r['candidate_tokens']['有']])
            no = dec(vector[r['candidate_tokens']['无']])
            exact = no - yes
            atom_error = abs(exact - dec(r['readout']['m']))
            max_atom_error = max(max_atom_error, atom_error)
            assert atom_error == 0, 'FP32 candidate subtraction must be exact in FP64'
            assert yes == dec(r['readout']['z_yes']) and no == dec(r['readout']['z_no'])
            assert r['readout']['raw_prediction'] == ('无' if exact > 0 else '有' if exact < 0 else None)
            margins[stage][r['condition_id']] = exact
            receipts[stage, r['condition_id']] = r
            counts[stage] += 1
    assert sum(counts.values()) == expected_count
    assert all(counts[p] == 540 for p in passes if p != 'production')
    differences = {}
    for stage in passes[1:]:
        if stage == 'production':
            assert all(abs(v - margins['reference'][cid]) <= dec(q['margin_error_bound'])
                       for cid, v in margins[stage].items())
            continue
        assert set(margins[stage]) == set(margins['reference'])
        error = max(abs(v - margins['reference'][cid]) for cid, v in margins[stage].items())
        assert float(error) == q['max_margin_differences'][stage]
        differences[stage] = float(error)
    assert differences['repeat'] == differences['reverse_order'] == 0
    assert max(differences['left_padding'], differences['right_padding']) <= 0.001
    expected_bound = max(dec(1e-6), Decimal(2) * max(map(dec, differences.values())))
    assert float(expected_bound) == q['margin_error_bound']

    analyzed, max_expression_error = [], Decimal(0)
    if results is not None:
        assert verified['status'] == 'complete', 'No query reference join before raw seal and release'
        frames = [('new-development-results.json', c.SCIENCE / 'analysis-plan.json', True)]
        if verified['model_key'] != 'qwen3-8b':
            frames.append(('legacy-replication-results.json', prepared / 'legacy-analysis.json', False))
        for filename, plan_path, is_new in frames:
            result_path = results / filename
            out, plan = c.read(result_path), c.read(plan_path)
            selected = [r for r in verified['production_records']
                        if r['condition_id'].startswith('CMAD-') == is_new]
            reconstructed = c.production_analysis(selected, plan)
            reconstructed['model_key'] = verified['model_key']
            assert reconstructed == out, 'Complete analysis reconstruction differs'
            refs = {r['query_id']: r for r in plan['references']}
            by_id = {r['condition_id']: r for r in out['scores']}
            assert len(by_id) == len(out['scores']) == len(selected)
            for row in out['scores']:
                m = margins['production'][row['condition_id']]
                ref = refs[row['query_id']]['adopted_label']
                aligned = m if ref == '无' else -m
                assert float(aligned) == row['reference_aligned_margin']
                assert row['adopted_reference'] == ref
                correct = (m > 0 and ref == '无') or (m < 0 and ref == '有')
                assert row['raw_correct'] == correct
                assert row['conservative_correct'] == (correct and abs(m) > expected_bound)
                assert row['exact_tie'] == (m == 0)
                assert row['numeric_resolution'] == sign(m, expected_bound)
                assert row['original_gold'] == refs[row['query_id']]['original_gold']
            for expression, frozen in zip(out['comparisons'], plan['comparisons'], strict=True):
                assert all(expression[k] == v for k, v in frozen.items())
                coefficients, physical = defaultdict(Decimal), {}
                for term in frozen['terms']:
                    cid = term['condition_id']
                    r = receipts['production', cid]
                    key = r['physical_score_id']
                    value = (margins['production'][cid], dec(r['readout']['margin_error_bound']))
                    assert key not in physical or physical[key] == value
                    physical[key] = value
                    coefficients[key] += dec(term['coefficient'])
                exact = sum((weight * physical[key][0] for key, weight in coefficients.items()), Decimal(0))
                bound = sum((abs(weight) * physical[key][1] for key, weight in coefficients.items()), Decimal(0))
                got = expression['effect']
                error = abs(exact - dec(got['value']))
                max_expression_error = max(max_expression_error, error)
                assert float(exact) == got['value'] and float(bound) == got['bound']
                assert got['resolution'] == sign(exact, bound)
                ref = refs[frozen['query_id']]['adopted_label']
                aligned = exact if ref == '无' else -exact
                assert float(aligned) == expression['reference_aligned_change']
                assert sign(aligned, bound) == expression['reference_aligned_resolution']
                expected_terms = [{'physical_score_id': key, 'coefficient': float(weight)}
                                  for key, weight in sorted(coefficients.items()) if weight]
                assert got['physical_terms'] == expected_terms
                terms = frozen['terms']
                transition = None
                if len(terms) == 2 and sorted(t['coefficient'] for t in terms) == [-1, 1]:
                    before = next(margins['production'][t['condition_id']] for t in terms if t['coefficient'] == -1)
                    after = next(margins['production'][t['condition_id']] for t in terms if t['coefficient'] == 1)
                    before_ok = (before > 0 and ref == '无') or (before < 0 and ref == '有')
                    after_ok = (after > 0 and ref == '无') or (after < 0 and ref == '有')
                    transition = ('unresolved_transition' if min(abs(before), abs(after)) <= expected_bound else
                                  'repair' if not before_ok and after_ok else
                                  'damage' if before_ok and not after_ok else
                                  'stable_correct' if before_ok else 'stable_wrong')
                assert expression['verified_classification_transition'] == transition
            analyzed.append({'source': c.info(result_path), 'scores': len(out['scores']),
                             'expressions': len(out['comparisons']), 'exact_reconstruction': True})
    return {'status': 'pass', 'model_key': verified['model_key'], 'run_status': verified['status'],
            'audit_source': c.info(Path(__file__).resolve()), 'binding': c.info(run / 'binding.json'),
            'qualification': c.info(run / 'qualification.json'),
            'resource_release': c.read(run / 'state.json')['resource_release'],
            'Decimal_precision': 60, 'atoms_by_pass': dict(counts), 'atoms': sum(counts.values()),
            'maximum_atom_error': str(max_atom_error), 'engineering_differences': differences,
            'margin_error_bound_exact_match': True, 'analyses': analyzed,
            'maximum_expression_error': str(max_expression_error),
            'query_reference_join_performed_by_this_audit': results is not None,
            'extra_GPU_forwards': 0,
            'scope': 'Independent Decimal audit covers the primary two candidate logits, margins, bounds, directions, predictions and classification transitions. Full-vocabulary auxiliary probabilities and provenance are reconstructed by the frozen checker.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with localcontext() as ctx:
        ctx.prec = 60
        report = audit(args.run.resolve(), args.results.resolve() if args.results else None)
    c.atomic(args.output, report)
    print(json.dumps({k: report[k] for k in ['status', 'model_key', 'run_status', 'atoms', 'maximum_atom_error', 'maximum_expression_error']}, ensure_ascii=False))
