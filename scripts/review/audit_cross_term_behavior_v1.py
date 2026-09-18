#!/usr/bin/env python3
"""Independent tokenizer/source and 60-digit arithmetic audits; CPU only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from decimal import Decimal, localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics import cross_term_behavior_execution_v1 as c


def need(value, message):
    if not value:
        raise AssertionError(message)


def source(entry):
    path = ROOT / entry['path']
    need(path.stat().st_size == entry['bytes'], f'byte size differs: {path}')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    need(h.hexdigest() == entry['sha256'], f'hash differs: {path}')


def arithmetic(scores, expressions):
    """Independent arithmetic: no production readout/effect/resolution helpers."""
    lookup = {r['condition_id']: r for r in scores}
    with localcontext() as ctx:
        ctx.prec = 60
        largest = Decimal(0)
        for r in scores:
            m = Decimal.from_float(r['z_no']) - Decimal.from_float(r['z_yes'])
            need(m == Decimal.from_float(r['m']), 'candidate margin arithmetic differs')
            b = Decimal.from_float(r['margin_error_bound'])
            prediction = '无' if m > 0 else '有' if m < 0 else None
            resolution = 'resolved_no' if m > b else 'resolved_yes' if m < -b else 'numerical_unresolved'
            need(r['raw_prediction'] == prediction and r['resolution'] == resolution and r['exact_tie'] == (m == 0),
                 'prediction/resolution differs')
            correct = prediction == r['adopted_reference']
            need(r['raw_correct'] == correct and r['conservative_correct'] == (correct and abs(m) > b),
                 'reference agreement differs')
        for row in expressions:
            coeff = defaultdict(Decimal); physical = {}
            for term in row['terms']:
                r = lookup[term['condition_id']]; key = r['physical_score_id']
                identity = (r['m'], r['margin_error_bound'], r['prompt_sha256'], r['qualification_ref'])
                need(key not in physical or physical[key] == identity, 'conflicting physical aliases')
                physical[key] = identity
                coeff[key] += Decimal(str(term['coefficient']))
            coeff = {k: a for k, a in coeff.items() if a}
            value = sum((a * Decimal.from_float(physical[k][0]) for k, a in coeff.items()), Decimal(0))
            bound = sum((abs(a) * Decimal.from_float(physical[k][1]) for k, a in coeff.items()), Decimal(0))
            err = abs(value - Decimal.from_float(row['effect']['value']))
            largest = max(largest, err)
            need(float(value) == row['effect']['value'] and float(bound) == row['effect']['bound'], 'expression value/bound differs')
            need(row['effect']['physical_terms'] == [dict(physical_score_id=k, coefficient=float(a)) for k, a in sorted(coeff.items())],
                 'physical coefficient cancellation differs')
            state = 'positive' if value > bound else 'negative' if value < -bound else 'numerical_unresolved'
            oriented = value * row['reference_direction_sign']
            oriented_state = 'positive' if oriented > bound else 'negative' if oriented < -bound else 'numerical_unresolved'
            need(row['effect']['resolution'] == state and float(oriented) == row['reference_aligned_change'] and
                 row['reference_aligned_resolution'] == oriented_state, 'direction differs')
            expected_transition = None
            if len(row['terms']) == 2 and sorted(t['coefficient'] for t in row['terms']) == [-1, 1]:
                control = lookup[next(t['condition_id'] for t in row['terms'] if t['coefficient'] == -1)]
                treatment = lookup[next(t['condition_id'] for t in row['terms'] if t['coefficient'] == 1)]
                if any(r['resolution'] == 'numerical_unresolved' for r in [control, treatment]):
                    expected_transition = 'unresolved_transition'
                else:
                    expected_transition = {(False, True): 'repair', (True, False): 'damage',
                        (True, True): 'stable_correct', (False, False): 'stable_wrong'}[(control['raw_correct'], treatment['raw_correct'])]
            need(row['verified_classification_transition'] == expected_transition, 'repair/damage transition differs')
    return {'precision': 60, 'score_predictions': len(scores), 'expression_values_and_bounds': len(expressions),
            'max_exact_expression_error': str(largest)}


def synthetic_registered_analysis(seed):
    """Known CPU numbers on all registered equations, including unresolved/tie cases."""
    conditions = c.read(c.MATERIAL / 'design.json')['conditions']
    plan = c.read(c.MATERIAL / 'analysis-plan.json')
    refs = {r['query_id']: r['human_reference']['task_label'] for r in plan['references']}
    rows = []
    for i, condition in enumerate(conditions):
        m = ((i * (seed + 3) + seed) % 17 - 8) / 8
        b = 1 / (32 if condition['input_status'] == 'historical_reuse' else 16)
        score = {'z_no': m, 'z_yes': 0.0, 'm': m, 'margin_error_bound': b,
            'raw_prediction': '无' if m > 0 else '有' if m < 0 else None, 'exact_tie': m == 0,
            'resolution': 'resolved_no' if m > b else 'resolved_yes' if m < -b else 'numerical_unresolved',
            'qualification_ref': {'synthetic_CPU_only': condition['input_status']}, 'legal_mass': 0.5}
        rows.append({'condition_id': condition['condition_id'], 'prompt_sha256': condition['prompt_sha256'],
                     'physical_score_id': f'SYNTHETIC-CPU:{i}', 'readout': score})
    return c.analysis_values(conditions, plan, rows)


def audit_preparation(prepared):
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU audit requires empty CUDA visibility')
    manifest = json.loads((prepared / 'manifest.json').read_text())
    for entry in manifest['artifacts'] + manifest['sources']:
        source(entry)
    new = [json.loads(s) for s in (prepared / 'model-inputs.jsonl').read_text().splitlines()]
    bridges = [json.loads(s) for s in (prepared / 'bridge-inputs.jsonl').read_text().splitlines()]
    need((prepared / 'model-inputs.jsonl').read_bytes() == (c.MATERIAL / 'model-inputs.jsonl').read_bytes(), 'reviewed new bytes changed')
    originals = {r['condition_id']: r for r in c.jsonl(c.OLD_FREEZE / 'model-inputs.jsonl')}
    need(all(r == originals[r['condition_id']] for r in bridges), 'historical input bytes/fields changed')
    need(len(new) == 36 and len(bridges) == 84 and len({r['condition_id'] for r in new + bridges}) == 120,
         'input coverage differs')
    design = json.loads((c.MATERIAL / 'design.json').read_text())
    need({r['condition_id'] for r in new + bridges} == {r['condition_id'] for r in design['conditions']}, 'reviewed core coverage differs')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'models/base/Qwen3-8B'), local_files_only=True, trust_remote_code=False)
    for row in new + bridges:
        prompt = tokenizer.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        need(prompt == row['chat_prompt'] and tokenizer.encode(prompt, add_special_tokens=False) == row['input_ids'], 'tokenizer reconstruction differs')
        for answer, token in [('有', 18830), ('无', 42192)]:
            need(tokenizer.encode(prompt + answer, add_special_tokens=False) == row['input_ids'] + [token], 'bare answer boundary differs')
    historical = c.read(prepared / 'historical-index.json')['records']
    import numpy as np
    for entry in historical:
        source(entry['score_ref']); source(entry['raw_logits'])
        receipt = json.loads((ROOT / entry['score_ref']['path']).read_text())
        vector = np.load(ROOT / entry['raw_logits']['path'], allow_pickle=False)
        need(vector.dtype == np.dtype('float32') and vector.shape == (151936,) and np.isfinite(vector).all(), 'old vector format differs')
        margin = Decimal.from_float(float(vector[42192])) - Decimal.from_float(float(vector[18830]))
        need(margin == Decimal.from_float(receipt['readout']['m']), 'old candidate margin differs')
        need(receipt['physical_score_id'] == entry['physical_score_id'] and
             receipt['readout']['qualification_ref'] == entry['qualification_ref'], 'old score identity differs')
    arithmetic_checks = []
    for seed in range(7):
        result = synthetic_registered_analysis(seed)
        arithmetic_checks.append(arithmetic(result['scores'], result['expressions']))
    plan = json.loads((prepared / 'plan.json').read_text())
    need(plan['allocation'] is None and plan['GPU_launch_authorized_by_this_preparation'] is False, 'GPU unexpectedly bound')
    expected_schedule = [(84, 'none', 'forward')] + [(36, pad, order) for pad, order in [
        ('none', 'forward'), ('none', 'forward'), ('left_to_next_strict_multiple_of_16', 'forward'),
        ('right_to_next_strict_multiple_of_16', 'forward'), ('none', 'reverse'), ('none', 'forward')]]
    schedule = json.loads((prepared / 'qualification-plan.json').read_text())['passes']
    need([(s['prompt_forwards'], s['padding'], s['request_order']) for s in schedule] == expected_schedule, '300-forward schedule differs')
    return {'status': 'pass', 'preparation_manifest': c.file_info(prepared / 'manifest.json'),
        'source_and_artifact_hashes': len(manifest['sources']) + len(manifest['artifacts']),
        'tokenizer_reconstructions': 120, 'bare_answer_boundaries': 240, 'historical_physical_margins': 84,
        'synthetic_equations_checked': sum(r['expression_values_and_bounds'] for r in arithmetic_checks),
        'synthetic_predictions_checked': sum(r['score_predictions'] for r in arithmetic_checks),
        'Decimal_precision': 60, 'synthetic_max_expression_error': max(r['max_exact_expression_error'] for r in arithmetic_checks),
        'new_GPU_forward_performed': False, 'GPU_inventory_called': False, 'model_weights_loaded': False}


def audit_results(prepared, run, output):
    c.assert_released(run)
    c.check_completed(prepared, run, full=True)
    result = json.loads((output / 'results.json').read_text())
    report = arithmetic(result['scores'], result['expressions'])
    import numpy as np
    records = []
    for spec in c.specs():
        q = c.old.qualification_with_ref(run) if spec['pass_id'] == 'science' else None
        records.extend(c.check_pass(prepared, run, spec, q, full=False))
    records.extend(c.read_historical(prepared, full=False))
    for record in records:
        source(record['raw_logits'])
        vector = np.load(ROOT / record['raw_logits']['path'], allow_pickle=False)
        with localcontext() as ctx:
            ctx.prec = 60
            value = Decimal.from_float(float(vector[42192])) - Decimal.from_float(float(vector[18830]))
        need(value == Decimal.from_float(record['readout']['m']), 'independent raw margin differs')
    # Reconstruct every emitted byte using the gated production CPU analyzer too.
    reconstruction = c.analyze(prepared, run, output)
    report.update(status='pass', raw_vectors=len(records), results=c.file_info(output / 'results.json'),
                  exact_analysis_reconstruction=reconstruction, new_GPU_forward_performed=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preparation', 'results'])
    parser.add_argument('--prepared', type=Path, default=c.WORK / 'prepared-01')
    parser.add_argument('--run', type=Path, default=c.WORK / 'run-01')
    parser.add_argument('--output', type=Path, default=c.WORK / 'results-01')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    result = audit_preparation(args.prepared) if args.action == 'preparation' else audit_results(args.prepared, args.run, args.output)
    if args.receipt:
        c.atomic_json(args.receipt, result, replace=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
