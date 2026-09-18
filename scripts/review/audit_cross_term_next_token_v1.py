#!/usr/bin/env python3
"""Independent Decimal/long-double audit of raw vectors and every registered contrast."""
import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.cross_term_next_token_v1 import atomic_json, check_completed, file_info, read, require


def audit(frozen, run, results):
    frozen, run, results = Path(frozen), Path(run), Path(results)
    check_completed(frozen, run, full=False)
    binding = read(run / 'binding.json')
    require(read(run / 'run_manifest.json')['status'] == 'complete', 'run not terminal complete')
    by_pass = {}; max_margin_error = Decimal(0); max_aux_error = np.longdouble(0); count = 0
    with localcontext() as context:
        context.prec = 60
        for directory in sorted((run / 'scores').iterdir()):
            if not directory.is_dir(): continue
            rows = {}
            for receipt in sorted(directory.glob('*.json')):
                r = read(receipt); vector = np.load(ROOT / r['raw_logits']['path'], allow_pickle=False)
                require(file_info(ROOT / r['raw_logits']['path']) == r['raw_logits'], 'raw vector hash changed')
                dy, dn = Decimal.from_float(float(vector[18830])), Decimal.from_float(float(vector[42192]))
                margin = dn - dy
                max_margin_error = max(max_margin_error, abs(Decimal.from_float(r['readout']['m']) - margin))
                require(r['readout']['m'] == float(margin), 'Decimal margin differs')
                values = vector.astype(np.longdouble)
                largest = values.max()
                normalizer_relative = np.log(np.exp(values - largest).sum(dtype=np.longdouble))
                logyes = values[18830] - largest - normalizer_relative
                logno = values[42192] - largest - normalizer_relative
                pair_largest = max(values[18830], values[42192])
                logmass = pair_largest - largest + np.log(np.exp(values[[18830, 42192]] - pair_largest).sum(dtype=np.longdouble)) - normalizer_relative
                pairno = 1 / (1 + np.exp(values[18830] - values[42192]))
                expected = {'log_p_yes': logyes, 'log_p_no': logno, 'log_legal_mass': logmass,
                            'legal_mass': np.exp(logmass), 'pair_support_no': pairno}
                for field, v in expected.items():
                    error = abs(np.longdouble(r['readout'][field]) - v)
                    max_aux_error = max(max_aux_error, error)
                    require(error < 1e-10, f'long-double readout differs: {field}')
                rows[r['condition_id']] = {'margin': margin, 'receipt': r}
                count += 1
            require(len(rows) == 120, 'incomplete pass')
            by_pass[directory.name] = rows
        require(count == 720 and len(by_pass) == 6, 'score inventory differs')
        ref = by_pass['engineering-reference']
        differences = {name: max(abs(rows[c]['margin'] - ref[c]['margin']) for c in ref)
                       for name, rows in by_pass.items() if name.startswith('engineering-') and name != 'engineering-reference'}
        require(differences['engineering-repeat'] == differences['engineering-reverse-request-order'] == 0, 'independent repeat/order gate failed')
        for name in ['engineering-left-padding', 'engineering-right-padding']:
            require(differences[name] <= Decimal.from_float(0.001), 'independent padding gate failed')
        bound = max(Decimal.from_float(0.000001), 2 * max(differences.values()))
        qualification = read(run / 'qualification.json')
        require(qualification['margin_error_bound'] == float(bound), 'independent bound differs')
        require({k: float(v) for k, v in differences.items()} == qualification['max_margin_differences'], 'gate maxima differ')
        data = read(results / 'results.json')
        calculated = {r['condition_id']: r for r in data['scores']}
        source_plan = read(ROOT / read(frozen / 'plan.json')['analysis_plan']['path'])
        references = {r['query_id']: r['human_reference']['task_label'] for r in source_plan['references']}
        max_expression_error = Decimal(0); max_bound_error = Decimal(0)
        for r in data['scores']:
            m = by_pass['science'][r['condition_id']]['margin']
            prediction = '无' if m > 0 else '有' if m < 0 else None
            resolution = 'resolved_no' if m > bound else 'resolved_yes' if m < -bound else 'numerical_unresolved'
            require(r['raw_prediction'] == prediction and r['resolution'] == resolution, 'independent prediction/resolution differs')
            require(r['raw_correct'] == (prediction == references[r['query_id']]), 'reference correctness differs')
            require(r['conservative_correct'] == (resolution != 'numerical_unresolved' and prediction == references[r['query_id']]), 'conservative correctness differs')
            require(r['original_gold'] is None and r['original_correct'] is None, 'original Gold invented')
        for r, original in zip(data['expressions'], source_plan['comparisons']):
            require(r['comparison_id'] == original['comparison_id'] and r['terms'] == original['terms'], 'registered expression changed')
            physical = defaultdict(Decimal); scores = {}
            for t in original['terms']:
                score = calculated[t['condition_id']]; key = score['physical_score_id']
                physical[key] += Decimal(t['coefficient']); scores[key] = by_pass['science'][t['condition_id']]['margin']
            physical = {k: v for k, v in physical.items() if v}
            value = sum((v * scores[k] for k, v in physical.items()), Decimal(0))
            total_bound = sum((abs(v) * bound for v in physical.values()), Decimal(0))
            max_expression_error = max(max_expression_error, abs(Decimal.from_float(r['effect']['value']) - value))
            max_bound_error = max(max_bound_error, abs(Decimal.from_float(r['effect']['bound']) - total_bound))
            require(r['effect']['value'] == float(value) and r['effect']['bound'] == float(total_bound), 'expression/bound differs')
            sign = 1 if references[r['query_id']] == '无' else -1
            aligned = value * sign
            state = 'positive' if aligned > total_bound else 'negative' if aligned < -total_bound else 'numerical_unresolved'
            require(r['reference_aligned_change'] == float(aligned) and r['reference_aligned_resolution'] == state, 'aligned direction differs')
            transition = None
            if len(original['terms']) == 2 and sorted(t['coefficient'] for t in original['terms']) == [-1, 1]:
                t = calculated[next(t['condition_id'] for t in original['terms'] if t['coefficient'] == 1)]
                c = calculated[next(t['condition_id'] for t in original['terms'] if t['coefficient'] == -1)]
                if 'numerical_unresolved' in [c['resolution'], t['resolution']]: transition = 'unresolved_transition'
                elif not c['raw_correct'] and t['raw_correct']: transition = 'repair'
                elif c['raw_correct'] and not t['raw_correct']: transition = 'damage'
                else: transition = 'stable_correct' if c['raw_correct'] else 'stable_wrong'
            require(r['verified_classification_transition'] == transition, 'transition differs')
        require(len(data['expressions']) == 252, 'expression coverage differs')
    generated = read(run / 'generation.json')
    generation_steps = 0
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / read(frozen / 'plan.json')['model_path']), local_files_only=True, trust_remote_code=False)
    requests = {r['condition_id']: r for r in [json.loads(line) for line in (frozen / 'model-inputs.jsonl').read_text().splitlines()]}
    for r in generated['records']:
        ids = list(requests[r['condition_id']]['input_ids']); emitted = []
        for step in r['steps']:
            path = ROOT / step['raw_logits']['path']; require(file_info(path) == step['raw_logits'], 'generation vector changed')
            raw = np.load(path, allow_pickle=False)
            require(raw.shape == (151936,) and np.isfinite(raw).all(), 'invalid generation vector')
            require(step['selected_token_id'] == int(raw.argmax()), 'greedy token differs')
            ids_bytes = json.dumps(ids, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            require(hashlib.sha256(ids_bytes).hexdigest() == step['input_ids_sha256'], 'generation prefix differs')
            ids.append(step['selected_token_id']); emitted.append(step['selected_token_id']); generation_steps += 1
        visible = tokenizer.decode(emitted, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        require(emitted == r['generated_ids'] and visible == r['visible_answer'] and r['strict_pass'] == (visible in ['有', '无']), 'generation compliance differs')
    return {'status': 'pass', 'decimal_precision': 60, 'raw_score_vectors': count, 'candidate_logit_scalars': 1440,
        'auxiliary_longdouble_readouts': 3600, 'max_margin_error': str(max_margin_error),
        'max_auxiliary_error': str(max_aux_error), 'expressions_and_bounds': 252,
        'max_expression_error': str(max_expression_error), 'max_bound_error': str(max_bound_error),
        'predictions_and_correctness': 120, 'generation_prompts': 12, 'generation_step_vectors': generation_steps,
        'references_read_only_after_complete_raw_seals': True, 'no_GPU_forward': True,
        'run_binding': file_info(run / 'binding.json'), 'results': file_info(results / 'results.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.freeze, args.run, args.results)
    atomic_json(args.output, result, replace=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))
