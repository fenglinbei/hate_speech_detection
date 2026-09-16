#!/usr/bin/env python3
"""Independent input reconstruction and Decimal arithmetic audit (no forward)."""
from __future__ import annotations
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'exps/causal_context/general_model_evidence_applicability_v1'
WORK = BASE / 'reviews/analysis-freeze-20260914/lexicon-scope-v1'


def read(path): return json.loads(Path(path).read_text())
def lines(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ensure(value, message):
    if not value: raise ValueError(message)


def verify_manifest(directory):
    manifest = read(directory / 'manifest.json')
    for relative, expected in manifest['source_files'].items():
        ensure(digest(ROOT / relative) == expected, 'source changed: ' + relative)
    for name, expected in manifest['artifacts'].items():
        ensure(digest(directory / name) == expected, 'artifact changed: ' + name)
    return manifest


def audit_inputs(directory):
    manifest = verify_manifest(directory)
    plan = read(directory / 'plan.json'); contexts = lines(directory / 'contexts.jsonl')
    source = read(directory / 'materials.json'); selected = lines(directory / 'historical-selected.jsonl')
    parent_pointer = read(BASE / 'content-decomposition-results-v1/current.json')
    parent_dir = BASE / parent_pointer['freeze_path']; parent_plan = read(parent_dir / 'plan.json')
    old = {c['record_id']: c for c in lines(parent_dir / 'contexts.jsonl')}
    old_raw = {r['record_id']: r for r in lines(BASE / parent_pointer['run_path'] / parent_plan['raw_pass'] / 'scores.jsonl')}
    fillers = {r['arm']: r['block'] for r in source['filler_entries']}
    notes = {r['arm']: r['block'] for r in source['definition_notes']}
    for k in ('USE_TORCH', 'USE_TF', 'USE_FLAX', 'USE_TORCH_XLA'): os.environ[k] = '0'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'models/base/Qwen3-8B'), local_files_only=True, trust_remote_code=False, use_fast=True)
    rebuilt, history_count = {}, 0
    for c in contexts:
        parent = old[c['source_record_id']]; arm = c['lexicon_arm']
        messages = [dict(m) for m in parent['messages']]
        old_block = source['original_entry_block']; user = messages[1]['content']
        ensure(user.count(old_block) == 1, 'ambiguous old entry')
        if arm == 'D': user = user.replace(old_block, '', 1)
        elif arm.startswith('N'): user = user.replace(old_block, fillers[arm], 1)
        elif arm.startswith('E'):
            user = user.replace(old_block, '', 1)
            user = user.replace('参考示例：\n', fillers['N' + arm[1:]] + '参考示例：\n', 1)
        elif arm.startswith(('P', 'X')): user = user.replace(old_block, notes[arm], 1)
        else: ensure(arm == 'O', 'unknown arm')
        messages[1]['content'] = user
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        ensure(messages == c['messages'] and prompt == c['prompt_text'] and tokens == c['prompt_token_ids'], 'independent prompt replay failed')
        ensure(user.split('参考示例：\n')[1] == parent['messages'][1]['content'].split('参考示例：\n')[1], 'example/query changed')
        expected_delta = -33 if arm == 'D' else (14 if arm in ('P1', 'X1') else (13 if arm in ('P2', 'X2') else 0))
        ensure(len(tokens) - parent['prompt_tokens'] == expected_delta, 'independent total length differs')
        for candidate in plan['catalog'][c['task']]:
            ensure(tokenizer.encode(prompt + candidate['canonical_answer'], add_special_tokens=False)
                   == tokens + candidate['answer_token_ids'], 'independent answer boundary failed')
        if arm == 'O':
            ensure(prompt == parent['prompt_text'] and tokens == parent['prompt_token_ids'], 'history changed')
            row = next(r for r in selected if r['record_id'] == c['record_id'])
            ensure(row['candidates'] == old_raw[parent['record_id']]['candidates'], 'historical candidate payload changed')
            history_count += 1
        rebuilt[c['record_id']] = tokens
    ensure(len(contexts) == len(rebuilt) == 320 and history_count == 32, 'independent coverage failed')
    proofs = lines(directory / 'matching-proofs.jsonl')
    for p in proofs:
        a, b = [rebuilt[r] for r in p['record_ids']]
        ensure(len(a) == len(b), 'matched token length failed')
        if p['kind'] == 'within_arm_word_pair': span = p['proof']['target_text_token_span']
        elif p['kind'] in ('matched_replacement', 'matched_scope_note'): span = p['proof']['matched_span']
        else: continue
        start, end = span
        ensure(a[:start] == b[:start] and a[end:] == b[end:], 'independent external token identity failed')
    ensure(plan['catalog'] == parent_plan['catalog'] and plan['numeric_policy'] == parent_plan['numeric_policy'], 'score contract changed')
    return {'status': 'passed', 'mode': 'inputs', 'source_files': len(manifest['source_files']), 'prompts': 320,
            'candidate_boundaries': 640, 'historical_payload_replays': 32, 'geometry_proofs': len(proofs), 'model_forward_executed': False}


def audit_results(directory, run, results):
    verify_manifest(directory); verify_manifest(results)
    state = read(run / 'run_manifest.json'); plan = read(directory / 'plan.json')
    ensure(state['status'] == 'complete' and state['numerical_validation_passed']
           and len(state['checks']) == 10 and len(state['derived_checks']) == 6
           and all(c['passed'] for c in state['checks'] + state['derived_checks']), 'run not numerically sealed')
    raw_path = run / plan['raw_pass'] / 'scores.jsonl'
    ensure(digest(raw_path) == state['raw_scores_sha256'], 'raw scores changed')
    contexts = {c['record_id']: c for c in lines(directory / 'contexts.jsonl')}
    refs = {(r['query_id'], r['task']): r for r in read(directory / 'analysis_references.json')}
    scalar_checks, direction_checks, prediction_checks, near_zero_rounding_cases, largest = 0, 0, 0, 0, Decimal(0)
    D = lambda x: Decimal(str(x))
    def close(observed, expected):
        nonlocal scalar_checks, largest
        error = abs(D(observed) - expected); largest = max(largest, error); scalar_checks += 1
        ensure(error < Decimal('1e-10'), 'independent Decimal arithmetic differs')
    def sign(value, bound): return 'unresolved' if abs(value) <= bound else ('positive' if value > 0 else 'negative')
    with localcontext() as ctx:
        ctx.prec = 50
        eps = D(plan['numeric_policy']['epsilon']); margins, candidates, values, bounds = {}, {}, {}, {}
        for row in lines(raw_path):
            c = contexts[row['record_id']]; computed = []
            for candidate in row['candidates']:
                ts = [D(x) for x in candidate['token_logprobs']]; eos = D(candidate['eos_logprob'])
                total = sum(ts); n = len(ts)
                scores = {'answer_sum': total, 'answer_mean': total / n, 'total_with_eos': total + eos,
                          'mean_with_eos': (total + eos) / (n + 1), 'eos_logprob': eos}
                for mode, value in scores.items(): close(candidate['scores'][mode], value)
                computed.append(scores)
            for mode in ('answer_sum', 'answer_mean', 'total_with_eos', 'mean_with_eos'):
                key = (c['root_condition'], c['encoding'], c['probe_id'], mode)
                margins[key] = computed[1][mode] - computed[0][mode]
                candidates[key] = (computed[0][mode], computed[1][mode])
                if c['probe_id'] is None:
                    view = c['encoding'] + '/' + mode
                    values[c['root_condition'], view] = margins[key]; bounds[view] = eps
        for row in lines(results / 'condition-scores.jsonl'):
            key = (row['condition'], row['encoding'], None, row['score_mode'])
            close(row['non_hate_margin'], margins[key])
            close(row['hate_score'], candidates[key][0]); close(row['non_hate_score'], candidates[key][1])
            prediction = 'non-hate' if margins[key] > 0 else 'hate'
            ensure(row['prediction'] == prediction, 'raw prediction differs')
            ensure(row['reviewed_correct'] == (prediction == refs['3169', 'hate']['adjudicated_label']), 'reference comparison differs')
            prediction_checks += 1
        probe_names = ('empty', 'space', 'na', 'mask', 'lorem')
        def bg(xs):
            p = sum(1 / (1 + (-x).exp()) for x in xs) / len(xs)
            return (p / (1 - p)).ln()
        for row in lines(results / 'ncc-conditions.jsonl'):
            c = row['condition']; real = margins[c, 'original', None, 'answer_mean']
            ps = {p: margins[c, 'original', p, 'answer_mean'] for p in probe_names}
            background = bg(list(ps.values())); ncc = real - background
            close(row['real_mean_margin'], real); close(row['background_margin'], background); close(row['ncc_margin'], ncc)
            # The catalog's tie rule applies to the actual FP64 exported margin.
            # A mathematically zero residual can straddle zero in finite-precision
            # sigmoid/logit evaluation, while remaining numerically unresolved.
            ensure(row['prediction'] == ('non-hate' if row['ncc_margin'] > 0 else 'hate'), 'NCC exported prediction differs')
            if abs(ncc) > Decimal('1e-10'):
                ensure(row['prediction'] == ('non-hate' if ncc > 0 else 'hate'), 'NCC Decimal prediction differs')
            else:
                ensure(row['numerically_unresolved'] and abs(D(row['ncc_margin'])) <= 2 * eps, 'near-zero result treated as resolved')
                near_zero_rounding_cases += 1
            prediction_checks += 1
            values[c, 'ncc'] = ncc; bounds['ncc'] = 2 * eps
            values[c, 'background'] = background; bounds['background'] = eps
            f, r = [values[c, e + '/answer_sum'] for e in ('ab_forward', 'ab_reverse')]
            values[c, 'ab_symmetric_sum'] = (f + r) / 2; bounds['ab_symmetric_sum'] = eps
            values[c, 'ab_mapping_gap_sum'] = f - r; bounds['ab_mapping_gap_sum'] = 2 * eps
            for p in probe_names:
                for name, value, field in [('single_probe', real - ps[p], 'probe_ncc_margins'),
                    ('leave_one_out', real - bg([v for k, v in ps.items() if k != p]), 'leave_one_out_ncc_margins')]:
                    close(row[field][p], value); values[c, name + '/' + p] = value; bounds[name + '/' + p] = 2 * eps
        comps = {c['contrast_id']: c for c in lines(directory / 'comparisons.jsonl')}
        calculated, calculated_bounds = {}, {}
        for row in lines(results / 'contrast-scores.jsonl') + lines(results / 'probe-contrasts.jsonl'):
            view = row.get('view') or row['kind'] + '/' + row['probe_id']
            terms = comps[row['contrast_id']]['terms']
            value = sum(D(t['coefficient']) * values[t['condition'], view] for t in terms)
            bound = sum(abs(D(t['coefficient'])) * bounds[view] for t in terms)
            close(row['effect'], value); close(row['numeric_bound'], bound)
            ensure(row['direction'] == sign(value, bound), 'contrast direction differs'); direction_checks += 1
            calculated[row['contrast_id'], view] = value; calculated_bounds[row['contrast_id'], view] = bound
        for row in lines(results / 'paired-changes.jsonl'):
            f, a, b, view = row['template'], row['from_arm'], row['to_arm'], row['view']
            ah, ao, bh, bo = [values[f'F{f}-{arm}-{form}', view] for arm, form in ((a, 'H'), (a, 'O'), (b, 'H'), (b, 'O'))]
            da, db, sh, so = ah - ao, bh - bo, bh - ah, bo - ao
            i, common, reduction = (bh - bo) - (ah - ao), (bh + bo - ah - ao) / 2, abs(da) - abs(db)
            for k, value in {'delta_before': da, 'delta_after': db, 'shift_H': sh, 'shift_O': so,
                'interaction': i, 'common_shift': common, 'absolute_magnitude_reduction': reduction}.items(): close(row[k], value)
            unit = bounds[view]
            for k, factor in {'delta_before_bound': 2, 'delta_after_bound': 2, 'interaction_bound': 4,
                             'common_shift_bound': 2, 'absolute_magnitude_reduction_bound': 4}.items(): close(row[k], unit * factor)
            for k, value, factor in [('delta_before_direction', da, 2), ('delta_after_direction', db, 2),
                ('interaction_direction', i, 4), ('common_shift_direction', common, 2), ('magnitude_reduction_direction', reduction, 4)]:
                ensure(row[k] == sign(value, unit * factor), 'paired result direction differs'); direction_checks += 1
            if abs(da) > 2 * unit:
                close(row['signed_remaining_ratio'], db / da); close(row['absolute_remaining_ratio'], abs(db / da))
            else: ensure(row['signed_remaining_ratio'] is None and row['absolute_remaining_ratio'] is None, 'ratio with unresolved denominator')
    return {'status': 'passed', 'mode': 'results', 'decimal_precision': 50, 'scalar_checks': scalar_checks,
            'direction_checks': direction_checks, 'prediction_checks': prediction_checks,
            'near_zero_prediction_rounding_cases': near_zero_rounding_cases,
            'maximum_decimal_export_error': str(largest), 'model_forward_executed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('inputs', 'results'))
    parser.add_argument('--plan', type=Path, default=WORK / 'frozen-01')
    parser.add_argument('--run', type=Path, default=WORK / 'run-01')
    parser.add_argument('--results', type=Path, default=WORK / 'results-01')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    receipt = audit_inputs(args.plan) if args.mode == 'inputs' else audit_results(args.plan, args.run, args.results)
    payload = json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    if args.receipt:
        ensure(not args.receipt.exists(), 'receipt already exists')
        args.receipt.parent.mkdir(parents=True, exist_ok=True); args.receipt.write_text(payload)
    print(payload, end='')


if __name__ == '__main__': main()
