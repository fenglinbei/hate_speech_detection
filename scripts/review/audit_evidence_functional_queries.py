#!/usr/bin/env python3
"""Independent token reconstruction and 50-digit Decimal result audit."""
from decimal import Decimal, localcontext
from pathlib import Path
import argparse
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'exps/causal_context/general_model_evidence_applicability_v1'
PREVIEW = BASE / 'reviews/functional-query-diagnostics-v1/draft-01'
PROBES = ('empty', 'space', 'na', 'mask', 'lorem')


def read(p): return json.loads(Path(p).read_bytes())
def lines(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ensure(value, msg):
    if not value: raise ValueError(msg)


def verify(directory):
    m = read(directory / 'manifest.json')
    for p, h in m['source_files'].items(): ensure(digest(ROOT / p) == h, 'source changed: ' + p)
    for p, h in m['artifacts'].items(): ensure(digest(directory / p) == h, 'artifact changed: ' + p)
    return m


def inputs(directory):
    m = verify(directory); plan = read(directory / 'plan.json')
    rows = lines(directory / 'contexts.jsonl')
    previews = {r['prompt_id']: r for r in lines(PREVIEW / 'prompt-previews.jsonl')}
    for name in ('USE_TORCH', 'USE_TF', 'USE_FLAX', 'USE_TORCH_XLA'): os.environ[name] = '0'
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BASE / 'reviews/analysis-freeze-20260914/lexicon-scope-v1/model-copy-01'),
                                       local_files_only=True, trust_remote_code=False, use_fast=True)
    for r in rows:
        p = previews[r['record_id']]
        prompt = tok.apply_chat_template(r['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(prompt, add_special_tokens=False)
        ensure(prompt == r['prompt_text'] == p['prompt_text'] and ids == r['prompt_token_ids'] == p['prompt_token_ids'], 'token/input preview differs')
        for candidate in plan['catalog'][r['task']]:
            ensure(tok.encode(prompt + candidate['canonical_answer'], add_special_tokens=False) == ids + candidate['answer_token_ids'], 'candidate boundary differs')
    matrix = lines(directory / 'input-matrix.jsonl'); ids = {r['condition_id'] for r in matrix}
    ensure(len(ids) == len(matrix), 'duplicate scientific condition')
    aliases = {(a['condition_id'], r['encoding'], r['probe_id']) for r in rows for a in r['bindings'] if a['condition_id'] in ids}
    ensure(len(aliases) == 8 * len(matrix), 'logical alias coverage differs')
    ensure(len(rows) == plan['counts']['unique_prompts'] and len({r['prompt_sha256'] for r in rows}) == len(rows), 'prompt dedup differs')
    return {'status': 'passed', 'stage': plan['stage'], 'source_files': len(m['source_files']),
            'prompt_reconstructions': len(rows), 'candidate_boundaries': len(rows) * 2,
            'logical_aliases': len(aliases), 'model_forward_executed': False}


def results(directory, run, output):
    verify(directory); verify(output)
    plan = read(directory / 'plan.json'); state = read(run / 'run_manifest.json')
    ensure(state['status'] == 'complete' and state['numerical_validation_passed']
           and len(state['checks']) == 10 and len(state['derived_checks']) == 6
           and all(r['passed'] for r in state['checks'] + state['derived_checks']), 'unsealed run')
    raw_path = run / plan['raw_pass'] / 'scores.jsonl'
    ensure(digest(raw_path) == state['raw_scores_sha256'], 'raw hash differs')
    contexts = {r['record_id']: r for r in lines(directory / 'contexts.jsonl')}
    raw = lines(raw_path); refs = {r['query_id']: r for r in read(directory / 'analysis_references.json')}
    matrix = {r['condition_id']: r for r in lines(directory / 'input-matrix.jsonl')}
    comps = {r['contrast_id']: r for r in lines(directory / 'comparisons.jsonl')}
    scalars, directions, predictions, rounded_ties = 0, 0, 0, 0
    largest = Decimal(0)
    D = lambda x: Decimal.from_float(x) if isinstance(x, float) else Decimal(x)
    def check(actual, expected):
        nonlocal scalars, largest
        error = abs(D(actual) - expected); largest = max(largest, error); scalars += 1
        ensure(error <= Decimal('1e-10'), 'Decimal scalar discrepancy: ' + str(error))
    with localcontext() as ctx:
        ctx.prec = 50
        values, aliases = {}, {}
        for r in raw:
            margins = {}; candidates = []
            for c in r['candidates']:
                total = sum(map(D, c['token_logprobs']), Decimal(0)); n = len(c['token_logprobs']); eos = D(c['eos_logprob'])
                scores = {'answer_sum': total, 'answer_mean': total / n,
                          'total_with_eos': total + eos, 'mean_with_eos': (total + eos) / (n + 1)}
                for mode, value in scores.items(): check(c['scores'][mode], value)
                candidates.append(scores)
            for mode in candidates[0]: margins[mode] = candidates[1][mode] - candidates[0][mode]
            c = contexts[r['record_id']]
            for a in c['bindings']:
                if a['condition_id'] in matrix: aliases[a['condition_id'], c['encoding'], c['probe_id']] = margins
        def bg(xs):
            probability = sum((1 / (1 + (-v).exp()) for v in xs), Decimal(0)) / len(xs)
            return (probability / (1 - probability)).ln()
        for cid in matrix:
            vv = {}
            for e in ('original', 'ab_forward', 'ab_reverse'):
                for mode, margin in aliases[cid, e, None].items(): vv[e + '/' + mode] = margin
            probe = [aliases[cid, 'original', p]['answer_mean'] for p in PROBES]
            real = vv['original/answer_mean']; prior = bg(probe)
            vv.update(background=prior, ncc=real - prior)
            for i, p in enumerate(PROBES):
                vv['single_probe/' + p] = real - probe[i]
                vv['leave_one_out/' + p] = real - bg(probe[:i] + probe[i + 1:])
            f, r = vv['ab_forward/answer_sum'], vv['ab_reverse/answer_sum']
            vv.update(ab_symmetric_sum=(f + r) / 2, ab_mapping_gap_sum=f - r)
            values[cid] = vv
        for r in lines(output / 'condition-scores.jsonl'):
            value = values[r['condition_id']][r['view']]; check(r['non_hate_margin'], value)
            ref = refs[r['query_id']]
            if r['query_id'] != '3169': ensure(r['original_reference'] is None and r['original_correct'] is None, 'invented original Gold')
            if r['prediction'] is not None:
                expected = 'non-hate' if value > 0 else 'hate'
                if expected != r['prediction']:
                    ensure(r['numerically_unresolved'] and abs(value) <= Decimal('1e-10'), 'prediction differs beyond rounding')
                    rounded_ties += 1
                ensure(r['reviewed_correct'] == (r['prediction'] == ref['adjudicated_label']), 'reference agreement differs')
                predictions += 1
        for r in lines(output / 'contrast-scores.jsonl'):
            c = comps[r['contrast_id']]; view = r['view']
            value = sum((D(t['coefficient']) * values[t['condition_id']][view] for t in c['terms']), Decimal(0))
            if c['ncc_background_cancels'] and (view == 'ncc' or view.startswith(('single_probe/', 'leave_one_out/'))):
                value = sum((D(t['coefficient']) * values[t['condition_id']]['original/answer_mean'] for t in c['terms']), Decimal(0))
            if c['ncc_background_cancels'] and view == 'background': value = Decimal(0)
            check(r['effect'], value)
            bound = D(r['numeric_bound']); expected = 'unresolved' if abs(value) <= bound else 'positive' if value > 0 else 'negative'
            ensure(r['direction'] == expected, 'contrast direction differs'); directions += 1
        for r in lines(output / 'paired-changes.jsonl'):
            check(r['interaction'], D(r['shift_H']) - D(r['shift_A']))
            check(r['interaction'], D(r['delta_after']) - D(r['delta_before']))
            check(r['common_shift'], (D(r['shift_H']) + D(r['shift_A'])) / 2)
            if r['delta_before_direction'] == 'unresolved': ensure(r['signed_remaining_ratio'] is None, 'unresolved denominator ratio')
    return {'status': 'passed', 'stage': plan['stage'], 'decimal_precision': 50, 'scalar_checks': scalars,
            'direction_checks': directions, 'prediction_checks': predictions, 'near_zero_rounding_cases': rounded_ties,
            'max_absolute_export_error': str(largest), 'model_forward_executed': False}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=('inputs', 'results'))
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--run', type=Path)
    p.add_argument('--results', type=Path); p.add_argument('--receipt', type=Path)
    a = p.parse_args(); result = inputs(a.plan) if a.mode == 'inputs' else results(a.plan, a.run, a.results)
    if a.receipt:
        a.receipt.parent.mkdir(parents=True, exist_ok=True)
        with a.receipt.open('x') as f: json.dump(result, f, ensure_ascii=False, sort_keys=True, indent=2); f.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
