"""Independent stdlib-only source/input and Decimal result verification."""
import argparse
from collections import Counter
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
WORK = Path(__file__).resolve().parents[1]
FREEZE = WORK / 'frozen-02'
PROBES = {'empty': '', 'space': ' ', 'na': 'N/A', 'mask': '[MASK]', 'lorem': 'Lorem ipsum'}
MARK = '\n\n待判断文本（JSON 字符串）：\n'


def read(path): return json.loads(path.read_text())
def lines(path): return [json.loads(x) for x in path.read_text().splitlines()]
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(obj): return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def check(condition, message):
    if not condition: raise ValueError(message)


def decimal_margin(row, mode):
    scores = []
    for c in row['candidates']:
        values = [Decimal.from_float(x) for x in c['token_logprobs']]
        if mode in ('total_with_eos', 'mean_with_eos'): values += [Decimal.from_float(c['eos_logprob'])]
        s = sum(values)
        if mode in ('answer_mean', 'mean_with_eos'): s /= len(values)
        scores.append(s)
    return scores[1] - scores[0]


def bg(margins):
    p = sum(Decimal(1) / (1 + (-m).exp()) for m in margins) / len(margins)
    return (p / (1 - p)).ln()


def results():
    state = read(WORK / 'run-02/run_manifest.json')
    check(state['status'] == 'complete' and state['numerical_validation_passed'] and len(state['checks']) == 10
          and len(state['derived_checks']) == 6 and all(c['passed'] for c in state['checks'] + state['derived_checks']), 'result read before gates')
    folder = WORK / 'results-02'; plan = read(FREEZE / 'plan.json')
    check(state['plan_id'] == plan['plan_id'], 'run plan identity differs')
    contexts = lines(FREEZE / 'contexts.jsonl'); raw = lines(WORK / 'run-02' / plan['raw_pass'] / 'scores.jsonl')
    check(digest(WORK / 'run-02' / plan['raw_pass'] / 'scores.jsonl') == state['raw_scores_sha256'], 'raw hash')
    refs = {r['query_id']: r for r in read(FREEZE / 'analysis_references.json') if r['task'] == 'hate'}
    check(len(raw) == len(contexts) and [r['record_id'] for r in raw] == [c['record_id'] for c in contexts], 'raw frame')
    modes = ('answer_sum', 'answer_mean', 'total_with_eos', 'mean_with_eos')
    all_margins = {(c['query_id'], c['root_condition'], c['encoding'], c['probe_id'], mode): decimal_margin(r, mode)
                   for c, r in zip(contexts, raw) for mode in modes}
    values, bounds = {}, {}; epsilon = Decimal.from_float(plan['numeric_policy']['epsilon'])
    check((len(contexts), len(raw)) == (208, 208), 'content result frame')
    scalar_count = 0; max_error = 0.
    def near(actual, expected):
        nonlocal scalar_count, max_error
        error = abs(float(actual) - float(expected)); scalar_count += 1; max_error = max(error, max_error)
        check(error <= 1e-12, f'independent scalar differs: {actual} versus {expected}')
    for r in lines(folder / 'condition-scores.jsonl'):
        value = all_margins[r['query_id'], r['condition'], r['encoding'], None, r['score_mode']]
        near(r['non_hate_margin'], value)
        pred = 'non-hate' if value > 0 else 'hate'
        check(r['prediction'] == pred and r['original_correct'] == (pred == refs[r['query_id']]['original_label'])
              and r['reviewed_correct'] == (pred == refs[r['query_id']]['adjudicated_label']), 'semantic prediction/reference differs')
        values[r['query_id'], r['condition'], r['view']] = value; bounds[r['view']] = epsilon
    for r in lines(folder / 'ncc-conditions.jsonl'):
        q, c = r['query_id'], r['condition']; real = all_margins[q, c, 'original', None, 'answer_mean']
        margins = [all_margins[q, c, 'original', p, 'answer_mean'] for p in PROBES]
        background = bg(margins); residual = real - background
        near(r['background_margin'], background); near(r['ncc_margin'], residual)
        near(r['calibrated_non_hate_probability'], Decimal(1) / (1 + (-residual).exp()))
        check(r['prediction'] == ('non-hate' if residual > 0 else 'hate'), 'calibrated prediction')
        for i, p in enumerate(PROBES):
            for name, value in [('single_probe', real - margins[i]), ('leave_one_out', real - bg(margins[:i] + margins[i + 1:]))]:
                field = 'probe_ncc_margins' if name == 'single_probe' else 'leave_one_out_ncc_margins'
                near(r[field][p], value); values[q, c, name + '/' + p] = value; bounds[name + '/' + p] = 2 * epsilon
        f = values[q, c, 'ab_forward/answer_sum']; rev = values[q, c, 'ab_reverse/answer_sum']
        for name, value, bound in [('ncc', residual, 2 * epsilon), ('background', background, epsilon),
                                  ('ab_symmetric_sum', (f + rev) / 2, epsilon), ('ab_mapping_gap_sum', f - rev, 2 * epsilon)]:
            values[q, c, name] = value; bounds[name] = bound
    comparisons = {r['contrast_id']: r for r in lines(FREEZE / 'comparisons.jsonl')}
    for file in ('contrast-scores.jsonl', 'probe-contrasts.jsonl'):
        for row in lines(folder / file):
            comp = comparisons[row['contrast_id']]; view = row.get('view', row['kind'] + '/' + row.get('probe_id', ''))
            value = sum(Decimal(t['coefficient']) * values[row['query_id'], t['condition'], view] for t in comp['terms'])
            bound = sum(abs(t['coefficient']) for t in comp['terms']) * bounds[view]
            near(row['effect'], value); near(row['numeric_bound'], bound)
            expected = 'unresolved' if abs(value) <= bound else ('positive' if value > 0 else 'negative')
            check(row['direction'] == expected, 'independent contrast direction differs')
    check(scalar_count == 2210, 'independent scalar coverage')
    rm = read(folder / 'manifest.json')
    for path, h in rm['source_files'].items(): check(digest(ROOT / path) == h, 'analysis source changed')
    for name, h in rm['artifacts'].items(): check(digest(folder / name) == h, 'analysis artifact changed')
    return {'status': 'passed', 'independent_arithmetic': 'stdlib Decimal, 50 decimal digits; explicit probability-space NCC',
            'scalar_checks': scalar_count, 'max_abs_error_vs_export': max_error, 'all_original_and_reviewed_predictions_checked': True,
            'contrast_and_probe_directions_checked': 780, 'model_forward_executed': False}


if __name__ == '__main__':
    with localcontext() as ctx:
        ctx.prec = 50
        output = results()
    output['verifier_sha256'] = digest(Path(__file__))
    output['plan_id'] = read(FREEZE / 'plan.json')['plan_id']
    output['raw_scores_sha256'] = read(WORK / 'run-02/run_manifest.json')['raw_scores_sha256']
    target = Path(__file__).parent / 'independent-results-two-gpu.json'
    data = json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    if target.exists(): check(target.read_text() == data, 'independent receipt changed')
    else:
        with target.open('x') as stream: stream.write(data)
    print(json.dumps(output, ensure_ascii=False))
