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
FREEZE = WORK / 'frozen-01'
PROBES = {'empty': '', 'space': ' ', 'na': 'N/A', 'mask': '[MASK]', 'lorem': 'Lorem ipsum'}
MARK = '\n\n待判断文本（JSON 字符串）：\n'


def read(path): return json.loads(path.read_text())
def lines(path): return [json.loads(x) for x in path.read_text().splitlines()]
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(obj): return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def check(condition, message):
    if not condition: raise ValueError(message)


def inputs():
    m = read(FREEZE / 'manifest.json'); plan = read(FREEZE / 'plan.json')
    for path, h in m['source_files'].items(): check(digest(ROOT / path) == h, 'source changed: ' + path)
    for name, h in m['artifacts'].items(): check(digest(FREEZE / name) == h, 'artifact changed: ' + name)
    oldfreeze = WORK.parents[1] / 'analysis-freeze-20260912/matched-input-freeze-v1/frozen-01'
    original = {r['record_id']: r for r in lines(oldfreeze / 'contexts.jsonl')}
    contexts = lines(FREEZE / 'contexts.jsonl')
    check(len(contexts) == len({r['record_id'] for r in contexts}) == len({r['prompt_sha256'] for r in contexts}) == 288, 'context coverage')
    counts = Counter()
    for c in contexts:
        check(hashlib.sha256(canonical({k: v for k, v in c.items() if k != 'context_sha256'})).hexdigest() == c['context_sha256'], 'context sha')
        source = original[c['source_record_id']]
        left = source['messages'][1]['content'].split(MARK); right = c['messages'][1]['content'].split(MARK)
        check(len(left) == len(right) == 2, 'query split')
        encoding = c['encoding']; counts[encoding + ('_probe' if c['probe_id'] is not None else '_real')] += 1
        if encoding == 'original':
            check(source['messages'][0] == c['messages'][0] and left[0] == right[0], 'probe changed background')
            wanted = left[1] if c['probe_id'] is None else json.dumps(PROBES[c['probe_id']], ensure_ascii=False)
            check(right[1] == wanted, 'probe query replacement')
            if c['probe_id'] is None:
                check(source['prompt_text'] == c['prompt_text'] and source['prompt_token_ids'] == c['prompt_token_ids'], 'original token replay')
        else:
            check(left[1] == right[1] and c['probe_id'] is None, 'AB query changed')
            reverse = {'A': 'hate', 'B': 'non-hate'} if encoding == 'ab_forward' else {'B': 'hate', 'A': 'non-hate'}
            a, b = left[0].splitlines(), right[0].splitlines(); check(len(a) == len(b), 'AB lines differ')
            answers = 0
            for x, y in zip(a, b):
                if x in ('输出："hate"', '输出："non-hate"'):
                    check(y.startswith('输出：') and reverse[json.loads(y[3:])] == json.loads(x[3:]), 'demo answer mapping')
                    answers += 1
                else: check(x == y, 'AB non-answer line changed')
            check(answers in (9, 10), 'AB demo coverage')
            catalog = plan['catalog'][c['task']]
            check([reverse[json.loads(x['canonical_answer'])] for x in catalog] == ['hate', 'non-hate'], 'candidate semantic decoding')
            check(all(x['answer_tokens'] == 2 for x in catalog), 'AB candidate length')
    check(dict(counts) == {'original_real': 36, 'original_probe': 180, 'ab_forward_real': 36, 'ab_reverse_real': 36}, 'encoding counts')
    geometry = {(r['query_id'], r['condition'], r['encoding'], r['probe_id']): r for r in lines(FREEZE / 'geometry.jsonl')}
    for q, c, e, p in geometry:
        if e == 'ab_forward':
            a, b = geometry[q, c, e, p], geometry[q, c, 'ab_reverse', None]
            check(a['layout'] == b['layout'] and a['prompt_tokens'] == b['prompt_tokens'], 'mapping token layout')
    return {'status': 'passed', 'sources_unchanged': len(m['source_files']), 'artifacts_verified': len(m['artifacts']),
            'contexts_checked': 288, 'original_replays': 36, 'probe_replacements': 180, 'AB_mappings': 72,
            'ab_layout_pairs': 36, 'query_references_parsed': False, 'model_forward_executed': False}


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
    state = read(WORK / 'run-01/run_manifest.json')
    check(state['status'] == 'complete' and state['numerical_validation_passed'] and len(state['checks']) == 10
          and len(state['derived_checks']) == 6 and all(c['passed'] for c in state['checks'] + state['derived_checks']), 'result read before gates')
    folder = WORK / 'results-01'; plan = read(FREEZE / 'plan.json')
    contexts = lines(FREEZE / 'contexts.jsonl'); raw = lines(WORK / 'run-01' / plan['raw_pass'] / 'scores.jsonl')
    check(digest(WORK / 'run-01' / plan['raw_pass'] / 'scores.jsonl') == state['raw_scores_sha256'], 'raw hash')
    refs = {r['query_id']: r for r in read(FREEZE / 'analysis_references.json')}
    check(len(raw) == len(contexts) and [r['record_id'] for r in raw] == [c['record_id'] for c in contexts], 'raw frame')
    modes = ('answer_sum', 'answer_mean', 'total_with_eos', 'mean_with_eos')
    all_margins = {(c['query_id'], c['root_condition'], c['encoding'], c['probe_id'], mode): decimal_margin(r, mode)
                   for c, r in zip(contexts, raw) for mode in modes}
    values, bounds = {}, {}; epsilon = Decimal.from_float(plan['numeric_policy']['epsilon'])
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
    rm = read(folder / 'manifest.json')
    for path, h in rm['source_files'].items(): check(digest(ROOT / path) == h, 'analysis source changed')
    for name, h in rm['artifacts'].items(): check(digest(folder / name) == h, 'analysis artifact changed')
    return {'status': 'passed', 'independent_arithmetic': 'stdlib Decimal, 50 decimal digits; explicit probability-space NCC',
            'scalar_checks': scalar_count, 'max_abs_error_vs_export': max_error, 'all_original_and_reviewed_predictions_checked': True,
            'contrast_and_probe_directions_checked': 1664, 'model_forward_executed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('inputs', 'results')); args = parser.parse_args()
    with localcontext() as ctx:
        ctx.prec = 50
        output = inputs() if args.stage == 'inputs' else results()
    output['verifier_sha256'] = digest(Path(__file__))
    target = Path(__file__).parent / ('independent-' + args.stage + '.json')
    data = json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    if target.exists(): check(target.read_text() == data, 'independent receipt changed')
    else: target.write_text(data)
    print(json.dumps(output, ensure_ascii=False))
