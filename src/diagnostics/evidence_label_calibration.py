"""Plan-bound full binary labels and gold-free NCC, with propagated bounds."""
from copy import deepcopy
import math

from diagnostics.general_model_evidence_evaluation import require, canonical, sha
from diagnostics.general_model_numeric_analysis import candidate_scores, SCORE_MODES

ENCODINGS = {'original': 'hate', 'ab_forward': 'hate_ab_forward', 'ab_reverse': 'hate_ab_reverse'}
PROBES = [('empty', ''), ('space', ' '), ('na', 'N/A'), ('mask', '[MASK]'), ('lorem', 'Lorem ipsum')]


def validate_block(row, context, catalog):
    for key in ('record_id', 'query_id', 'task', 'condition', 'context_sha256', 'prompt_sha256'):
        require(row.get(key) == context[key], 'score/input identity differs: ' + key)
    wanted = catalog[context['task']]
    require(len(row['candidates']) == len(wanted) == 2, 'incomplete binary catalog')
    require([c['candidate_id'] for c in wanted] == ['hate', 'non-hate'], 'semantic candidate order differs')
    for c, expected in zip(row['candidates'], wanted):
        require(all(c.get(k) == v for k, v in expected.items()), 'candidate differs from frozen encoding')
        require(c.get('finite_target_logits_checked') and c.get('token_boundary_checked'), 'candidate boundary/logits not checked')
        require(len(c['token_logprobs']) == len(c['answer_token_ids']) == c['answer_tokens'], 'candidate token count differs')
        computed = candidate_scores(c['token_logprobs'], c['eos_logprob'])
        require(c['scores'] == computed and all(c.get(k) == v for k, v in computed.items()), 'candidate score does not reproduce')
    readouts(row)


def readouts(row):
    candidates = row['candidates']
    require([c['candidate_id'] for c in candidates] == ['hate', 'non-hate'], 'candidate semantics/order differs')
    values = {}
    for c in candidates:
        computed = candidate_scores(c['token_logprobs'], c['scores']['eos_logprob'])
        require(computed == c['scores'], 'stored score differs from token scores')
        for i, v in enumerate(c['token_logprobs']): values[f"{c['candidate_id']}/token/{i}"] = v
        for k, v in computed.items(): values[f"{c['candidate_id']}/{k}"] = v
    for mode in SCORE_MODES:
        values['margin/' + mode] = candidates[1]['scores'][mode] - candidates[0]['scores'][mode]
    require(all(math.isfinite(v) for v in values.values()), 'nonfinite numeric readout')
    return values


def compare_passes(reference, other):
    require(len(reference) == len(other), 'comparison coverage differs')
    differences, maximum, largest = [], 0.0, None
    for a, b in zip(reference, other):
        require(a['record_id'] == b['record_id'] and a['task'] == b['task'], 'comparison order/task differs')
        for x, y in zip(a['candidates'], b['candidates']):
            require(all(x[k] == y[k] for k in ('candidate_id', 'canonical_answer', 'answer_token_ids', 'eos_token_id')), 'comparison label identity differs')
        left, right = readouts(a), readouts(b)
        require(left.keys() == right.keys(), 'numeric readout frame differs')
        values = {k: right[k] - left[k] for k in left}
        for k, v in values.items():
            require(math.isfinite(v), 'nonfinite difference')
            if abs(v) > maximum: maximum, largest = abs(v), {'record_id': a['record_id'], 'metric': k, 'difference': v}
        differences.append({'record_id': a['record_id'], 'differences': values})
    return {'max_abs_error': maximum, 'largest_error': largest, 'blocks': differences}


def reference_rows(rows):
    result = deepcopy(rows)
    for r in result:
        for c in r['candidates']:
            c.update(c['reference_scores'])
            c['scores'] = candidate_scores(c['token_logprobs'], c['eos_logprob'])
    return result


def logsumexp(xs):
    require(bool(xs) and all(math.isfinite(x) for x in xs), 'invalid logsumexp input')
    m = max(xs)
    return m + math.log(math.fsum(math.exp(x - m) for x in xs))


def log_sigmoid(x):
    return -math.log1p(math.exp(-x)) if x >= 0 else x - math.log1p(math.exp(x))


def background_margin(probe_margins):
    # logit(mean(sigmoid(m_j))), using normalized label probabilities.
    return logsumexp([log_sigmoid(x) for x in probe_margins]) - logsumexp([log_sigmoid(-x) for x in probe_margins])


def calibrated_rows(contexts, raw):
    by_id = {r['record_id']: r for r in raw}
    require(len(by_id) == len(raw) == len(contexts) and by_id.keys() == {c['record_id'] for c in contexts}, 'calibration frame differs')
    index = {}
    for c in contexts:
        row = by_id[c['record_id']]
        require(row['context_sha256'] == c['context_sha256'] and row['prompt_sha256'] == c['prompt_sha256'], 'calibration input binding differs')
        index[c['query_id'], c['root_condition'], c['encoding'], c['probe_id']] = readouts(row)
    output = []
    keys = sorted({(c['query_id'], c['root_condition']) for c in contexts})
    for q, condition in keys:
        original = index[q, condition, 'original', None]
        probes = [index[q, condition, 'original', p]['margin/answer_mean'] for p, _ in PROBES]
        bg = background_margin(probes)
        real = original['margin/answer_mean']
        output.append({'query_id': q, 'condition': condition, 'real_mean_margin': real,
                       'background_margin': bg, 'ncc_margin': real - bg,
                       'probe_margins': dict(zip((p for p, _ in PROBES), probes)),
                       'probe_ncc_margins': {p: real - probes[i] for i, (p, _) in enumerate(PROBES)},
                       'leave_one_out_ncc_margins': {p: real - background_margin(probes[:i] + probes[i + 1:]) for i, (p, _) in enumerate(PROBES)}})
    return output


def derived_readouts(contexts, raw, comparisons):
    rows = calibrated_rows(contexts, raw)
    index = {(r['query_id'], r['condition']): r for r in rows}
    values = {}
    for r in rows:
        prefix = r['query_id'] + '/' + r['condition']
        for name, factor in (('background_margin', 1), ('ncc_margin', 2)):
            values[prefix + '/' + name] = (r[name], factor)
        for kind in ('probe_ncc_margins', 'leave_one_out_ncc_margins'):
            for p, v in r[kind].items(): values[prefix + '/' + kind + '/' + p] = (v, 2)
    for comp in comparisons:
        terms = comp['terms']; l1 = sum(abs(t['coefficient']) for t in terms)
        for field, factor in (('background_margin', 1), ('ncc_margin', 2)):
            v = math.fsum(t['coefficient'] * index[comp['query_id'], t['condition']][field] for t in terms)
            values[comp['contrast_id'] + '/' + field] = (v, factor * l1)
        for field in ('probe_ncc_margins', 'leave_one_out_ncc_margins'):
            for p, _ in PROBES:
                v = math.fsum(t['coefficient'] * index[comp['query_id'], t['condition']][field][p] for t in terms)
                values[comp['contrast_id'] + '/' + field + '/' + p] = (v, 2 * l1)
    return values


def compare_derived(contexts, reference, other, comparisons, epsilon):
    left, right = [derived_readouts(contexts, rows, comparisons) for rows in (reference, other)]
    require(left.keys() == right.keys(), 'derived readout frame differs')
    rows = []
    for key, (a, factor) in left.items():
        b, other_factor = right[key]; require(factor == other_factor, 'derived bound differs')
        error, bound = abs(b - a), epsilon * factor
        rows.append({'metric': key, 'difference': b - a, 'bound': bound, 'passed': error <= bound})
    return {'passed': all(r['passed'] for r in rows), 'readouts': len(rows),
            'max_bound_fraction': max(abs(r['difference']) / r['bound'] for r in rows), 'differences': rows}
