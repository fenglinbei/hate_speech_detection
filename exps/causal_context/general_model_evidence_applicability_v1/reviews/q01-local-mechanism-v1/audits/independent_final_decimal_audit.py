"""Independent stdlib Decimal audit of the completed Q01 scoring and selection.

Reads sealed artifacts only. No project scoring helpers, model, or GPU imports.
All expression atoms are rebuilt from the recorded candidate token logprobs.
"""
from pathlib import Path
from decimal import Decimal, localcontext
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json

WORK = Path(__file__).resolve().parents[1]
RUN, RESULTS, FREEZE = WORK / 'run-01', WORK / 'results-01', WORK / 'frozen-01'
D = Decimal.from_float
MODES = ('answer_sum', 'answer_mean', 'total_with_eos', 'mean_with_eos')


def read(path): return json.loads(path.read_text())
def lines(path): return [json.loads(line) for line in path.open()]
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    state, plan = read(RUN / 'run_manifest.json'), read(FREEZE / 'plan.json')
    assert state['status'] == 'complete' and state['numerical_validation_passed']
    assert state['completed_passes'] == [s['pass_id'] for s in plan['schedule']]
    assert sha(RUN / 'science-reference/scores.jsonl') == state['raw_scores_sha256']
    assert all(i['all_workers_exited'] and i['all_workers_normal_exit'] for i in state['invocations'])
    manifest = read(RESULTS / 'manifest.json')
    assert manifest['run_manifest_sha256'] == sha(RUN / 'run_manifest.json')
    for name, h in manifest['artifacts'].items(): assert sha(RESULTS / name) == h
    for spec in plan['schedule']:
        target = RUN / spec['pass_id']
        m = read(target / 'manifest.json')
        assert m['status'] == 'complete'
        for name, h in m['artifacts'].items(): assert sha(target / name) == h
        a = read(target / 'acceptance.json')
        assert a['status'] == 'passed' and all(g['passed'] for g in a['gates'])
        assert not a.get('derived') or a['derived']['passed']
    raw = lines(RUN / 'science-reference/scores.jsonl')
    expressions = lines(RESULTS / 'comparisons.jsonl')
    saved_atoms = read(RESULTS / 'atoms.json')
    nomination = read(RESULTS / 'nomination.json')
    margins, atoms, maxima = {}, {}, defaultdict(lambda: Decimal(0))
    checked = defaultdict(int)
    eps = D(plan['numeric_policy']['epsilon'])

    def close(actual, stored, kind, tolerance='1e-10'):
        delta = abs(actual - D(stored))
        assert delta < Decimal(tolerance), (kind, str(delta), str(actual), stored)
        maxima[kind] = max(maxima[kind], delta)
        checked[kind] += 1

    with localcontext() as ctx:
        ctx.prec = 60
        for row in raw:
            scores = []
            for c in row['candidates']:
                answer = sum((D(v) for v in c['token_logprobs']), Decimal(0))
                eos, n = D(c['eos_logprob']), Decimal(c['answer_tokens'])
                values = {'answer_sum': answer, 'answer_mean': answer / n,
                          'total_with_eos': answer + eos, 'mean_with_eos': (answer + eos) / (n + 1)}
                for mode, value in values.items(): close(value, c['scores'][mode], 'candidate_scores')
                scores.append(values)
            margins[row['request_id']] = {mode: scores[1][mode] - scores[0][mode] for mode in MODES}
        for key, source in saved_atoms['source_terms'].items():
            if source['kind'] == 'margin':
                value = margins[source['request_id']][source['mode']]
            else:
                assert source['kind'] == 'logit_mean_sigmoid'
                xs = [margins[rid]['answer_mean'] for rid in source['request_ids']]
                p = sum((1 / (1 + (-x).exp()) for x in xs), Decimal(0)) / len(xs)
                value = (p / (1 - p)).ln()
            atoms[key] = value
            close(value, saved_atoms['values'][key], 'atoms')

        def coefficients(record):
            return {k: D(v) for k, v in record['coefficients'].items() if v}

        def combine(*terms):
            result = defaultdict(lambda: Decimal(0))
            for weight, record in terms:
                for key, value in coefficients(record).items(): result[key] += weight * value
            return {k: v for k, v in result.items() if v}

        def evaluate(record):
            coef = coefficients(record)
            value = sum((v * atoms[k] for k, v in coef.items()), Decimal(0))
            bound = sum(map(abs, coef.values()), Decimal(0)) * eps
            close(value, record['value'], 'expression_values')
            close(bound, record['numeric_bound'], 'expression_bounds')
            return value, bound

        primary = defaultdict(list)
        interim = {(r['group'], r['module'], r['layer'], r['role'], r['view'], r['direction'], r['metric']): r
                   for r in csv.DictReader((WORK / 'interim-01/all-comparisons.csv').open())}
        visited = set()
        for r in expressions:
            for fields in r['arms'].values():
                for record in fields.values(): evaluate(record)
                assert coefficients(fields['actual_shift']) == combine((1, fields['patched']), (-1, fields['recipient']))
                sign = 1 if r['direction'] == 'R' else -1
                assert coefficients(fields['effect']) == combine((sign, fields['actual_shift']))
                assert coefficients(fields['target']) == combine((sign, fields['donor_under_estimand']), (-sign, fields['recipient']))
            for metric, weights in [('C', (Decimal('0.5'), Decimal('0.5'))), ('I', (1, -1))]:
                m = r['metrics'][metric]
                for field in ('target', 'effect'):
                    assert coefficients(m[field]) == combine((weights[0], r['arms']['H'][field]), (weights[1], r['arms']['A'][field]))
                assert coefficients(m['residual']) == combine((1, m['target']), (-1, m['effect']))
                t, tb = evaluate(m['target']); e, eb = evaluate(m['effect']); z, zb = evaluate(m['residual'])
                resolved = abs(t) > tb
                aligned = resolved and abs(e) > eb and t * e > 0
                improvement, ib = abs(t) - abs(z), tb + zb
                eligible = aligned and improvement > ib
                assert resolved == m['target_resolved'] and aligned == m['aligned_beyond_bound']
                assert eligible == m['eligible_direction']
                close(improvement, m['absolute_residual_improvement'], 'residual_improvement')
                close(ib, m['improvement_numeric_bound'], 'improvement_bound')
                gain = 1 - abs(z) / abs(t) if resolved else None
                if resolved:
                    close(e / t, m['effect_over_target'], 'ratio', '1e-8')
                    close(gain, m['closeness_gain'], 'gain', '1e-8')
                else: assert m['effect_over_target'] is None and m['closeness_gain'] is None
                key = (r['group'], r['module'], str(r['layer']), r['role'], r['view'], r['direction'], metric)
                old = interim[key]; visited.add(key)
                for field in ('target', 'effect', 'residual'):
                    assert float(old[field]) == m[field]['value']
                assert float(old['closeness_gain']) == m['closeness_gain'] if old['closeness_gain'] else m['closeness_gain'] is None
                checked['unchanged_interim_metric_rows'] += 1
                if r['category'] == 'primary' and r['view'] == 'original/answer_sum':
                    primary[metric, r['layer'], r['role']].append((r['group'], r['direction'], eligible, gain))
        assert visited == interim.keys()

        role_order = ['lexicon_end', 'demo_answer_end', 'query_hehe', 'pre_answer']
        winners = {}
        for metric in ('C', 'I'):
            eligible = []
            for (what, layer, role), group in primary.items():
                if what != metric: continue
                assert len(group) == len({(x[0], x[1]) for x in group}) == 8
                if all(x[2] for x in group): eligible.append((min(x[3] for x in group), layer, role))
            eligible.sort(key=lambda x: (-x[0], x[1], role_order.index(x[2])))
            saved = nomination['rankings'][metric]
            assert len(eligible) == len(saved['eligible_ranking'])
            for value, actual in zip(eligible, saved['eligible_ranking']):
                assert value[1:] == (actual['layer'], actual['role'])
                close(value[0], actual['worst_closeness_gain'], 'ranking')
            winner = eligible[0] if eligible else None
            assert (winner[1], winner[2]) == (saved['winner']['layer'], saved['winner']['role'])
            winners[metric] = {'layer': winner[1], 'role': winner[2], 'eligible_units': len(eligible)}
        for row in lines(RESULTS / 'margins.jsonl'):
            for mode, actual in margins[row['request_id']].items():
                close(actual, row['margins'][mode], 'margins')
                assert ('non-hate' if actual > 0 else 'hate') == row['predictions'][mode]
                assert (abs(actual) > eps) == row['numerically_resolved'][mode]
                checked['predictions'] += 1

    receipt = {'status': 'passed', 'created_at': datetime.now(timezone.utc).isoformat(),
        'decimal_precision': 60, 'project_math_helpers_used': False, 'gpu_forward_executed': False,
        'counts': dict(checked), 'max_absolute_errors': {k: str(v) for k, v in maxima.items()},
        'independent_winners': winners, 'run_manifest_sha256': sha(RUN / 'run_manifest.json'),
        'results_manifest_sha256': sha(RESULTS / 'manifest.json'), 'script_sha256': sha(Path(__file__))}
    with (WORK / 'audits/independent-final-decimal-audit.json').open('x') as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__': main()
