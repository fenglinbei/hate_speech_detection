"""Independent stdlib-only audit of the sealed module engineering reference pass."""
from pathlib import Path
from decimal import Decimal, localcontext
import hashlib
import json

ROOT = Path(__file__).resolve().parents[5]
# Locate the workspace by its experiment directory, without importing project code.
while not (ROOT / 'src/diagnostics/q01_module_package.py').is_file():
    if ROOT == ROOT.parent:
        raise RuntimeError('workspace not found')
    ROOT = ROOT.parent
WORK = ROOT / 'exps/causal_context/general_model_evidence_applicability_v1/reviews/q01-module-refinement-v1'
FREEZE = WORK / 'frozen-01'
TARGET = WORK / 'run-01/engineering-reference'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def number(value):
    return Decimal.from_float(value) if isinstance(value, float) else Decimal(value)


def scores(tokens, eos):
    total = sum(map(number, tokens), Decimal(0))
    return {'answer_sum': total, 'answer_mean': total / len(tokens),
            'total_with_eos': total + number(eos),
            'mean_with_eos': (total + number(eos)) / (len(tokens) + 1), 'eos_logprob': number(eos)}


def readout(row):
    output = {}
    margins = {}
    for c in row['candidates']:
        values = scores(c['token_logprobs'], c['eos_logprob'])
        margins[c['candidate_id']] = values
        for key, value in values.items():
            output[c['candidate_id'] + '/' + key] = value
        for i, value in enumerate(c['token_logprobs']):
            output[c['candidate_id'] + '/token/' + str(i)] = number(value)
    for mode in ('answer_sum', 'answer_mean', 'total_with_eos', 'mean_with_eos'):
        output['margin/' + mode] = margins['non-hate'][mode] - margins['hate'][mode]
    return output


def difference(left, right):
    require(left.keys() == right.keys(), 'readout coverage differs')
    return max(abs(left[k] - right[k]) for k in left)


def audit():
    plan = read(FREEZE / 'plan.json')
    manifest = read(TARGET / 'manifest.json')
    require(manifest['status'] == 'complete', 'reference pass is not sealed')
    for name, expected in manifest['artifacts'].items():
        require(sha(TARGET / name) == expected, 'sealed pass bytes changed: ' + name)
    data = rows(TARGET / 'scores.jsonl')
    requests = {r['request_id']: r for r in rows(FREEZE / 'requests.jsonl')}
    spec = plan['schedule'][0]
    require([r['request_id'] for r in data] == spec['request_ids'], 'reference frame differs')
    baselines = {requests[r['request_id']]['recipient']: readout(r) for r in data if requests[r['request_id']]['kind'] == 'baseline'}
    history = {r['record_id']: readout(r) for r in rows(FREEZE / 'historical-scores.jsonl')}
    require(baselines.keys() == history.keys() and len(baselines) == 96, 'history bridge frame differs')
    scalar_count, candidate_count, controls = 0, 0, 0
    max_arithmetic = max_reference = max_control = Decimal(0)
    max_history = max(difference(v, history[k]) for k, v in baselines.items())
    for row in data:
        require(row['query_reference_loaded'] is False and row['formal_test_or_reserve_access'] is False, 'reference isolation differs')
        for c in row['candidates']:
            candidate_count += 1
            raw = scores(c['token_logprobs'], c['eos_logprob'])
            reference = scores(c['reference_token_logprobs'], c['reference_eos_logprob'])
            for computed, recorded in ((raw, c), (raw, c['scores']), (reference, c['reference_scores'])):
                for name, value in computed.items():
                    scalar_count += 1
                    error = abs(value - number(recorded[name]))
                    max_arithmetic = max(max_arithmetic, error)
                    require(error < Decimal('1e-10'), 'independent score arithmetic differs')
            max_reference = max(max_reference, difference(raw, reference),
                max(abs(number(a) - number(b)) for a, b in zip(c['token_logprobs'], c['reference_token_logprobs'], strict=True)))
        request = requests[row['request_id']]
        if request['expected_baseline']:
            controls += 1
            max_control = max(max_control, difference(readout(row), baselines[request['expected_baseline']]))
    require(controls == 2784 and candidate_count == 5760, 'engineering coverage differs')
    require(max_history <= number(plan['numeric_policy']['epsilon']) and max_control <= number(plan['numeric_policy']['epsilon']), 'engineering expectation failed')
    require(max_reference <= number(plan['numeric_policy']['reference_abs_tolerance']), 'FP64 arithmetic check failed')
    return {'status': 'passed', 'plan_id': plan['plan_id'], 'pass_manifest_sha256': sha(TARGET / 'manifest.json'),
        'auditor_sha256': sha(Path(__file__)), 'decimal_precision': 60, 'candidate_scores': candidate_count,
        'reconstructed_scalars': scalar_count, 'historical_sources': 96, 'engineering_expectations': controls,
        'max_reconstruction_error': str(max_arithmetic), 'max_fp64_error': str(max_reference),
        'max_historical_error': str(max_history), 'max_engineering_error': str(max_control),
        'project_scoring_helpers_imported': False, 'gpu_forward_executed': False,
        'analysis_reference_parsed': False, 'only_reference_pass_audited': True, 'all_six_engineering_passes_complete': False}


if __name__ == '__main__':
    with localcontext() as context:
        context.prec = 60
        print(json.dumps(audit(), ensure_ascii=False))
