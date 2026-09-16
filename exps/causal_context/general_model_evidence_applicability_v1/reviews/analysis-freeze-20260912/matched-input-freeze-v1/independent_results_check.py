"""Independent standard-library audit; refuses to inspect unsealed scores."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import csv
import json
import math
from pathlib import Path

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[5]
FREEZE, RUN, RESULTS = (WORK / name for name in ('frozen-01', 'run-01', 'results-01'))
MODES = ('answer_sum', 'answer_mean', 'total_with_eos', 'mean_with_eos')
read = lambda p: json.loads(p.read_text())
lines = lambda p: [json.loads(s) for s in p.read_text().splitlines()]
digest = lambda p: sha256(p.read_bytes()).hexdigest()
checks = 0


def same(actual, expected):
    global checks
    assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10), (actual, expected)
    checks += 1


state = read(RUN / 'run_manifest.json')
assert state['status'] == 'complete' and state['numerical_validation_passed']
assert len(state['checks']) == 10 and all(c['passed'] and c['max_abs_error'] <= c['limit'] for c in state['checks'])
fm, rm = read(FREEZE / 'manifest.json'), read(RESULTS / 'manifest.json')
plan = read(FREEZE / 'plan.json')
assert state['plan_id'] == rm['plan_id'] == plan['plan_id']
assert rm['freeze_manifest_sha256'] == digest(FREEZE / 'manifest.json')
assert rm['run_manifest_sha256'] == digest(RUN / 'run_manifest.json')
raw_path = RUN / plan['raw_pass'] / 'scores.jsonl'
assert rm['raw_scores_sha256'] == state['raw_scores_sha256'] == digest(raw_path)
for name, expected in fm['source_files'].items():
    assert digest(ROOT / name) == expected, name
for directory, manifest in ((FREEZE, fm), (RESULTS, rm)):
    assert {p.name for p in directory.iterdir()} == set(manifest['artifacts']) | {'manifest.json'}
    for name, expected in manifest['artifacts'].items():
        assert digest(directory / name) == expected, name

contexts = lines(FREEZE / 'contexts.jsonl')
raw = lines(raw_path)
assert len(contexts) == len(raw) == 36
assert [c['record_id'] for c in contexts] == [r['record_id'] for r in raw]
reference = {(r['query_id'], r['task']): r for r in read(FREEZE / 'analysis_references.json')}
assert len(reference) == 2 and all(r['original_label'] == r['adjudicated_label'] == 'non-hate' for r in reference.values())
score_lookup, context_lookup = {}, {}
for c, block in zip(contexts, raw):
    for key in ('record_id', 'query_id', 'task', 'condition', 'prompt_sha256', 'context_sha256'):
        assert c[key] == block[key], key
    assert [x['candidate_id'] for x in block['candidates']] == ['hate', 'non-hate']
    for x, catalog in zip(block['candidates'], plan['catalog']['hate']):
        assert x['answer_token_ids'] == catalog['answer_token_ids']
        assert x['prompt_token_ids_sha256'] == c['prompt_token_ids_sha256']
        values, eos = x['token_logprobs'], x['eos_logprob']
        assert len(values) == len(x['answer_token_ids']) == x['answer_tokens']
        total = math.fsum(values)
        with_eos = math.fsum([*values, eos])
        expected = {'answer_sum': total, 'answer_mean': total / len(values), 'eos_logprob': eos,
                    'total_with_eos': with_eos, 'mean_with_eos': with_eos / (len(values) + 1)}
        for mode, value in expected.items():
            same(x['scores'][mode], value)
            same(x[mode], value)
        score_lookup[c['record_id'], x['candidate_id']] = expected
    context_lookup[c['record_id']] = c

conditions = lines(RESULTS / 'conditions.jsonl')
assert len(conditions) == 144 and len({(r['record_id'], r['score_mode']) for r in conditions}) == 144
margin_lookup = {}
for r in conditions:
    c, mode = context_lookup[r['record_id']], r['score_mode']
    assert mode in MODES and all(r[k] == c[k] for k in ('query_id', 'task', 'condition', 'prompt_sha256', 'prompt_tokens', 'baseline_replay'))
    h, n = [score_lookup[r['record_id'], label][mode] for label in ('hate', 'non-hate')]
    margin = n - h
    prediction = 'hate' if h >= n else 'non-hate'
    assert r['prediction'] == prediction and r['prediction_ordinal'] == (0 if prediction == 'hate' else 1)
    assert r['original_reference'] == r['reviewed_reference'] == 'non-hate'
    assert r['original_correct'] == r['reviewed_correct'] == (prediction == 'non-hate')
    assert r['tied_top_count'] == (2 if h == n else 1)
    for key in ('non_hate_margin', 'original_margin', 'reviewed_margin'):
        same(r[key], margin)
    same(r['hate_direction'], -margin)
    same(r['top_gap'], abs(margin))
    same(r['label_margins']['hate'], -margin)
    margin_lookup[r['query_id'], r['condition'], mode] = margin

comparisons = lines(FREEZE / 'comparisons.jsonl')
effects = lines(RESULTS / 'contrasts.jsonl')
specs = {r['contrast_id']: r for r in comparisons}
assert len(comparisons) == len(specs) == 64 and len(effects) == 256
assert {(r['contrast_id'], r['score_mode']) for r in effects} == {(r, mode) for r in specs for mode in MODES}
for r in effects:
    spec = specs[r['contrast_id']]
    assert r['terms'] == spec['terms'] and r['query_id'] == spec['query_id'] and not r['human_mechanism_judgment']
    value = math.fsum(t['coefficient'] * margin_lookup[r['query_id'], t['condition'], r['score_mode']] for t in spec['terms'])
    bound = sum(abs(t['coefficient']) for t in spec['terms']) * plan['numeric_policy']['epsilon']
    detail = r['non_hate_margin_effect']
    for actual in (detail['value'], r['original_margin_effect'], r['reviewed_margin_effect']):
        same(actual, value)
    same(detail['numeric_bound'], bound)
    assert detail['raw_sign'] == (1 if value > 0 else -1 if value < 0 else 0)
    assert detail['direction'] == ('numerically_unresolved' if abs(value) <= bound else 'positive' if value > 0 else 'negative')
    assert detail['statistical_interval'] is False

for name, count in (('conditions', 144), ('contrasts', 256)):
    with (RESULTS / (name + '.csv')).open() as f:
        assert len(list(csv.DictReader(f))) == count
summary = read(RESULTS / 'summary.json')
assert summary['primary_prediction_counts'] == dict(Counter(r['prediction'] for r in conditions if r['score_mode'] == 'answer_sum'))
report = {'schema_version': 'evidence-matched-results-independent-audit/v1', 'status': 'passed',
          'recorded_at': datetime.now(timezone.utc).isoformat(), 'plan_id': plan['plan_id'],
          'freeze_manifest_sha256': digest(FREEZE / 'manifest.json'),
          'run_manifest_sha256': digest(RUN / 'run_manifest.json'),
          'results_manifest_sha256': digest(RESULTS / 'manifest.json'),
          'raw_scores_sha256': digest(raw_path), 'checker_sha256': digest(Path(__file__)),
          'conditions': 36, 'candidates': 72, 'condition_readouts': 144, 'contrast_readouts': 256,
          'scalar_checks': checks, 'source_files_verified': len(fm['source_files']),
          'new_model_forward_executed_by_audit': False, 'production_writes': 0}
print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
