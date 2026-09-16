"""Build separate descriptive tables and closeout receipts from sealed results."""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import csv
import hashlib
import json

WORK = Path(__file__).resolve().parents[1]
RUN, RESULTS = WORK / 'run-01', WORK / 'results-01'
DEST = WORK / 'interpretation-01'
MAIN = ('original/answer_sum', 'original/answer_mean', 'ncc_recalibrated',
        'ab_forward/answer_sum', 'ab_reverse/answer_sum')


def read(p): return json.loads(p.read_text())
def lines(p): return [json.loads(x) for x in p.open()]
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def write_json(p, data):
    with p.open('x') as f: json.dump(data, f, ensure_ascii=False, indent=2); f.write('\n')
def write_csv(name, rows):
    assert rows
    with (DEST / name).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    state = read(RUN / 'run_manifest.json')
    plan = read(WORK / 'frozen-01/plan.json')
    assert state['status'] == 'complete' and state['numerical_validation_passed']
    assert state['completed_passes'] == [x['pass_id'] for x in plan['schedule']]
    result_manifest = read(RESULTS / 'manifest.json')
    assert result_manifest['run_manifest_sha256'] == sha(RUN / 'run_manifest.json')
    for name, h in result_manifest['artifacts'].items(): assert sha(RESULTS / name) == h
    decimal = read(WORK / 'audits/independent-final-decimal-audit.json')
    assert decimal['status'] == 'passed' and decimal['results_manifest_sha256'] == sha(RESULTS / 'manifest.json')
    reconstruction = read(WORK / 'audits/final-analysis-reconstruction-01.log')
    comparisons = lines(RESULTS / 'comparisons.jsonl')
    nomination = read(RESULTS / 'nomination.json')
    assert reconstruction['comparison_rows'] == len(comparisons)
    assert reconstruction['selected_units'] == nomination['selected_units']
    DEST.mkdir(exist_ok=False)
    grouped = defaultdict(list)
    index = {}
    for row in comparisons:
        ident = (row['group'], row['module'], row['layer'], row['role'], row['category'], row['view'], row['direction'])
        assert ident not in index
        index[ident] = row
        for metric, values in row['metrics'].items():
            grouped[(row['category'], row['module'], row['layer'], row['role'], metric, row['view'], row['direction'])].append(values)
    summary = []
    for key, rows in sorted(grouped.items(), key=str):
        out = dict(zip(('category', 'module', 'layer', 'role', 'metric', 'view', 'direction'), key))
        out.update(groups=len(rows), target_resolved=sum(m['target_resolved'] for m in rows),
                   effect_resolved=sum(abs(m['effect']['value']) > m['effect']['numeric_bound'] for m in rows),
                   aligned=sum(m['aligned_beyond_bound'] for m in rows),
                   eligible=sum(m['eligible_direction'] for m in rows))
        for name in ('target', 'effect', 'residual'):
            values = [m[name]['value'] for m in rows]
            out[name + '_min'], out[name + '_max'] = min(values), max(values)
            out[name + '_abs_min'], out[name + '_abs_max'] = min(map(abs, values)), max(map(abs, values))
        for name in ('closeness_gain', 'effect_over_target'):
            values = [m[name] for m in rows if m[name] is not None]
            out[name + '_min'], out[name + '_max'] = (min(values), max(values)) if values else (None, None)
        summary.append(out)
    write_csv('all-view-summary.csv', summary)
    selected_keys = {(u['module'], u['layer'], u['role'], metric)
                     for u in nomination['selected_units'] for metric in u['selected_for']}
    selected_main = [r for r in summary if r['category'] == 'primary' and r['view'] in MAIN
                     and (r['module'], r['layer'], r['role'], r['metric']) in selected_keys]
    assert len(selected_main) == 20
    write_csv('selected-main-views.csv', selected_main)
    write_csv('all-unit-nomination.csv', [{'metric': metric, **u} for metric, s in nomination['rankings'].items() for u in s['all_units']])
    exceptions = [{k: u[k] for k in ('module', 'layer', 'role')} | e
                  for u in nomination['secondary_checks'] for e in u['direction_exceptions']]
    assert len(exceptions) == decimal['counts']['secondary_exceptions']
    write_csv('secondary-exceptions.csv', exceptions)
    unit_keys = {(u['module'], u['layer'], u['role']) for u in nomination['selected_units']}
    arm_rows, counts = [], defaultdict(list)
    for row in comparisons:
        if row['category'] != 'primary' or (row['module'], row['layer'], row['role']) not in unit_keys: continue
        for surface, fields in row['arms'].items():
            out = {k: row[k] for k in ('group', 'module', 'layer', 'role', 'view', 'direction')}
            out['surface'] = surface
            for name in ('recipient', 'donor_under_estimand', 'patched'):
                value, bound = fields[name]['value'], fields[name]['numeric_bound']
                out[name + '_margin'] = value
                out[name + '_label'] = 'non-hate' if value > 0 else 'hate'
                out[name + '_resolved'] = abs(value) > bound
            out['matches_donor_under_estimand_label'] = out['patched_label'] == out['donor_under_estimand_label']
            out['changed_recipient_label'] = out['patched_label'] != out['recipient_label']
            arm_rows.append(out)
            counts[row['module'], row['layer'], row['role'], row['view'], row['direction']].append(out)
    write_csv('selected-primary-arms.csv', arm_rows)
    prediction_summary = []
    for key, rows in sorted(counts.items(), key=str):
        out = dict(zip(('module', 'layer', 'role', 'view', 'direction'), key))
        out['correlated_arms'] = len(rows)
        for name in ('recipient', 'donor_under_estimand', 'patched'):
            out[name + '_non_hate'] = sum(r[name + '_label'] == 'non-hate' for r in rows)
            out[name + '_resolved'] = sum(r[name + '_resolved'] for r in rows)
        out['donor_label_mismatches'] = sum(not r['matches_donor_under_estimand_label'] for r in rows)
        out['changed_recipient_label'] = sum(r['changed_recipient_label'] for r in rows)
        prediction_summary.append(out)
    write_csv('selected-label-counts.csv', prediction_summary)
    fixed_equal = 0
    for ident, row in index.items():
        if row['view'] != 'ncc_fixed_recipient': continue
        peer = index[(*ident[:5], 'original/answer_mean', ident[-1])]
        assert row['metrics'] == peer['metrics']
        fixed_equal += 1
    budgets = []
    for spec in plan['schedule']:
        target = RUN / spec['pass_id']
        manifest, acceptance = read(target / 'manifest.json'), read(target / 'acceptance.json')
        for name, h in manifest['artifacts'].items(): assert sha(target / name) == h
        rows = lines(target / 'scores.jsonl')
        assert [r['request_id'] for r in rows] == spec['request_ids']
        assert all(not r['query_reference_loaded'] and not r['formal_test_or_reserve_access'] for r in rows)
        candidates = sum(len(r['candidates']) for r in rows)
        forwards = 0
        for row in rows:
            if spec['mode'] == 'prefix':
                calls = {c['prefix_unique_forward_count'] for c in row['candidates']}
                assert len(calls) == 1
                forwards += calls.pop()
            else: forwards += len(row['candidates'])
        captures = sum(read(p)['prompt_only_forward_calls'] for p in target.glob('shards/*/captures-*.json'))
        assert candidates == spec['candidate_evaluations'] == acceptance['candidate_evaluations']
        assert all(g['passed'] for g in acceptance['gates'])
        assert not acceptance['derived'] or acceptance['derived']['passed']
        budgets.append({'pass_id': spec['pass_id'], 'candidate_evaluations': candidates,
                        'scoring_forwards': forwards, 'prompt_only_captures': captures,
                        'gates': acceptance['gates'], 'derived': acceptance['derived']})
    old = read(WORK / 'audits/window-closeout-01.json')
    prefix_checks = []
    for phase in old['phases']:
        for shard in phase['shards']:
            path = RUN / phase['pass_id'] / 'shards' / str(shard['physical_gpu_index']) / 'scores.jsonl'
            content = path.read_bytes()
            assert hashlib.sha256(content[:shard['prefix_bytes']]).hexdigest() == shard['scores_sha256']
            if phase['sealed']: assert len(content) == shard['prefix_bytes']
            prefix_checks.append({'pass_id': phase['pass_id'], **shard, 'old_prefix_unchanged': True})
    closeout = {'schema_version': 'q01-module-final-closeout/v1', 'status': 'complete',
                'created_at': datetime.now(timezone.utc).isoformat(), 'plan_id': state['plan_id'],
                'run_manifest_sha256': sha(RUN / 'run_manifest.json'),
                'results_manifest_sha256': sha(RESULTS / 'manifest.json'),
                'completed_passes': len(budgets), 'passes': budgets,
                'candidate_evaluations': sum(b['candidate_evaluations'] for b in budgets),
                'scoring_forward_calls': sum(b['scoring_forwards'] for b in budgets),
                'prompt_only_capture_calls': sum(b['prompt_only_captures'] for b in budgets),
                'extra_prompt_only_captures_due_to_resume': sum(b['prompt_only_captures'] - 384 for b in budgets),
                'old_reused_partial_requests': 1711, 'checkpoint_prefix_checks': prefix_checks,
                'analysis_reconstruction_passed': True, 'independent_decimal_audit_passed': True,
                'fixed_ncc_equals_original_mean_comparisons': fixed_equal,
                'selected_units': nomination['selected_units'], 'secondary_exception_rows': len(exceptions),
                'gpu_forward_executed_by_this_audit': False, 'mechanism_ready': False,
                'artifacts': {name: sha(WORK / 'audits' / name) for name in (
                    'independent-final-decimal-audit.json', 'independent_final_decimal_audit.py',
                    'resource-release-final-01.json', 'final-analysis-reconstruction-01.log',
                    'build_final_review_01.py')}}
    assert closeout['candidate_evaluations'] == 67200
    assert closeout['scoring_forward_calls'] == 100640
    assert closeout['prompt_only_capture_calls'] == 4976
    write_json(WORK / 'audits/final-closeout-01.json', closeout)
    print(json.dumps({k: closeout[k] for k in ('status', 'candidate_evaluations', 'scoring_forward_calls',
        'prompt_only_capture_calls', 'extra_prompt_only_captures_due_to_resume',
        'fixed_ncc_equals_original_mean_comparisons', 'secondary_exception_rows')}, ensure_ascii=False))


if __name__ == '__main__': main()
