"""Publish a new combined result pointer only after both stages and audits pass."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json

WORK = Path(__file__).resolve().parents[1]
BASE = WORK.parents[2]
ROOT = BASE.parents[2]
PUBLIC = BASE / 'functional-query-results-v1'
read = lambda p: json.loads(p.read_text())
lines = lambda p: [json.loads(s) for s in p.read_text().splitlines()]
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
relative = lambda p: str(p.relative_to(BASE))


def verify(p):
    m = read(p / 'manifest.json')
    for name, h in m['source_files'].items(): assert digest(ROOT / name) == h, name
    for name, h in m['artifacts'].items(): assert digest(p / name) == h, name
    return m


def main():
    assert not (PUBLIC / 'current.json').exists(), 'result pointer already published'
    stages, facts, inputs, comparison_ids = [], [], [], []
    independent = Counter()
    views = ['original/answer_sum', 'original/answer_mean', 'ncc', 'ab_forward/answer_sum', 'ab_reverse/answer_sum']
    for stage in (1, 2):
        freeze, run, output = [WORK / f'{name}-stage-{stage}-01' for name in ('frozen', 'run', 'results')]
        fm, rm = verify(freeze), verify(output)
        plan, state, summary = read(freeze / 'plan.json'), read(run / 'run_manifest.json'), read(output / 'summary.json')
        assert state['status'] == summary['status'] == 'complete'
        assert len(state['checks']) == 10 and len(state['derived_checks']) == 6
        assert state['numerical_validation_passed'] and all(r['passed'] for r in state['checks'] + state['derived_checks'])
        assert digest(run / plan['raw_pass'] / 'scores.jsonl') == state['raw_scores_sha256']
        release = read(run / 'worker-release.json')
        assert release['all_workers_exited'] and len(release['workers']) == 4
        assert all(w['exited'] and w['exitcode'] == 0 for w in release['workers'])
        evals = sum(len(r['candidates']) for spec in plan['schedule'] for r in lines(run / spec['pass'] / 'scores.jsonl'))
        assert evals == plan['counts']['candidate_evaluations']
        audit_path = WORK / f'audits/independent-results-stage-{stage}-01.json'
        audit = read(audit_path)
        assert audit['status'] == 'passed' and not audit['model_forward_executed']
        for key in ('scalar_checks', 'direction_checks', 'prediction_checks', 'near_zero_rounding_cases'): independent[key] += audit[key]
        inputs.append({r['record_id']: r for r in lines(freeze / 'contexts.jsonl')})
        comparison_ids.append({r['contrast_id'] for r in lines(freeze / 'comparisons.jsonl')})
        refs = read(output / 'analysis_references.json')
        assert all(r['original_label'] is None for r in refs if r['query_id'] != '3169')
        cs = [r for r in lines(output / 'condition-scores.jsonl') if r['query_id'] != '3169']
        counts = {arm: {v: sum(r['reviewed_correct'] for r in cs if r['lexicon_arm'] == arm and r['view'] == v)
                        for v in views} for arm in sorted({r['lexicon_arm'] for r in cs})}
        facts.append({'stage': stage, 'correct_conditions_by_arm': counts,
                      'primary_views': views, 'not_independent_case_accuracy': True})
        stages.append({'stage': stage, 'plan_id': plan['plan_id'], 'counts': plan['counts'],
            'freeze_path': relative(freeze), 'freeze_manifest_sha256': digest(freeze / 'manifest.json'),
            'run_path': relative(run), 'run_manifest_sha256': digest(run / 'run_manifest.json'),
            'raw_scores_sha256': state['raw_scores_sha256'], 'results_path': relative(output),
            'results_manifest_sha256': digest(output / 'manifest.json'), 'source_files_unchanged': len(fm['source_files']),
            'raw_gates': state['checks'], 'derived_gates': state['derived_checks'],
            'independent_results_path': relative(audit_path), 'independent_results_sha256': digest(audit_path)})
    common = inputs[0].keys() & inputs[1].keys()
    assert len(common) == 80 and all(inputs[0][k] == inputs[1][k] for k in common)
    assert len(comparison_ids[0] | comparison_ids[1]) == 608
    assert len(comparison_ids[0] & comparison_ids[1]) == 12
    bridge = read(WORK / 'run-stage-2-01/stage-bridge-differences.json')
    assert bridge['passed'] and bridge['shared_prompts'] == 80
    gpu_release = WORK / 'audits/gpu-release-01.json'
    release = read(gpu_release)
    assert release['all_owned_workers_exited'] and not release['owned_worker_pids_still_in_nvidia_smi']
    cpu_path = WORK / 'audits/cpu-acceptance-01.json'
    cpu = read(cpu_path)
    assert cpu['status'] == 'passed' and cpu['unittest_tests'] == 7
    with (PUBLIC / 'report-facts.json').open('x') as f:
        json.dump({'author': 'assistant', 'phase': 'post-outcome', 'stages': facts}, f, ensure_ascii=False, indent=2, sort_keys=True); f.write('\n')
    artifacts = [PUBLIC / name for name in ('README.md', 'INTERPRETATION.md', 'stage-1-C-I.png', 'stage-2-C-I.png', 'report-facts.json')]
    artifacts += [cpu_path, gpu_release, Path(__file__).resolve(), WORK / 'audits/plot-results-01.py',
                  BASE / 'functional-query-diagnostics-v1/feedback-01.json', BASE / 'functional-query-diagnostics-v1/execution-01.json']
    pointer = {'schema_version': 'evidence-functional-query-results-current/v1', 'status': 'complete',
        'recorded_at': datetime.now(timezone.utc).isoformat(), 'stages': stages,
        'scientific_queries': 8, 'new_scientific_conditions': 192, 'unique_prompt_inputs': len(inputs[0].keys() | inputs[1].keys()),
        'candidate_evaluations': sum(s['counts']['candidate_evaluations'] for s in stages),
        'stage_bridge': {k: bridge[k] for k in ('shared_prompts', 'max_abs_error', 'limit', 'passed')},
        'all_608_draft_comparisons_covered': True, 'independent_audit_totals': dict(independent),
        'cpu_tests_passed': 7, 'numerical_validation_passed': True, 'gpu_workers_exited': True,
        'device_indices': [0, 1, 2, 3], 'model_forward_executed': True,
        'bulk_adoption_preserves_ai_authorship': True, 'human_fields_changed_by_analysis': 0,
        'old_pointer_changes': 0, 'new_queries_have_no_original_gold': True, 'online_writeback': False,
        'formal_test_or_reserve_access': False, 'mechanism_ready': False,
        'artifacts': {relative(p): digest(p) for p in artifacts}}
    assert pointer['candidate_evaluations'] == 9712 and pointer['unique_prompt_inputs'] == 708
    with (PUBLIC / 'current.json').open('x') as f:
        json.dump(pointer, f, ensure_ascii=False, sort_keys=True, indent=2); f.write('\n')
    print(json.dumps({k: pointer[k] for k in ('status', 'scientific_queries', 'new_scientific_conditions', 'candidate_evaluations', 'stage_bridge', 'independent_audit_totals')}, ensure_ascii=False))


if __name__ == '__main__': main()
