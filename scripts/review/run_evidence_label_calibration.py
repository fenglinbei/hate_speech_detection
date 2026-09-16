#!/usr/bin/env python3
"""Execute/check the separate full-label NCC and A/B frozen plan."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from data.stage1_data import sha256_file
from diagnostics.general_model_package import read_json, read_jsonl, write_json
from diagnostics.general_model_evidence_evaluation import require
from diagnostics.evidence_label_calibration import compare_passes, reference_rows, compare_derived
from scripts.review.freeze_evidence_label_calibration import WORK, load_frozen as load_plan
from scripts.review.run_evidence_interventions import SCHEDULE, gate, now, verified_sources

def execute(directory, output):
    from diagnostics.general_model_numeric import atomic_json, progress
    from diagnostics.evidence_label_calibration_execution import score_pass, replica_proof
    from diagnostics.evidence_label_calibration_execution import validate_geometry
    from diagnostics.general_model_nolabel_execution import validate_runtime
    from diagnostics.evidence_label_calibration_execution import PersistentNumericPool, validate_sealed_pass

    plan, contexts, historical = load_plan(directory)
    comparisons = read_jsonl(directory / 'comparisons.jsonl')
    require(output.resolve().is_relative_to(WORK) and not output.resolve().is_relative_to(directory.resolve())
            and not directory.resolve().is_relative_to(output.resolve()), 'run overlaps immutable preparation or old inputs')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.writer.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = output / 'run_manifest.json'
    if state_path.exists():
        previous = read_json(state_path)
        require(previous['plan_id'] == plan['plan_id'], 'existing run belongs to another plan')
        require(previous['status'] in ('running', 'interrupted'), 'terminal run is sealed; use check, not a new forward')
    state = {'schema_version': 'evidence-label-calibration-run/v1', 'plan_id': plan['plan_id'],
             'plan_path': str(directory.relative_to(ROOT)), 'status': 'running', 'started_at': now(),
             'query_gold_loaded_during_scoring': False, 'formal_test_or_reserve_access': False,
             'model_forward_executed': False, 'checks': [], 'derived_checks': [], 'raw_pass': plan['raw_pass']}
    atomic_json(state_path, state)
    baseline_contexts = [r for r in contexts if r['baseline_replay']]
    selected = {'baseline': baseline_contexts, 'all': contexts}
    pool = None
    references = {}
    policy = plan['numeric_policy']
    try:
        progress('evidence-loading-fp32-replicas', conditions=288, baselines=36)
        pool = PersistentNumericPool(plan['runtime_parent_plan'], plan['config']['runtime'], [0, 1, 2, 3], ROOT)
        validate_runtime(plan, pool.identity)
        atomic_json(output / 'runtime_identity.json', pool.identity)
        for spec in plan['schedule']:
            name, selection, options, check = (spec[k] for k in ('pass', 'selection', 'options', 'check'))
            progress('evidence-pass-start', pass_name=name, blocks=len(selected[selection]))
            state.update(active_pass=name, model_forward_executed=True)
            atomic_json(state_path, state)
            rows, receipt = score_pass(pool, selected[selection], plan, output / name, batch_size=1, **options)
            require(validate_sealed_pass(output / name, plan) == receipt, 'sealed pass validation failed')
            proof = validate_geometry(rows, selected[selection], plan, **options)
            write_json(output / (name + '-geometry.json'), proof)
            if check == 'same_logits_reference':
                references[selection] = rows
                comparison = compare_passes(reference_rows(rows), rows)
                limit = policy['reference_abs_tolerance']
            else:
                comparison = compare_passes(references[selection], rows)
                limit = policy['repeat_abs_tolerance'] if check == 'repeat' else policy['epsilon']
            write_json(output / (name + '-differences.json'), comparison)
            info = {'pass': name, 'check': check, 'max_abs_error': comparison['max_abs_error'],
                    'limit': limit, 'passed': comparison['max_abs_error'] <= limit,
                    'manifest_sha256': sha256_file(output / name / 'manifest.json')}
            state['checks'].append(info)
            atomic_json(state_path, state)
            gate(comparison, limit, name)
            if selection == 'all':
                target = reference_rows(rows) if check == 'same_logits_reference' else references['all']
                derived = compare_derived(contexts, target, rows, comparisons, limit)
                write_json(output / (name + '-derived-differences.json'), derived)
                state['derived_checks'].append({'pass': name, 'passed': derived['passed'],
                    'readouts': derived['readouts'], 'max_bound_fraction': derived['max_bound_fraction']})
                atomic_json(state_path, state)
                require(derived['passed'], 'NCC propagated numerical gate failed: ' + name)
            if name == 'regression-b1-r0':
                comparison = compare_passes(historical, rows)
                write_json(output / 'historical-differences.json', comparison)
                info = {'pass': name, 'check': 'historical_replay', 'max_abs_error': comparison['max_abs_error'],
                        'limit': policy['epsilon'], 'passed': comparison['max_abs_error'] <= policy['epsilon']}
                state['checks'].append(info)
                atomic_json(state_path, state)
                gate(comparison, policy['epsilon'], 'historical_replay')
            if name == plan['raw_pass']:
                comparison = compare_passes(references['baseline'], [r for r in rows if r['record_id'] in {c['record_id'] for c in baseline_contexts}])
                write_json(output / 'raw-baseline-differences.json', comparison)
                info = {'pass': name, 'check': 'raw_baseline_replay', 'max_abs_error': comparison['max_abs_error'],
                        'limit': policy['epsilon'], 'passed': comparison['max_abs_error'] <= policy['epsilon']}
                state['checks'].append(info)
                atomic_json(state_path, state)
                gate(comparison, policy['epsilon'], 'raw_baseline_replay')
            if check == 'replica':
                write_json(output / 'replica-producer-proof.json', replica_proof(references['all'], rows))
            progress('evidence-pass-verified', **info)
        verified_sources(plan['code_sha256'])
        state.update(status='complete', completed_at=now(), active_pass=None, numerical_validation_passed=True,
                     raw_scores_sha256=sha256_file(output / plan['raw_pass'] / 'scores.jsonl'),
                     raw_manifest_sha256=sha256_file(output / plan['raw_pass'] / 'manifest.json'),
                     blocks=288, candidates=576, mechanism_ready=False)
        atomic_json(state_path, state)
        progress('evidence-scoring-complete', blocks=288, candidates=576)
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error_type=type(error).__name__, error=str(error), stopped_at=now())
        atomic_json(state_path, state)
        raise
    finally:
        if pool is not None:
            pool.close(terminate=state['status'] != 'complete')
        lock.close()

def check_run(directory, output):
    from diagnostics.evidence_label_calibration import compare_passes
    from diagnostics.evidence_label_calibration_execution import replica_proof
    from diagnostics.evidence_label_calibration_execution import validate_geometry
    from diagnostics.general_model_nolabel_execution import validate_runtime
    from diagnostics.evidence_label_calibration_execution import validate_sealed_pass
    plan, contexts, historical = load_plan(directory)
    comparisons = read_jsonl(directory / 'comparisons.jsonl')
    state = read_json(output / 'run_manifest.json')
    require(state['status'] == 'complete' and state['plan_id'] == plan['plan_id'], 'run is not complete')
    require(state['numerical_validation_passed'] and len(state['checks']) == 10
            and all(c['passed'] for c in state['checks']), 'terminal gates are incomplete')
    require(len(state['derived_checks']) == 6 and all(r['passed'] for r in state['derived_checks']), 'derived gates incomplete')
    derived_receipts = []
    validate_runtime(plan, read_json(output / 'runtime_identity.json'))
    selected = {'all': contexts, 'baseline': [r for r in contexts if r['baseline_replay']]}
    references = {}
    for spec in plan['schedule']:
        name, selection, options, check = (spec[k] for k in ('pass', 'selection', 'options', 'check'))
        validate_sealed_pass(output / name, plan)
        rows = read_jsonl(output / name / 'scores.jsonl')
        require(validate_geometry(rows, selected[selection], plan, **options) == read_json(output / (name + '-geometry.json')), 'geometry differs')
        if check == 'same_logits_reference':
            references[selection] = rows
            comparison = compare_passes(reference_rows(rows), rows)
            limit = plan['numeric_policy']['reference_abs_tolerance']
        else:
            comparison = compare_passes(references[selection], rows)
            limit = plan['numeric_policy']['repeat_abs_tolerance'] if check == 'repeat' else plan['numeric_policy']['epsilon']
        require(comparison == read_json(output / (name + '-differences.json')), 'numerical comparison differs')
        gate(comparison, limit, name)
        if selection == 'all':
            target = reference_rows(rows) if check == 'same_logits_reference' else references['all']
            derived = compare_derived(contexts, target, rows, comparisons, limit)
            require(derived == read_json(output / (name + '-derived-differences.json')) and derived['passed'], 'derived comparison differs')
            derived_receipts.append({'pass': name, 'passed': derived['passed'],
                'readouts': derived['readouts'], 'max_bound_fraction': derived['max_bound_fraction']})
        if check == 'replica':
            require(replica_proof(references['all'], rows) == read_json(output / 'replica-producer-proof.json'), 'replica producer proof differs')
    for name, comparison in (
        ('historical-differences.json', compare_passes(historical, references['baseline'])),
        ('raw-baseline-differences.json', compare_passes(references['baseline'], [r for r in references['all'] if r['record_id'] in {c['record_id'] for c in selected['baseline']}]))):
        require(comparison == read_json(output / name), 'baseline comparison differs')
        gate(comparison, plan['numeric_policy']['epsilon'], name)
    require(sha256_file(output / plan['raw_pass'] / 'scores.jsonl') == state['raw_scores_sha256']
            and sha256_file(output / plan['raw_pass'] / 'manifest.json') == state['raw_manifest_sha256'], 'terminal raw hashes differ')
    require(derived_receipts == state['derived_checks'], 'derived gate receipt differs')
    print(json.dumps({'status': 'verified', 'passes': len(SCHEDULE), 'blocks': 288, 'candidates': 576, 'model_forward_executed_by_check': False}))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('validate', 'run', 'check', 'analyze'))
    parser.add_argument('--plan', type=Path, default=WORK / 'frozen-01')
    parser.add_argument('--run', type=Path, default=WORK / 'run-01')
    parser.add_argument('--output', type=Path, default=WORK / 'results-01')
    args = parser.parse_args()
    if args.command == 'validate':
        plan, contexts, historical = load_plan(args.plan)
        print(json.dumps({'status': 'validated', 'plan_id': plan['plan_id'], 'contexts': len(contexts),
                          'historical': len(historical), 'model_forward_executed': False}))
    elif args.command == 'run': execute(args.plan, args.run)
    elif args.command == 'check': check_run(args.plan, args.run)
    else:
        from scripts.review.analyze_evidence_label_calibration import analyze
        analyze(args.plan, args.run, args.output)


if __name__ == '__main__': main()
