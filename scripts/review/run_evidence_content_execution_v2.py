#!/usr/bin/env python3
"""Versioned two-GPU execution of the identical frozen content experiment."""
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
from scripts.review.freeze_evidence_content_execution_v2 import WORK, load_frozen as load_plan
from scripts.review.run_evidence_interventions import SCHEDULE, gate, now, verified_sources


def gpu_preflight(plan):
    import csv
    import subprocess
    indices = plan['config']['execution']['device_indices']
    require(indices == [1, 2], 'this execution version is bound to physical GPUs 1 and 2')
    command = ['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
               '--format=csv,noheader,nounits', '--id', ','.join(map(str, indices))]
    output = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30).stdout
    rows = list(csv.reader(output.strip().splitlines(), skipinitialspace=True))
    require(len(rows) == 2 and [int(r[0]) for r in rows] == indices, 'GPU preflight inventory differs')
    expected = plan['execution_amendment']['selected_gpu_uuids']
    for r in rows:
        require(r[1].strip() == expected[r[0]], 'GPU UUID differs from authorized allocation')
        require(int(r[3]) - int(r[2]) >= 36000 and int(r[4]) <= 5, 'selected GPU is no longer idle: ' + r[0])
    return {'checked_at': now(), 'command': command,
            'devices': [{'index': int(r[0]), 'uuid': r[1].strip(), 'used_mib': int(r[2]),
                         'total_mib': int(r[3]), 'utilization_percent': int(r[4])} for r in rows]}

def execute(directory, output):
    from diagnostics.general_model_numeric import atomic_json, progress
    from diagnostics.evidence_label_calibration_execution import score_pass, replica_proof
    from diagnostics.evidence_label_calibration_execution import validate_geometry
    from diagnostics.general_model_nolabel_execution import validate_runtime
    from diagnostics.evidence_label_calibration_execution import PersistentNumericPool, validate_sealed_pass

    directory, output = directory.resolve(), output.resolve()
    plan, contexts, historical = load_plan(directory)
    comparisons = read_jsonl(directory / 'comparisons.jsonl')
    require(output.resolve().parent == WORK and output.name.startswith('run-') and not output.resolve().is_relative_to(directory.resolve())
            and not directory.resolve().is_relative_to(output.resolve()), 'run overlaps immutable preparation or old inputs')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.writer.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = output / 'run_manifest.json'
        require(state_path.exists() or {p.name for p in output.iterdir()} == {'.writer.lock'}, 'nonempty output has no resumable run manifest')
        if state_path.exists():
            previous = read_json(state_path)
            require(previous['plan_id'] == plan['plan_id'], 'existing run belongs to another plan')
            require(previous['status'] in ('running', 'interrupted'), 'terminal run is sealed; use check, not a new forward')
        state = {'schema_version': 'evidence-content-decomposition-run/v2', 'plan_id': plan['plan_id'],
                 'plan_path': str(directory.relative_to(ROOT)), 'status': 'running', 'started_at': now(),
                 'query_gold_loaded_during_scoring': False, 'formal_test_or_reserve_access': False,
                 'model_forward_executed': False, 'checks': [], 'derived_checks': [], 'raw_pass': plan['raw_pass'], 'device_indices': plan['config']['execution']['device_indices']}
        atomic_json(state_path, state)
        baseline_contexts = [r for r in contexts if r['baseline_replay']]
        selected = {'baseline': baseline_contexts, 'all': contexts}
        pool = None
        references = {}
        policy = plan['numeric_policy']
        try:
            inventory = gpu_preflight(plan)
            atomic_json(output / 'launch_gpu_inventory.json', inventory)
            progress('evidence-loading-fp32-replicas', conditions=len(contexts), baselines=len(baseline_contexts))
            pool = PersistentNumericPool(plan['runtime_parent_plan'], plan['config']['runtime'], plan['config']['execution']['device_indices'], ROOT)
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
            load_plan(directory)  # Recheck the complete freeze/source closure before sealing.
            state.update(status='complete', completed_at=now(), active_pass=None, numerical_validation_passed=True,
                         raw_scores_sha256=sha256_file(output / plan['raw_pass'] / 'scores.jsonl'),
                         raw_manifest_sha256=sha256_file(output / plan['raw_pass'] / 'manifest.json'),
                         blocks=len(contexts), candidates=sum(len(plan['catalog'][c['task']]) for c in contexts), mechanism_ready=False)
            atomic_json(state_path, state)
            progress('evidence-scoring-complete', blocks=len(contexts), candidates=sum(len(plan['catalog'][c['task']]) for c in contexts))
        except BaseException as error:
            state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                         error_type=type(error).__name__, error=str(error), stopped_at=now())
            atomic_json(state_path, state)
            raise
        finally:
            if pool is not None:
                pool.close(terminate=state['status'] != 'complete')

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
    raw_receipts = []
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
        raw_receipts.append({'pass': name, 'check': check, 'max_abs_error': comparison['max_abs_error'],
            'limit': limit, 'passed': True, 'manifest_sha256': sha256_file(output / name / 'manifest.json')})
        if name == 'regression-b1-r0':
            diff = compare_passes(historical, rows)
            raw_receipts.append({'pass': name, 'check': 'historical_replay', 'max_abs_error': diff['max_abs_error'],
                'limit': plan['numeric_policy']['epsilon'], 'passed': diff['max_abs_error'] <= plan['numeric_policy']['epsilon']})
        if name == plan['raw_pass']:
            ids = {c['record_id'] for c in selected['baseline']}
            diff = compare_passes(references['baseline'], [r for r in rows if r['record_id'] in ids])
            raw_receipts.append({'pass': name, 'check': 'raw_baseline_replay', 'max_abs_error': diff['max_abs_error'],
                'limit': plan['numeric_policy']['epsilon'], 'passed': diff['max_abs_error'] <= plan['numeric_policy']['epsilon']})
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
    require(raw_receipts == state['checks'], 'raw gate receipt differs')
    require(state['blocks'] == len(contexts) and state['candidates'] == plan['counts']['candidates']
            and state['model_forward_executed'] and not state['query_gold_loaded_during_scoring'], 'terminal run coverage/provenance differs')
    print(json.dumps({'status': 'verified', 'passes': len(plan['schedule']), 'blocks': len(contexts), 'candidates': plan['counts']['candidates'], 'model_forward_executed_by_check': False}))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('validate', 'run', 'check', 'analyze'))
    parser.add_argument('--plan', type=Path, default=WORK / 'frozen-02')
    parser.add_argument('--run', type=Path, default=WORK / 'run-02')
    parser.add_argument('--output', type=Path, default=WORK / 'results-02')
    args = parser.parse_args()
    if args.command == 'validate':
        plan, contexts, historical = load_plan(args.plan)
        print(json.dumps({'status': 'validated', 'plan_id': plan['plan_id'], 'contexts': len(contexts),
                          'historical': len(historical), 'model_forward_executed': False}))
    elif args.command == 'run': execute(args.plan, args.run)
    elif args.command == 'check': check_run(args.plan, args.run)
    else:
        from scripts.review.analyze_evidence_content_decomposition import analyze
        analyze(args.plan, args.run, args.output)


if __name__ == '__main__': main()
