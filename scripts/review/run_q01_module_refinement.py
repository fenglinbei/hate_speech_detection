#!/usr/bin/env python3
"""Module refinement: CPU commands plus explicit, deadline-bound GPU execution."""
from pathlib import Path
from datetime import datetime
import argparse
import json
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.general_model_evidence_evaluation import require, file_sha
from diagnostics.q01_module_package import load_frozen, FREEZE, WORK, read_json, read_lines
from diagnostics.q01_module_execution import (execute, check_run, analyze, verify_completed_passes,
    verify_capture_audit, check_geometry, numeric_difference)
from diagnostics.q01_mechanism_inputs import digest


def preparation_check(directory, output, require_engineering=False):
    """Recheck every sealed pass and any safely flushed partial pass, without labels."""
    directory, output = Path(directory).resolve(), Path(output).resolve()
    plan, contexts, _, requests = load_frozen(directory)
    state = read_json(output / 'run_manifest.json')
    require(state['status'] in ('paused', 'complete'), 'checkpoint audit needs a released paused/complete run')
    require(all(i.get('all_workers_normal_exit') and i.get('all_workers_exited') for i in state['invocations']),
            'owned worker release incomplete')
    refs = verify_completed_passes(plan, contexts, requests, directory, output, state, require_complete=False)
    engineering = [p['pass_id'] for p in plan['schedule'] if p['phase'] == 'engineering']
    engineering_passed = all(p in state['completed_passes'] for p in engineering)
    if require_engineering:
        require(engineering_passed, 'all six engineering passes are required')
    result = {'status': 'passed', 'plan_id': plan['plan_id'], 'run_manifest_sha256': file_sha(output / 'run_manifest.json'),
              'engineering_acceptance_passed': engineering_passed, 'completed_passes': state['completed_passes'],
              'science_complete': state['status'] == 'complete', 'query_analysis_reference_parsed': False,
              'gpu_forward_executed': False, 'reusable_checkpoint_requests': 0, 'partial_prefix_max_error': None}
    if len(state['completed_passes']) == len(plan['schedule']):
        return result
    spec = plan['schedule'][len(state['completed_passes'])]
    target = output / spec['pass_id']
    by_request = {r['request_id']: r for r in requests}
    by_context = {c['record_id']: c for c in contexts}
    reference = {r['request_id']: r for r in refs.get(spec['phase'], [])}
    rows, artifacts, maximum = [], {}, 0.0
    for ordinal, index in enumerate(state['device_indices']):
        shard = target / 'shards' / str(index)
        path = shard / 'scores.jsonl'
        part = read_lines(path) if path.exists() else []
        expected = [rid for i, rid in enumerate(spec['request_ids'])
                    if (i + spec['options']['replica_shift']) % len(state['device_indices']) == ordinal]
        require([r['request_id'] for r in part] == expected[:len(part)], 'partial shard request prefix differs')
        identity = next(r for r in state['runtime_identities'] if r['physical_gpu_index'] == index)
        for row in part:
            request = by_request[row['request_id']]
            check_geometry(row, request, by_context[request['recipient']], plan, spec['options'])
            require(row['physical_gpu_index'] == index and row['physical_gpu_uuid'] == identity['hardware']['uuid']
                    and row['runtime_sha256'] == digest(identity['identity']), 'partial producer/runtime differs')
            if reference:
                maximum = max(maximum, numeric_difference(row, reference[row['request_id']]))
        rows.extend(part)
        if shard.exists():
            artifacts.update({str(p.relative_to(target)): file_sha(p) for p in shard.iterdir() if p.is_file()})
    limit = plan['numeric_policy']['repeat_abs_tolerance'] if spec['mode'] == 'repeat' else plan['numeric_policy']['epsilon']
    require(not reference or maximum <= limit, 'partial numerical challenge failed')
    if rows:
        verify_capture_audit(plan, contexts, requests, rows, target, {'artifacts': artifacts}, state, spec)
    result.update(next_pass=spec['pass_id'], reusable_checkpoint_requests=len(rows),
                  partial_prefix_max_error=maximum if reference else None,
                  remaining_candidate_evaluations=sum(p['candidate_evaluations'] for p in plan['schedule'][len(state['completed_passes']):]) - 2 * len(rows))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('validate', 'run', 'resume-check', 'engineering-check', 'check', 'analyze'))
    p.add_argument('--plan', type=Path, default=FREEZE)
    p.add_argument('--run', type=Path, default=WORK / 'run-01')
    p.add_argument('--output', type=Path, default=WORK / 'results-01')
    p.add_argument('--gpus', type=int, nargs='+')
    p.add_argument('--stop-at')
    p.add_argument('--phase', choices=('engineering', 'full'), default='engineering')
    args = p.parse_args()
    if args.command == 'validate':
        plan, _, _, _ = load_frozen(args.plan)
        result = {'status': 'validated', 'plan_id': plan['plan_id'], 'gpu_forward_executed': False}
    elif args.command in ('resume-check', 'engineering-check'):
        result = preparation_check(args.plan, args.run, args.command == 'engineering-check')
    elif args.command == 'run':
        require(args.gpus == [0, 1, 2, 3] and args.stop_at, 'this preparation requires four cards and an explicit deadline')
        stop = datetime.fromisoformat(args.stop_at)
        require(stop.tzinfo is not None, 'deadline needs timezone')
        def stop_owned_run(signum, frame):
            raise KeyboardInterrupt('termination: preserve completed pairs and release owned workers')
        signal.signal(signal.SIGTERM, stop_owned_run)
        if (args.run / 'run_manifest.json').exists():
            preparation_check(args.plan, args.run)
        result = execute(args.plan, args.run, args.gpus, stop_epoch=stop.timestamp(),
                         through_pass='engineering-replica' if args.phase == 'engineering' else None)
    elif args.command == 'check':
        check_run(args.plan, args.run)
        result = {'status': 'verified', 'gpu_forward_executed': False}
    else:
        result = analyze(args.plan, args.run, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
