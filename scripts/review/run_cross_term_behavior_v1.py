#!/usr/bin/env python3
"""Prepare on CPU; bind/run only in a separately authorized future GPU window."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics import cross_term_behavior_execution_v1 as c


def validate_resume_state(state, resume, preparation, binding):
    c.require(resume and state['status'] == 'paused', 'only an explicit paused resume is allowed; terminal runs cannot restart')
    c.require(state['preparation_manifest'] == preparation and state['binding'] == binding, 'resume source binding differs')


def supervise(prepared, bound, run, resume=False):
    prepared, bound, run = Path(prepared).resolve(), Path(bound).resolve(), Path(run).resolve()
    c.require(run.parent == c.WORK and run.name.startswith('run-'), 'new experiment owned run directory required')
    c.WORK.mkdir(parents=True, exist_ok=True)
    with (c.WORK / '.executor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, _ = c.check_bound(prepared, bound, weights=True)
        state_path = run / 'run_manifest.json'
        c.require(not (run / 'STOP').exists(), 'STOP remains present')
        if state_path.exists():
            state = c.read(state_path)
            validate_resume_state(state, resume, c.file_info(prepared / 'manifest.json'), c.file_info(run / 'binding.json'))
            c.require(c.read(run / 'binding.json')['device_binding'] == c.file_info(bound), 'resume device binding differs')
            c.resume_check(prepared, run)
        else:
            c.require(not resume and not run.exists(), 'new, absent run directory required')
            state = None
        inventory = c.old.gpu_inventory()
        device = c.old.idle_device(plan['allocation']['index'], inventory)
        c.require({k: device[k] for k in plan['allocation']} == plan['allocation'], 'original bound GPU identity differs')
        if state is None:
            run.mkdir()
            c.atomic_json(run / 'binding.json', {'schema_version': 'cross-term-behavior-run-binding/v1',
                'run_id': str(uuid.uuid4()), 'preparation_manifest': c.file_info(prepared / 'manifest.json'),
                'device_binding': c.file_info(bound), 'allocation': plan['allocation'], 'runtime': plan['runtime'],
                'started_at': c.now()}, replace=False)
            state = {'schema_version': 'cross-term-behavior-run-state/v1', 'status': 'created',
                'preparation_manifest': c.file_info(prepared / 'manifest.json'), 'binding': c.file_info(run / 'binding.json'),
                'invocations': [], 'completed_passes': [], 'analysis_reference_join_performed': False}
        invocation = str(uuid.uuid4())
        state['invocations'].append({'invocation_id': invocation, 'controller_pid': os.getpid(),
                                      'worker_pid': None, 'started_at': c.now()})
        state.update(status='launching', updated_at=c.now())
        c.atomic_json(state_path, state)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=plan['allocation']['uuid'], CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONHASHSEED='0')
        command = [sys.executable, str(Path(__file__).resolve()), '_worker', '--prepared', str(prepared),
                   '--bound', str(bound), '--run', str(run), '--invocation', invocation]
        with (run / 'gpu.log').open('ab', buffering=0) as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            print(json.dumps({'worker_pid': process.pid, 'run': str(run), 'allocation': plan['allocation']}), flush=True)
            code = process.wait()
        state = c.read(state_path)
        inventory = c.old.gpu_inventory()
        absent = not Path(f'/proc/{process.pid}').exists() and not any(
            len(r) > 1 and r[1].strip() == str(process.pid) for r in inventory['compute_processes'])
        c.require(absent, 'owned CUDA worker remains after wait')
        release_path = run / f'resource-release-{invocation}.json'
        c.atomic_json(release_path, {'worker_pid': process.pid, 'worker_exit_code': code,
            'owned_worker_absent': True, 'inventory': inventory, 'checked_at': c.now()}, replace=False)
        state.update(worker_exit_code=code, owned_worker_absent=True, resource_release=c.file_info(release_path),
                     worker_exit_checked_at=c.now())
        if code == 0 and state['status'] == 'scoring_complete_releasing':
            state.update(status='complete', completed_at=c.now())
        elif code != 0:
            state.update(status='failed', failure='worker failed; no automatic retries')
        c.atomic_json(state_path, state)
        c.require(code == 0 and state['status'] in ['complete', 'paused'], 'worker failed; inspect gpu.log')
        return {'status': state['status'], 'run_manifest': c.file_info(state_path), 'resource_release': state['resource_release']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'validate', 'bind', 'run', '_worker', 'resume-check', 'check', 'analyze'])
    parser.add_argument('--prepared', type=Path, default=c.WORK / 'prepared-01')
    parser.add_argument('--bound', type=Path, default=c.WORK / 'bound-01.json')
    parser.add_argument('--run', type=Path, default=c.WORK / 'run-01')
    parser.add_argument('--output', type=Path, default=c.WORK / 'results-01')
    parser.add_argument('--authorization-note')
    parser.add_argument('--weights', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--invocation')
    args = parser.parse_args()
    if args.action == 'prepare':
        result = c.create_preparation(args.prepared)
    elif args.action == 'validate':
        plan, requests = c.check_prepared(args.prepared, args.weights)
        result = {'status': 'pass', 'new_inputs': len(requests['new']), 'historical_bridge_inputs': len(requests['bridge']),
                  'allocation': None, 'GPU_qualification': False, 'budget': plan['budget']}
    elif args.action == 'bind':
        result = c.bind_gpu(args.prepared, args.bound, args.authorization_note)
    elif args.action == 'run':
        result = supervise(args.prepared, args.bound, args.run, args.resume)
    elif args.action == '_worker':
        c.require(args.invocation, 'worker invocation identity required')
        from diagnostics.cross_term_behavior_gpu_v1 import worker
        result = {'status': worker(args.prepared, args.bound, args.run, args.invocation)}
    elif args.action == 'resume-check':
        result = c.resume_check(args.prepared, args.run)
    elif args.action == 'check':
        result = c.check_completed(args.prepared, args.run)
    else:
        result = c.analyze(args.prepared, args.run, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
