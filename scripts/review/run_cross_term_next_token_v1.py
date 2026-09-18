#!/usr/bin/env python3
"""Freeze, execute and verify the separately registered cross-term next-token run."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.cross_term_next_token_v1 import (
    WORK, analyze, atomic_json, check_completed, check_freeze, create_freeze,
    file_info, gpu_inventory, idle_device, now, read, require, sha,
)


def supervise(frozen, run, resume=False):
    frozen, run = Path(frozen).resolve(), Path(run).resolve()
    require(run.parent == WORK and run.name.startswith('run-'), 'dedicated owned run directory required')
    run.mkdir(parents=True, exist_ok=True)
    with (WORK / '.executor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, _ = check_freeze(frozen, weights=True)
        device = idle_device(plan['allocation']['index'])
        require({k: device[k] for k in plan['allocation']} == plan['allocation'], 'bound GPU identity differs')
        require(not (run / 'STOP').exists(), 'STOP remains present')
        state_path = run / 'run_manifest.json'
        if state_path.exists():
            state = read(state_path)
            require(resume and state['status'] == 'paused', 'only explicit same-binding paused resume is allowed')
            require(state['frozen_manifest'] == file_info(frozen / 'manifest.json'), 'resume freeze differs')
        else:
            require(not resume, 'no run exists to resume')
            binding = {'schema_version': 'cross-term-next-token-run-binding/v1', 'run_id': str(uuid.uuid4()),
                'frozen_manifest': file_info(frozen / 'manifest.json'), 'allocation': plan['allocation'],
                'runtime': plan['runtime'], 'started_at': now()}
            atomic_json(run / 'binding.json', binding, replace=False)
            state = {'schema_version': 'cross-term-next-token-run-state/v1', 'status': 'created',
                'frozen_manifest': file_info(frozen / 'manifest.json'), 'binding': file_info(run / 'binding.json'),
                'invocations': [], 'completed_passes': [], 'analysis_reference_join_performed': False}
        invocation = str(uuid.uuid4())
        state['invocations'].append({'invocation_id': invocation, 'controller_pid': os.getpid(), 'worker_pid': None, 'started_at': now()})
        state.update(status='launching', updated_at=now())
        atomic_json(state_path, state)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=plan['allocation']['uuid'], CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONHASHSEED='0')
        command = [sys.executable, str(Path(__file__).resolve()), '_worker', '--freeze', str(frozen),
                   '--run', str(run), '--invocation', invocation]
        with (run / 'gpu.log').open('ab', buffering=0) as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            print(json.dumps({'worker_pid': process.pid, 'run': str(run), 'allocation': plan['allocation']}), flush=True)
            code = process.wait()
        state = read(state_path)
        state.update(worker_exit_code=code, owned_worker_absent=not Path(f'/proc/{process.pid}').exists(),
                     worker_exit_checked_at=now())
        inventory = gpu_inventory()
        selected = next(d for d in inventory['devices'] if d['uuid'] == plan['allocation']['uuid'])
        owned_visible = any(len(r) > 1 and r[1].strip() == str(process.pid) for r in inventory['compute_processes'])
        require(not owned_visible and state['owned_worker_absent'], 'owned CUDA process remains after wait')
        atomic_json(run / f'resource-release-{invocation}.json', {'worker_pid': process.pid,
            'worker_exit_code': code, 'owned_worker_absent': True, 'selected_gpu_after_exit': selected,
            'inventory': inventory}, replace=False)
        if code == 0 and state['status'] == 'scoring_complete_releasing':
            state.update(status='complete', completed_at=now())
        elif code != 0:
            state.update(status='failed', failure='worker failed; no automatic retries')
        atomic_json(state_path, state)
        require(code == 0 and state['status'] in ['complete', 'paused'], 'worker failed; inspect gpu.log')
        return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['freeze', 'validate', 'run', '_worker', 'check', 'analyze'])
    p.add_argument('--freeze', type=Path, default=WORK / 'frozen-01')
    p.add_argument('--run', type=Path, default=WORK / 'run-01')
    p.add_argument('--output', type=Path, default=WORK / 'results-01')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--weights', action='store_true')
    p.add_argument('--invocation')
    args = p.parse_args()
    if args.action == 'freeze':
        result = create_freeze(args.freeze, args.gpu)
    elif args.action == 'validate':
        plan, requests = check_freeze(args.freeze, args.weights)
        result = {'status': 'pass', 'requests': len(requests), 'allocation': plan['allocation']}
    elif args.action == 'run':
        result = supervise(args.freeze, args.run, args.resume)
    elif args.action == '_worker':
        from diagnostics.cross_term_next_token_gpu_v1 import worker
        result = {'status': worker(args.freeze, args.run, args.invocation)}
    elif args.action == 'check':
        result = check_completed(args.freeze, args.run)
    else:
        result = analyze(args.freeze, args.run, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
