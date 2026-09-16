#!/usr/bin/env python3
"""Poll the original four GPUs every 15 minutes; launch the paused run once."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.general_model_evidence_evaluation import require, file_sha
from diagnostics.q01_module_package import WORK, load_frozen, read_json, atomic_json
from diagnostics.q01_module_poll_v2 import verify_amendment, validate_config, decision, command, epoch
from diagnostics.q01_module_execution import preflight


def now():
    return datetime.now(timezone.utc).isoformat()


def terminate_owned(process):
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def supervise(config_path):
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    validate_config(config)
    verify_amendment(config['amendment_sha256'])
    directory = config_path.parent
    require(directory.parent == WORK and directory.name.startswith('window-'), 'dedicated window directory required')
    state_path = directory / 'state.json'
    # One new polling supervisor per experiment, in addition to the executor writer lock.
    with (WORK / '.idle-resume-v2.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not state_path.exists(), 'this polling supervisor already has a recorded invocation')
        state = {'schema_version': 'q01-module-idle-state/v2', 'status': 'validating_cpu',
                 'config_sha256': file_sha(config_path), 'pid': os.getpid(), 'process_group': os.getpgrp(),
                 'started_at': now(), 'poll_interval_seconds': 900, 'checks': [],
                 'gpu_launch_attempted': False, 'notification_policy': 'quiet while busy; record start, completion or failure'}
        atomic_json(state_path, state)
        process = None
        try:
            require(file_sha(Path(config['plan']) / 'manifest.json') == config['plan_manifest_sha256'], 'plan changed')
            plan, _, _, _ = load_frozen(config['plan'])
            require(file_sha(config['cpu_receipt']) == config['cpu_receipt_sha256'], 'CPU checkpoint receipt changed')
            receipt = read_json(config['cpu_receipt'])
            require(receipt['status'] == 'passed' and receipt['plan_id'] == plan['plan_id'], 'CPU checkpoint not accepted')
            require(file_sha(Path(config['run']) / 'run_manifest.json') == config['paused_run_manifest_sha256'], 'paused run changed')
            require(read_json(Path(config['run']) / 'run_manifest.json')['status'] == 'paused', 'run is not resumable')
            next_check = epoch(config['first_check_at'])
            state.update(status='waiting', next_check_at=datetime.fromtimestamp(next_check, timezone.utc).isoformat())
            atomic_json(state_path, state)
            while True:
                if (directory / 'CANCEL').exists():
                    state['status'] = 'cancelled_before_launch'
                    break
                if time.time() < next_check:
                    time.sleep(min(30, next_check - time.time()))
                    continue
                action = decision(config, time.time())
                if action['action'] == 'window_closed':
                    state['status'] = 'window_closed_without_launch'
                    break
                verify_amendment(config['amendment_sha256'])
                require(file_sha(config_path) == state['config_sha256'], 'polling configuration changed in place')
                require(file_sha(Path(config['run']) / 'run_manifest.json') == config['paused_run_manifest_sha256'],
                        'another executor changed the run while polling; duplicate launch prohibited')
                inventory = None
                try:
                    inventory = preflight(config['device_indices'])
                except (ValueError, subprocess.SubprocessError, OSError) as error:
                    state['checks'].append({'checked_at': now(), 'available': False, 'reason': str(error)})
                else:
                    allocation = [{'physical_gpu_index': d['physical_gpu_index'], 'uuid': d['uuid']} for d in inventory['devices']]
                    require(allocation == config['allocation'], 'original physical GPU UUIDs changed')
                    state['checks'].append({'checked_at': now(), 'available': True, 'inventory': inventory})
                next_check = action['next_check_epoch']
                state.update(next_check_at=datetime.fromtimestamp(next_check, timezone.utc).isoformat())
                atomic_json(state_path, state)
                if inventory is None:
                    continue
                # The original executor repeats preflight and validates its complete source lease.
                state.update(status='launching', gpu_launch_attempted=True, gpu_launch_attempted_at=now())
                atomic_json(state_path, state)
                with (directory / 'gpu-run.log').open('ab') as log:
                    process = subprocess.Popen(command(config_path, config), cwd=ROOT, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    state.update(status='running', run_pid=process.pid, run_process_group=process.pid, next_check_at=None)
                    atomic_json(state_path, state)
                    while process.poll() is None:
                        release = epoch(config.get('release_deadline'))
                        if (release is not None and time.time() >= release) or (directory / 'CANCEL').exists():
                            terminate_owned(process)
                            state.update(status='stopped_owned_run', reason='release deadline or explicit cancellation')
                            break
                        time.sleep(5)
                state['run_exit_code'] = process.returncode
                run_state = read_json(Path(config['run']) / 'run_manifest.json')
                state.update(run_status=run_state.get('status'), completed_passes=run_state.get('completed_passes', []))
                if state['status'] == 'stopped_owned_run':
                    break
                require(process.returncode == 0 and run_state['status'] in ('paused', 'complete'), 'run failed; automatic retries disabled')
                require(all(i.get('all_workers_exited') and i.get('all_workers_normal_exit') for i in run_state['invocations']),
                        'normal worker release incomplete')
                action = 'analyze' if run_state['status'] == 'complete' else 'resume-check'
                state.update(status='verifying_cpu')
                atomic_json(state_path, state)
                args = [str(ROOT / '.conda/stage1-p0/bin/python'), str(ROOT / 'scripts/review/run_q01_module_refinement.py'),
                        action, '--plan', config['plan'], '--run', config['run']]
                if action == 'analyze':
                    args += ['--output', config['results']]
                with (directory / ('analysis.log' if action == 'analyze' else 'checkpoint-check.log')).open('ab') as log:
                    checked = subprocess.run(args, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                require(checked.returncode == 0, 'post-run CPU verification failed')
                state.update(status='complete' if action == 'analyze' else 'paused_at_deadline',
                             results=config['results'] if action == 'analyze' else None)
                break
        except BaseException as error:
            if process is not None and process.poll() is None:
                terminate_owned(process)
            state.update(status='failed', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            state['ended_at'] = now()
            atomic_json(state_path, state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(supervise(args.config), ensure_ascii=False))
