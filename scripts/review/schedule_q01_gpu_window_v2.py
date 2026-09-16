#!/usr/bin/env python3
"""One authorized server window: wait, check four GPUs, run once, checkpoint, stop."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.general_model_evidence_evaluation import require, file_sha
from diagnostics.q01_mechanism_package import load_frozen, read_json, atomic_json, WORK
from diagnostics.q01_mechanism_execution_v2 import preflight
from diagnostics.q01_mechanism_resume_v2 import verify_amendment


def epoch(value):
    dt = datetime.fromisoformat(value)
    require(dt.tzinfo is not None, 'window time must include timezone')
    return dt.timestamp()


def decision(config, timestamp):
    start, stop = epoch(config['first_check_at']), epoch(config['checkpoint_at'])
    if timestamp < start: return {'action': 'wait', 'next_check_epoch': start}
    if timestamp >= stop: return {'action': 'window_closed', 'next_check_epoch': None}
    interval = config['poll_interval_seconds']
    require(interval == 1800, 'user authorized half-hour checks')
    return {'action': 'check', 'next_check_epoch': min(start + (int((timestamp - start) // interval) + 1) * interval, stop)}


def command(config):
    require(config['device_indices'] == [0, 1, 2, 3], 'this window is for all four GPUs')
    return [str(ROOT / '.conda/stage1-p0/bin/python'), str(ROOT / 'scripts/review/run_q01_local_mechanism_v2.py'),
            'run', '--plan', config['plan'], '--run', config['run'], '--gpus', '0', '1', '2', '3',
            '--stop-at', config['checkpoint_at']]


def supervise(config_path):
    config_path = Path(config_path).resolve(); config = read_json(config_path)
    verify_amendment()
    require(epoch(config['release_deadline']) + 55 < epoch(config['user_window_end']), 'release fallback must finish before user window ends')
    directory = config_path.parent
    require(directory.parent == WORK and directory.name.startswith('window-'), 'scheduler must use its dedicated experiment directory')
    require(epoch(config['first_check_at']) < epoch(config['checkpoint_at']) < epoch(config['release_deadline']), 'invalid window ordering')
    state_path = directory / 'state.json'
    with (directory / '.supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not state_path.exists(), 'window supervisor already has a recorded invocation')
        state = {'schema_version': 'q01-gpu-window-state/v1', 'status': 'waiting',
                 'config_sha256': file_sha(config_path), 'pid': os.getpid(),
                 'process_group': os.getpgrp(), 'started_at': datetime.now(timezone.utc).isoformat(),
                 'model_forward_started': False, 'checks': [], 'poll_interval_seconds': 1800}
        atomic_json(state_path, state)
        try:
            require(file_sha(Path(config['plan']) / 'manifest.json') == config['plan_manifest_sha256'], 'scheduled plan changed')
            plan, _, _, _ = load_frozen(config['plan'])
            require(file_sha(config['cpu_receipt']) == config['cpu_receipt_sha256'], 'CPU acceptance receipt changed')
            receipt = read_json(config['cpu_receipt'])
            require(receipt['status'] == 'passed' and receipt['plan_id'] == plan['plan_id'], 'CPU preparation not accepted')
            next_check = epoch(config['first_check_at'])
            while time.time() < epoch(config['checkpoint_at']):
                if (directory / 'CANCEL').exists():
                    state['status'] = 'cancelled'; break
                if time.time() < next_check:
                    time.sleep(min(60, next_check - time.time())); continue
                action = decision(config, time.time())
                if action['action'] == 'window_closed': break
                inventory = None
                try: inventory = preflight(config['device_indices'])
                except (ValueError, subprocess.SubprocessError, OSError) as error:
                    state['checks'].append({'checked_at': datetime.now(timezone.utc).isoformat(), 'available': False, 'reason': str(error)})
                else:
                    state['checks'].append({'checked_at': datetime.now(timezone.utc).isoformat(), 'available': True, 'inventory': inventory})
                next_check = action['next_check_epoch']
                state['next_check_at'] = datetime.fromtimestamp(next_check, timezone.utc).isoformat()
                atomic_json(state_path, state)
                if inventory is None: continue
                state.update(status='running', gpu_launch_attempted_at=datetime.now(timezone.utc).isoformat())
                atomic_json(state_path, state)
                with (directory / 'gpu-run.log').open('ab') as log:
                    process = subprocess.Popen(command(config), cwd=ROOT, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    state.update(run_pid=process.pid, run_process_group=process.pid)
                    atomic_json(state_path, state)
                    while process.poll() is None:
                        if time.time() >= epoch(config['release_deadline']) or (directory / 'CANCEL').exists():
                            # Only this newly created process group, including its owned model workers.
                            process.send_signal(signal.SIGTERM)
                            try: process.wait(timeout=45)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=10)
                            state.update(status='failed', error='release deadline or explicit cancellation; owned run stopped')
                            break
                        time.sleep(5)
                state['run_exit_code'] = process.returncode
                run_path = Path(config['run']) / 'run_manifest.json'
                run_state = read_json(run_path) if run_path.exists() else {}
                state['run_status'] = run_state.get('status')
                state['gpu_runtime_initialized'] = bool(run_state.get('runtime_identities'))
                state['scored_requests_observed'] = bool(run_state.get('completed_passes') or run_state.get('checkpointed_requests'))
                state.pop('model_forward_started', None)
                if state['status'] == 'failed': break
                require(process.returncode == 0 and run_state.get('status') in ('complete', 'paused'), 'GPU run failed; automatic retries disabled')
                require(all(i.get('all_workers_exited') for i in run_state['invocations']), 'owned worker release is incomplete')
                if run_state['status'] == 'paused':
                    state.update(status='paused_at_window_end', completed_passes=run_state['completed_passes'])
                    break
                state.update(status='verifying_cpu', completed_passes=run_state['completed_passes'])
                atomic_json(state_path, state)
                args = command(config)
                # analyze rechecks all sealed passes before parsing the reference; it performs no GPU forward.
                cpu = args[:2] + ['analyze', '--plan', config['plan'], '--run', config['run'], '--output', config['results']]
                with (directory / 'analysis.log').open('ab') as log:
                    checked = subprocess.run(cpu, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                require(checked.returncode == 0, 'sealed CPU analysis failed')
                state.update(status='complete', results=config['results'])
                break
            else:
                state['status'] = 'window_closed_without_launch'
            if state['status'] == 'waiting': state['status'] = 'window_closed_without_launch'
        except BaseException as error:
            state.update(status='failed', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            state['ended_at'] = datetime.now(timezone.utc).isoformat()
            atomic_json(state_path, state)
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    args = p.parse_args(); print(json.dumps(supervise(args.config), ensure_ascii=False))


if __name__ == '__main__': main()
