"""Separate Q01 executor: scoped hooks, checkpointed passes and fail-closed gates."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing
from multiprocessing.connection import wait
import os
from pathlib import Path
import subprocess
import time
import traceback
import uuid

from diagnostics.evidence_label_calibration import readouts, reference_rows, validate_block
from diagnostics.general_model_evidence_evaluation import require, file_sha, json_bytes, jsonl
from diagnostics.q01_mechanism_inputs import digest
from diagnostics.q01_mechanism_package import ROOT, WORK, load_frozen, read_json, read_lines, verify_sources, atomic_json
from diagnostics.q01_mechanism_scoring import compare_derived, ReadoutBuilder, nominate, MODES


def now(): return datetime.now(timezone.utc).isoformat()


def emit(**fields): print(json.dumps({'time': now(), **fields}, ensure_ascii=False), flush=True)


def preflight(indices):
    import csv
    require(2 <= len(indices) <= 4 and len(set(indices)) == len(indices) and
            all(type(i) is int and i >= 0 for i in indices), 'assign two to four distinct physical GPUs')
    output = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
                             '--format=csv,noheader,nounits', '--id', ','.join(map(str, indices))],
                            check=True, capture_output=True, text=True, timeout=30).stdout
    devices = {}
    for r in csv.reader(output.strip().splitlines(), skipinitialspace=True):
        require(len(r) == 6, 'ambiguous GPU inventory')
        index = int(r[0]); require(index not in devices, 'duplicate GPU inventory')
        devices[index] = {'physical_gpu_index': index, 'uuid': r[1].strip(), 'name': r[2].strip(),
                          'used_mib': int(r[3]), 'total_mib': int(r[4]), 'utilization_percent': int(r[5])}
    require(devices.keys() == set(indices), 'GPU inventory does not cover assignment')
    for d in devices.values():
        require(d['used_mib'] <= 1024 and d['total_mib'] - d['used_mib'] >= 36000 and d['utilization_percent'] <= 5,
                f"GPU {d['physical_gpu_index']} is not available")
    require(len({d['uuid'] for d in devices.values()}) == len(indices), 'duplicate physical GPU UUID')
    return {'checked_at': now(), 'devices': [devices[i] for i in indices]}


def validate_numeric_identity(plan, identity):
    require(identity['source_generation_runtime'] == plan['runtime_parent_plan']['generation_runtime_identity'], 'source generation identity differs')
    require(identity['numeric_runtime'] == plan['runtime'] and identity['transformer_dtype'] == 'torch.float32'
            and identity['lm_head_dtype'] == 'torch.float32' and identity['use_cache'] is False,
            'numeric runtime differs from frozen FP32 scorer')
    require(identity['tf32_matmul'] is False and identity['tf32_cudnn'] is False, 'TF32 is prohibited')


def check_geometry(row, request, context, plan, options):
    require(row['request_id'] == request['request_id'], 'wrong request result')
    require(row.get('query_reference_loaded') is False and row.get('formal_test_or_reserve_access') is False,
            'scoring result contains forbidden reference access')
    validate_block(row, context, plan['catalog'])
    require(row['scoring_options'] == options, 'score challenge options differ')
    for c in row['candidates']:
        require(c['prompt_tokens'] == context['prompt_tokens'] and
                c['prompt_token_ids_sha256'] == context['prompt_token_ids_sha256'] and
                c['eos_token_id'] == plan['eos_token_id'], 'candidate/source geometry differs')
        require(c['batch_size'] == 1 and c['use_cache'] is False and c['causal_shift'] == 1,
                'scorer must be uncached, shifted, batch-one')
        require(c['model_logits_dtype'] == 'torch.float32' and c['reference_checked'] == options['reference'], 'score precision/reference mode differs')
        if options['prefix']:
            require(c['scoring_implementation'] == 'uncached-prefix-only' and c['padded_sequence_tokens'] is None, 'prefix challenge not executed')
        else:
            expected = context['prompt_tokens'] + c['answer_tokens'] + 1 + options['padding_extra']
            require(c['padded_sequence_tokens'] == expected, 'padding geometry differs')
    hook = row['hook_receipt']
    if request['kind'] == 'baseline': require(hook is None, 'baseline has a replacement hook')
    elif request['kind'] == 'capture_only':
        require(hook['observation_counts'] and all(v > 0 for v in hook['observation_counts'].values())
                and hook['donor_cache_written_by_observer'] is False, 'capture-only hook was not exercised')
    else:
        require(hook['candidate_independent'] and hook['source_is_prompt_only'] and hook['donor_positions'] == request['positions'], 'donor/position contract differs')
        prefixes = {tuple(c['answer_token_ids'][:i]) for c in plan['catalog'][context['task']]
                    for i in range(c['answer_tokens'] + 1)}
        require(hook['hook_calls'] == (len(prefixes) if options['prefix'] else 2), 'candidate branches did not share one hook/cache')


def numeric_difference(a, b):
    left, right = readouts(a), readouts(b)
    require(left.keys() == right.keys(), 'numeric readout coverage differs')
    maximum = max(abs(left[k] - right[k]) for k in left)
    return maximum


def accept_pass(plan, contexts, requests, rows, spec, history, reference=None, engineering_baseline=None):
    by_context = {c['record_id']: c for c in contexts}
    by_request = {r['request_id']: r for r in requests}
    require([r['request_id'] for r in rows] == spec['request_ids'], 'pass coverage/order differs')
    for row in rows:
        r = by_request[row['request_id']]
        check_geometry(row, r, by_context[r['recipient']], plan, spec['options'])
    old = {r['record_id']: r for r in history}
    baselines = {by_request[r['request_id']]['recipient']: r for r in rows if by_request[r['request_id']]['kind'] == 'baseline'}
    require(baselines.keys() == old.keys() == by_context.keys(), '96-source regression is incomplete')
    eps = plan['numeric_policy']['epsilon']
    gates = [{'name': 'historical_replay', 'max_error': max(numeric_difference(old[cid], row) for cid, row in baselines.items()), 'limit': eps}]
    if engineering_baseline is not None:
        gates.append({'name': 'phase_bridge', 'max_error': max(numeric_difference(engineering_baseline[cid], row) for cid, row in baselines.items()), 'limit': eps})
    if spec['mode'] == 'reference':
        ref = reference_rows(rows)
        gates.append({'name': 'same_logits_fp64', 'max_error': max(numeric_difference(a, b) for a, b in zip(ref, rows, strict=True)),
                      'limit': plan['numeric_policy']['reference_abs_tolerance']})
    else:
        require(reference is not None and [r['request_id'] for r in reference] == spec['request_ids'], 'phase reference pass missing')
        ref = reference
        gates.append({'name': spec['mode'], 'max_error': max(numeric_difference(a, b) for a, b in zip(ref, rows, strict=True)),
                      'limit': plan['numeric_policy']['repeat_abs_tolerance'] if spec['mode'] == 'repeat' else eps})
    if spec['phase'] == 'engineering':
        controls = [r for r in rows if by_request[r['request_id']]['expected_baseline']]
        gates.append({'name': 'registered_engineering_expectations',
                      'max_error': max(numeric_difference(baselines[by_request[r['request_id']]['expected_baseline']], r) for r in controls),
                      'limit': eps, 'requests': len(controls)})
    if spec['mode'] == 'replica':
        require(all(a['physical_gpu_uuid'] != b['physical_gpu_uuid'] and a['physical_gpu_index'] != b['physical_gpu_index']
                    for a, b in zip(reference, rows, strict=True)), 'replica pass did not move every physical producer')
        gates.append({'name': 'physical_producer_rotation', 'max_error': 0.0, 'limit': 0.0, 'requests': len(rows)})
    for g in gates: g['passed'] = g['max_error'] <= g['limit']
    require(all(g['passed'] for g in gates), 'numeric gate failed: ' + json.dumps(gates))
    derived = None
    if spec['phase'] == 'science':
        derived = compare_derived(contexts, requests, ref, rows,
                    plan['numeric_policy']['reference_abs_tolerance'] if spec['mode'] == 'reference' else
                    plan['numeric_policy']['repeat_abs_tolerance'] if spec['mode'] == 'repeat' else eps)
        require(derived['passed'], 'propagated C/I/NCC numerical gate failed')
    return {'status': 'passed', 'pass_id': spec['pass_id'], 'gates': gates, 'derived': derived,
            'rows': len(rows), 'candidate_evaluations': len(rows) * 2}


def _worker(connection, plan, contexts, positions, physical_index, expected_gpu, root):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(physical_index)
    try:
        from diagnostics.general_model_numeric_kernel_v2 import NumericRunner
        from diagnostics.evidence_label_calibration_execution import hardware_identity
        from diagnostics.q01_mechanism_hooks import HookRuntime
        verify_sources(plan['code_sha256'])
        runner = NumericRunner(plan['runtime_parent_plan'], plan['runtime'], 'cuda:0', Path(root))
        hardware = hardware_identity(runner.torch, physical_index)
        require(hardware['uuid'] == expected_gpu['uuid'], 'physical GPU differs after preflight')
        validate_numeric_identity(plan, runner.identity)
        connection.send({'kind': 'ready', 'physical_gpu_index': physical_index, 'identity': runner.identity, 'hardware': hardware})
        while True:
            job = connection.recv()
            if job['kind'] == 'stop': break
            require(job['kind'] == 'pass', 'unknown worker operation')
            verify_sources(plan['code_sha256'])
            directory = Path(job['directory']); directory.mkdir(parents=True, exist_ok=True)
            rows_path = directory / 'scores.jsonl'
            existing = read_lines(rows_path) if rows_path.exists() else []
            expected = [r['request_id'] for r in job['requests']]
            require([r['request_id'] for r in existing] == expected[:len(existing)], 'checkpoint is not a valid request prefix')
            for r, row in zip(job['requests'], existing):
                check_geometry(row, r, next(c for c in contexts if c['record_id'] == r['recipient']), plan, job['options'])
                require(row['physical_gpu_uuid'] == hardware['uuid'] and row['runtime_sha256'] == digest(runner.identity), 'checkpoint producer/runtime differs')
            hooks = HookRuntime(runner, contexts, positions, plan['plan_id'])
            invocation = str(uuid.uuid4())
            completed = len(existing)
            with rows_path.open('ab') as f:
                for request in job['requests'][completed:]:
                    if job['stop_epoch'] is not None and time.time() >= job['stop_epoch']: break
                    row = hooks.score(request, plan['catalog'], job['options'])
                    row.update(physical_gpu_index=physical_index, physical_gpu_uuid=hardware['uuid'],
                               runtime_sha256=digest(runner.identity), capture_invocation=invocation)
                    f.write(jsonl([row])); f.flush(); os.fsync(f.fileno())
                    completed += 1
                    if completed % 25 == 0:
                        connection.send({'kind': 'progress', 'physical_gpu_index': physical_index,
                                         'completed': completed, 'total': len(job['requests'])})
            atomic_json(directory / ('captures-' + invocation + '.json'), {
                'physical_gpu_index': physical_index, 'physical_gpu_uuid': hardware['uuid'],
                'runtime_sha256': digest(runner.identity), 'capture_invocation': invocation,
                'prompt_only_forward_calls': hooks.forward_capture_count, 'receipts': hooks.capture_receipts})
            connection.send({'kind': 'complete', 'physical_gpu_index': physical_index, 'completed': completed,
                             'total': len(job['requests']), 'scores_sha256': file_sha(rows_path)})
    except BaseException as error:
        try: connection.send({'kind': 'error', 'physical_gpu_index': physical_index, 'error': str(error), 'traceback': traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError): pass
        raise
    finally:
        connection.close()


class HookPool:
    def __init__(self, plan, contexts, positions, inventory, ownership_path=None):
        self.workers = []
        self.identities = []
        self.contexts = contexts
        self.closed = False
        ctx = multiprocessing.get_context('spawn')
        try:
            for d in inventory['devices']:
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_worker, args=(child, plan, contexts, positions,
                    d['physical_gpu_index'], d, str(ROOT)), name=f"q01-hook-{d['physical_gpu_index']}")
                process.start(); child.close()
                self.workers.append({'connection': parent, 'process': process, 'index': d['physical_gpu_index']})
                if ownership_path:
                    atomic_json(ownership_path, {'status': 'loading', 'workers': [
                        {'physical_gpu_index': w['index'], 'pid': w['process'].pid} for w in self.workers]})
            self.identities = sorted(self.collect('ready', timeout=900), key=lambda r: r['physical_gpu_index'])
            require(all(r['identity'] == self.identities[0]['identity'] for r in self.identities), 'numeric identities differ across workers')
        except BaseException:
            release = self.close(terminate=True)
            if ownership_path: atomic_json(ownership_path, {'status': 'startup_failed', 'release': release,
                                                          'all_workers_exited': all(w['exited'] for w in release)})
            raise

    def collect(self, kind, timeout=None):
        pending = {w['connection']: w for w in self.workers}; result = []; started = time.monotonic()
        while pending:
            if timeout is not None: require(time.monotonic() - started < timeout, 'GPU startup timeout')
            for connection in wait(list(pending), timeout=1):
                row = connection.recv()
                require(row['physical_gpu_index'] == pending[connection]['index'], 'worker producer differs')
                if row['kind'] == 'progress': emit(event='progress', **{k: v for k, v in row.items() if k != 'kind'}); continue
                require(row['kind'] == kind, 'worker failed: ' + json.dumps(row))
                result.append(row); del pending[connection]
            require(all(w['process'].is_alive() for w in pending.values()), 'worker exited before returning its result')
        return result

    def run_pass(self, plan, requests, spec, directory, stop_epoch):
        by_id = {r['request_id']: r for r in requests}
        selected = [by_id[rid] for rid in spec['request_ids']]
        assignments = [[] for _ in self.workers]
        for ordinal, r in enumerate(selected): assignments[(ordinal + spec['options']['replica_shift']) % len(self.workers)].append(r)
        for w, shard in zip(self.workers, assignments, strict=True):
            w['connection'].send({'kind': 'pass', 'requests': shard, 'options': spec['options'],
                                   'directory': str(directory / 'shards' / str(w['index'])), 'stop_epoch': stop_epoch})
        completed = self.collect('complete')
        rows, artifacts = {}, {}
        for w, assignment in zip(self.workers, assignments, strict=True):
            path = directory / 'shards' / str(w['index']) / 'scores.jsonl'
            receipt = next(c for c in completed if c['physical_gpu_index'] == w['index'])
            require(file_sha(path) == receipt['scores_sha256'], 'worker shard digest differs')
            shard_rows = read_lines(path)
            require([r['request_id'] for r in shard_rows] == [r['request_id'] for r in assignment[:len(shard_rows)]], 'shard prefix differs')
            for row in shard_rows:
                require(row['request_id'] not in rows and row['physical_gpu_index'] == w['index'], 'duplicate/wrong-producer score')
                rows[row['request_id']] = row
            for p in path.parent.iterdir():
                if p.is_file(): artifacts[str(p.relative_to(directory))] = file_sha(p)
        return [rows[r['request_id']] for r in selected if r['request_id'] in rows], artifacts, all(c['completed'] == c['total'] for c in completed)

    def close(self, terminate=False):
        if self.closed: return []
        self.closed = True
        for w in self.workers:
            if terminate and w['process'].is_alive(): w['process'].terminate()
            elif w['process'].is_alive():
                try: w['connection'].send({'kind': 'stop'})
                except (BrokenPipeError, EOFError, OSError): pass
        deadline = time.monotonic() + 30
        for w in self.workers: w['process'].join(max(0, deadline - time.monotonic()))
        for w in self.workers:
            if w['process'].is_alive(): w['process'].terminate(); w['process'].join(5)
            if w['process'].is_alive(): w['process'].kill(); w['process'].join(5)
            w['connection'].close()
        return [{'physical_gpu_index': w['index'], 'pid': w['process'].pid,
                 'exited': not w['process'].is_alive(), 'exitcode': w['process'].exitcode} for w in self.workers]


def execute(directory, output, indices, *, stop_epoch=None, through_pass=None):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    plan, contexts, positions, requests = load_frozen(directory)
    require(output.parent == WORK and output.name.startswith('run-') and not output.is_relative_to(directory), 'run output must be a new dedicated run directory')
    require(through_pass is None or through_pass in {p['pass_id'] for p in plan['schedule']}, 'unknown stopping pass')
    if stop_epoch is not None: require(stop_epoch > time.time(), 'requested checkpoint deadline has passed')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.writer.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = output / 'run_manifest.json'
        if state_path.exists():
            state = read_json(state_path)
            require(state['status'] == 'paused' and state['plan_id'] == plan['plan_id'], 'only an explicitly paused run may resume; terminal runs are sealed')
            require(state['device_indices'] == indices, 'resume requires the same physical allocation')
            verify_completed_passes(plan, contexts, requests, directory, output, state, require_complete=False)
        else:
            require({p.name for p in output.iterdir()} == {'.writer.lock'}, 'nonempty run directory has no manifest')
            state = {'schema_version': 'q01-mechanism-run/v1', 'status': 'prepared', 'plan_id': plan['plan_id'],
                'plan_manifest_sha256': file_sha(directory / 'manifest.json'), 'device_indices': indices,
                'started_at': now(), 'completed_passes': [], 'invocations': [],
                'query_reference_loaded_during_scoring': False, 'formal_test_or_reserve_access': False, 'mechanism_ready': False}
        invocation = {'id': str(uuid.uuid4()), 'started_at': now(), 'stop_epoch': stop_epoch, 'through_pass': through_pass}
        state['invocations'].append(invocation)
        state.update(status='running', active_pass=None)
        atomic_json(state_path, state)
        pool = None
        try:
            inventory = preflight(indices)
            allocation = [{'physical_gpu_index': d['physical_gpu_index'], 'uuid': d['uuid']} for d in inventory['devices']]
            require('allocation' not in state or state['allocation'] == allocation, 'physical UUID changed on resume')
            state['allocation'] = allocation
            atomic_json(output / ('inventory-' + invocation['id'] + '.json'), inventory)
            ownership_path = output / ('ownership-' + invocation['id'] + '.json')
            pool = HookPool(plan, contexts, positions, inventory, ownership_path)
            require('runtime_identities' not in state or state['runtime_identities'] == pool.identities, 'runtime changed on resume')
            state['runtime_identities'] = pool.identities
            invocation['workers'] = [{'physical_gpu_index': w['index'], 'pid': w['process'].pid} for w in pool.workers]
            atomic_json(state_path, state)
            history = read_lines(directory / 'historical-scores.jsonl')
            phase_refs = {}
            engineering_baseline = None
            by_request = {r['request_id']: r for r in requests}
            for spec in plan['schedule']:
                name = spec['pass_id']
                if name in state['completed_passes']:
                    if spec['mode'] == 'reference':
                        phase_refs[spec['phase']] = read_lines(output / name / 'scores.jsonl')
                        if spec['phase'] == 'engineering':
                            engineering_baseline = {by_request[r['request_id']]['recipient']: r for r in phase_refs['engineering']
                                                    if by_request[r['request_id']]['kind'] == 'baseline'}
                    continue
                if stop_epoch is not None and time.time() >= stop_epoch:
                    state['status'] = 'paused'; break
                state.update(active_pass=name); atomic_json(state_path, state)
                emit(event='pass_start', pass_id=name, requests=len(spec['request_ids']))
                target = output / name; target.mkdir(exist_ok=True)
                rows, artifacts, complete = pool.run_pass(plan, requests, spec, target, stop_epoch)
                if not complete:
                    state.update(status='paused', checkpointed_requests=len(rows)); break
                verify_capture_audit(plan, contexts, requests, rows, target, {'artifacts': artifacts}, state, spec)
                acceptance = accept_pass(plan, contexts, requests, rows, spec, history,
                                         phase_refs.get(spec['phase']), engineering_baseline if spec['phase'] == 'science' else None)
                with (target / 'scores.jsonl').open('xb') as f: f.write(jsonl(rows))
                atomic_json(target / 'acceptance.json', acceptance)
                artifacts.update({'scores.jsonl': file_sha(target / 'scores.jsonl'), 'acceptance.json': file_sha(target / 'acceptance.json')})
                atomic_json(target / 'manifest.json', {'status': 'complete', 'plan_id': plan['plan_id'],
                    'pass_spec_sha256': digest(spec), 'artifacts': artifacts, 'completed_at': now()})
                state['completed_passes'].append(name)
                if spec['mode'] == 'reference':
                    phase_refs[spec['phase']] = rows
                    if spec['phase'] == 'engineering':
                        engineering_baseline = {by_request[r['request_id']]['recipient']: r for r in rows if by_request[r['request_id']]['kind'] == 'baseline'}
                atomic_json(state_path, state); emit(event='pass_accepted', pass_id=name)
                if name == through_pass and len(state['completed_passes']) < len(plan['schedule']):
                    state['status'] = 'paused'; break
            else:
                load_frozen(directory)
                state.update(status='complete', numerical_validation_passed=True,
                             raw_scores_sha256=file_sha(output / 'science-reference/scores.jsonl'), completed_at=now())
            state.update(active_pass=None)
        except BaseException as error:
            state.update(status='failed', error_type=type(error).__name__, error=str(error), stopped_at=now())
            raise
        finally:
            if pool is not None:
                release = pool.close(terminate=state['status'] == 'failed')
                invocation['release'] = release
                invocation['all_workers_exited'] = all(r['exited'] for r in release)
                invocation['all_workers_normal_exit'] = all(r['exited'] and r['exitcode'] == 0 for r in release)
                atomic_json(output / ('ownership-' + invocation['id'] + '.json'), {
                    'status': 'released', 'release': release, 'all_workers_exited': invocation['all_workers_exited']})
                if not invocation['all_workers_exited']:
                    state.update(status='failed', error='one or more owned workers did not exit')
                elif state['status'] in ('complete', 'paused') and not invocation['all_workers_normal_exit']:
                    state.update(status='failed', error='owned worker required forced termination or had a nonzero exit')
            invocation['ended_at'] = now()
            atomic_json(state_path, state)
    emit(event='run_exit', status=state['status'], completed_passes=len(state['completed_passes']))
    return state


def verify_completed_passes(plan, contexts, requests, directory, output, state, *, require_complete=True):
    expected = [p['pass_id'] for p in plan['schedule']]
    require(state['plan_id'] == plan['plan_id'] and state['plan_manifest_sha256'] == file_sha(directory / 'manifest.json'), 'run plan binding differs')
    require(state['completed_passes'] == expected[:len(state['completed_passes'])], 'completed passes do not form a schedule prefix')
    if require_complete:
        require(state['status'] == 'complete' and state['numerical_validation_passed'] and state['completed_passes'] == expected,
                'analysis/check requires all 12 sealed passes')
        require(state['raw_scores_sha256'] == file_sha(output / 'science-reference/scores.jsonl'), 'raw score seal changed')
    require(state['query_reference_loaded_during_scoring'] is False and state['formal_test_or_reserve_access'] is False, 'scoring/reference separation changed')
    if state['completed_passes']:
        require(len(state['runtime_identities']) == len(state['device_indices']) >= 2, 'runtime producer frame differs')
        for runtime in state['runtime_identities']: validate_numeric_identity(plan, runtime['identity'])
    history = read_lines(directory / 'historical-scores.jsonl')
    references, engineering_baseline = {}, None
    by_request = {r['request_id']: r for r in requests}
    for spec in plan['schedule'][:len(state['completed_passes'])]:
        target = output / spec['pass_id']; manifest = read_json(target / 'manifest.json')
        require(manifest['status'] == 'complete' and manifest['plan_id'] == plan['plan_id'] and manifest['pass_spec_sha256'] == digest(spec), 'pass identity differs')
        for name, h in manifest['artifacts'].items(): require(file_sha(target / name) == h, 'sealed pass artifact changed: ' + name)
        rows = read_lines(target / 'scores.jsonl')
        verify_capture_audit(plan, contexts, requests, rows, target, manifest, state, spec)
        for ordinal, row in enumerate(rows):
            index = state['device_indices'][(ordinal + spec['options']['replica_shift']) % len(state['device_indices'])]
            producer = next(p for p in state['runtime_identities'] if p['physical_gpu_index'] == index)
            require(row['physical_gpu_index'] == index and row['physical_gpu_uuid'] == producer['hardware']['uuid']
                    and row['runtime_sha256'] == digest(producer['identity']), 'sealed producer assignment changed')
        acceptance = accept_pass(plan, contexts, requests, rows, spec, history, references.get(spec['phase']),
                                 engineering_baseline if spec['phase'] == 'science' else None)
        require(read_json(target / 'acceptance.json') == json.loads(json.dumps(acceptance, allow_nan=False)), 'gate reconstruction differs')
        if spec['mode'] == 'reference':
            references[spec['phase']] = rows
            if spec['phase'] == 'engineering':
                engineering_baseline = {by_request[r['request_id']]['recipient']: r for r in rows if by_request[r['request_id']]['kind'] == 'baseline'}
    if require_complete:
        require(all(i.get('all_workers_exited', False) and i.get('all_workers_normal_exit', False)
                    for i in state['invocations']), 'normal worker release incomplete')
    return references


def check_run(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    plan, contexts, _, requests = load_frozen(directory)
    return verify_completed_passes(plan, contexts, requests, directory, output, read_json(output / 'run_manifest.json'))


def verify_capture_audit(plan, contexts, requests, rows, target, manifest, state, spec, *, hidden_size=4096):
    """Reconnect each replaced vector to its candidate-free capture and physical shard."""
    by_context = {c['record_id']: c for c in contexts}
    by_request = {r['request_id']: r for r in requests}
    captures, shard_rows = {}, {}
    for name in manifest['artifacts']:
        parts = Path(name).parts
        if len(parts) != 3 or parts[0] != 'shards': continue
        index = int(parts[1])
        if parts[2] == 'scores.jsonl':
            for row in read_lines(target / name):
                require(row['request_id'] not in shard_rows and row['physical_gpu_index'] == index, 'shard coverage/producer differs')
                shard_rows[row['request_id']] = row
        elif parts[2].startswith('captures-'):
            capture = read_json(target / name)
            require(capture['physical_gpu_index'] == index and
                    capture['prompt_only_forward_calls'] == len(capture['receipts']), 'capture forward count differs')
            sources = {}
            for receipt in capture['receipts']:
                cid = receipt['source_record_id']; c = by_context[cid]
                require(cid not in sources and receipt['source_is_prompt_only'] and receipt['forward_calls'] == 1
                        and receipt['prompt_tokens'] == c['prompt_tokens'], 'capture is not one prompt-only source')
                states = {}
                for item in receipt['states']:
                    ident = item['identity']
                    require(ident['plan_id'] == plan['plan_id'] and ident['source_record_id'] == cid
                            and ident['source_prompt_sha256'] == c['prompt_sha256']
                            and ident['source_context_sha256'] == c['context_sha256']
                            and ident['source_prompt_token_ids_sha256'] == c['prompt_token_ids_sha256']
                            and ident['source_prompt_tokens'] == c['prompt_tokens']
                            and ident['runtime_sha256'] == capture['runtime_sha256']
                            and ident['candidate_independent'] and ident['source_is_prompt_only'], 'capture source identity differs')
                    require(item['shape'] == [len(ident['positions']), hidden_size]
                            and all(0 <= p < c['prompt_tokens'] for p in ident['positions']), 'capture extends beyond prompt')
                    states[digest(ident)] = item
                sources[cid] = states
            key = (index, capture['capture_invocation'])
            require(key not in captures, 'duplicate capture invocation')
            captures[key] = {**capture, 'sources': sources}
    require(shard_rows.keys() == {r['request_id'] for r in rows}, 'merged/shard frame differs')
    for row in rows:
        require(shard_rows[row['request_id']] == row, 'merged row differs from shard bytes')
        r = by_request[row['request_id']]
        if r['donor'] is None: continue
        capture = captures[row['physical_gpu_index'], row['capture_invocation']]
        require(capture['physical_gpu_uuid'] == row['physical_gpu_uuid'] and capture['runtime_sha256'] == row['runtime_sha256'], 'donor producer differs from scorer')
        hook = row['hook_receipt']
        item = capture['sources'][r['donor']][hook['donor_identity_sha256']]
        require(item['vector_sha256'] == hook['donor_vector_sha256'], 'scored donor differs from captured vector')
        ident = item['identity']
        require(all(ident[k] == r[k] for k in ('module', 'layer', 'role', 'encoding', 'probe_id'))
                and set(r['positions']) <= set(ident['positions']), 'capture/patch role differs')


def analyze(directory, output, destination):
    directory, output, destination = (Path(p).resolve() for p in (directory, output, destination))
    plan, contexts, _, requests = load_frozen(directory)
    references = check_run(directory, output)
    # First parse of the analysis reference in this executable workflow occurs only here.
    reference = read_json(directory / 'analysis-reference.json')
    require(reference['query_id'] == 'FD-3169-Q01' and reference['original_label'] is None, 'Q01 must not inherit dataset Gold')
    rows = references['science']
    builder = ReadoutBuilder(contexts, requests, rows, plan['numeric_policy']['epsilon'])
    comparisons = builder.comparisons()
    selected = nominate(comparisons)
    from diagnostics.general_model_evidence_evaluation import write_output
    margins = []
    by_request = {r['request_id']: r for r in requests}
    for row in rows:
        values = readouts(row)
        margins.append({'request_id': row['request_id'], **by_request[row['request_id']],
            'margins': {m: values['margin/' + m] for m in MODES},
            'predictions': {m: ('non-hate' if values['margin/' + m] > 0 else 'hate') for m in MODES},
            'numerically_resolved': {m: abs(values['margin/' + m]) > plan['numeric_policy']['epsilon'] for m in MODES},
            'hook_receipt': row['hook_receipt']})
    files = {'comparisons.jsonl': jsonl(comparisons), 'margins.jsonl': jsonl(margins),
             'atoms.json': json_bytes({'values': builder.atoms, 'source_terms': builder.atom_sources}),
             'nomination.json': json_bytes(selected), 'analysis-reference.json': json_bytes(reference),
             'summary.json': json_bytes({'status': 'complete', 'plan_id': plan['plan_id'],
                 'raw_scores_sha256': file_sha(output / 'science-reference/scores.jsonl'),
                 'analysis_reference_loaded_after_all_gates': True, 'comparison_rows': len(comparisons),
                 'independent_queries': 1, 'new_original_gold': False, 'mechanism_ready': False})}
    from scripts.review.prepare_q01_local_mechanism import csv_data
    compact = []
    for r in comparisons:
        for metric, m in r['metrics'].items():
            compact.append({k: r[k] for k in ('group', 'layer', 'role', 'category', 'view', 'direction')} |
                           {'metric': metric, 'target': m['target']['value'], 'effect': m['effect']['value'],
                            'effect_bound': m['effect']['numeric_bound'], 'residual': m['residual']['value'],
                            'residual_bound': m['residual']['numeric_bound'], 'ratio': m['effect_over_target'],
                            'closeness_gain': m['closeness_gain'], 'eligible_direction': m['eligible_direction']})
    files['comparison-summary.csv'] = csv_data(compact)
    text = ['# Q01 局部机制结果', '', '全部 12 遍封存和数值验收已通过；这里仍只涉及一个已暴露查询。', '',
            '| 目标 | 提名单元 | 四组双向最小 closeness gain |', '|---|---|---:|']
    for metric in ('C', 'I'):
        winner = selected['rankings'][metric]['winner']
        text.append(f"| {metric} | block {winner['layer']} / {winner['role']} | {winner['worst_closeness_gain']:.6f} |" if winner else f'| {metric} | 无合格单元 | — |')
    text.extend(['', 'A/B、NCC、探针和控制结果均保留在 comparison-summary.csv 与 comparisons.jsonl；辅助口径异常见 nomination.json。',
                 '提名依据原总分的数值可分辨双向恢复；没有按分类翻转选择，也没有自动替换赢家或启动模块细化。',
                 'C 与 I 分别解释。机制 readiness 仍为 false，不能从单例局部干预推出独立词形原因或完整证据使用机制。', ''])
    files['RESULTS.md'] = '\n'.join(text).encode()
    files['manifest.json'] = json_bytes({'status': 'complete', 'plan_id': plan['plan_id'],
        'run_manifest_sha256': file_sha(output / 'run_manifest.json'),
        'artifacts': {name: __import__('hashlib').sha256(data).hexdigest() for name, data in files.items()}})
    if destination.exists():
        require({p.name for p in destination.iterdir()} == set(files), 'existing analysis inventory differs')
        for name, data in files.items(): require((destination / name).read_bytes() == data, 'analysis reconstruction differs: ' + name)
    else: write_output(destination, files)
    return {'comparison_rows': len(comparisons), 'selected_units': selected['selected_units']}
