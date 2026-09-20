"""Explicitly bound future GPU lifecycle; all check/analysis entry points are CPU."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import numpy as np

from diagnostics import case_content_replacement_inputs_v1 as c
from diagnostics import case_attention_capture_v1 as capture
from diagnostics.cross_model_applicability_execution_v1 import readout, gpu_inventory, runtime_versions


def idle(allocation, inventory):
    devices = {d['uuid']: d for d in inventory['devices']}
    for a in allocation:
        c.require(a['uuid'] in devices, 'Bound device absent')
        d = devices[a['uuid']]
        c.require(all(d[k] == v for k, v in a.items()) and d['used_mib'] == 0 and d['utilization'] == 0
                  and not any(a['uuid'] in r for r in inventory['compute_processes']), 'Bound GPU changed or occupied')


def bind(prepared, gpu, output, authorization_note):
    c.require(authorization_note.strip(), 'Record the future explicit GPU window authorization')
    plan, profile, _ = c.validate(prepared, weights=True)
    inv = gpu_inventory()
    device = next((d for d in inv['devices'] if d['index'] == gpu), None)
    c.require(device is not None and device['total_mib'] >= 44000, 'Need one device with at least 44,000 MiB for pinned FP32 layout')
    allocation = [{k: device[k] for k in ('index', 'uuid', 'name', 'total_mib')}]
    idle(allocation, inv)
    c.atomic(output, {'prepared_manifest': c.info(Path(prepared) / 'manifest.json'), 'model_key': profile['model_key'],
                      'allocation': allocation, 'runtime_versions': plan['runtime_versions'], 'authorization_note': authorization_note,
                      'inventory': inv, 'created_at_unix': time.time()})
    return c.info(output)


def save_array(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The receipt is the commit marker. A crash with an uncommitted file is a
    # failed run, not permission to silently overwrite or reuse that file.
    with path.open('xb') as f:
        np.save(f, array, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    return c.info(path)


def record_path(run, stage, rid):
    return Path(run) / 'records' / stage / (rid + '.json')


def load_record(run, stage, req, binding, profile, role=None):
    rid = req['request_id'] + (':' + role if role else '')
    r = c.read(record_path(run, stage, rid))
    c.require(r['request_id'] == rid and r['stage'] == stage and r['binding_sha256'] == binding
              and r['prompt_sha256'] == req['prompt_sha256'] and r['input_ids_sha256'] == req['input_ids_sha256']
              and r['prefix_role'] == role, 'Record binding differs')
    c.verify(r['producer'])
    runtime = c.read(r['producer']['path'])
    c.require(runtime['runtime_identity'] == c.read(Path(run) / 'runtime-identity.json'), 'Producer runtime identity differs')
    c.verify(r['vector'])
    vector = np.load(r['vector']['path'], allow_pickle=False)
    c.require(vector.dtype == np.float32 and vector.shape == (profile['vocab_size'],) and np.isfinite(vector).all(), 'Bad saved logits')
    c.require(r['readout'] == (None if role else readout(vector, profile['candidate_tokens'])), 'Saved score reconstruction failed')
    attention = None
    expected_roles = req['roles'] if not role else {k: req['roles'][k] if k == role else [] for k in c.ROLES}
    c.require(r['roles'] == expected_roles, 'Recorded role geometry differs')
    if r['attention']:
        c.verify(r['attention'])
        attention = np.load(r['attention']['path'], allow_pickle=False)
        length = req['roles'][role][0] + 1 if role else req['prompt_tokens']
        c.require(attention.shape == (profile['layers'], profile['heads'], len(c.ROLES), length), 'Saved attention geometry differs')
        capture.validate_attention(attention, req, expected_roles)
    c.require((attention is None) == (stage == 'baseline'), 'Unexpected capture availability')
    return r, vector, attention


def check_engineering(prepared, run):
    plan, profile, requests = c.validate(prepared)
    run = Path(run)
    binding = c.sha(run / 'binding.json')
    comparisons, records = [], []
    for req in requests:
        base, v0, _ = load_record(run, 'baseline', req, binding, profile)
        ref, v, att = load_record(run, 'capture', req, binding, profile)
        records.extend([c.info(record_path(run, 'baseline', req['request_id'])), c.info(record_path(run, 'capture', req['request_id']))])
        comparisons.append({'kind': 'hook', 'request_id': req['request_id'],
                            'margin_difference': abs(base['readout']['m'] - ref['readout']['m']),
                            'vector_max_difference': float(np.abs(v0.astype(np.float64) - v).max()), 'attention': None})
        for stage in ('repeat', 'reverse', 'left', 'right'):
            other, _, oa = load_record(run, stage, req, binding, profile)
            records.append(c.info(record_path(run, stage, req['request_id'])))
            comparisons.append({'kind': stage, 'request_id': req['request_id'],
                                'margin_difference': abs(other['readout']['m'] - ref['readout']['m']),
                                'attention': capture.difference(att, oa, req['roles'])})
        for proof in req['prefix_proofs']:
            role = proof['role']
            _, _, pa = load_record(run, 'prefix', req, binding, profile, role)
            records.append(c.info(record_path(run, 'prefix', req['request_id'] + ':' + role)))
            ri = c.ROLES.index(role)
            comparisons.append({'kind': 'prefix', 'request_id': req['request_id'], 'role': role, 'margin_difference': None,
                                'attention': capture.difference(att[:, :, ri:ri + 1, :pa.shape[-1]], pa[:, :, ri:ri + 1])})
    q = capture.qualify(comparisons, plan['acceptance'])
    q.update(comparisons=comparisons, record_files=records, prepared_manifest=c.info(Path(prepared) / 'manifest.json'),
             binding_sha256=binding, input_count=len(requests))
    return q


def check_format(run, req, profile, binding):
    path = Path(run) / 'format' / (req['request_id'] + '.json')
    rec = c.read(path)
    c.require(rec['request_id'] == req['request_id'] and rec['binding_sha256'] == binding
              and rec['input_ids_sha256'] == req['input_ids_sha256'], 'Format binding differs')
    reference, first, _ = load_record(run, 'capture', req, binding, profile)
    c.verify(rec['producer'])
    c.require(c.read(rec['producer']['path'])['runtime_identity'] == c.read(Path(run) / 'runtime-identity.json'), 'Format producer mismatch')
    tokens = [int(first.argmax())]
    for step in rec['steps']:
        c.require(tokens[-1] not in profile['eos_token_ids'] and step['prefix_tokens'] == tokens, 'Format prefix mismatch')
        c.verify(step['vector'])
        v = np.load(step['vector']['path'], allow_pickle=False)
        c.require(v.dtype == np.float32 and v.shape == (profile['vocab_size'],) and np.isfinite(v).all(), 'Bad format vector')
        tokens.append(int(v.argmax()))
    valid = len(tokens) == 2 and tokens[0] in profile['candidate_tokens'].values() and tokens[1] in profile['eos_token_ids']
    c.require(tokens == rec['tokens'] and rec['exact_label_then_eos'] == valid and len(tokens) <= 8, 'Format reconstruction differs')
    c.require(valid, f'{req["request_id"]}: exact single-label-then-EOS gate failed')
    return c.info(path)


def seal_equal(path, value):
    if Path(path).exists():
        c.require(c.read(path) == value, 'Existing seal differs')
    else:
        c.atomic(path, value)


def worker(prepared, bound, run, invocation, phase):
    from diagnostics.cross_model_applicability_models_v1 import load_checkpoint
    prepared, run = Path(prepared), Path(run)
    plan, profile, requests = c.validate(prepared, weights=True)
    b, state = c.read(bound), c.read(run / 'state.json')
    c.verify(b['prepared_manifest'])
    c.require(b['prepared_manifest'] == c.info(prepared / 'manifest.json') and c.read(run / 'binding.json') == b, 'Preparation binding mismatch')
    c.require(state['status'] == 'launching' and state['invocations'][-1]['id'] == invocation, 'Unregistered worker')
    binding, stopping = c.sha(run / 'binding.json'), False
    state['worker_pid'] = os.getpid()
    forwards = reused = 0

    def update(**values):
        state.update(values, updated_at_unix=time.time(), new_forwards=forwards, reused_records=reused)
        c.atomic(run / 'state.json', state, replace=True)

    def stop_signal(signum, frame):
        nonlocal stopping
        stopping = True

    class Paused(Exception):
        pass

    def checkpoint():
        if stopping or (run / 'STOP').exists():
            raise Paused()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop_signal)
    try:
        checkpoint()
        update(status='loading_model')
        model, identity = load_checkpoint(profile, b['allocation'])
        seal_equal(run / 'runtime-identity.json', identity)
        producer_path = run / 'invocations' / (invocation + '.json')
        c.atomic(producer_path, {'runtime_identity': identity, 'worker_pid': os.getpid(), 'invocation': invocation,
                               'model_loaded_at_unix': time.time(), 'binding_sha256': binding})
        producer = c.info(producer_path)
        leases = [(Path(r['path']), Path(r['path']).stat()) for r in profile['metadata_sources'] + profile['weight_sources']]

        def lease_check():
            for p, old in leases:
                st = p.stat()
                c.require((st.st_ino, st.st_dev, st.st_size, st.st_mtime_ns) ==
                          (old.st_ino, old.st_dev, old.st_size, old.st_mtime_ns), 'Model file changed while loaded')

        for stage in ('baseline', 'capture', 'repeat', 'reverse', 'left', 'right', 'prefix'):
            ordered = list(reversed(requests)) if stage == 'reverse' else requests
            for req in ordered:
                prefix_roles = [p['role'] for p in req['prefix_proofs']] if stage == 'prefix' else [None]
                for role in prefix_roles:
                    checkpoint()
                    lease_check()
                    rid = req['request_id'] + (':' + role if role else '')
                    rp = record_path(run, stage, rid)
                    if rp.exists():
                        load_record(run, stage, req, binding, profile, role)
                        reused += 1
                        continue
                    update(status='engineering', stage=stage, current_request=rid)
                    start = time.monotonic()
                    vec, att, roles = capture.forward(model, req, profile['pad_token_id'],
                        padding=stage if stage in ('left', 'right') else 'none', capture=stage != 'baseline', prefix_role=role)
                    forwards += 1
                    if att is not None:
                        capture.validate_attention(att, req, roles, plan['acceptance']['attention_row_sum_cap'])
                    vp = save_array(rp.with_suffix('.logits.npy'), vec)
                    ap = save_array(rp.with_suffix('.attention.npy'), att) if att is not None else None
                    c.atomic(rp, {'stage': stage, 'request_id': rid, 'prefix_role': role, 'roles': roles,
                                  'binding_sha256': binding, 'prompt_sha256': req['prompt_sha256'],
                                  'input_ids_sha256': req['input_ids_sha256'], 'vector': vp, 'attention': ap,
                                  'producer': producer, 'readout': None if role else readout(vec, profile['candidate_tokens']),
                                  'elapsed_seconds': time.monotonic() - start})
        q = check_engineering(prepared, run)
        seal_equal(run / 'qualification.json', q)
        for req in requests:
            checkpoint()
            path = run / 'format' / (req['request_id'] + '.json')
            if path.exists():
                check_format(run, req, profile, binding)
                continue
            update(status='format', current_request=req['request_id'])
            _, first, _ = load_record(run, 'capture', req, binding, profile)
            tokens, steps = [int(first.argmax())], []
            while tokens[-1] not in profile['eos_token_ids'] and len(tokens) < 8:
                lease_check()
                generated_request = dict(req, input_ids=req['input_ids'] + tokens)
                v, _, _ = capture.forward(model, generated_request, profile['pad_token_id'], capture=False)
                forwards += 1
                step_path = path.with_name(path.stem + f'.step-{len(tokens)}.npy')
                steps.append({'prefix_tokens': list(tokens), 'vector': save_array(step_path, v)})
                tokens.append(int(v.argmax()))
            valid = len(tokens) == 2 and tokens[0] in profile['candidate_tokens'].values() and tokens[1] in profile['eos_token_ids']
            c.atomic(path, {'request_id': req['request_id'], 'binding_sha256': binding, 'producer': producer,
                            'input_ids_sha256': req['input_ids_sha256'], 'tokens': tokens, 'steps': steps, 'exact_label_then_eos': valid})
            check_format(run, req, profile, binding)
        seal_equal(run / 'engineering-seal.json', {'qualification': c.info(run / 'qualification.json'),
                   'format': [check_format(run, r, profile, binding) for r in requests], 'binding_sha256': binding})
        if phase == 'full':
            for req in requests:
                checkpoint()
                lease_check()
                path = record_path(run, 'production', req['request_id'])
                if path.exists():
                    load_record(run, 'production', req, binding, profile)
                    reused += 1
                    continue
                update(status='production', current_request=req['request_id'])
                v, a, roles = capture.forward(model, req, profile['pad_token_id'])
                forwards += 1
                ref, _, refa = load_record(run, 'capture', req, binding, profile)
                difference = {'kind': 'production_replay', 'margin_difference': abs(readout(v, profile['candidate_tokens'])['m'] - ref['readout']['m']),
                              'attention': capture.difference(a, refa, roles)}
                capture.qualify([difference], plan['acceptance'])
                c.atomic(path, {'stage': 'production', 'request_id': req['request_id'], 'prefix_role': None, 'roles': roles,
                    'binding_sha256': binding, 'prompt_sha256': req['prompt_sha256'], 'input_ids_sha256': req['input_ids_sha256'],
                    'vector': save_array(path.with_suffix('.logits.npy'), v), 'attention': save_array(path.with_suffix('.attention.npy'), a),
                    'producer': producer, 'readout': readout(v, profile['candidate_tokens']), 'replay': difference})
            seal_equal(run / 'raw-seal.json', {'records': [c.info(record_path(run, 'production', r['request_id'])) for r in requests],
                                             'engineering_seal': c.info(run / 'engineering-seal.json'), 'binding_sha256': binding})
        lease_check()
        update(status='complete_releasing' if phase == 'full' else 'qualified_releasing')
    except Paused:
        update(status='paused')
    except Exception as exc:
        update(status='failed', failure=str(exc))
        raise


def supervise(prepared, bound, run, phase='engineering', resume=False):
    prepared, bound, run = (Path(p).absolute() for p in (prepared, bound, run))
    c.require(run.parent == c.WORK and run.name.startswith('run-'), 'Use a new directory in this experiment only')
    c.validate(prepared)
    b = c.read(bound)
    c.verify(b['prepared_manifest'])
    c.require(b['prepared_manifest'] == c.info(prepared / 'manifest.json') and b['runtime_versions'] == runtime_versions()
              and b['authorization_note'], 'Missing/stale GPU binding')
    c.require(not (run / 'STOP').exists(), 'STOP is still present')
    lockdir = c.WORK / '.locks'
    lockdir.mkdir(exist_ok=True)
    handles = []
    try:
        for name in (run.name, b['allocation'][0]['uuid']):
            h = (lockdir / (name + '.lock')).open('a+')
            handles.append(h)
            fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idle(b['allocation'], gpu_inventory())
        if run.exists():
            state = c.read(run / 'state.json')
            c.require(state['binding'] == c.info(bound), 'Cannot change run binding')
            c.require((resume and state['status'] == 'paused' and state['phase'] == phase)
                      or (not resume and state['status'] == 'qualified' and phase == 'full'), 'Failed/complete runs are terminal')
            c.require(state['owned_worker_absent'] and state['worker_exit_code'] == 0, 'Previous worker not normally released')
            if phase == 'full' and not resume:
                check_run(prepared, run)
        else:
            c.require(not resume and phase == 'engineering', 'New runs start with engineering only')
            run.mkdir()
            c.atomic(run / 'binding.json', b)
            state = {'status': 'created', 'binding': c.info(bound), 'invocations': [], 'query_reference_join_performed': False}
        invocation = str(uuid.uuid4())
        state['invocations'].append({'id': invocation, 'controller_pid': os.getpid(), 'started_at_unix': time.time()})
        state.update(status='launching', phase=phase, owned_worker_absent=False)
        c.atomic(run / 'state.json', state, replace=True)
        env = dict(os.environ, PYTHONUTF8='1', CUDA_VISIBLE_DEVICES=b['allocation'][0]['uuid'], CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')
        # A CPU preparation shell may have disabled torch imports; workers must
        # explicitly use the pinned torch runtime, never inherit that disable.
        env['USE_TORCH'] = '1'
        cli = c.ROOT / 'scripts/review/run_case_content_replacement_v1.py'
        cmd = [sys.executable, str(cli), '_worker', '--prepared', str(prepared), '--bound', str(bound), '--run', str(run),
               '--phase', phase, '--invocation', invocation]
        with (run / 'gpu.log').open('ab', buffering=0) as log:
            proc = subprocess.Popen(cmd, cwd=c.ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = proc.wait()
        inventory = gpu_inventory()
        absent = not Path(f'/proc/{proc.pid}').exists() and not any(r.split(',')[-1].strip() == str(proc.pid) for r in inventory['compute_processes'])
        c.require(absent, 'Owned worker still present')
        release = run / f'release-{invocation}.json'
        c.atomic(release, {'worker_pid': proc.pid, 'worker_exit_code': code, 'owned_worker_absent': absent,
                           'inventory': inventory, 'checked_at_unix': time.time()})
        state = c.read(run / 'state.json')
        state.update(owned_worker_absent=True, worker_exit_code=code, resource_release=c.info(release))
        state['invocations'][-1].update(worker_pid=proc.pid, ended_at_unix=time.time())
        if code == 0 and state['status'] in ('complete_releasing', 'qualified_releasing'):
            state['status'] = state['status'].removesuffix('_releasing')
        elif code != 0:
            state['status'] = 'failed'
        c.atomic(run / 'state.json', state, replace=True)
        c.require(code == 0, 'Worker failed; no automatic retry')
        return state
    finally:
        for h in reversed(handles):
            h.close()


def check_run(prepared, run):
    run = Path(run)
    plan, profile, reqs = c.validate(prepared)
    state, b = c.read(run / 'state.json'), c.read(run / 'binding.json')
    c.require(b['prepared_manifest'] == c.info(Path(prepared) / 'manifest.json'), 'Run preparation differs')
    c.require(state['status'] in ('qualified', 'complete') and state['owned_worker_absent'] and state['worker_exit_code'] == 0,
              'CPU result check requires normal owned-worker release')
    c.verify(state['resource_release'])
    release = c.read(state['resource_release']['path'])
    c.require(release['owned_worker_absent'] and release['worker_exit_code'] == 0, 'Invalid release receipt')
    q = check_engineering(prepared, run)
    c.require(q == c.read(run / 'qualification.json'), 'Qualification reconstruction differs')
    expected = {'qualification': c.info(run / 'qualification.json'),
                'format': [check_format(run, r, profile, c.sha(run / 'binding.json')) for r in reqs],
                'binding_sha256': c.sha(run / 'binding.json')}
    c.require(c.read(run / 'engineering-seal.json') == expected, 'Engineering seal differs')
    if state['status'] == 'complete':
        for req in reqs:
            r, _, att = load_record(run, 'production', req, c.sha(run / 'binding.json'), profile)
            ref, _, refa = load_record(run, 'capture', req, c.sha(run / 'binding.json'), profile)
            d = {'kind': 'production_replay', 'margin_difference': abs(r['readout']['m'] - ref['readout']['m']),
                 'attention': capture.difference(att, refa, req['roles'])}
            c.require(r['replay'] == d, 'Production replay receipt differs')
            capture.qualify([d], plan['acceptance'])
        expected = {'records': [c.info(record_path(run, 'production', r['request_id'])) for r in reqs],
                    'engineering_seal': c.info(run / 'engineering-seal.json'), 'binding_sha256': c.sha(run / 'binding.json')}
        c.require(c.read(run / 'raw-seal.json') == expected, 'Raw seal differs')
    return {'status': state['status'], 'qualification': q, 'CPU_reconstructed': True, 'query_reference_join_performed': False}
