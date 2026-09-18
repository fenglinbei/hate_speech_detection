"""GPU-only execution adapter; unchanged native scorer, no human-reference reads."""
from __future__ import annotations

import gc
import json
import os
import signal
from pathlib import Path

from diagnostics import cross_term_behavior_execution_v1 as c
from diagnostics import cross_term_next_token_gpu_v1 as kernel


class Paused(Exception):
    pass


def worker(prepared, bound, run, invocation_id):
    prepared, run = Path(prepared).resolve(), Path(run).resolve()
    plan, _ = c.check_bound(prepared, bound, weights=True)
    c.check_run_binding(prepared, run)
    state_path = run / 'run_manifest.json'
    state = c.read(state_path)
    c.require(state['status'] == 'launching' and state['invocations'][-1]['invocation_id'] == invocation_id,
              'unregistered worker launch')
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == plan['allocation']['uuid'], 'GPU visibility differs')
    state['invocations'][-1]['worker_pid'] = os.getpid()
    state['worker_pid'] = os.getpid()
    binding = c.read(run / 'binding.json'); binding_sha = c.sha(run / 'binding.json')
    stop = False

    def on_signal(signum, frame):
        nonlocal stop
        stop = True

    def checkpoint():
        if stop or (run / 'STOP').exists():
            raise Paused('stop requested; committed requests preserved')

    def update(**values):
        state.update(values, updated_at=c.now())
        c.atomic_json(state_path, state)

    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, on_signal)
    update(status='loading_model')
    model = tokenizer = None
    new_forwards = reused = 0
    try:
        checkpoint()
        model, tokenizer, identity = kernel.load_model(plan)
        c.require(identity == c.read(c.ROOT / plan['required_producer_identity']['path']),
                  'loaded GPU runtime/layout differs from original producer')
        identity_path = run / 'invocations' / f'{invocation_id}.json'
        c.atomic_json(identity_path, identity, replace=False)
        producer = {'uuid': plan['allocation']['uuid'], 'pid': os.getpid(), 'invocation_id': invocation_id,
                    'runtime_identity': c.file_info(identity_path)}
        state['invocations'][-1]['runtime_identity'] = producer['runtime_identity']
        update()
        maths = c.math_module()
        lease = [(c.ROOT / e['path'], (c.ROOT / e['path']).stat()) for e in plan['model_files']]
        for spec in c.specs():
            checkpoint()
            c.check_prepared(prepared)
            for path, before in lease:
                after = path.stat()
                c.require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                          (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'model changed during invocation')
            if spec['pass_id'] != 'historical-bridge':
                c.require(c.build_bridge(prepared, run, full=False) == c.read(run / 'bridge-seal.json'),
                          'historical bridge not accepted before new scoring')
            q = None
            if spec['pass_id'] == 'science':
                c.require(c.build_qualification(prepared, run, full=False) == c.read(run / 'qualification.json'),
                          'qualification not sealed before science')
                q = c.old.qualification_with_ref(run)
            requests = c.pass_requests(prepared, spec)
            update(status='running', current_pass=spec['pass_id'], pass_completed_requests=0)
            for i, request in enumerate(requests, 1):
                checkpoint()
                receipt_path, vector_path = c.old.record_paths(run, spec['pass_id'], request['request_id'])
                if receipt_path.exists():
                    record = c.old.check_record(run, spec, request, binding_sha, q, full=True)
                    c.check_producer(run, record, identity)
                    reused += 1
                else:
                    prepared_input, vector = kernel.forward_logits(model, request['input_ids'], spec['padding'], 'cuda:0')
                    score = maths.readout(vector, q['margin_error_bound'] if q else None, q['receipt_ref'] if q else None)
                    kernel.save_vector(vector_path, vector)
                    receipt = {'schema_version': 'cross-term-next-token-score/v1', 'scored_at': c.now(),
                        'request_id': request['request_id'], 'condition_id': request['condition_id'],
                        'physical_score_id': f"{binding['run_id']}:{spec['pass_id']}:{request['request_id']}",
                        'pass_id': spec['pass_id'], 'prompt_sha256': request['prompt_sha256'],
                        'input_ids_sha256': request['input_ids_sha256'], 'binding_sha256': binding_sha,
                        'prepared_input': prepared_input, 'producer': producer, 'readout': score,
                        'raw_logits': c.file_info(vector_path), 'candidate_forward_calls': 1,
                        'candidates_share_forward': True, 'vocab_size': len(vector)}
                    c.atomic_json(receipt_path, receipt, replace=False)
                    new_forwards += 1
                update(pass_completed_requests=i, new_prompt_forwards_this_invocation=new_forwards,
                       reused_requests_this_invocation=reused)
                if i % 12 == 0 or i == len(requests):
                    print(json.dumps({'pass': spec['pass_id'], 'complete': i, 'of': len(requests), 'at': c.now()}), flush=True)
            if spec['pass_id'] == 'historical-bridge':
                c.seal(run / 'bridge-seal.json', c.build_bridge(prepared, run, full=True))
            elif spec['pass_id'] == c.ENGINEERING[-1]:
                c.seal(run / 'qualification.json', c.build_qualification(prepared, run, full=True))
            elif spec['pass_id'] == 'science':
                c.seal(run / 'science-seal.json', c.check_science(prepared, run, full=True)[1])
            if spec['pass_id'] not in state['completed_passes']:
                state['completed_passes'].append(spec['pass_id'])
            update()
        c.check_prepared(prepared, weights=True)
        update(status='scoring_complete_releasing', scoring_completed_at=c.now())
    except Paused as error:
        update(status='paused', pause_reason=str(error))
    except BaseException as error:
        update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        model = tokenizer = None
        gc.collect()
        import torch
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        state['invocations'][-1].update(ended_at=c.now(), new_prompt_forwards=new_forwards,
            reused_requests=reused, model_references_released=True)
        c.atomic_json(state_path, state)
    return state['status']
