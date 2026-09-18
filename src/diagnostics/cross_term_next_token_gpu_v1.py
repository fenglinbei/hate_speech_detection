"""Single-device FP32 prompt-only worker; no query-reference file is parsed here."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import signal
import time
from pathlib import Path

from diagnostics.cross_term_next_token_v1 import (
    ROOT, atomic_json, build_qualification, canonical, check_freeze, check_record,
    check_science, file_info, math_module, now, padded_input, pass_specs,
    qualification_with_ref, read, record_paths, request_order, require, sha,
)


class Paused(Exception):
    pass


def prepare_tensors(token_ids, padding, device):
    import torch
    prepared = padded_input(token_ids, padding)
    return prepared, {k: torch.tensor([prepared[k]], dtype=torch.long, device=device)
                      for k in ['input_ids', 'attention_mask', 'position_ids']}


def forward_logits(model, token_ids, padding, device):
    import torch
    prepared, tensors = prepare_tensors(token_ids, padding, device)
    keep = torch.tensor([prepared['last_valid_index']], dtype=torch.long, device=device)
    with torch.inference_mode():
        result = model(**tensors, use_cache=False, return_dict=True, logits_to_keep=keep)
        require(result.past_key_values is None, 'unexpected KV cache')
        require(result.logits.shape == (1, 1, model.config.vocab_size), 'unexpected selected logits shape')
        require(result.logits.dtype == torch.float32, 'scorer produced reduced precision logits')
        vector = result.logits[0, 0].detach().cpu().numpy().copy()
    return prepared, vector


def save_vector(path, vector):
    import numpy as np
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    # An orphan vector has no committed receipt and cannot be reused as a score.
    if path.exists():
        orphan = path.parent / 'uncommitted-vectors'
        orphan.mkdir(exist_ok=True)
        os.replace(path, orphan / f'{path.stem}-{time.time_ns()}.npy')
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, vector, allow_pickle=False)
        stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path); temporary.unlink()


def configure_torch():
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'CUBLAS determinism environment absent')
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return torch


def load_model(plan):
    torch = configure_torch()
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one bound visible CUDA device required')
    props = torch.cuda.get_device_properties(0)
    observed_uuid = str(props.uuid).lower().removeprefix('gpu-')
    require(observed_uuid == plan['allocation']['uuid'].lower().removeprefix('gpu-'), 'CUDA and NVML UUID differ')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model, loading = AutoModelForCausalLM.from_pretrained(str(ROOT / plan['model_path']),
        local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.float32, attn_implementation='eager', device_map={'': 0},
        low_cpu_mem_usage=True, output_loading_info=True)
    require(not any(loading.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']), 'checkpoint loading mismatch')
    require(sorted(model.state_dict()) == plan['checkpoint_tensor_names'], 'checkpoint tensor inventory differs')
    model.eval()
    require(model.config._attn_implementation == 'eager', 'attention implementation differs')
    layout = []
    for kind, items in [('parameter', model.named_parameters()), ('buffer', model.named_buffers())]:
        for name, tensor in items:
            require(tensor.device.type == 'cuda' and tensor.device.index == 0, f'tensor placement differs: {name}')
            require(not tensor.is_floating_point() or tensor.dtype == torch.float32, f'non-FP32 tensor: {name}')
            layout.append({'kind': kind, 'name': name, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'device': str(tensor.device)})
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / plan['model_path']), local_files_only=True, trust_remote_code=False)
    require(tokenizer.encode('有', add_special_tokens=False) == [18830] and tokenizer.encode('无', add_special_tokens=False) == [42192], 'candidate tokens differ')
    identity = {'uuid': plan['allocation']['uuid'], 'name': props.name, 'capability': list(torch.cuda.get_device_capability(0)),
        'torch_cuda': torch.version.cuda, 'torch_total_memory_bytes': props.total_memory,
        'nvml_total_mib': plan['allocation']['total_mib'], 'compute_dtype': 'float32', 'attention': 'eager',
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32,
        'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'], 'torch_threads': torch.get_num_threads(),
        'all_model_tensors_on_bound_gpu': True, 'all_floating_tensors_fp32': True,
        'tensor_layout_sha256': hashlib.sha256(canonical(layout)).hexdigest(), 'tensor_count': len(layout),
        'loading_info': loading}
    return model, tokenizer, identity


def worker(frozen, run, invocation_id):
    frozen, run = Path(frozen).resolve(), Path(run).resolve()
    plan, requests = check_freeze(frozen, weights=True)
    state_path = run / 'run_manifest.json'
    state = read(state_path)
    require(state['status'] == 'launching' and state['invocations'][-1]['invocation_id'] == invocation_id, 'unregistered worker launch')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == plan['allocation']['uuid'], 'device visibility differs')
    binding_sha = sha(run / 'binding.json')
    state['invocations'][-1]['worker_pid'] = os.getpid()
    state.update(status='loading_model', worker_pid=os.getpid(), updated_at=now())
    atomic_json(state_path, state)
    stop_requested = False

    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def checkpoint():
        if stop_requested or (run / 'STOP').exists():
            raise Paused('requested stop at completed-record boundary')

    def update(**values):
        state.update(values, updated_at=now()); atomic_json(state_path, state)

    model = tokenizer = None
    new_forwards = reused = generated_forwards = 0
    try:
        model, tokenizer, identity = load_model(plan)
        atomic_json(run / 'invocations' / f'{invocation_id}.json', identity, replace=False)
        producer = {'uuid': plan['allocation']['uuid'], 'pid': os.getpid(), 'invocation_id': invocation_id,
                    'runtime_identity': file_info(run / 'invocations' / f'{invocation_id}.json')}
        state['invocations'][-1]['runtime_identity'] = producer['runtime_identity']
        atomic_json(state_path, state)
        maths = math_module()
        # A cheap stat lease detects changes while the already-hashed weights are in use.
        lease = [(ROOT / e['path'], (ROOT / e['path']).stat()) for e in plan['model_files']]
        for spec in pass_specs(frozen):
            checkpoint()
            check_freeze(frozen)
            for path, before in lease:
                after = path.stat()
                require((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) ==
                        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns), 'model file changed during invocation')
            q = qualification_with_ref(run) if spec['pass_id'] == 'science' else None
            if spec['pass_id'] == 'science':
                require(build_qualification(frozen, run, full=False) == read(run / 'qualification.json'), 'qualification changed before science')
            update(status='running', current_pass=spec['pass_id'], pass_completed_requests=0)
            for i, request in enumerate(request_order(requests, spec), 1):
                checkpoint()
                receipt_path, vector_path = record_paths(run, spec['pass_id'], request['request_id'])
                if receipt_path.exists():
                    check_record(run, spec, request, binding_sha, q, full=True)
                    reused += 1
                else:
                    prepared, vector = forward_logits(model, request['input_ids'], spec['padding'], 'cuda:0')
                    score = maths.readout(vector, q['margin_error_bound'] if q else None, q['receipt_ref'] if q else None)
                    save_vector(vector_path, vector)
                    receipt = {'schema_version': 'cross-term-next-token-score/v1', 'scored_at': now(),
                        'request_id': request['request_id'], 'condition_id': request['condition_id'],
                        'physical_score_id': f'{read(run / "binding.json")["run_id"]}:{spec["pass_id"]}:{request["request_id"]}',
                        'pass_id': spec['pass_id'], 'prompt_sha256': request['prompt_sha256'],
                        'input_ids_sha256': request['input_ids_sha256'], 'binding_sha256': binding_sha,
                        'prepared_input': prepared, 'producer': producer, 'readout': score,
                        'raw_logits': file_info(vector_path), 'candidate_forward_calls': 1,
                        'candidates_share_forward': True, 'vocab_size': len(vector)}
                    atomic_json(receipt_path, receipt, replace=False)
                    new_forwards += 1
                update(pass_completed_requests=i, new_prompt_forwards_this_invocation=new_forwards,
                       reused_requests_this_invocation=reused)
                if i % 20 == 0:
                    print(json.dumps({'pass': spec['pass_id'], 'complete': i, 'of': 120, 'at': now()}), flush=True)
            if spec['pass_id'] not in state['completed_passes']:
                state['completed_passes'].append(spec['pass_id'])
            update()
            if spec['pass_id'] == 'engineering-reverse-request-order':
                update(status='engineering_cpu_verification')
                qualification = build_qualification(frozen, run, full=True)
                if (run / 'qualification.json').exists():
                    require(read(run / 'qualification.json') == qualification, 'sealed qualification differs')
                else:
                    atomic_json(run / 'qualification.json', qualification, replace=False)
                print(json.dumps({'engineering': 'pass', 'bound': qualification['margin_error_bound']}), flush=True)
            if spec['pass_id'] == 'science':
                _, seal = check_science(frozen, run, full=True)
                if (run / 'science-seal.json').exists():
                    require(read(run / 'science-seal.json') == seal, 'science seal changed')
                else:
                    atomic_json(run / 'science-seal.json', seal, replace=False)
        update(status='generation_diagnostic')
        lookup = {r['condition_id']: r for r in requests}
        config = plan['generation_diagnostic']; generated = []
        for ci, condition in enumerate(config['condition_ids']):
            checkpoint()
            request = lookup[condition]; ids = list(request['input_ids']); output = []; step_records = []
            receipt_path = run / 'generation' / f'{request["request_id"]}.json'
            if receipt_path.exists():
                r = read(receipt_path)
                require(r['condition_id'] == condition and r['prompt_sha256'] == request['prompt_sha256'], 'generation checkpoint identity differs')
                for step in r['steps']:
                    verify = step['raw_logits']
                    require(file_info(ROOT / verify['path']) == verify, 'generation checkpoint vector differs')
                generated.append(r)
                continue
            for step in range(config['max_new_tokens']):
                checkpoint()
                prepared, vector = forward_logits(model, ids, 'none', 'cuda:0')
                import numpy as np
                token = int(np.argmax(vector))
                vector_path = run / 'generation' / f'{request["request_id"]}-step-{step}.npy'
                save_vector(vector_path, vector)
                step_records.append({'step': step, 'raw_logits': file_info(vector_path), 'selected_token_id': token,
                                     'input_ids_sha256': hashlib.sha256(canonical(ids)).hexdigest()})
                ids.append(token); output.append(token); generated_forwards += 1
                if token == config['eos_token_id']:
                    break
            visible = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            r = {'condition_id': condition, 'prompt_sha256': request['prompt_sha256'], 'generated_ids': output,
                'decoded_with_special_tokens': tokenizer.decode(output, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                'visible_answer': visible, 'strict_pass': visible in config['strict_pass_values'], 'steps': step_records,
                'producer': producer, 'terminated_by_eos': bool(output and output[-1] == config['eos_token_id'])}
            atomic_json(receipt_path, r, replace=False); generated.append(r)
            update(generation_completed_requests=ci + 1)
        gen = {'primary_metric': False, 'records': generated, 'forward_calls': sum(len(r['steps']) for r in generated)}
        if (run / 'generation.json').exists():
            require(read(run / 'generation.json') == gen, 'generation seal differs')
        else:
            atomic_json(run / 'generation.json', gen, replace=False)
        check_freeze(frozen, weights=True)
        update(status='scoring_complete_releasing', completed_at=now())
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
            torch.cuda.synchronize(); torch.cuda.empty_cache()
        state['invocations'][-1].update(ended_at=now(), new_prompt_forwards=new_forwards,
            reused_requests=reused, generation_forwards=generated_forwards, model_references_released=True)
        atomic_json(state_path, state)
    return state['status']
