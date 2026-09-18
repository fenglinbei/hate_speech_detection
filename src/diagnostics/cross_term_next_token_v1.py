"""Versioned cross-term next-token execution contracts and CPU verification."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT / 'docs/research/experiment-plans/cross-term-execution-prep-20260917-v1'
MATERIAL = ROOT / 'docs/research/experiment-plans/cross-term-joint-v1/frozen-01'
WORK = ROOT / 'reviews/cross-term-next-token-v1'
PASS_NAMES = ['engineering-reference', 'engineering-repeat', 'engineering-left-padding',
              'engineering-right-padding', 'engineering-reverse-request-order', 'science']
IMPLEMENTATION = [
    'src/diagnostics/cross_term_next_token_v1.py',
    'src/diagnostics/cross_term_next_token_gpu_v1.py',
    'src/diagnostics/__init__.py',
    'scripts/review/run_cross_term_next_token_v1.py',
    'scripts/review/test_cross_term_next_token_v1.py',
    'scripts/review/audit_cross_term_next_token_v1.py',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def file_info(path):
    path = Path(path)
    return {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            'bytes': path.stat().st_size, 'sha256': sha(path)}


def verify_file(entry):
    path = ROOT / entry['path']
    require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], f'file lease differs: {path}')


def atomic_json(path, value, *, replace=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(replace or not path.exists(), f'refusing to overwrite sealed record: {path}')
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    if replace:
        os.replace(temp, path)
    else:
        os.link(temp, path); temp.unlink()


def math_module():
    path = PREP / 'next_token_math.py'
    spec = importlib.util.spec_from_file_location('cross_term_frozen_math', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def input_records(frozen):
    return [json.loads(line) for line in (Path(frozen) / 'model-inputs.jsonl').read_text().splitlines()]


def runtime_snapshot():
    packages = {n: importlib.metadata.version(n) for n in
                ['torch', 'transformers', 'accelerate', 'numpy', 'safetensors', 'tokenizers', 'huggingface-hub']}
    return {'python': sys.version, 'executable': str(Path(sys.executable).resolve()),
            'platform': platform.platform(), 'packages': packages}


def runtime_source_files():
    site = Path(importlib.util.find_spec('transformers').origin).parent
    names = ['models/qwen3/modeling_qwen3.py', 'models/qwen3/configuration_qwen3.py',
             'modeling_utils.py', 'masking_utils.py', 'modeling_attn_mask_utils.py',
             'modeling_rope_utils.py', 'cache_utils.py']
    return [file_info(site / name) for name in names]


def gpu_inventory():
    import subprocess
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version',
                                   '--format=csv,noheader,nounits'], text=True)
    devices = []
    for r in csv.reader(raw.splitlines()):
        require(len(r) == 7, 'unexpected GPU inventory row')
        devices.append(dict(index=int(r[0]), uuid=r[1].strip(), name=r[2].strip(),
            total_mib=int(r[3]), used_mib=int(r[4]), utilization=int(r[5]), driver=r[6].strip()))
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    processes = [r for r in csv.reader(raw.splitlines()) if r]
    return {'checked_at': now(), 'devices': devices, 'compute_processes': processes}


def idle_device(index, inventory=None):
    inventory = inventory or gpu_inventory()
    device = next(d for d in inventory['devices'] if d['index'] == index)
    require(device['used_mib'] == 0 and device['utilization'] == 0, f'GPU {index} is busy')
    require(not any(r[0].strip() == device['uuid'] for r in inventory['compute_processes']), 'selected GPU has a compute process')
    return device


def create_freeze(output, gpu=0):
    import shutil
    import runpy
    output = Path(output).resolve()
    require(output.parent == WORK and not output.exists(), 'new owned freeze directory required')
    material_check = runpy.run_path(str(MATERIAL / 'validate_freeze.py'))['check']()
    prep_check = runpy.run_path(str(PREP / 'validate_preparation.py'))['check']()
    inventory = gpu_inventory(); device = idle_device(gpu, inventory)
    old_plan = read(PREP / 'plan.json')
    model_path = ROOT / old_plan['model']['path']
    index = read(model_path / 'model.safetensors.index.json')
    weights = [file_info(model_path / name) for name in sorted(set(index['weight_map'].values()))]
    require(len(weights) == 5, 'checkpoint shard inventory differs')
    output.mkdir(parents=True)
    shutil.copyfile(PREP / 'model-inputs.jsonl', output / 'model-inputs.jsonl')
    shutil.copyfile(PREP / 'qualification-plan.json', output / 'qualification-plan.json')
    authorization = {'recorded_at': now(), 'user_message_verbatim': '那现在可以开始GPU 评分器实现及运行绑定，目前GPU四卡已空闲，可以自由使用',
        'scope': 'Implement, bind and run the new scorer under the prepared engineering-before-science protocol. Four available GPUs authorized; this execution uses only GPU 0.',
        'fixed_deadline': None, 'prior_experiment_restart_authorized': False}
    atomic_json(output / 'authorization.json', authorization, replace=False)
    plan = {'schema_version': 'cross-term-next-token-execution/v1', 'created_at': now(),
        'preparation_manifest': file_info(PREP / 'manifest.json'), 'material_manifest': file_info(MATERIAL / 'manifest.json'),
        'model_input': file_info(output / 'model-inputs.jsonl'), 'qualification_plan': file_info(output / 'qualification-plan.json'),
        'analysis_plan': file_info(PREP / 'analysis-plan.json'), 'authorization': file_info(output / 'authorization.json'),
        'runtime': runtime_snapshot(), 'runtime_sources': runtime_source_files(),
        'model_path': str(model_path.relative_to(ROOT)), 'model_files': weights + [file_info(model_path / name) for name in
            ['config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']],
        'checkpoint_tensor_names': sorted(index['weight_map']),
        'allocation': {k: device[k] for k in ['index', 'uuid', 'name', 'total_mib', 'driver']},
        'required_runtime': old_plan['required_runtime'], 'candidate_tokens': old_plan['candidate_tokens'],
        'generation_diagnostic': old_plan['generation_diagnostic'], 'primary_budget': old_plan['primary_budget'],
        'passes': PASS_NAMES, 'dtype_override': 'float32 even though checkpoint config says bfloat16',
        'logits_forward': 'native Qwen3 forward with tensor logits_to_keep selecting last valid prompt position; both candidates share one full-vocabulary vector',
        'checkpoint_policy': 'Per-request atomic raw-vector then receipt. Resume only on same frozen sources/runtime/UUID; reuse every valid saved request. Failed and terminal complete runs reject execution.',
        'interpretation_scope': 'All frozen development inputs; no rule-fit strata, label mapping change, NCC or independent replication claim.'}
    atomic_json(output / 'plan.json', plan, replace=False)
    atomic_json(output / 'cpu-preparation.json', {'material': material_check, 'preparation': prep_check,
        'weight_bytes_hashed': sum(e['bytes'] for e in weights), 'inventory_at_binding': inventory}, replace=False)
    artifacts = [file_info(p) for p in sorted(output.iterdir()) if p.is_file()]
    sources = [file_info(ROOT / name) for name in IMPLEMENTATION] + [file_info(PREP / 'next_token_math.py'),
        file_info(PREP / 'manifest.json'), file_info(PREP / 'analysis-plan.json'), file_info(MATERIAL / 'manifest.json')]
    manifest = {'schema_version': 'cross-term-next-token-freeze/v1', 'created_at': now(),
        'artifacts': artifacts, 'sources': sources, 'runtime_sources': plan['runtime_sources'],
        'weight_sources': plan['model_files'], 'scientific_inputs': 120, 'primary_prompt_forwards': 720,
        'numerical_qualification_performed': False}
    atomic_json(output / 'manifest.json', manifest, replace=False)
    return {'freeze': str(output), 'manifest': file_info(output / 'manifest.json'), 'allocation': plan['allocation']}


def check_freeze(frozen, weights=False):
    frozen = Path(frozen).resolve(); manifest = read(frozen / 'manifest.json')
    for e in manifest['artifacts'] + manifest['sources'] + manifest['runtime_sources']:
        verify_file(e)
    if weights:
        for e in manifest['weight_sources']:
            verify_file(e)
    plan = read(frozen / 'plan.json')
    require(plan['runtime'] == runtime_snapshot(), 'Python/runtime identity differs')
    require(plan['passes'] == PASS_NAMES and plan['candidate_tokens'] == {'有': 18830, '无': 42192}, 'scoring plan changed')
    require(sha(frozen / 'model-inputs.jsonl') == sha(PREP / 'model-inputs.jsonl'), 'model-visible input changed')
    requests = input_records(frozen)
    require(len(requests) == len({r['condition_id'] for r in requests}) == 120, 'input inventory differs')
    return plan, requests


def pass_specs(frozen):
    specs = read(Path(frozen) / 'qualification-plan.json')['passes']
    return specs + [{'pass_id': 'science', 'request_order': 'forward', 'padding': 'none'}]


def request_order(requests, spec):
    return list(reversed(requests)) if spec['request_order'] == 'reverse' else list(requests)


def padded_input(token_ids, padding):
    n = len(token_ids)
    require(0 < n <= 40960, 'empty or overlong input; truncation prohibited')
    if padding == 'none':
        ids, mask = list(token_ids), [1] * n
    else:
        require(padding in ['left_to_next_strict_multiple_of_16', 'right_to_next_strict_multiple_of_16'], 'unknown padding')
        extra = (n // 16 + 1) * 16 - n
        if padding.startswith('left'):
            ids, mask = [151643] * extra + list(token_ids), [0] * extra + [1] * n
        else:
            ids, mask = list(token_ids) + [151643] * extra, [1] * n + [0] * extra
    positions = []; seen = 0
    for m in mask:
        seen += m; positions.append(max(seen - 1, 0))
    last = max(i for i, m in enumerate(mask) if m)
    return {'input_ids': ids, 'attention_mask': mask, 'position_ids': positions,
            'last_valid_index': last, 'valid_tokens': n, 'tensor_tokens': len(ids)}


def record_paths(run, pass_id, request_id):
    directory = Path(run) / 'scores' / pass_id
    return directory / (request_id + '.json'), directory / (request_id + '.npy')


def check_record(run, spec, request, binding, qualification=None, full=True):
    import numpy as np
    receipt_path, vector_path = record_paths(run, spec['pass_id'], request['request_id'])
    r = read(receipt_path)
    require(r['request_id'] == request['request_id'] and r['condition_id'] == request['condition_id'], 'request identity differs')
    require(r['prompt_sha256'] == request['prompt_sha256'] and r['pass_id'] == spec['pass_id'], 'prompt/pass identity differs')
    require(r['binding_sha256'] == binding and r['input_ids_sha256'] == request['input_ids_sha256'], 'input or run binding differs')
    binding_data = read(Path(run) / 'binding.json')
    require(r['physical_score_id'] == f'{binding_data["run_id"]}:{spec["pass_id"]}:{request["request_id"]}', 'physical score identity differs')
    require(r['candidate_forward_calls'] == 1 and r['candidates_share_forward'] is True and r['vocab_size'] == 151936, 'candidate forward contract differs')
    require(r['prepared_input'] == padded_input(request['input_ids'], spec['padding']), 'mask/position/score index differs')
    require(r['raw_logits'] == file_info(vector_path), 'raw vector changed')
    if full:
        values = np.load(vector_path, allow_pickle=False)
        require(values.shape == (151936,) and values.dtype == np.dtype('float32') and np.isfinite(values).all(), 'invalid raw logits')
        bound = qualification['margin_error_bound'] if qualification else None
        qref = qualification['receipt_ref'] if qualification else None
        expected = math_module().readout(values, bound, qref)
        require(r['readout'] == expected, 'score does not reconstruct from raw logits')
    return r


def check_pass(frozen, run, spec, qualification=None, full=True):
    requests = request_order(input_records(frozen), spec)
    binding = sha(Path(run) / 'binding.json')
    directory = Path(run) / 'scores' / spec['pass_id']
    expected = {r['request_id'] + '.json' for r in requests}
    require({p.name for p in directory.glob('*.json')} == expected, 'pass not complete or contains unexpected records')
    require({p.name for p in directory.glob('*.npy')} == {name.replace('.json', '.npy') for name in expected}, 'raw vector inventory differs')
    records = [check_record(run, spec, r, binding, qualification, full) for r in requests]
    binding_data = read(Path(run) / 'binding.json')
    invocations = {x['invocation_id']: x for x in read(Path(run) / 'run_manifest.json')['invocations']}
    for r in records:
        require(r['producer']['uuid'] == binding_data['allocation']['uuid'], 'physical producer differs')
        require(r['producer']['invocation_id'] in invocations, 'unregistered producer invocation')
        require(r['producer']['pid'] == invocations[r['producer']['invocation_id']]['worker_pid'], 'producer PID differs')
        verify_file(r['producer']['runtime_identity'])
        identity = read(ROOT / r['producer']['runtime_identity']['path'])
        require(identity['uuid'] == binding_data['allocation']['uuid'] and identity['all_floating_tensors_fp32']
                and identity['deterministic_algorithms'] and not identity['tf32_matmul'] and not identity['tf32_cudnn'], 'producer runtime contract differs')
    return records


def qualification_values(by_pass, config):
    expected = set(by_pass['engineering-reference'])
    require(len(expected) == 120 and set(by_pass) == set(PASS_NAMES[:-1]), 'engineering pass coverage incomplete')
    differences = {}; identity_error = 0.0
    for pass_id, rows in by_pass.items():
        require(set(rows) == expected, 'engineering input coverage differs')
        for score in rows.values():
            require(all(math.isfinite(score[k]) for k in ['m', 'z_no', 'z_yes', 'log_p_no', 'log_p_yes', 'legal_mass', 'log_legal_mass', 'pair_support_no']), 'nonfinite readout')
            identity_error = max(identity_error, abs(score['m'] - (score['log_p_no'] - score['log_p_yes'])))
        if pass_id != 'engineering-reference':
            differences[pass_id] = max(abs(rows[k]['m'] - by_pass['engineering-reference'][k]['m']) for k in expected)
    rules = config['acceptance']
    require(differences['engineering-repeat'] <= rules['repeat_and_reverse_order_max_margin_difference'], 'repeat numerical gate failed')
    require(differences['engineering-reverse-request-order'] <= rules['repeat_and_reverse_order_max_margin_difference'], 'request order numerical gate failed')
    for key in ['engineering-left-padding', 'engineering-right-padding']:
        require(differences[key] <= rules['padding_max_abs_margin_difference_cap'], f'{key} numerical gate failed')
    require(identity_error <= rules['full_vocab_logprob_identity_max_abs_error'], 'log probability identity gate failed')
    return {'status': 'pass', 'engineering_requests': 600, 'unique_inputs': 120,
            'max_margin_differences': differences, 'max_log_probability_identity_error': identity_error,
            'margin_error_bound': max(0.000001, 2 * max(differences.values())),
            'bound_meaning': 'engineering reproducibility envelope, not a statistical confidence interval'}


def build_qualification(frozen, run, full=True):
    by_pass = {}; seals = []
    for spec in pass_specs(frozen)[:-1]:
        records = check_pass(frozen, run, spec, full=full)
        by_pass[spec['pass_id']] = {r['condition_id']: r['readout'] for r in records}
        seals.append({'pass_id': spec['pass_id'], 'receipt_inventory_sha256': hashlib.sha256(canonical([
            file_info(record_paths(run, spec['pass_id'], r['request_id'])[0]) for r in records])).hexdigest()})
    result = qualification_values(by_pass, read(Path(frozen) / 'qualification-plan.json'))
    result.update(binding_sha256=sha(Path(run) / 'binding.json'), frozen_manifest_sha256=sha(Path(frozen) / 'manifest.json'),
                  pass_receipts=seals, qualification_policy=read(Path(frozen) / 'qualification-plan.json')['acceptance'])
    return result


def qualification_with_ref(run):
    p = Path(run) / 'qualification.json'
    q = read(p)
    return dict(q, receipt_ref=file_info(p))


def check_science(frozen, run, full=True):
    q = qualification_with_ref(run)
    records = check_pass(frozen, run, pass_specs(frozen)[-1], q, full)
    refs = {r['condition_id']: r['readout'] for r in check_pass(frozen, run, pass_specs(frozen)[0], full=False)}
    maximum = max(abs(r['readout']['m'] - refs[r['condition_id']]['m']) for r in records)
    require(maximum <= q['margin_error_bound'], 'science replay differs from qualified reference')
    return records, {'science_requests': 120, 'max_reference_margin_difference': maximum,
                     'margin_error_bound': q['margin_error_bound'], 'raw_inventory_sha256': hashlib.sha256(canonical([
                         file_info(record_paths(run, 'science', r['request_id'])[0]) for r in records])).hexdigest()}


def check_completed(frozen, run, full=True):
    check_freeze(frozen)
    require(build_qualification(frozen, run, full) == read(Path(run) / 'qualification.json'), 'qualification receipt reconstruction differs')
    records, seal = check_science(frozen, run, full)
    require(seal == read(Path(run) / 'science-seal.json'), 'raw science seal differs')
    generated = read(Path(run) / 'generation.json')
    plan = read(Path(frozen) / 'plan.json')
    require([r['condition_id'] for r in generated['records']] == plan['generation_diagnostic']['condition_ids'], 'generation coverage differs')
    require(generated['primary_metric'] is False and len(generated['records']) == 12, 'generation mixed into primary')
    return {'status': 'pass', 'all_passes': 6, 'primary_prompt_forwards': 720, 'candidate_values': 1440,
            'qualification': read(Path(run) / 'qualification.json'), 'science': seal,
            'strict_generation_passes': sum(r['strict_pass'] for r in generated['records']),
            'generation_prompts': 12, 'generation_forwards': generated['forward_calls']}


def analyze(frozen, run, output):
    """Reference join is permitted only after all raw/gate seals and worker release."""
    frozen, run, output = Path(frozen), Path(run), Path(output)
    state = read(run / 'run_manifest.json')
    require(state['status'] == 'complete' and state['worker_exit_code'] == 0 and state['owned_worker_absent'], 'GPU worker not completely released')
    checked = check_completed(frozen, run, full=True)
    plan = read(frozen / 'plan.json'); verify_file(plan['analysis_plan'])
    analysis_plan = read(ROOT / plan['analysis_plan']['path'])
    refs = {r['query_id']: r for r in analysis_plan['references']}
    conditions = {r['condition_id']: r for r in analysis_plan['condition_map']}
    math_api = math_module(); score_rows = []; score_map = {}; arms = defaultdict(list)
    for r in check_pass(frozen, run, pass_specs(frozen)[-1], qualification_with_ref(run), full=False):
        c = conditions[r['condition_id']]; reference = refs[c['query_id']]
        answer = reference['human_reference']['task_label']
        score = dict(r['readout'], physical_score_id=r['physical_score_id'], prompt_sha256=r['prompt_sha256'])
        score_map[r['condition_id']] = score
        row = dict(c, family_id=reference['family_id'], adopted_reference=answer,
            adopted_severity=reference['human_reference']['attack_severity'], original_gold=None, original_correct=None,
            **score, **math_api.reference_status(score, answer))
        score_rows.append(row); arms[(c['lexicon_arm'], c['demo_arm'])].append(row)
    expressions = []; grouped = defaultdict(lambda: defaultdict(list))
    for c in analysis_plan['comparisons']:
        effect = math_api.linear_effect(c['terms'], score_map)
        ref = refs[c['query_id']]; sign = c['correct_direction_sign']
        aligned = sign * effect['value']
        aligned_resolution = math_api.resolution(aligned, effect['bound'])
        transition = None
        if len(c['terms']) == 2 and sorted(t['coefficient'] for t in c['terms']) == [-1, 1]:
            treatment = next(t['condition_id'] for t in c['terms'] if t['coefficient'] == 1)
            control = next(t['condition_id'] for t in c['terms'] if t['coefficient'] == -1)
            transition = math_api.transition(score_map[control], score_map[treatment], ref['human_reference']['task_label'])
        sid = c['query_id'].split('-')[1]
        contrast_type = c['comparison_id'].replace(f'CTDJ-legacy-CTDD-{sid}-', '').replace(f'CTDJ-{sid}-', '')
        row = dict(c, family_id=ref['family_id'], contrast_type=contrast_type, effect=effect,
            reference_aligned_change=aligned, reference_aligned_resolution=aligned_resolution,
            verified_classification_transition=transition)
        expressions.append(row); grouped[(c['scope'], contrast_type)][ref['family_id']].append(row)
    summaries = []
    for (scope, kind), families in sorted(grouped.items()):
        family_rows = []
        for family, rows in sorted(families.items()):
            n = len(rows)
            family_rows.append({'family_id': family, 'comparisons': n,
                'mean_effect': math.fsum(r['effect']['value'] for r in rows) / n,
                'mean_reference_aligned_change': math.fsum(r['reference_aligned_change'] for r in rows) / n,
                'mean_bound': math.fsum(r['effect']['bound'] for r in rows) / n})
        count = len(family_rows)
        all_rows = [r for rows in families.values() for r in rows]
        summaries.append({'scope': scope, 'contrast_type': kind, 'family_weighting': 'equal',
            'family_count': count, 'families': family_rows,
            'family_equal_mean_effect': math.fsum(f['mean_effect'] for f in family_rows) / count,
            'family_equal_mean_reference_aligned_change': math.fsum(f['mean_reference_aligned_change'] for f in family_rows) / count,
            'family_equal_bound': math.fsum(f['mean_bound'] for f in family_rows) / count,
            'aligned_resolved_positive': sum(r['reference_aligned_resolution'] == 'positive' for r in all_rows),
            'aligned_resolved_negative': sum(r['reference_aligned_resolution'] == 'negative' for r in all_rows),
            'numerical_unresolved': sum(r['reference_aligned_resolution'] == 'numerical_unresolved' for r in all_rows),
            'comparison_count': len(all_rows)})
    arm_rows = []
    for (lex, demo), rows in sorted(arms.items()):
        arm_rows.append({'lexicon_arm': lex, 'demo_arm': demo, 'conditions': len(rows),
            'raw_correct': sum(r['raw_correct'] for r in rows),
            'conservative_correct': sum(r['conservative_correct'] for r in rows),
            'numerical_unresolved': sum(r['resolution'] == 'numerical_unresolved' for r in rows),
            'exact_ties': sum(r['exact_tie'] for r in rows),
            'min_legal_mass': min(r['legal_mass'] for r in rows),
            'mean_legal_mass': math.fsum(r['legal_mass'] for r in rows) / len(rows),
            'independent_samples': False})
    result = {'schema_version': 'cross-term-next-token-results/v1', 'frozen_manifest': file_info(frozen / 'manifest.json'),
        'binding': file_info(run / 'binding.json'), 'qualification': file_info(run / 'qualification.json'),
        'science_seal': file_info(run / 'science-seal.json'), 'analysis_plan': plan['analysis_plan'],
        'checks': checked, 'arm_readouts': arm_rows, 'scores': score_rows, 'expressions': expressions,
        'family_equal_summaries': summaries,
        'scope': 'Development materials, three exposed terms with shared constructions; descriptive correlated conditions, not independent confirmation. No rule-fit stratification or pure semantics/applicability causation.'}
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'results.json'
    if target.exists():
        require(read(target) == result, 'final analysis bytes cannot be reconstructed')
    else:
        atomic_json(target, result, replace=False)
    lines = ['# 跨词项新评分实验结果', '', '主分数 m=z无−z有；采用已采纳的12条查询参考。原始Gold为空。各条件共享查询及材料，不能作为独立样本。', '',
        f'工程与正式评分共720次prompt-only forward；数值界b={checked["qualification"]["margin_error_bound"]:.12g}。自由生成严格通过{checked["strict_generation_passes"]}/12。', '',
        '| 词典槽 | 示例背景 | 正确／条件数 | 数值未决 | 最小合法输出质量 |', '|---|---|---:|---:|---:|']
    for r in arm_rows:
        lines.append(f'| {r["lexicon_arm"]} | {r["demo_arm"]} | {r["conservative_correct"]}/{r["conditions"]} | {r["numerical_unresolved"]} | {r["min_legal_mass"]:.6g} |')
    lines += ['', '## 固定示例背景下的比较', '', '| 比较 | 向参考方向移动 | 反向移动 | 数值未决 | 家族等权平均参考方向变化 |', '|---|---:|---:|---:|---:|']
    for r in summaries:
        if r['scope'] == 'within_query_fixed_D':
            lines.append(f'| {r["contrast_type"]} | {r["aligned_resolved_positive"]} | {r["aligned_resolved_negative"]} | {r["numerical_unresolved"]} | {r["family_equal_mean_reference_aligned_change"]:.6g} |')
    lines += ['', '全部120项读数、252项表达式、家族内外汇总、修复/损伤与未决转移见 results.json。全部反向或未决行保留。L−N含长度与措辞差异；A/B为整包内容诊断。数值可分辨不等于实际重要性或统计显著。']
    report = '\n'.join(lines) + '\n'
    if (output / 'RESULTS.md').exists():
        require((output / 'RESULTS.md').read_text() == report, 'report reconstruction differs')
    else:
        (output / 'RESULTS.md').write_text(report)
    atomic_json(output / 'analysis-reconstruction.json', {'status': 'pass', 'results': file_info(target),
        'report': file_info(output / 'RESULTS.md'), 'scored_conditions': len(score_rows), 'expressions': len(expressions)})
    return {'results': file_info(target), 'report': file_info(output / 'RESULTS.md'), 'arms': arm_rows}
