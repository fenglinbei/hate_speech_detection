"""CPU preparation, provenance, numerical gates and analysis for the 36-input extension.

The reviewed scientific freeze and the completed first run are immutable sources.
Preparation never inventories GPUs; allocation is a separate, explicit future step.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import runpy
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from diagnostics import cross_term_next_token_v1 as old

ROOT = old.ROOT
MATERIAL = ROOT / 'docs/research/experiment-plans/cross-term-behavior-discrimination-v1/frozen-01'
WORK = ROOT / 'reviews/cross-term-behavior-execution-v1'
OLD_FREEZE = ROOT / 'reviews/cross-term-next-token-v1/frozen-01'
OLD_RUN = ROOT / 'reviews/cross-term-next-token-v1/run-01'
MATERIAL_SHA = '9e0e03ac66b4815c95d6466041cf4d886e3a7bf2bf4f1a814f862df88fe0b722'
OLD_FREEZE_SHA = '3fcb4793f42a62f15d66bef4294907c61c3919c8d0c543606682377292e5cbfb'
ENGINEERING = old.PASS_NAMES[:-1]
PASSES = ['historical-bridge'] + ENGINEERING + ['science']
IMPLEMENTATION = [
    'src/diagnostics/cross_term_behavior_execution_v1.py',
    'src/diagnostics/cross_term_behavior_gpu_v1.py',
    'scripts/review/run_cross_term_behavior_v1.py',
    'scripts/review/test_cross_term_behavior_v1.py',
    'scripts/review/audit_cross_term_behavior_v1.py',
]
require, read, sha = old.require, old.read, old.sha
file_info, verify_file, atomic_json = old.file_info, old.verify_file, old.atomic_json
canonical, now, math_module = old.canonical, old.now, old.math_module


def specs():
    padding = {'engineering-left-padding': 'left_to_next_strict_multiple_of_16',
               'engineering-right-padding': 'right_to_next_strict_multiple_of_16'}
    return [dict(pass_id=p, request_order='reverse' if p == ENGINEERING[-1] else 'forward',
                 padding=padding.get(p, 'none'), inputs='bridge' if p == 'historical-bridge' else 'new',
                 prompt_forwards=84 if p == 'historical-bridge' else 36) for p in PASSES]


def qualification_policy():
    return {
        'historical_bridge_max_absolute_margin_difference': 0.0,
        'repeat_and_reverse_order_max_margin_difference': 0.0,
        'padding_max_abs_margin_difference_cap': 0.001,
        'full_vocab_logprob_identity_max_abs_error': 1e-10,
        'new_margin_bound': 'max(0.000001, 2*max(new-input engineering margin differences))',
        'historical_bound': 'Retain each original physical score and its original qualification receipt/bound.',
        'bridge_meaning': 'All 84 reused margins must reproduce exactly on the original UUID/runtime. '
                          'No bridge score replaces an original endpoint; no transport allowance is added.',
        'science_replay': 'All 36 production margins must agree with engineering-reference within the new bound.',
        'failure_action': 'Stop and preserve evidence; no automatic retry, threshold relaxation or historical rescore substitution.',
        'not_qualified': ['another GPU UUID', 'batch size >1', 'KV reuse', 'different runtime/dtype/kernel',
                          'statistical significance', 'practical importance', 'independent confirmation'],
    }


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def inventory_digest(paths):
    return hashlib.sha256(canonical([file_info(p) for p in paths])).hexdigest()


def unique_refs(entries):
    result = {}
    for e in entries:
        require(e['path'] not in result or result[e['path']] == e, 'conflicting file references')
        result[e['path']] = e
    return [result[k] for k in sorted(result)]


def validate_input_sets(new, bridge):
    require(len(new) == 36 and len(bridge) == 84, '36 new / 84 historical inputs required')
    rows = new + bridge
    require(len({r['condition_id'] for r in rows}) == len({r['request_id'] for r in rows}) == 120,
            'input aliases or overlapping new/historical IDs')
    require(len({r['prompt_sha256'] for r in rows}) == 120, 'unexpected duplicate prompt')
    for r in rows:
        require(set(r) == {'request_id', 'condition_id', 'messages', 'chat_prompt', 'prompt_sha256',
                           'input_ids', 'input_ids_sha256', 'prompt_tokens', 'last_input_token_index',
                           'next_token_position', 'candidate_tokens'}, 'scorer input fields differ')
        require(hashlib.sha256(r['chat_prompt'].encode()).hexdigest() == r['prompt_sha256'], 'prompt hash differs')
        require(hashlib.sha256(canonical(r['input_ids'])).hexdigest() == r['input_ids_sha256'], 'token hash differs')
        n = len(r['input_ids'])
        require(n == r['prompt_tokens'] == r['next_token_position'] and r['last_input_token_index'] == n - 1,
                'candidate boundary differs')
        require(r['candidate_tokens'] == {'有': 18830, '无': 42192}, 'answer token mapping differs')
        old.padded_input(r['input_ids'], 'right_to_next_strict_multiple_of_16')


def create_preparation(output):
    """Hash weights on CPU, verify existing evidence and pin code; no CUDA calls."""
    output = Path(output).resolve()
    require(output.parent == WORK and not output.exists(), 'new preparation directory required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'prepare with CUDA_VISIBLE_DEVICES empty')
    require(sha(MATERIAL / 'manifest.json') == MATERIAL_SHA and sha(OLD_FREEZE / 'manifest.json') == OLD_FREEZE_SHA,
            'reviewed/historical freeze identity differs')
    adoption = runpy.run_path(str(ROOT / 'scripts/review/freeze_cross_term_behavior_discrimination_v1.py'))['check'](tokenize=True)
    old_plan, old_inputs = old.check_freeze(OLD_FREEZE, weights=True)
    state = read(OLD_RUN / 'run_manifest.json')
    require(state['status'] == 'complete' and state['worker_exit_code'] == 0 and state['owned_worker_absent'],
            'historical run is not terminal complete')
    historical_checks = old.check_completed(OLD_FREEZE, OLD_RUN, full=True)
    bridges = read(MATERIAL / 'historical-bridges.json')
    source_inputs = {r['request_id']: r for r in old_inputs}
    bridge_inputs = [source_inputs[r['request_id']] for r in bridges['records']]
    new = jsonl(MATERIAL / 'model-inputs.jsonl')
    validate_input_sets(new, bridge_inputs)
    sources = []
    for directory, groups in [(MATERIAL, ['artifacts', 'sources', 'implementation']),
                               (OLD_FREEZE, ['artifacts', 'sources', 'runtime_sources'])]:
        manifest = read(directory / 'manifest.json')
        sources.append(file_info(directory / 'manifest.json'))
        for group in groups:
            sources.extend(manifest[group])
    for name in ['binding.json', 'run_manifest.json', 'qualification.json', 'science-seal.json', 'generation.json']:
        sources.append(file_info(OLD_RUN / name))
    historical = []
    for b, request in zip(bridges['records'], bridge_inputs):
        verify_file(b['score_ref']); verify_file(b['input_ref']); verify_file(b['results_ref'])
        receipt = read(ROOT / b['score_ref']['path'])
        require(all(request[k] == b[k] == receipt[k] for k in
                    ['condition_id', 'request_id', 'prompt_sha256', 'input_ids_sha256']), 'bridge input identity differs')
        historical.append(dict(condition_id=b['condition_id'], request_id=b['request_id'],
                               prompt_sha256=b['prompt_sha256'], score_ref=b['score_ref'],
                               raw_logits=receipt['raw_logits'], physical_score_id=receipt['physical_score_id'],
                               qualification_ref=receipt['readout']['qualification_ref']))
        sources.extend([b['score_ref'], receipt['raw_logits'], receipt['producer']['runtime_identity'], b['results_ref']])
    sources = unique_refs(sources + [file_info(ROOT / n) for n in IMPLEMENTATION])
    for entry in sources:
        verify_file(entry)
    output.mkdir(parents=True)
    shutil.copyfile(MATERIAL / 'model-inputs.jsonl', output / 'model-inputs.jsonl')
    # Preserve each original line, including its JSON encoding and newline.
    old_lines = {json.loads(line)['request_id']: line for line in (OLD_FREEZE / 'model-inputs.jsonl').read_text().splitlines(keepends=True)}
    (output / 'bridge-inputs.jsonl').write_text(''.join(old_lines[r['request_id']] for r in bridge_inputs))
    atomic_json(output / 'historical-index.json', {'records': historical, 'original_run': str(OLD_RUN.relative_to(ROOT)),
        'old_N_inputs_retained_externally': bridges['old_N_inputs'],
        'original_results': bridges['records'][0]['results_ref']}, replace=False)
    atomic_json(output / 'qualification-plan.json', {'passes': specs(), 'acceptance': qualification_policy()}, replace=False)
    old_identity = read(ROOT / historical[0]['score_ref']['path'])['producer']['runtime_identity']
    plan = {'schema_version': 'cross-term-behavior-preparation/v1', 'created_at': now(),
        'status': 'CPU_prepared_GPU_binding_and_acceptance_pending',
        'user_instruction': '请开始下一步，直至GPU任务需求前停止，目前GPU暂未空闲',
        'GPU_launch_authorized_by_this_preparation': False, 'allocation': None,
        'required_allocation': old_plan['allocation'], 'required_producer_identity': old_identity,
        'runtime': old_plan['runtime'], 'runtime_sources': old_plan['runtime_sources'],
        'model_path': old_plan['model_path'], 'model_files': old_plan['model_files'],
        'checkpoint_tensor_names': old_plan['checkpoint_tensor_names'],
        'required_runtime': old_plan['required_runtime'], 'candidate_tokens': old_plan['candidate_tokens'],
        'material_manifest': file_info(MATERIAL / 'manifest.json'), 'historical_manifest': file_info(OLD_FREEZE / 'manifest.json'),
        'analysis_plan': file_info(MATERIAL / 'analysis-plan.json'), 'design': file_info(MATERIAL / 'design.json'),
        'passes': PASSES, 'budget': {'historical_bridge_forwards': 84, 'new_engineering_forwards': 180,
            'new_science_forwards': 36, 'total_prompt_forwards': 300, 'candidate_values': 600,
            'new_scientific_conditions': 36, 'reused_scientific_conditions': 84, 'core_conditions': 120,
            'registered_comparisons': 168, 'additional_generation_forwards': 0},
        'generation_diagnostic': 'No new generation: the unchanged 12 empty baselines retain the old 12/12 diagnostic. '
                                 'This does not qualify free generation on the 36 new prompts; all new legal masses are reported.',
        'analysis_reference_join': 'Only after bridge/engineering/science seals and owned-worker release.',
        'source_policy': 'Never mutate reviewed inputs, old bounds, physical score IDs, old runs or old selectors.',
        'checkpoint_policy': 'Reuse every committed request on same binding/runtime/UUID; resume only an explicit paused run. '
                             'Failed or complete runs cannot restart. STOP checkpoints before the next request.',
    }
    atomic_json(output / 'plan.json', plan, replace=False)
    atomic_json(output / 'cpu-preparation.json', {'status': 'pass', 'adoption': adoption,
        'historical_full_CPU_reconstruction': historical_checks,
        'weight_bytes_hashed': sum(e['bytes'] for e in old_plan['model_files'] if e['path'].endswith('.safetensors')),
        'model_tensors_loaded': 0, 'GPU_inventory_called': False, 'new_GPU_forwards': 0,
        'new_inputs': 36, 'bridge_inputs': 84}, replace=False)
    (output / 'README.md').write_text(preparation_readme(), encoding='utf-8')
    manifest = {'schema_version': 'cross-term-behavior-preparation-manifest/v1', 'created_at': now(),
        'artifacts': [file_info(p) for p in sorted(output.iterdir())], 'sources': sources,
        'model_files': old_plan['model_files'], 'GPU_bound': False, 'GPU_qualification': False,
        'new_GPU_forwards': 0}
    atomic_json(output / 'manifest.json', manifest, replace=False)
    check_prepared(output)
    return {'status': 'CPU_prepared_only', 'manifest': file_info(output / 'manifest.json'), 'budget': plan['budget']}


def preparation_readme():
    return '''# 第二轮跨词项行为实验：GPU 前准备

36 条新输入、84 条历史端点和 168 项比较沿用人工已采纳的科学冻结包；全部提示与答案保持原字节。这里冻结执行实现和数值验收方案，实际设备绑定、GPU 验收及新评分均未进行。没有启动或排队 GPU 任务。

## 执行顺序与预算

1. 在新获准的 GPU 窗口中，核验第一轮使用的 GPU 0 UUID、驱动、Python、权重及源码完全相同，并确认该卡空闲。只需要该张 L20，不需要四卡。
2. 重放全部 84 条历史端点。主分数 m=z无−z有 必须逐项精确重现，最大差为 0；不覆盖旧分数，不把桥接计为新增科学条件。失败即停止。
3. 对 36 条新输入各做 reference、repeat、左 padding、右 padding、逆请求顺序五遍，共 180 次 forward。重复和逆序误差必须为 0，padding 上限沿用 0.001。新界 b=max(0.000001,2×新输入最大工程差)，不继承旧界。
4. 封存桥接和资格凭据后，正式评分新 36 条，各一次 forward，并与 reference 比较。每次只取同一全词表 FP32 向量，两个合法答案共享一次 forward，FP64 归约。
5. 封存原始评分并释放 worker，才允许 CPU 分析读入人工参考，复用 84 个旧物理分数及其原界，计算全部 168 项比较。共享物理项先抵消，再累加误差界；修复、损伤、连续移动与未决分开报告。

合计 **300 次 prompt-only forward / 600 个候选值**，包括 84 桥接 + 180 工程 + 36 正式。全词表向量约 183 MB（十进制），另加凭据。旧 36 条 N 条件通过历史结果完整保留，不进入新核心。原 12 个空示例基线的自由生成诊断保留为历史证据，本轮无额外生成；不能据此声称新输入自由生成已验收。

旧端点的界仍为 0.00045013427734375，保留其原资格文件；新端点界等待新 GPU 工程验收。桥接只认可同设备同运行环境的精确主分数重现，不增加事后容差。任何失败保留证据、停止，不自动重试或放宽阈值。

## 入口

CPU 核查：
```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 .conda/stage1-p0/bin/python scripts/review/run_cross_term_behavior_v1.py validate
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 .conda/stage1-p0/bin/python scripts/review/test_cross_term_behavior_v1.py
```

以下步骤必须等待用户重新给出可用窗口；当前未执行，也没有空闲轮询：
```bash
.conda/stage1-p0/bin/python scripts/review/run_cross_term_behavior_v1.py bind --authorization-note '填写实际新授权原话'
.conda/stage1-p0/bin/python scripts/review/run_cross_term_behavior_v1.py run
CUDA_VISIBLE_DEVICES='' .conda/stage1-p0/bin/python scripts/review/run_cross_term_behavior_v1.py check
CUDA_VISIBLE_DEVICES='' .conda/stage1-p0/bin/python scripts/review/run_cross_term_behavior_v1.py analyze
CUDA_VISIBLE_DEVICES='' .conda/stage1-p0/bin/python scripts/review/audit_cross_term_behavior_v1.py results
```

bind 新建独立 bound-01.json，prepared-01 保持不变；run 只读取这个绑定。暂停用新 run 目录中的 STOP 文件；只在显式移除 STOP 并重新授权后用 run --resume，已完成和失败的 run 拒绝启动。不会重启任何历史 run。

所有材料都是三个已暴露词项上的开发条件；A/B、同词/异词改变整包文字，不能解释为纯义项或规则适用性的因果效应。没有新增独立确认、内部干预或机制结论。CPU 通过不等于 GPU 数值验收通过。
'''


def check_prepared(prepared, weights=False):
    prepared = Path(prepared).resolve()
    manifest = read(prepared / 'manifest.json')
    for entry in manifest['artifacts'] + manifest['sources']:
        verify_file(entry)
    require({p.name for p in prepared.iterdir()} == {'manifest.json'} | {Path(e['path']).name for e in manifest['artifacts']},
            'preparation inventory differs')
    if weights:
        for entry in manifest['model_files']:
            verify_file(entry)
    plan = read(prepared / 'plan.json')
    require(plan['runtime'] == old.runtime_snapshot(), 'Python/runtime identity differs')
    require(plan['allocation'] is None and not plan['GPU_launch_authorized_by_this_preparation'], 'CPU preparation claims GPU authority')
    require(plan['passes'] == PASSES and read(prepared / 'qualification-plan.json') ==
            {'passes': specs(), 'acceptance': qualification_policy()}, 'qualification plan differs')
    new, bridge = jsonl(prepared / 'model-inputs.jsonl'), jsonl(prepared / 'bridge-inputs.jsonl')
    validate_input_sets(new, bridge)
    return plan, {'new': new, 'bridge': bridge}


def bind_gpu(prepared, output, authorization_note):
    """Future explicit action. This is the first GPU inventory in this implementation."""
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    require(output.parent == WORK and not output.exists(), 'new bound file required')
    require(authorization_note and authorization_note.strip(), 'record the actual new GPU-window authorization')
    plan, _ = check_prepared(prepared, weights=True)
    inventory = old.gpu_inventory()
    device = old.idle_device(plan['required_allocation']['index'], inventory)
    allocation = {k: device[k] for k in plan['required_allocation']}
    require(allocation == plan['required_allocation'], 'original GPU/driver identity required for historical bridge')
    bound = {'schema_version': 'cross-term-behavior-device-binding/v1', 'bound_at': now(),
        'preparation_manifest': file_info(prepared / 'manifest.json'), 'allocation': allocation,
        'authorization_note': authorization_note, 'inventory_at_binding': inventory, 'GPU_forward_performed': False}
    atomic_json(output, bound, replace=False)
    return bound


def check_bound(prepared, bound_path, weights=False):
    plan, requests = check_prepared(prepared, weights)
    bound = read(bound_path)
    require(bound['preparation_manifest'] == file_info(Path(prepared) / 'manifest.json'), 'device binding preparation differs')
    require(bound['allocation'] == plan['required_allocation'] and bool(bound['authorization_note'].strip()), 'GPU binding differs')
    return dict(plan, allocation=bound['allocation']), requests


def check_run_binding(prepared, run):
    run = Path(run)
    binding = read(run / 'binding.json')
    require(binding['preparation_manifest'] == file_info(Path(prepared) / 'manifest.json'), 'run preparation differs')
    verify_file(binding['device_binding'])
    plan, requests = check_bound(prepared, ROOT / binding['device_binding']['path'])
    require(binding['allocation'] == plan['allocation'] and binding['runtime'] == plan['runtime'], 'run device/runtime differs')
    state = read(run / 'run_manifest.json')
    require(state['binding'] == file_info(run / 'binding.json') and state['preparation_manifest'] == binding['preparation_manifest'],
            'state/run binding differs')
    return plan, requests


def pass_requests(prepared, spec):
    rows = jsonl(Path(prepared) / ('bridge-inputs.jsonl' if spec['inputs'] == 'bridge' else 'model-inputs.jsonl'))
    return old.request_order(rows, spec)


def check_producer(run, record, expected_identity):
    binding = read(Path(run) / 'binding.json')
    invocations = {v['invocation_id']: v for v in read(Path(run) / 'run_manifest.json')['invocations']}
    producer = record['producer']
    require(producer['invocation_id'] in invocations, 'unregistered producer invocation')
    invocation = invocations[producer['invocation_id']]
    require(producer['pid'] == invocation['worker_pid'] and producer['uuid'] == binding['allocation']['uuid'], 'producer PID/UUID differs')
    require(producer['runtime_identity'] == invocation['runtime_identity'], 'producer identity binding differs')
    verify_file(producer['runtime_identity'])
    require(read(ROOT / producer['runtime_identity']['path']) == expected_identity, 'producer numeric runtime differs from original')


def check_pass(prepared, run, spec, qualification=None, full=True):
    requests = pass_requests(prepared, spec)
    directory = Path(run) / 'scores' / spec['pass_id']
    expected = {r['request_id'] for r in requests}
    require({p.stem for p in directory.glob('*.json')} == expected and
            {p.stem for p in directory.glob('*.npy')} == expected, 'pass incomplete or has unexpected records')
    plan = read(Path(prepared) / 'plan.json')
    identity = read(ROOT / plan['required_producer_identity']['path'])
    records = []
    for request in requests:
        record = old.check_record(run, spec, request, sha(Path(run) / 'binding.json'), qualification, full)
        check_producer(run, record, identity)
        records.append(record)
    return records


def read_historical(prepared, full=True):
    index = read(Path(prepared) / 'historical-index.json')
    requests = {r['condition_id']: r for r in jsonl(Path(prepared) / 'bridge-inputs.jsonl')}
    old_run = ROOT / index['original_run']
    q = old.qualification_with_ref(old_run)
    records = []
    for entry in index['records']:
        verify_file(entry['score_ref']); verify_file(entry['raw_logits'])
        record = old.check_record(old_run, {'pass_id': 'science', 'padding': 'none'}, requests[entry['condition_id']],
                                  sha(old_run / 'binding.json'), q, full)
        require(record['physical_score_id'] == entry['physical_score_id'] and record['prompt_sha256'] == entry['prompt_sha256']
                and record['readout']['qualification_ref'] == entry['qualification_ref'], 'historical provenance differs')
        records.append(record)
    return records


def bridge_values(historical, replay):
    require(len(historical) == len(replay) == 84 and set(historical) == set(replay), 'historical bridge coverage differs')
    differences = {k: abs(historical[k]['m'] - replay[k]['m']) for k in historical}
    require(all(math.isfinite(v) for v in differences.values()) and max(differences.values()) == 0.0,
            'historical bridge numerical gate failed; original margins must reproduce exactly')
    identity_error = max(abs(r['m'] - (r['log_p_no'] - r['log_p_yes'])) for r in replay.values())
    require(math.isfinite(identity_error) and identity_error <= 1e-10, 'bridge log probability identity gate failed')
    return {'status': 'pass', 'historical_inputs': 84, 'max_margin_difference': max(differences.values()),
            'max_log_probability_identity_error': identity_error, 'transport_allowance': 0.0,
            'old_physical_scores_replaced': False, 'condition_differences': differences}


def build_bridge(prepared, run, full=True):
    originals = read_historical(prepared, full)
    replays = check_pass(prepared, run, specs()[0], full=full)
    values = bridge_values({r['condition_id']: r['readout'] for r in originals},
                           {r['condition_id']: r['readout'] for r in replays})
    values.update(binding_sha256=sha(Path(run) / 'binding.json'),
                  original_index=file_info(Path(prepared) / 'historical-index.json'),
                  replay_receipts_sha256=inventory_digest([old.record_paths(run, 'historical-bridge', r['request_id'])[0] for r in replays]))
    return values


def qualification_values(by_pass):
    require(set(by_pass) == set(ENGINEERING), 'engineering pass coverage incomplete')
    reference = by_pass['engineering-reference']
    require(len(reference) == 36, 'new-input qualification requires all 36 inputs')
    differences = {}; identity_error = 0.0
    for name, scores in by_pass.items():
        require(set(scores) == set(reference), 'engineering input coverage differs')
        for score in scores.values():
            require(all(math.isfinite(score[k]) for k in ['m', 'z_no', 'z_yes', 'log_p_no', 'log_p_yes',
                    'legal_mass', 'log_legal_mass', 'pair_support_no']), 'nonfinite readout')
            identity_error = max(identity_error, abs(score['m'] - (score['log_p_no'] - score['log_p_yes'])))
        if name != 'engineering-reference':
            differences[name] = max(abs(scores[k]['m'] - reference[k]['m']) for k in reference)
    require(differences['engineering-repeat'] == differences[ENGINEERING[-1]] == 0.0, 'repeat/order numerical gate failed')
    require(max(differences[k] for k in ENGINEERING[2:4]) <= 0.001, 'padding numerical gate failed')
    require(identity_error <= 1e-10, 'log probability identity gate failed')
    return {'status': 'pass', 'engineering_requests': 180, 'unique_inputs': 36,
            'max_margin_differences': differences, 'max_log_probability_identity_error': identity_error,
            'margin_error_bound': max(0.000001, 2 * max(differences.values())),
            'bound_meaning': 'new-input engineering envelope; not an effect threshold or statistical interval'}


def build_qualification(prepared, run, full=True):
    require(build_bridge(prepared, run, full) == read(Path(run) / 'bridge-seal.json'), 'bridge seal differs')
    by_pass, inventories = {}, []
    for spec in specs()[1:-1]:
        records = check_pass(prepared, run, spec, full=full)
        by_pass[spec['pass_id']] = {r['condition_id']: r['readout'] for r in records}
        inventories.append({'pass_id': spec['pass_id'], 'receipt_inventory_sha256': inventory_digest([
            old.record_paths(run, spec['pass_id'], r['request_id'])[0] for r in records])})
    result = qualification_values(by_pass)
    result.update(binding_sha256=sha(Path(run) / 'binding.json'),
        preparation_manifest=file_info(Path(prepared) / 'manifest.json'),
        bridge_seal=file_info(Path(run) / 'bridge-seal.json'), pass_receipts=inventories,
        qualification_policy=qualification_policy())
    return result


def check_science(prepared, run, full=True):
    q = old.qualification_with_ref(run)
    records = check_pass(prepared, run, specs()[-1], q, full)
    refs = {r['condition_id']: r['readout'] for r in check_pass(prepared, run, specs()[1], full=False)}
    maximum = max(abs(r['readout']['m'] - refs[r['condition_id']]['m']) for r in records)
    require(maximum <= q['margin_error_bound'], 'science replay exceeds new qualification bound')
    return records, {'science_requests': 36, 'max_reference_margin_difference': maximum,
        'margin_error_bound': q['margin_error_bound'], 'qualification': file_info(Path(run) / 'qualification.json'),
        'raw_inventory_sha256': inventory_digest([old.record_paths(run, 'science', r['request_id'])[0] for r in records])}


def seal(path, value):
    if Path(path).exists():
        require(read(path) == value, f'sealed receipt reconstruction differs: {path}')
    else:
        atomic_json(path, value, replace=False)


def check_completed(prepared, run, full=True):
    check_run_binding(prepared, run)
    require({p.name for p in (Path(run) / 'scores').iterdir()} == set(PASSES), 'unexpected pass inventory')
    require(build_qualification(prepared, run, full) == read(Path(run) / 'qualification.json'), 'qualification reconstruction differs')
    _, science = check_science(prepared, run, full)
    require(science == read(Path(run) / 'science-seal.json'), 'science seal differs')
    return {'status': 'pass', 'passes': 7, 'prompt_forwards': 300, 'candidate_values': 600,
            'reused_core_endpoints': 84, 'new_core_endpoints': 36,
            'bridge': file_info(Path(run) / 'bridge-seal.json'),
            'qualification': file_info(Path(run) / 'qualification.json'),
            'science': file_info(Path(run) / 'science-seal.json')}


def resume_check(prepared, run):
    """Validate all saved requests before loading weights in a resumed worker."""
    plan, _ = check_run_binding(prepared, run)
    state = read(Path(run) / 'run_manifest.json')
    require(state['status'] == 'paused', 'only a paused run may resume')
    completed = state['completed_passes']
    require(completed == PASSES[:len(completed)], 'sealed passes are not an ordered prefix')
    identity = read(ROOT / plan['required_producer_identity']['path'])
    count = 0
    score_dir = Path(run) / 'scores'
    require(not score_dir.exists() or {p.name for p in score_dir.iterdir()} <= set(PASSES), 'unknown saved pass')
    for i, spec in enumerate(specs()):
        requests = {r['request_id']: r for r in pass_requests(prepared, spec)}
        directory = score_dir / spec['pass_id']
        saved = sorted(directory.glob('*.json'))
        require({p.stem for p in saved} <= set(requests), 'unknown checkpoint request')
        require(i <= len(completed) or not saved, 'checkpoint after an unsealed pass')
        q = old.qualification_with_ref(run) if spec['pass_id'] == 'science' and saved else None
        for path in saved:
            record = old.check_record(run, spec, requests[path.stem], sha(Path(run) / 'binding.json'), q, full=True)
            check_producer(run, record, identity)
            count += 1
        if spec['pass_id'] in completed:
            check_pass(prepared, run, spec, q, full=False)
    if 'historical-bridge' in completed:
        require(build_bridge(prepared, run, full=True) == read(Path(run) / 'bridge-seal.json'), 'saved bridge seal differs')
    if ENGINEERING[-1] in completed:
        require(build_qualification(prepared, run, full=False) == read(Path(run) / 'qualification.json'), 'saved qualification differs')
    if 'science' in completed:
        require(check_science(prepared, run, full=False)[1] == read(Path(run) / 'science-seal.json'), 'saved science seal differs')
    return {'status': 'pass', 'saved_requests': count, 'sealed_passes': completed, 'new_GPU_forwards': 0}


def assert_released(run):
    state = read(Path(run) / 'run_manifest.json')
    require(state['status'] == 'complete' and state.get('worker_exit_code') == 0 and state.get('owned_worker_absent'),
            'reference join requires completed run and released worker')
    verify_file(state['resource_release'])
    released = read(ROOT / state['resource_release']['path'])
    require(released['owned_worker_absent'] and released['worker_exit_code'] == 0 and
            released['worker_pid'] == state['invocations'][-1]['worker_pid'], 'resource-release evidence differs')


def analysis_values(conditions, analysis_plan, records):
    """Pure CPU join; production caller is guarded by raw seals and release."""
    maths = math_module(); refs = {r['query_id']: r for r in analysis_plan['references']}
    by_condition = {r['condition_id']: r for r in records}
    require(len(by_condition) == len(records) == len(conditions) == 120, 'analysis endpoint inventory differs')
    require(set(by_condition) == {c['condition_id'] for c in conditions}, 'analysis endpoints differ')
    scores, rows, expressions = {}, [], []
    for condition in conditions:
        record = by_condition[condition['condition_id']]
        require(record['prompt_sha256'] == condition['prompt_sha256'], 'analysis prompt identity differs')
        score = dict(record['readout'], physical_score_id=record['physical_score_id'], prompt_sha256=record['prompt_sha256'])
        require(score['margin_error_bound'] is not None and score['qualification_ref'], 'analysis has unqualified input')
        ref = refs[condition['query_id']]; answer = ref['human_reference']['task_label']
        require(ref['original_gold'] is None and ref['original_correct'] is None, 'synthetic reference provenance changed')
        scores[condition['condition_id']] = score
        rows.append(dict(condition, **score, adopted_reference=answer,
            adopted_severity=ref['human_reference']['attack_severity'], original_gold=None, original_correct=None,
            **maths.reference_status(score, answer)))
    require(len(analysis_plan['comparisons']) == 168, 'registered comparison inventory differs')
    for comparison in analysis_plan['comparisons']:
        effect = maths.linear_effect(comparison['terms'], scores)
        ref = refs[comparison['query_id']]['human_reference']['task_label']
        sign = comparison['reference_direction_sign']
        require(sign == (1 if ref == '无' else -1), 'reference orientation differs')
        transition = None
        if len(comparison['terms']) == 2 and sorted(t['coefficient'] for t in comparison['terms']) == [-1, 1]:
            control = next(scores[t['condition_id']] for t in comparison['terms'] if t['coefficient'] == -1)
            treatment = next(scores[t['condition_id']] for t in comparison['terms'] if t['coefficient'] == 1)
            transition = maths.transition(control, treatment, ref)
        aligned = sign * effect['value']
        expressions.append(dict(comparison, effect=effect, reference_aligned_change=aligned,
            reference_aligned_resolution=maths.resolution(aligned, effect['bound']),
            verified_classification_transition=transition))
    arms = []
    for lex, demo in sorted({(r['lexicon_arm'], r['demo_arm']) for r in rows}):
        subset = [r for r in rows if (r['lexicon_arm'], r['demo_arm']) == (lex, demo)]
        require(len(subset) == 12, 'unequal arm query coverage')
        arms.append({'lexicon_arm': lex, 'demo_arm': demo, 'conditions': 12,
            'raw_correct': sum(r['raw_correct'] for r in subset),
            'conservative_correct': sum(r['conservative_correct'] for r in subset),
            'numerical_unresolved': sum(r['resolution'] == 'numerical_unresolved' for r in subset),
            'exact_ties': sum(r['exact_tie'] for r in subset),
            'min_legal_mass': min(r['legal_mass'] for r in subset),
            'mean_legal_mass': math.fsum(r['legal_mass'] for r in subset) / 12,
            'independent_samples': False})
    summaries = []
    for kind in sorted({e['kind'] for e in expressions}):
        subset = [e for e in expressions if e['kind'] == kind]
        families = []
        for family in sorted({e['family_id'] for e in subset}):
            group = [e for e in subset if e['family_id'] == family]
            families.append({'family_id': family, 'comparisons': len(group),
                'mean_effect': math.fsum(e['effect']['value'] for e in group) / len(group),
                'mean_reference_aligned_change': math.fsum(e['reference_aligned_change'] for e in group) / len(group),
                'mean_bound': math.fsum(e['effect']['bound'] for e in group) / len(group)})
        summaries.append({'kind': kind, 'comparisons': len(subset), 'family_weighting': 'equal', 'families': families,
            'family_equal_mean_effect': math.fsum(f['mean_effect'] for f in families) / len(families),
            'family_equal_mean_reference_aligned_change': math.fsum(f['mean_reference_aligned_change'] for f in families) / len(families),
            'family_equal_bound': math.fsum(f['mean_bound'] for f in families) / len(families),
            'directions': dict(Counter(e['reference_aligned_resolution'] for e in subset)),
            'classification_transitions': dict(Counter(e['verified_classification_transition'] for e in subset
                                                      if e['verified_classification_transition'] is not None))})
    return {'scores': rows, 'expressions': expressions, 'arm_readouts': arms, 'family_equal_summaries': summaries}


def write_exact(path, payload):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == payload, f'output byte reconstruction differs: {path}')
    else:
        with path.open('xb') as stream:
            stream.write(payload)


def tsv_bytes(rows, fields):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter='\t', lineterminator='\n', extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def analyze(prepared, run, output):
    prepared, run, output = Path(prepared), Path(run), Path(output)
    assert_released(run)
    checked = check_completed(prepared, run, full=True)
    # First parsing of reference-bearing files in the execution lifecycle occurs here.
    plan = read(prepared / 'plan.json')
    analysis_plan = read(ROOT / plan['analysis_plan']['path'])
    conditions = read(ROOT / plan['design']['path'])['conditions']
    historical = read_historical(prepared, full=False)
    science = check_pass(prepared, run, specs()[-1], old.qualification_with_ref(run), full=False)
    result = analysis_values(conditions, analysis_plan, historical + science)
    result.update(schema_version='cross-term-behavior-results/v1', checks=checked,
        preparation_manifest=file_info(prepared / 'manifest.json'), binding=file_info(run / 'binding.json'),
        analysis_plan=plan['analysis_plan'], resource_release=read(run / 'run_manifest.json')['resource_release'],
        historical_N=read(prepared / 'historical-index.json')['old_N_inputs_retained_externally'],
        historical_results=read(prepared / 'historical-index.json')['original_results'],
        scope='Three outcome-exposed development terms; correlated conditions, no independent confirmation, '
              'no binary rule-fit causal groups or pure same/other-word semantic manipulation.',
        additional_generation_diagnostic=False)
    output.mkdir(parents=True, exist_ok=True)
    write_exact(output / 'results.json', (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())
    write_exact(output / 'scores.tsv', tsv_bytes(result['scores'], ['condition_id', 'query_short_id', 'family_id',
        'lexicon_arm', 'demo_arm', 'input_status', 'm', 'margin_error_bound', 'raw_prediction', 'resolution',
        'adopted_reference', 'raw_correct', 'conservative_correct', 'legal_mass', 'physical_score_id']))
    flat = [dict(e, value=e['effect']['value'], bound=e['effect']['bound'], resolution=e['effect']['resolution']) for e in result['expressions']]
    write_exact(output / 'comparisons.tsv', tsv_bytes(flat, ['comparison_id', 'kind', 'query_short_id', 'family_id',
        'priority', 'exposure', 'value', 'bound', 'resolution', 'reference_aligned_change',
        'reference_aligned_resolution', 'verified_classification_transition']))
    lines = ['# 第二轮跨词项行为评分结果', '', '84 个旧端点保留原评分身份与原界，36 个新端点使用新资格凭据。全部 168 项比较见 comparisons.tsv，全部 120 个条件见 scores.tsv。', '',
             '| 词典 | 示例 | 保守正确 / 12 | 数值未决 |', '|---|---|---:|---:|']
    lines += [f"| {r['lexicon_arm']} | {r['demo_arm']} | {r['conservative_correct']} | {r['numerical_unresolved']} |" for r in result['arm_readouts']]
    lines += ['', '分类修复／损伤与连续分数移动分别报告。旧 36 条 N 条件仍在 historical_results 所指的历史结果内。所有条件共享 12 条查询与三个已暴露词项，不是独立确认样本；不作纯义项、纯规则适用性或内部机制推断。原始 Gold 与 original_correct 保持空。']
    write_exact(output / 'RESULTS.md', ('\n'.join(lines) + '\n').encode())
    seal(output / 'manifest.json', {'schema_version': 'cross-term-behavior-result-seal/v1',
        'artifacts': [file_info(output / n) for n in ['results.json', 'scores.tsv', 'comparisons.tsv', 'RESULTS.md']],
        'preparation': result['preparation_manifest'], 'binding': result['binding'], 'checks': checked,
        'reference_join_performed': True, 'GPU_forward_performed_by_analysis': False})
    return {'status': 'pass', 'manifest': file_info(output / 'manifest.json'), 'scores': 120, 'expressions': 168}
