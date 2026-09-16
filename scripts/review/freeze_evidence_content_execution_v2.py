#!/usr/bin/env python3
"""Freeze a two-GPU allocation amendment without changing scientific inputs."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.general_model_evidence_evaluation import (
    canonical, file_sha, json_bytes, read_json, require, sha, write_output,
)
from scripts.review.freeze_evidence_content_decomposition import (
    WORK, PUBLIC, load_frozen as original_load,
)

PARENT = WORK / 'frozen-01'
PARENT_HASH = 'cf49b35baf350614c77e4e1eef1471a13e9bd1755570acfd62f6515c360cab88'
WINDOW = PUBLIC / 'gpu-window-two-card-01.json'
CODE = tuple('scripts/review/' + name + '.py' for name in (
    'freeze_evidence_content_execution_v2', 'run_evidence_content_execution_v2',
    'test_evidence_content_execution_v2',
))
CHANGED_ARTIFACTS = {'plan.json', 'execution_source.json', 'audit.json', 'FREEZE.md'}


def build():
    require(file_sha(PARENT / 'manifest.json') == PARENT_HASH, 'preparation identity differs')
    parent, contexts, historical = original_load(PARENT)
    manifest = read_json(PARENT / 'manifest.json')
    window = read_json(WINDOW)
    require(window['authorized_now'] and window['selected_physical_devices'] == [1, 2]
            and not window['scientific_inputs_or_threshold_changes'], 'two-GPU authorization differs')
    sources = dict(manifest['source_files'])
    for name, h in {**manifest['artifacts'], 'manifest.json': PARENT_HASH}.items():
        sources[str((PARENT / name).relative_to(ROOT))] = h
    sources[str(WINDOW.relative_to(ROOT))] = file_sha(WINDOW)
    code = dict(parent['code_sha256'])
    for name in CODE: sources[name] = code[name] = file_sha(ROOT / name)
    amendment = {'schema_version': 'evidence-content-execution-amendment/v2', 'author': 'assistant',
        'authorization': window, 'preparation_plan_id': parent['plan_id'],
        'preparation_manifest_sha256': PARENT_HASH, 'selected_gpu_uuids': window['selected_gpu_uuids'],
        'old_device_indices': [0, 1, 2, 3], 'device_indices': [1, 2], 'replica_shift': 1,
        'dynamic_device_growth': False, 'inputs_catalog_protocol_comparisons_unchanged': True,
        'numeric_kernels_and_thresholds_unchanged': True,
        'interpretation': 'Allocation changes only; every context is still scored at batch one, and the replica pass swaps its physical GPU. This run does not assert new four-GPU replication.',
        'new_individual_human_fields': 0}
    plan = deepcopy(parent); plan.pop('plan_id')
    plan['config']['execution']['device_indices'] = [1, 2]
    plan.update(source_files=sources, code_sha256=code, execution_amendment=amendment,
                preparation_plan_id=parent['plan_id'], preparation_manifest_sha256=PARENT_HASH)
    plan['plan_id'] = 'evidence-content-decomposition-' + sha(canonical(plan).encode())
    files = {name: (PARENT / name).read_bytes() for name in manifest['artifacts'] if name not in CHANGED_ARTIFACTS}
    audit = {**read_json(PARENT / 'audit.json'), 'execution_device_indices': [1, 2],
             'unchanged_scientific_artifacts': sorted(files), 'parent_manifest_sha256': PARENT_HASH,
             'allocation_only_amendment': True, 'model_forward_executed': False}
    files.update({'plan.json': json_bytes(plan), 'execution-amendment.json': json_bytes(amendment),
        'execution_source.json': json_bytes({p: {'sha256': h, 'text': (ROOT / p).read_text()} for p, h in code.items()}),
        'audit.json': json_bytes(audit),
        'FREEZE.md': ('# 内容拆分：两卡执行版本\n\n'
            '用户已通知 GPU 窗口，目前仅两卡空闲。独立冻结本次物理 GPU 1、2 的分配；原四卡准备及全部材料保留。'
            '208 提示、416 候选、80 历史重放、30 比较以及五探针／两种 A/B 映射逐字沿用 frozen-01。'
            'FP32 内核、batch size 1、原始及派生数值阈值、八轮检查、查询参考读取顺序均不改。\n\n'
            '两卡分片各覆盖 104 个全量提示；副本轮 shift=1 将每个提示移到另一张物理卡。'
            '后续另外两卡空闲时，本次分片仍按两卡完成；本次不宣称四卡复现。\n\n'
            '准备时的 GPU 暂缓记录保留在原分析协议中；当前启动授权及其作用范围见 execution-amendment.json。'
            '本文件生成时未运行模型，真实运行状态由独立 run-02 记录。\n\n'
            f'计划 ID：`{plan["plan_id"]}`。\n').encode()})
    for p, h in sources.items(): require(file_sha(ROOT / p) == h, 'source changed: ' + p)
    return files, sources


def load_frozen(directory):
    plan, contexts, historical = original_load(directory)
    require(plan['preparation_manifest_sha256'] == PARENT_HASH
            and plan['config']['execution']['device_indices'] == [1, 2], 'wrong allocation version')
    require(read_json(directory / 'execution-amendment.json') == plan['execution_amendment'], 'amendment identity differs')
    for name, h in read_json(PARENT / 'manifest.json')['artifacts'].items():
        if name not in CHANGED_ARTIFACTS:
            require(file_sha(directory / name) == h, 'scientific artifact changed: ' + name)
    return plan, contexts, historical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=WORK / 'frozen-02')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(); target = args.output.resolve()
    require(target.parent == WORK and target.name == 'frozen-02', 'use the independent allocation version')
    files, sources = build()
    if args.check:
        m = read_json(target / 'manifest.json')
        require(m['source_files'] == sources and set(m['artifacts']) == set(files), 'inventory differs')
        for name, raw in files.items(): require((target / name).read_bytes() == raw and m['artifacts'][name] == sha(raw), 'reconstruction differs: ' + name)
        load_frozen(target)
    else:
        files['manifest.json'] = json_bytes({'schema_version': 'evidence-content-decomposition-freeze/v1', 'status': 'frozen',
            'created_at': datetime.now(timezone.utc).isoformat(), 'source_files': sources, 'artifacts': {k: sha(v) for k, v in files.items()}})
        write_output(target, files)
    print(canonical({'status': 'verified' if args.check else 'frozen', 'device_indices': [1, 2],
                     'contexts': 208, 'candidates': 416, 'model_forward_executed': False}))


if __name__ == '__main__': main()
