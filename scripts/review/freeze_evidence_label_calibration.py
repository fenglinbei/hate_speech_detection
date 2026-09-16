#!/usr/bin/env python3
"""Freeze original, normalized contextual calibration and two label mappings."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.general_model_evidence_evaluation import (
    EXPERIMENT, Sources, canonical, csv_bytes, file_sha, json_bytes, jsonl,
    local_path, read_json, read_lines, require, sha, write_output,
)
from diagnostics.evidence_label_calibration import ENCODINGS, PROBES, SCORE_MODES
from scripts.review.freeze_evidence_matched_inputs import load_frozen as load_parent, CONDITIONS, FORBIDDEN
from scripts.review.prepare_evidence_matched_materials import layout
from scripts.review.run_evidence_interventions import DESCRIPTOR, SCHEDULE

BASE = ROOT / EXPERIMENT
WORK = BASE / 'reviews/analysis-freeze-20260914/label-calibration-v1'
PARENT_MANIFEST = '922ab56320a3f64cbba3e371855de9b4aa716115a318521f1af2b50f54fd6462'
QUERY = '\n\n待判断文本（JSON 字符串）：\n'
ORIGINAL_RULE = '只输出一个 JSON 字符串："hate" 或 "non-hate"。'
MAPPINGS = {'original': {'hate': 'hate', 'non-hate': 'non-hate'},
            'ab_forward': {'hate': 'A', 'non-hate': 'B'},
            'ab_reverse': {'hate': 'B', 'non-hate': 'A'}}
NEW_CODE = ('scripts/review/freeze_evidence_label_calibration.py',
            'scripts/review/run_evidence_label_calibration.py',
            'scripts/review/analyze_evidence_label_calibration.py',
            'scripts/review/test_evidence_label_calibration.py',
            'src/diagnostics/evidence_label_calibration.py',
            'src/diagnostics/evidence_label_calibration_execution.py')


def no_gold(value):
    if isinstance(value, dict):
        require(not FORBIDDEN & value.keys(), 'query reference in scoring inputs')
        for v in value.values(): no_gold(v)
    elif isinstance(value, list):
        for v in value: no_gold(v)


def transformed_messages(messages, encoding, probe=None):
    require([m['role'] for m in messages] == ['system', 'user'], 'unexpected message roles')
    result = deepcopy(messages)
    system, user = (m['content'] for m in messages)
    require(user.count(QUERY) == 1 and system.count(ORIGINAL_RULE) == 1, 'ambiguous prompt structure')
    prefix, query_json = user.split(QUERY)
    require(type(json.loads(query_json)) is str and json.dumps(json.loads(query_json), ensure_ascii=False) == query_json,
            'query must be exactly one canonical JSON string')
    mapping = MAPPINGS[encoding]
    if encoding != 'original':
        rule = (f'类别编码约定：仇恨言论对应 "{mapping["hate"]}"，非仇恨言论对应 "{mapping["non-hate"]}"。'
                '只输出一个 JSON 字符串："A" 或 "B"。')
        result[0]['content'] = system.replace(ORIGINAL_RULE, rule)
        matches = list(re.finditer(r'^输出：("hate"|"non-hate")$', prefix, flags=re.M))
        require(len(matches) in (9, 10), 'unexpected demonstration answer count')
        prefix = re.sub(r'^输出：("hate"|"non-hate")$',
                        lambda m: '输出：' + json.dumps(mapping[json.loads(m[1])]), prefix, flags=re.M)
        restored = re.sub(r'^输出：("A"|"B")$',
                          lambda m: '输出：' + json.dumps({v: k for k, v in mapping.items()}[json.loads(m[1])]), prefix, flags=re.M)
        require(restored == user.split(QUERY)[0], 'non-answer demo text changed')
    if probe is not None:
        require(encoding == 'original' and probe in dict(PROBES), 'unregistered probe/encoding')
        query_json = json.dumps(dict(PROBES)[probe], ensure_ascii=False)
    result[1]['content'] = prefix + QUERY + query_json
    return result


def make_catalog(original, tokenizer):
    result = {}
    for encoding, task in ENCODINGS.items():
        rows = deepcopy(original)
        for c in rows:
            answer = json.dumps(MAPPINGS[encoding][c['candidate_id']])
            tokens = tokenizer.encode(answer, add_special_tokens=False)
            c.update(canonical_answer=answer, answer_sha256=sha(answer.encode()), answer_token_ids=tokens,
                     answer_token_ids_sha256=sha(canonical(tokens).encode()), answer_tokens=len(tokens))
        result[task] = rows
    require(result['hate'] == original, 'original candidate bytes changed')
    require(all(c['answer_tokens'] == 2 for t in ('hate_ab_forward', 'hate_ab_reverse') for c in result[t]), 'A/B lengths differ')
    return result


def input_context(source, rich, tokenizer, catalog, encoding, probe):
    messages = transformed_messages(source['messages'], encoding, probe)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    tokens = encoded['input_ids']
    mapping = MAPPINGS[encoding]
    materials = {'demo:' + d: {'source': {'text': rich['texts_by_demo'][d],
                 'original_answer': {'hate': mapping[rich['answers_by_demo'][d]]}}} for d in rich['demo_ids']}
    positions = layout(prompt, encoded['offset_mapping'], rich['demo_ids'], materials, None)
    if encoding == 'original' and probe is None:
        require(prompt == source['prompt_text'] and tokens == source['prompt_token_ids'] and positions == rich['layout'], 'original replay differs')
    for candidate in catalog[ENCODINGS[encoding]]:
        require(tokenizer.encode(prompt + candidate['canonical_answer'], add_special_tokens=False) == tokens + candidate['answer_token_ids'], 'full-label boundary changed')
    require(len(tokens) + max(c['answer_tokens'] for c in catalog[ENCODINGS[encoding]]) + 65 <= 8192, 'padded sequence too long')
    item = {'record_id': f'ELC-{source["query_id"]}:{encoding}:{source["condition"]}:{probe or "real"}',
            'query_id': source['query_id'], 'task': ENCODINGS[encoding], 'semantic_task': 'hate',
            'condition': source['condition'] + ('@' + probe if probe else ''), 'root_condition': source['condition'],
            'family': source['family'], 'encoding': encoding, 'probe_id': probe, 'messages': messages,
            'prompt_text': prompt, 'prompt_sha256': sha(prompt.encode()), 'prompt_tokens': len(tokens),
            'prompt_token_ids': tokens, 'prompt_token_ids_sha256': sha(canonical(tokens).encode()),
            'protocol_id': 'ELC-' + source['query_id'] + '-hate-v1',
            'baseline_replay': encoding == 'original' and probe is None, 'tokenization_status': 'verified',
            'source_record_id': source['record_id'], 'source_context_sha256': source['context_sha256']}
    no_gold(item)
    item['context_sha256'] = sha(canonical(item).encode())
    geometry = {'record_id': item['record_id'], 'query_id': item['query_id'], 'condition': source['condition'],
                'encoding': encoding, 'probe_id': probe, 'layout': positions, 'prompt_tokens': len(tokens),
                'demo_ids': rich['demo_ids'], 'texts_by_demo_sha256': sha(canonical(rich['texts_by_demo']).encode()),
                'entry_ids': rich['entry_ids'], 'prompt_sha256': item['prompt_sha256']}
    return item, geometry


def load_frozen(directory):
    """Verify source bytes and load the gold-free plan only."""
    manifest = read_json(directory / 'manifest.json')
    require(manifest['schema_version'] == 'evidence-label-calibration-freeze/v1' and manifest['status'] == 'frozen', 'wrong freeze schema')
    require({p.name for p in directory.iterdir()} == {*manifest['artifacts'], 'manifest.json'}, 'unexpected freeze files')
    for name, h in manifest['artifacts'].items():
        require(Path(name).name == name and file_sha(directory / name) == h, 'frozen artifact changed: ' + name)
    for name, h in manifest['source_files'].items():
        require(file_sha(local_path(ROOT, name)) == h, 'frozen source changed: ' + name)
    plan = read_json(directory / 'plan.json'); identity = dict(plan); pid = identity.pop('plan_id')
    require(pid == 'evidence-label-calibration-' + sha(canonical(identity).encode()), 'plan identity differs')
    contexts = read_lines(directory / 'contexts.jsonl'); historical = read_lines(directory / 'historical-selected.jsonl')
    require(len(contexts) == 288 and len(historical) == sum(c['baseline_replay'] for c in contexts) == 36, 'input coverage differs')
    require(sha(canonical(contexts).encode()) == plan['contexts_sha256'] and sha(canonical(historical).encode()) == plan['historical_selected_sha256'], 'input digest differs')
    expected = {(q, c, e, p) for q in ('541', '3169') for c in CONDITIONS
                for e in ENCODINGS for p in ([None] + [p for p, _ in PROBES] if e == 'original' else [None])}
    require({(c['query_id'], c['root_condition'], c['encoding'], c['probe_id']) for c in contexts} == expected, 'factorial coverage differs')
    for c in contexts:
        no_gold(c)
        require(sha(canonical({k: v for k, v in c.items() if k != 'context_sha256'}).encode()) == c['context_sha256'], 'context digest differs')
    return plan, contexts, historical


def build():
    sources = Sources(ROOT)
    def take(path, expected=None): return sources.verify(path, expected or file_sha(path))
    pointer = read_json(take(BASE / 'matched-input-results-v1/current.json'))
    require(pointer['status'] == 'complete' and pointer['freeze_manifest_sha256'] == PARENT_MANIFEST, 'wrong parent result')
    folder, run = BASE / pointer['freeze_path'], BASE / pointer['run_path']
    manifest = read_json(take(folder / 'manifest.json', PARENT_MANIFEST))
    for name, h in manifest['source_files'].items(): take(local_path(ROOT, name), h)
    for name, h in manifest['artifacts'].items(): take(folder / name, h)
    parent, originals, _ = load_parent(folder)
    state = read_json(take(run / 'run_manifest.json', pointer['run_manifest_sha256']))
    require(state['status'] == 'complete' and state['numerical_validation_passed'] and state['plan_id'] == parent['plan_id'], 'parent run unsealed')
    old_raw = read_lines(take(run / parent['raw_pass'] / 'scores.jsonl', pointer['raw_scores_sha256']))
    rich_pointer = read_json(take(BASE / 'position-length-materials-v1/current.json'))
    rich_folder = BASE / rich_pointer['materials_path']
    rich = {sha(canonical(r).encode()): r for r in read_lines(take(rich_folder / 'contexts.jsonl'))}
    for flag in ('USE_TORCH', 'USE_TF', 'USE_FLAX', 'USE_TORCH_XLA'): os.environ[flag] = '0'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'models/base/Qwen3-8B'), local_files_only=True, trust_remote_code=False, use_fast=True)
    catalog = make_catalog(parent['catalog']['hate'], tokenizer)
    contexts, geometries = [], []
    # The complete original frame is first, then all NCC probes, then forward/reverse A/B.
    variants = [('original', None)] + [('original', p) for p, _ in PROBES] + [('ab_forward', None), ('ab_reverse', None)]
    for encoding, probe in variants:
        for source in originals:
            c, g = input_context(source, rich[source['source_prepared_context_sha256']], tokenizer, catalog, encoding, probe)
            contexts.append(c); geometries.append(g)
    require(not {'torch', 'tensorflow', 'jax'} & set(sys.modules), 'freeze imported a model framework')
    require(len({c['record_id'] for c in contexts}) == len({c['prompt_sha256'] for c in contexts}) == 288, 'duplicate prompt/record identity')
    geometry_index = {(r['query_id'], r['condition'], r['encoding'], r['probe_id']): r for r in geometries}
    bridges = []
    for source in originals:
        key = source['query_id'], source['condition']
        a, f, r = [geometry_index[*key, encoding, None] for encoding in ENCODINGS]
        require(f['layout'] == r['layout'] and f['prompt_tokens'] == r['prompt_tokens'], 'A/B forward/reverse geometry differs')
        bridges.append({'query_id': key[0], 'condition': key[1], 'original_tokens': a['prompt_tokens'],
                        'ab_tokens': f['prompt_tokens'], 'ab_minus_original_tokens': f['prompt_tokens'] - a['prompt_tokens'],
                        'ab_forward_reverse_layout_equal': True, 'original_ab_layout_equal': a['layout'] == f['layout'],
                        'non_answer_demo_texts_equal': True, 'query_and_dictionary_texts_equal': True})
    raw_by_id = {r['record_id']: r for r in old_raw}; historical = []
    for c in contexts[:36]:
        old = deepcopy(raw_by_id[c['source_record_id']])
        require(old['prompt_sha256'] == c['prompt_sha256'] and all(x['prompt_token_ids_sha256'] == c['prompt_token_ids_sha256'] for x in old['candidates']), 'historical input differs')
        old['historical_record_id'] = old['record_id']; old['record_id'] = c['record_id']; historical.append(old)
    comparisons = read_lines(folder / 'comparisons.jsonl')
    require(len(comparisons) == 64, 'comparison frame differs')
    protocol = {
        'schema_version': 'evidence-label-calibration-protocol/v1', 'author': 'assistant', 'status': 'frozen_before_new_scores',
        'authorization': {'user_verbatim': '请帮我按这个顺序执行，弄一组新的结果，其中若有任何需要冻结项可以询问我',
                          'scope': 'three parallel score groups on existing 36 inputs; five background probes; two A/B mappings; freeze, test, run and analyze'},
        'source_method': {'title': 'Mitigating Label Length Bias in Large Language Models', 'method': 'NCC',
                          'url': 'https://aclanthology.org/2025.ijcnlp-long.78/',
                          'implementation_status': 'explicit local normalize-then-average specification; not claimed bit-identical to unavailable author code'},
        'groups': ['original_full_label_sum_and_mean', 'original_full_label_NCC', 'AB_forward_and_reverse'],
        'score_modes': list(SCORE_MODES), 'primary_original_and_ab_score': 'answer_sum', 'ncc_base_score': 'answer_mean',
        'ncc_steps': ['s_y = sum(log p(each token of complete JSON label including quotes)) / number of label tokens; EOS excluded',
                      'q_y = exp(s_y) / sum(exp(s_k)) for each real or probe prompt',
                      'pi_y(c) = arithmetic mean of q_y over the five frozen probes within the same condition c',
                      'q_cal_y = (q_real_y / pi_y) / sum(q_real_k / pi_k)',
                      'NCC margin = log(q_cal_non_hate / q_cal_hate) = original mean margin - log(pi_non_hate/pi_hate)'],
        'probes': [{'id': p, 'text': t, 'query_json': json.dumps(t, ensure_ascii=False)} for p, t in PROBES],
        'probe_replacement': 'replace only query JSON string; retain each condition dictionary, demonstration texts/answers, order and system; no retrieval rerun',
        'background_unit': 'per condition, never one shared prior across all conditions',
        'probe_diagnostics': 'all five individual-probe calibrations and five leave-one-out means; same 64 contrasts; no probe selection after scoring',
        'mappings': MAPPINGS, 'mapping_scope': 'system output instruction, every demonstration answer, complete output candidates; preserve other bytes',
        'ab_geometry': 'forward/reverse share full token layout; bridge to original changes instruction and label lengths, recorded separately',
        'ab_symmetric': '(forward semantic answer_sum margin + reverse semantic answer_sum margin)/2; auxiliary, both components mandatory',
        'ab_mapping_gap': 'forward semantic answer_sum margin minus reverse; auxiliary sensitivity measure',
        'prediction': 'argmax complete semantic candidates; exact tie selects semantic ordinal 0 (hate), flag tie separately',
        'numeric_policy': {'base_epsilon': parent['numeric_policy']['epsilon'],
                           'raw_margin_error_bound': 'epsilon', 'background_error_bound': 'epsilon',
                           'ncc_margin_error_bound': '2*epsilon', 'contrast_bound': 'per_margin_bound*sum(abs(coefficients))',
                           'derivation': 'B=logit(mean(sigmoid(m_j))) is 1-Lipschitz in sup norm: sum derivatives = mean(p_j*(1-p_j))/(pbar*(1-pbar)) <= 1',
                           'unresolved': 'abs(value)<=bound; numerical stability threshold, not statistical significance'},
        'references': 'copy immutable parent reference bytes; load only after all raw and NCC gates pass and raw scores are sealed',
        'interpretation': ['Compare sign, prediction and probe/mapping stability across score groups; scales differ.',
                           'Retain original, background and residual contrasts so prompt effects on priors remain visible.',
                           'Two exposed discovery cases; 64 contrasts are not independent samples.',
                           'Preserve position partner bridge, two-example swap and wording/length confounds from parent.',
                           'NCC correction is not evidence of a correct annotation or a purely semantic causal effect.',
                           'No new semantic materials, human field decisions or activation-patching readiness.'],
        'new_human_fields': 0, 'mechanism_ready': False}
    code = dict(parent['code_sha256'])
    for name in NEW_CODE: take(ROOT / name); code[name] = file_sha(ROOT / name)
    descriptors = [{k: c[k] for k in DESCRIPTOR} for c in contexts]
    schedule = [{'pass': n.replace('interventions-', 'calibration-'), 'selection': s, 'options': o, 'check': c} for n, s, o, c in SCHEDULE]
    plan = {'schema_version': 'evidence-label-calibration-plan/v1', 'status': 'frozen', 'cases': ['541', '3169'], 'tasks': list(ENCODINGS.values()),
            'source_files': sources.hashes, 'code_sha256': code, 'parent_plan_id': parent['plan_id'], 'parent_manifest_sha256': PARENT_MANIFEST,
            'blocks': descriptors, 'catalog': catalog, 'eos_token_id': parent['eos_token_id'], 'pad_token_id': parent['pad_token_id'],
            'generation_runtime_identity': parent['generation_runtime_identity'], 'config': parent['config'], 'numeric_policy': parent['numeric_policy'],
            'runtime_parent_plan': {'blocks': descriptors, 'catalog': catalog, 'package_path': parent['runtime_parent_plan']['package_path'],
                                    'generation_runtime_identity': parent['generation_runtime_identity']},
            'schedule': schedule, 'raw_pass': 'calibration-b1-r0', 'scoring_batch_size': 1,
            'query_gold_loaded_during_scoring': False, 'formal_test_or_reserve_access': False, 'analysis_after_all_gates_and_raw_seal': True,
            'epsilon_recalibration': False, 'automatic_profile_search': False, 'mechanism_ready': False,
            'contexts_sha256': sha(canonical(contexts).encode()), 'historical_selected_sha256': sha(canonical(historical).encode()),
            'analysis_protocol_sha256': sha(json_bytes(protocol)), 'comparisons_sha256': sha(jsonl(comparisons)),
            'counts': {'real_conditions': 36, 'original_real': 36, 'background_probes': 180, 'ab_real': 72,
                       'contexts': 288, 'candidates': 576, 'historical_contexts': 36, 'passes': 8,
                       'candidate_evaluations_across_passes': 3600, 'comparisons': 64}}
    plan['plan_id'] = 'evidence-label-calibration-' + sha(canonical(plan).encode())
    audit = {'status': 'passed', **plan['counts'], 'full_label_boundaries': 576, 'original_token_replays': 36,
             'ab_geometry_pairs': 36, 'torch_imported': False, 'model_forward_executed': False,
             'references_separate': True, 'old_pointer_changes': 0, 'human_fields_changed': 0}
    files = {'plan.json': json_bytes(plan), 'contexts.jsonl': jsonl(contexts), 'historical-selected.jsonl': jsonl(historical),
             'analysis_protocol.json': json_bytes(protocol), 'analysis_references.json': (folder / 'analysis_references.json').read_bytes(),
             'comparisons.jsonl': jsonl(comparisons), 'geometry.jsonl': jsonl(geometries), 'encoding-bridges.csv': csv_bytes(bridges),
             'audit.json': json_bytes(audit), 'execution_source.json': json_bytes({p: {'sha256': h, 'text': (ROOT / p).read_text()} for p, h in code.items()})}
    files['FREEZE.md'] = ('# 原标签、NCC、A/B 正反映射冻结\n\n'
        '沿用两案例的 36 个条件与 64 项比较，先复现全部原标签基线，再评估五探针 NCC 与 A/B 正反映射。'
        '授权范围来自用户本轮执行请求；计算细则为助手编写的可重建实施协议，不新增逐项人工语义裁决。\n\n'
        'NCC 使用完整 JSON 标签的 token 平均对数概率，在类别间归一化，再将五探针概率做算术平均。'
        '逐条件估计背景；保留原效应、背景效应、校准残差和全部单探针／留一探针结果。协议详见 analysis_protocol.json。\n\n'
        'A/B 同步变更任务输出说明、示例输出和候选；正反映射逐锚点等位置。与原标签的说明长度及答案长度差异写入 encoding-bridges.csv。\n\n'
        '基础 epsilon 沿用 0.0013427734375；NCC 单条件误差界为 2 epsilon，两项差值 4 epsilon，四项交互 8 epsilon。'
        '所有背景及残差也重做数值复核。以上是数值界，不是统计区间。\n\n'
        '288 提示、576 候选，8 个 pass 共 3600 候选评估；仍使用原 FP32 forward 内核。'
        '参考标签单独保存，仅在原始分数封存且全部检查通过后进入分析。旧材料、旧结果及人审记录保留。\n\n'
        f'计划 ID：`{plan["plan_id"]}`。\n').encode()
    sources.unchanged()
    return files, sources.hashes, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=WORK / 'frozen-01')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(); target = args.output.resolve()
    require(target.parent == WORK and target.name.startswith('frozen-'), 'output must be a new calibration freeze')
    files, sources, audit = build()
    if args.check:
        m = read_json(target / 'manifest.json')
        require(m['source_files'] == sources and set(m['artifacts']) == set(files), 'source/artifact inventory differs')
        for name, raw in files.items(): require((target / name).read_bytes() == raw and m['artifacts'][name] == sha(raw), 'reconstruction differs: ' + name)
        load_frozen(target); print('Byte reconstruction passed: 288 contexts, 576 boundaries, 36 historical inputs, 64 comparisons; no forward.')
    else:
        files['manifest.json'] = json_bytes({'schema_version': 'evidence-label-calibration-freeze/v1', 'status': 'frozen',
            'created_at': datetime.now(timezone.utc).isoformat(), 'source_files': sources, 'artifacts': {k: sha(v) for k, v in files.items()}})
        write_output(target, files); print(canonical(audit))


if __name__ == '__main__': main()
