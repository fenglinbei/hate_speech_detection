#!/usr/bin/env python3
"""Freeze both accepted material batches and a reference-separated scoring plan."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.general_model_evidence_evaluation import (
    EXPERIMENT, Sources, canonical, csv_bytes, file_sha, json_bytes, jsonl,
    local_path, read_json, read_lines, require, sha, unique, write_output,
)
from diagnostics.general_model_numeric_analysis import SCORE_MODES, _validated_candidates
from scripts.review.run_evidence_interventions import DESCRIPTOR, SCHEDULE

BASE = ROOT / EXPERIMENT
WORK = BASE / 'reviews/analysis-freeze-20260912/matched-input-freeze-v1'
BATCHES = (
    ('matched-materials-v1', 'cd88b7e34598080cac502aa62627edab620855c5a28ceee37842307842741e78',
     'batch-01-feedback-01.json', 'ddaee8155f31b52a4f2dfd7eace5f40ea9f71eb5aa7dea0453d0898d6dab7dd0', 8),
    ('position-length-materials-v1', '02c16bf1c38d266e6c32ec390df7ddbf032316abc24594856561ca1acb95e675',
     'batch-02-feedback-01.json', 'd7cdbe8e227352661fea69dba5d12acec1c5fa3c3b91bf2e48d5d1ce19828365', 6),
)
CONDITIONS = ('R-A', 'R-B', 'R-C1', 'R-C2', 'R-D1', 'R-D2',
              'P0-A', 'P1-A', 'P0-C1', 'P1-C1', 'P0-C2', 'P1-C2',
              'P0-D1', 'P1-D1', 'P0-D2', 'P1-D2', 'S', 'L')
FORBIDDEN = {'gold', 'reference_label', 'original_reference_label', 'reviewed_label', 'adjudicated_label', 'query_analysis_reference'}


def verify_feedback(feedback, rows, manifest_sha):
    require(feedback['source_manifest_sha256'] == manifest_sha and feedback['batch_overall_no_objection'] is True, 'feedback does not accept this batch')
    require(feedback['individual_field_decisions'] == [] and feedback['new_mechanism_judgments'] == 0, 'unexpected field-level feedback')
    expected = {r['material_id']: r['text_sha256'] for r in rows}
    actual = {r['material_id']: r['text_sha256'] for r in feedback['materials']}
    require(len(expected) == len(rows) == len(actual) == len(feedback['materials']) and actual == expected, 'feedback text coverage differs')
    for r in rows:
        require(sha(r['text'].encode()) == expected[r['material_id']], 'accepted text hash differs')


def scoring_contexts(rows):
    require(len(rows) == 36 and {(r['query_id'], r['task'], r['condition']) for r in rows} ==
            {(q, 'hate', c) for q in ('541', '3169') for c in CONDITIONS}, 'expected the exact 36-condition frame')
    def no_reference(value):
        if isinstance(value, dict):
            require(not FORBIDDEN & value.keys(), 'query reference in scoring input')
            for v in value.values(): no_reference(v)
        elif isinstance(value, list):
            for v in value: no_reference(v)
    keys = ('record_id', 'query_id', 'task', 'condition', 'family', 'messages', 'prompt_text',
            'prompt_sha256', 'prompt_tokens', 'prompt_token_ids', 'prompt_token_ids_sha256')
    result = []
    for row in sorted(rows, key=lambda r: (r['condition'], int(r['query_id']))):
        no_reference(row)
        require(sha(row['prompt_text'].encode()) == row['prompt_sha256'], 'prompt identity drift')
        require(sha(canonical(row['prompt_token_ids']).encode()) == row['prompt_token_ids_sha256'] and
                len(row['prompt_token_ids']) == row['prompt_tokens'], 'token identity drift')
        item = {k: deepcopy(row[k]) for k in keys}
        item.update(protocol_id=f'EMF-{row["query_id"]}-hate-v1',
                    baseline_replay=row['condition'] in ('R-A', 'R-B'), tokenization_status='verified',
                    source_prepared_context_sha256=sha(canonical(row).encode()))
        item['context_sha256'] = sha(canonical(item).encode())
        result.append(item)
    require(len({c['record_id'] for c in result}) == len({c['prompt_sha256'] for c in result}) == 36, 'duplicate context identity')
    return result


def registered_contrasts(second_batch, epsilon):
    rows = deepcopy(second_batch)
    for q in ('541', '3169'):
        def add(name, kind, weights):
            rows.append({'contrast_id': f'EMF-{q}:{name}', 'query_id': q, 'task': 'hate', 'kind': kind,
                         'terms': [{'condition': c, 'coefficient': w} for c, w in weights],
                         'author': 'assistant', 'status': 'frozen_before_new_scores', 'human_mechanism_judgment': False})
        add('natural-deletion', 'historical_natural_deletion_total_effect', [('R-B', 1), ('R-A', -1)])
        for variant in ('C1', 'C2', 'D1', 'D2'):
            add(variant + '-vs-A', 'fixed_slot_content_change', [('R-' + variant, 1), ('R-A', -1)])
            add(variant + '-vs-B', 'restored_example_vs_deletion_total_effect', [('R-' + variant, 1), ('R-B', -1)])
        for k in (1, 2):
            add('semantic-original-' + str(k), 'paired_content_paraphrase_comparison', [('R-D' + str(k), 1), ('R-C' + str(k), -1)])
        for family in ('C', 'D'):
            add('wording-' + family, 'within_family_wording_sensitivity', [('R-' + family + '2', 1), ('R-' + family + '1', -1)])
    require(len(rows) == len({r['contrast_id'] for r in rows}) == 64, 'comparison coverage differs')
    for row in rows:
        terms = row['terms']
        require(len({t['condition'] for t in terms}) == len(terms) and
                all(t['condition'] in CONDITIONS and type(t['coefficient']) is int and t['coefficient'] in (-1, 1) for t in terms) and
                sum(t['coefficient'] for t in terms) == 0, 'invalid comparison coefficients')
        row.update(status='frozen_before_new_scores', primary_score='answer_sum', score_modes=list(SCORE_MODES),
                   analysis_variable='non_hate_margin', numeric_bound=epsilon * sum(abs(t['coefficient']) for t in terms),
                   numerically_unresolved_if='abs(effect) <= numeric_bound', statistical_interval=False)
    return rows


def load_frozen(directory):
    """Validate the freeze, but parse no query-reference or semantic-assessment data."""
    manifest = read_json(directory / 'manifest.json')
    require(manifest['schema_version'] == 'evidence-matched-input-freeze/v1' and manifest['status'] == 'frozen', 'not a frozen input plan')
    require({p.name for p in directory.iterdir()} == {*manifest['artifacts'], 'manifest.json'}, 'unexpected freeze artifacts')
    for name, expected in manifest['artifacts'].items():
        require(Path(name).name == name and file_sha(directory / name) == expected, 'frozen artifact changed: ' + name)
    for name, expected in manifest['source_files'].items():
        require(file_sha(local_path(ROOT, name)) == expected, 'frozen source changed: ' + name)
    plan = read_json(directory / 'plan.json')
    identity = dict(plan); plan_id = identity.pop('plan_id')
    require(plan_id == 'evidence-matched-inputs-' + sha(canonical(identity).encode()), 'plan identity changed')
    contexts, historical = read_lines(directory / 'contexts.jsonl'), read_lines(directory / 'historical-selected.jsonl')
    require(sha(canonical(contexts).encode()) == plan['contexts_sha256'] and
            sha(canonical(historical).encode()) == plan['historical_selected_sha256'], 'scoring input binding differs')
    require(len(contexts) == 36 and sum(c['baseline_replay'] for c in contexts) == len(historical) == 4, 'scoring coverage differs')
    for c in contexts:
        require(sha(canonical({k: v for k, v in c.items() if k != 'context_sha256'}).encode()) == c['context_sha256'], 'context digest differs')
    return plan, contexts, historical


def build():
    sources = Sources(ROOT)
    def take(path, expected=None): return sources.verify(path, expected or file_sha(path))
    feedbacks, adopted, folders = [], [], []
    for name, manifest_sha, feedback_name, feedback_sha, count in BATCHES:
        pointer = read_json(take(BASE / name / 'current.json'))
        require(pointer['manifest_sha256'] == manifest_sha, 'wrong material version')
        folder = local_path(ROOT, str(EXPERIMENT / pointer['materials_path'])); folders.append(folder)
        manifest = read_json(take(folder / 'manifest.json', manifest_sha))
        for path, h in manifest['source_files'].items(): take(local_path(ROOT, path), h)
        for path, h in manifest['artifacts'].items():
            require(Path(path).name == path, 'invalid material artifact')
            take(folder / path, h)
        feedback = read_json(take(BASE / 'matched-materials-review-v1' / feedback_name, feedback_sha))
        rows = read_lines(folder / 'materials.jsonl'); require(len(rows) == count, 'material count changed')
        verify_feedback(feedback, rows, manifest_sha); feedbacks.append(feedback)
        for row in rows:
            adopted.append({'material_id': row['material_id'], 'text': row['text'], 'text_sha256': row['text_sha256'],
                            'source_material_manifest_sha256': manifest_sha, 'source_assessment': row['ai_assessment'],
                            'source_assessment_author': 'assistant', 'source_human_fields': row['human_review'],
                            'input_use_status': 'accepted_by_batch_no_objection',
                            'batch_acceptance': {'feedback_id': feedback['feedback_id'], 'feedback_sha256': feedback_sha,
                                                 'user_verbatim': feedback['user_verbatim'], 'scope': feedback['scope']},
                            'independent_human_field_adjudication': False, 'human_mechanism_judgment': False})
    original_rows = read_lines(folders[1] / 'contexts.jsonl')
    contexts = scoring_contexts(original_rows)
    first_rows = {(r['query_id'], r['condition']): r for r in read_lines(folders[0] / 'contexts.jsonl')}
    for c in contexts:
        if c['condition'].startswith('R-'):
            old = first_rows[c['query_id'], c['condition'][2:]]
            require(c['prompt_text'] == old['prompt_text'] and c['prompt_token_ids'] == old['prompt_token_ids'], 'first-batch bridge mismatch')
    previous = read_json(take(BASE / 'input-interventions-v1/current.json'))
    require(previous['status'] == 'complete', 'previous run is incomplete')
    old_folder, old_run = BASE / previous['plan_path'], BASE / previous['run_path']
    old_manifest = read_json(take(old_folder / 'manifest.json', previous['plan_manifest_sha256']))
    for name, expected in old_manifest['files'].items(): take(old_folder / name, expected)
    old = read_json(old_folder / 'plan.json')
    package = Path(old['runtime_parent_plan']['package_path'])
    for name in ('models.json', 'config.resolved.json'):
        path = package / name
        take(path, old['source_files'][str(path.relative_to(ROOT))])
    old_state = read_json(take(old_run / 'run_manifest.json', previous['run_manifest_sha256']))
    require(old_state['status'] == 'complete' and old_state['numerical_validation_passed'] and old_state['plan_id'] == old['plan_id'], 'old numeric run not sealed')
    raw = read_lines(take(old_run / old['raw_pass'] / 'scores.jsonl', previous['raw_scores_sha256']))
    selected_raw = {(r['query_id'], r['condition']): r for r in raw if r['task'] == 'hate' and r['query_id'] in ('541', '3169')}
    historical = []
    for c in contexts:
        if not c['baseline_replay']: continue
        source_condition = 'original' if c['condition'] == 'R-A' else 'demo_removed_dictionary_fixed'
        source = selected_raw[c['query_id'], source_condition]
        _validated_candidates('hate', source['candidates'])
        require(source['prompt_sha256'] == c['prompt_sha256'] and
                all(r['prompt_token_ids_sha256'] == c['prompt_token_ids_sha256'] for r in source['candidates']), 'historical numeric input mismatch')
        row = deepcopy(source)
        row['historical_record_id'] = row['record_id']; row['record_id'] = c['record_id']
        historical.append(row)
    require(len(historical) == 4, 'historical replay coverage differs')
    rp = read_json(BASE / 'analysis-reference-v1/current.json')
    overlays = unique(read_lines(BASE / rp['freeze_path'] / 'gold_overlay.jsonl'), lambda r: (r['query_id'], r['task']))
    refs = [overlays[q, 'hate'] for q in ('541', '3169')]
    require(all(r['analysis_reference_eligible'] and r['original_label'] == r['adjudicated_label'] == 'non-hate' for r in refs), 'frozen reference differs')
    # The tokenizer and its exact tree were bound by both parents. No model framework.
    for name in ('USE_TORCH', 'USE_TF', 'USE_FLAX', 'USE_TORCH_XLA'): os.environ[name] = '0'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'models/base/Qwen3-8B'), local_files_only=True, trust_remote_code=False, use_fast=True)
    require(not {'torch', 'tensorflow', 'jax'} & set(sys.modules), 'freeze imported model framework')
    for c in contexts:
        require(tokenizer.apply_chat_template(c['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False) == c['prompt_text'] and
                tokenizer.encode(c['prompt_text'], add_special_tokens=False) == c['prompt_token_ids'], 'frozen token replay differs')
        for candidate in old['catalog']['hate']:
            require(tokenizer.encode(c['prompt_text'] + candidate['canonical_answer'], add_special_tokens=False) == c['prompt_token_ids'] + candidate['answer_token_ids'], 'candidate boundary differs')
    comparisons = registered_contrasts(read_lines(folders[1] / 'contrasts.jsonl'), old['numeric_policy']['epsilon'])
    analysis_protocol = {
        'schema_version': 'evidence-matched-analysis-protocol/v1', 'status': 'frozen_before_new_scores', 'author': 'assistant',
        'primary_score': 'answer_sum', 'score_modes': list(SCORE_MODES),
        'prediction': 'full_two_candidate_argmax; smallest_canonical_ordinal_on_exact_tie',
        'analysis_variable': 'non_hate_margin = score(non-hate) - score(hate)',
        'positive_effect_meaning': 'greater_support_for_non_hate; both_original_and_reviewed_query_references_are_non_hate',
        'reference_handling': 'load analysis_references.json only after raw scores are sealed and all gates pass; retain both references',
        'numeric_direction': {'base_epsilon': old['numeric_policy']['epsilon'], 'bound': 'epsilon * sum(abs(coefficients))',
                              'unresolved': 'abs(effect) <= bound', 'is_statistical_interval': False},
        'case_hypotheses': {
            '541': 'D1/C1 and D2/C2 positive non-hate-margin differences would support a contribution of the preserved topic-plus-countergeneralization semantics within these inputs.',
            '3169': 'D1/C1 and D2/C2 negative non-hate-margin differences would support a harmful contribution of the original semantic/cue combination; target changes prevent attribution to 嘿嘿 alone.'},
        'alternatives': ['generic_demonstration_or_answer_presence', 'wording_specificity', 'partner_content_and_order',
                         'joint_movement_of_two_examples', 'wording_density_and_downstream_positions', 'score_definition_sensitivity'],
        'interpretation_rules': ['Report both matched variant pairs; discordant directions are wording sensitivity, not a reason to select one.',
                                 'Report P0-minus-R partner rewriting separately from P1-minus-P0 position changes.',
                                 'Short/long edits do not isolate pure length. Position swaps do not isolate one absolute position.',
                                 'No-effect, reverse, near-tie and score-mode-discordant results stay in every report.',
                                 'Two exposed discovery cases are not an independent confirmatory cohort; conditions are not independent cases.',
                                 'Numerical direction thresholds are not effect-size or statistical significance thresholds.',
                                 'Input effects do not establish an internal mechanism or activation-patching readiness.'],
        'condition_rows_expected': 144, 'contrast_rows_expected': 256, 'contrast_count': 64,
        'new_human_mechanism_judgments': 0,
    }
    code_hashes = dict(old['code_sha256'])
    for name, expected in code_hashes.items(): take(ROOT / name, expected)
    for name in ('scripts/review/freeze_evidence_matched_inputs.py', 'scripts/review/run_evidence_matched_inputs.py',
                 'scripts/review/analyze_evidence_interventions.py',
                 'scripts/review/test_freeze_evidence_matched_inputs.py'):
        take(ROOT / name); code_hashes[name] = file_sha(ROOT / name)
    descriptors = [{k: c[k] for k in DESCRIPTOR} for c in contexts]
    schedule = [{'pass': name.replace('interventions-', 'matched-'), 'selection': selection, 'options': options, 'check': check}
                for name, selection, options, check in SCHEDULE]
    plan = {'schema_version': 'evidence-matched-input-plan/v1', 'status': 'frozen', 'cases': ['541', '3169'], 'tasks': ['hate'],
            'material_acceptance': 'two_scoped_batch_no_objection_receipts; original_AI_field_authorship_retained',
            'material_manifest_sha256': [b[1] for b in BATCHES], 'feedback_sha256': [b[3] for b in BATCHES],
            'blocks': descriptors, 'catalog': {'hate': old['catalog']['hate']}, 'source_files': sources.hashes,
            'code_sha256': code_hashes, 'eos_token_id': old['eos_token_id'], 'pad_token_id': old['pad_token_id'],
            'generation_runtime_identity': old['generation_runtime_identity'], 'config': old['config'],
            'numeric_policy': old['numeric_policy'], 'scoring_batch_size': 1,
            'runtime_parent_plan': {'blocks': descriptors, 'catalog': {'hate': old['catalog']['hate']},
                                    'package_path': old['runtime_parent_plan']['package_path'],
                                    'generation_runtime_identity': old['generation_runtime_identity']},
            'schedule': schedule, 'raw_pass': 'matched-b1-r0', 'historical_comparison_limit': old['numeric_policy']['epsilon'],
            'query_gold_loaded_during_scoring': False, 'formal_test_or_reserve_access': False,
            'analysis_after_all_gates_and_raw_seal': True, 'epsilon_recalibration': False, 'automatic_profile_search': False,
            'contexts_sha256': sha(canonical(contexts).encode()), 'historical_selected_sha256': sha(canonical(historical).encode()),
            'counts': {'materials': 14, 'contexts': 36, 'candidates': 72, 'historical_contexts': 4, 'historical_candidates': 8,
                       'passes': 8, 'candidate_evaluations_across_passes': 448, 'comparisons': 64},
            'gpu_execution_started': False, 'mechanism_ready': False,
            'analysis_protocol_sha256': sha(json_bytes(analysis_protocol)), 'comparisons_sha256': sha(jsonl(comparisons))}
    plan['plan_id'] = 'evidence-matched-inputs-' + sha(canonical(plan).encode())
    audit = {'schema_version': 'evidence-matched-input-freeze-audit/v1', 'status': 'passed', 'batch_acceptance_receipts': 2,
             'accepted_material_texts': 14, 'new_individual_human_fields': 0, 'frozen_contexts': 36,
             'all_prompt_and_token_replays': 36, 'candidate_boundary_checks': 72, 'historical_numeric_contexts_bound': 4,
             'historical_candidate_scores_validated': 8, 'comparisons_frozen': 64,
             'worker_contexts_contain_query_references': False, 'references_stored_separately': True,
             'input_plan_ready': True, 'current_model_numeric_replay': 'not_executed', 'new_model_forward_executed': False,
             'production_writes': 0, 'source_pointer_changes': 0, 'mechanism_ready': False}
    files = {'plan.json': json_bytes(plan), 'contexts.jsonl': jsonl(contexts), 'historical-selected.jsonl': jsonl(historical),
             'accepted_materials.jsonl': jsonl(adopted), 'feedback_receipts.json': json_bytes(feedbacks),
             'analysis_references.json': json_bytes(refs), 'analysis_protocol.json': json_bytes(analysis_protocol),
             'comparisons.jsonl': jsonl(comparisons), 'comparisons.csv': csv_bytes(comparisons), 'audit.json': json_bytes(audit),
             'execution_source.json': json_bytes({p: {'sha256': h, 'text': (ROOT / p).read_text()} for p, h in code_hashes.items()})}
    files['FREEZE.md'] = report(plan).encode()
    sources.unchanged()
    return files, sources.hashes, audit


def report(plan):
    return '\n'.join([
        '# 两批输入材料与评分方案冻结', '',
        '用户先后回复“第一批没什么问题，还有下一批吗？”及“第二批也没有问题”。两次整体认可分别绑定 8 段与 6 段正文；14 段新文本按批次接受用于已说明的输入对照。AI 标签、严重度和解释保留原作者来源，没有补造逐项独立人审或机制裁决。', '',
        '## 冻结范围', '',
        '仅 541/hate、3169/hate，共 36 个不重复输入：12 个第一批条件（其中原始与删除 4 个、内容替换与同义改写 8 个），20 个共同背景下的位置条件，4 个短／长条件。词典及所有原输入正文保持准备时的字节。', '',
        '64 项比较：保留第二批预列的 38 项位置、背景、语义与长度比较，再补齐第一批 26 项删除、恢复示例、等长语义与措辞敏感性比较。comparisons.jsonl 列出每项条件和系数。多个条件共享两条查询，不能按 64 个独立样本解读。', '',
        '## 评分与解释', '',
        '本地 Qwen3-8B，沿用已核验的 float32 参数转换、eager attention、batch=1、无缓存、动态右填充、种子 42 及 4 个副本。完整评分 hate/non-hate 两个候选，答案 token 对数概率之和为主口径，另保留平均分及含 EOS 的两种口径；精确并列按候选原序决定，同时标记并列。', '',
        '主要连续量为 non-hate 分数减 hate 分数。两案例的原参考与审核参考均为 non-hate，两套来源仍单独保留；查询参考仅在封存原始分数后用于分析。', '',
        'epsilon 固定为 0.0013427734375。两项差值用 2 epsilon=0.002685546875，四项交互用 4 epsilon=0.00537109375；范围内标记数值未决。这不是统计置信区间。重复和同 logits 的 CPU 归一化参考阈值为 0.0001，其余沿用历史 epsilon，不重新挑选阈值。', '',
        '541 检验保留反泛化语义的 D 是否比 C 更支持 non-hate；3169 检验原语义／同形词组合是否使 D 比 C 更偏向 hate。两套措辞都保留，方向不一致报告为措辞敏感。位置效应只用共同背景 P1−P0，配对改写 P0−R 单列；长短变化包含措辞和后文位置的影响。', '',
        '## 执行顺序', '',
        '1. 重放之前已评分的两个原始和两个删除条件，并核对 CPU 归一化参考及重复。',
        '2. 同一次新运行评分全部 36 条件、72 候选，再核对重复、额外 64 token 填充、答案前缀、候选顺序及 GPU 副本交换。共 8 个 pass、448 次候选评估，前缀检查内部可能包含多次 forward。',
        '3. 全部检查通过并封存原始分数后，按四口径输出 144 条条件读数与 256 条比较读数，保留无效、反向、并列和口径不一致结果。', '',
        '本次完成 CPU 冻结与输入校验，尚未进行本轮 GPU 重放或评分。执行入口单独读取此冻结版本；旧 26 条件运行及所有材料草稿、反馈与参考指针保留。仍未开展激活采集或 patching。', '',
        f"计划 ID：`{plan['plan_id']}`。", ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=WORK / 'frozen-01')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(); target = args.output.resolve()
    require(target.parent == WORK and target.name.startswith('frozen-'), 'output must be a new freeze version')
    files, sources, audit = build()
    if args.check:
        manifest = read_json(target / 'manifest.json')
        require(manifest['schema_version'] == 'evidence-matched-input-freeze/v1' and manifest['status'] == 'frozen' and manifest['source_files'] == sources, 'freeze source inventory differs')
        require(set(manifest['artifacts']) == set(files) and {p.name for p in target.iterdir()} == {*files, 'manifest.json'}, 'freeze output inventory differs')
        for name, raw in files.items():
            require((target / name).read_bytes() == raw and manifest['artifacts'][name] == sha(raw), 'freeze reconstruction differs: ' + name)
        load_frozen(target)
        print('Freeze reproduces byte for byte: 14 accepted texts, 36 contexts, 72 boundaries, 64 comparisons; no model forward.')
    else:
        files['manifest.json'] = json_bytes({'schema_version': 'evidence-matched-input-freeze/v1', 'status': 'frozen',
                                            'created_at': datetime.now(timezone.utc).isoformat(), 'source_files': sources,
                                            'artifacts': {k: sha(v) for k, v in files.items()}})
        write_output(target, files); print(canonical(audit))


if __name__ == '__main__': main()
