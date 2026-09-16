#!/usr/bin/env python3
"""Prepare AI-drafted matched materials, with CPU full-prompt position proofs.

This command does not load model weights, adopt human fields, update prior
freezes, or start a review server. Existing output directories are immutable.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.general_model_evidence_evaluation import (
    EXPERIMENT, Sources, canonical, csv_bytes, file_sha, json_bytes, jsonl,
    local_path, read_json, read_lines, require, sha, unique, write_output,
)
from scripts.review.implement_evidence_case_uses import TOKENIZER_FILES
from scripts.review.update_evidence_case_uses import accepted_field

BASE = ROOT / EXPERIMENT
SPEC = BASE / 'matched-materials-v1'
PREF = BASE / 'ai_reviews/v2-reannotation-20260911'
CONDITIONS = ('A', 'B', 'C1', 'C2', 'D1', 'D2')


def replace_text(source, old, new):
    """A single demo-text edit; every other prompt/message byte is protected."""
    require(old != new and new.strip() == new and bool(new), 'empty or unchanged replacement')
    require(not any(s in new for s in ('\n', '\r', '<|', '|>')), 'replacement changes prompt structure')
    require(source['prompt_text'].count(old) == 1, 'ambiguous target text in full prompt')
    messages = deepcopy(source['messages'])
    require([m['role'] for m in messages] == ['system', 'user'], 'unsupported messages')
    require(messages[1]['content'].count(old) == 1 and old not in messages[0]['content'], 'ambiguous message target')
    messages[1]['content'] = messages[1]['content'].replace(old, new)
    prompt = source['prompt_text'].replace(old, new)
    return messages, prompt


def token_span(offsets, start, end):
    indices = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
    require(indices and indices == list(range(indices[0], indices[-1] + 1)), 'noncontiguous token span')
    return [indices[0], indices[-1] + 1]


def unique_span(prompt, part, offsets):
    require(prompt.count(part) == 1, 'ambiguous structural anchor')
    start = prompt.index(part)
    return token_span(offsets, start, start + len(part))


def layout(prompt, offsets, demo_ids, materials, target_id, replacement=None):
    demos = {}
    for slot, d in enumerate(demo_ids, 1):
        text = replacement if d == target_id and replacement is not None else materials['demo:' + d]['source']['text']
        answer = json.dumps(materials['demo:' + d]['source']['original_answer']['hate'], ensure_ascii=False)
        block = f'示例 {slot}\n文本：{text}\n输出：{answer}'
        require(prompt.count(block) == 1, 'demo block changed or absent')
        start = prompt.index(block)
        text_start = start + len(f'示例 {slot}\n文本：')
        answer_start = start + len(block) - len(answer)
        demos[d] = {'slot': slot, 'block': token_span(offsets, start, start + len(block)),
                    'text': token_span(offsets, text_start, text_start + len(text)),
                    'answer': token_span(offsets, answer_start, start + len(block))}
    query_start = prompt.rindex('待判断文本（JSON 字符串）：\n')
    query_end = prompt.index('<|im_end|>', query_start)
    return {'demos': demos, 'query_block': token_span(offsets, query_start, query_end),
            'generation_start': len(offsets)}


def alignment_proof(original, edited, target):
    """Total length is necessary but not sufficient: verify every anchor and ID."""
    left, right = original['prompt_token_ids'], edited['prompt_token_ids']
    a, b = original['layout'], edited['layout']
    require(len(left) == len(right), 'full prompt token count changed')
    require(a == b, 'demo, answer, query, or generation positions changed')
    start, end = a['demos'][target]['text']
    require(left[:start] == right[:start], 'tokens before target text changed')
    require(left[end:] == right[end:], 'tokens after target text changed')
    for d, positions in a['demos'].items():
        if d != target:
            s, e = positions['block']
            require(left[s:e] == right[s:e], 'untouched demo tokens changed')
    require(left != right, 'replacement has identical input tokens')
    return {'passed': True, 'full_prompt_tokens': len(left), 'target_text_token_span': [start, end],
            'all_demo_block_text_and_answer_positions_equal': True,
            'prefix_token_ids_equal': True, 'suffix_token_ids_equal': True,
            'all_untouched_demo_token_ids_equal': True,
            'query_block_token_span': a['query_block'], 'generation_start_token': a['generation_start'],
            'internal_semantic_token_alignment_claimed': False,
            'offset_note': 'Spans cover every overlapping token, including punctuation/newline merges at a text boundary.'}


def review_draft(selection, selected, text, query_text, old_text, basis, context):
    q, condition = selected['id'].split('-')
    case = next(c for c in selection['cases'] if c['query_id'] == q)
    kind = 'content_replacement' if condition.startswith('C') else 'semantic_preserving_paraphrase'
    required = {field['field']: None for field in selection['review_fields']}
    return {'material_id': selected['id'], 'query_id': q, 'task': 'hate', 'condition': condition,
            'target_demo_id': case['target_demo_id'], 'kind': kind,
            'author': 'assistant', 'status': 'draft_pending_human_review',
            'text': text, 'text_sha256': sha(text.encode()), 'query_text': query_text,
            'original_demo_text': old_text, 'source_record_ids': basis,
            'source_round': selected['round'], 'prompt_sha256': context['prompt_sha256'],
            'prompt_tokens': context['prompt_tokens'], 'alignment': context['alignment'],
            'fixed_demo_answer': case['demo_answer'],
            'ai_assessment': {'hate': case['demo_answer'], 'attack_severity': selected['ai_attack_severity'],
                              'rationale': selected['ai_rationale'],
                              'semantic_intent': case['preserve_semantics'],
                              'intent_operation': 'remove_this_semantic_combination' if kind == 'content_replacement' else 'preserve_this_semantic_combination',
                              'limitations': case['semantic_limit'], 'human_confirmed': False},
            'human_review': {'status': 'pending', 'values': required, 'events': []},
            'input_scoring_eligible': False, 'mechanism_ready': False}


def build(tokenizer_path):
    sources = Sources(ROOT)
    def take(path, expected=None):
        return sources.verify(path, expected or file_sha(path))
    def parent(name, key, expected):
        p = read_json(take(BASE / name / 'current.json'))
        require(p['manifest_sha256'] == expected, 'source version differs: ' + name)
        folder = local_path(ROOT, str(EXPERIMENT / p[key]))
        manifest = read_json(take(folder / 'manifest.json', expected))
        return folder, manifest
    selection = read_json(take(SPEC / 'selection.json'))
    require(selection['status'] == 'draft_pending_human_review', 'material adoption requires a separate version')
    require([(c['query_id'], c['task'], c['target_demo_id']) for c in selection['cases']] ==
            [('541', 'hate', '826'), ('3169', 'hate', '3660')], 'unexpected scope')
    freeze, fm = parent('analysis-reference-v1', 'freeze_path', '4e6ef030d6dfdd8d90087f3e32160de41f58a8c2a2d47905eddf99d60fd7a3af')
    previous, pm = parent('case-uses-implementation-v1', 'implementation_path', 'f3ec6698223a453f0ef0bc428cb4fae89e35906ea27431f445bbec3a3333c71c')
    materials = unique(read_lines(take(freeze / 'material_reviews.jsonl', fm['artifacts']['material_reviews.jsonl'])), lambda r: r['record_id'])
    overlays = unique(read_lines(take(freeze / 'gold_overlay.jsonl', fm['artifacts']['gold_overlay.jsonl'])), lambda r: (r['query_id'], r['task']))
    old_contexts = unique(read_lines(take(previous / 'contexts.jsonl', pm['artifacts']['contexts.jsonl'])), lambda r: (r['query_id'], r['task'], r['condition']))
    take(previous / 'interventions.jsonl', pm['artifacts']['interventions.jsonl'])
    # Bind completed natural-edit execution as provenance; do not inspect scores to select drafts.
    run = read_json(take(BASE / 'input-interventions-v1/current.json'))
    require(run['status'] == 'complete' and run['result_manifest_sha256'] == '09e8ddca7b3d9d310f1b5d2db9fa01e79d5d6fdb06cef5bb7d9d66dc5d268fba', 'unexpected prior execution')
    take(BASE / run['results_path'] / 'manifest.json', run['result_manifest_sha256'])
    for directory, id_key in [('preferences', 'preference_snapshot_id'), ('resource-reviews-v1/relations', 'relation_preference_snapshot_id')]:
        pointer = read_json(take(PREF / directory / 'current.json'))
        require(pointer['snapshot_id'] == selection[id_key], 'adjudication preference version changed')
        for ref in pointer['artifacts'].values():
            take(PREF / ref['path'], ref['sha256'])
    for name in ('attack-severity-v1.json', 'severity-hate-default-mapping-v1.json'):
        take(PREF / 'policies' / name)
    take(ROOT / 'docs/research/annotation-guidelines/evidence-applicability-annotation-policy-v2.md')
    ref = read_json(take(ROOT / 'exps/causal_context/general_model_ld_nolabel_v1/plan_ref.json'))
    plan = read_json(take(Path(ref['target_path']) / 'plan.json', ref['plan_sha256']))
    inventory = []
    for p in sorted(tokenizer_path.rglob('*')):
        if p.is_file() and p.name in TOKENIZER_FILES:
            take(p)
            inventory.append({'path': str(p.relative_to(tokenizer_path)), 'size': p.stat().st_size, 'sha256': file_sha(p)})
    tree_sha = sha(canonical(inventory).encode())
    require(tree_sha == plan['generation_runtime_identity']['tokenizer_tree_sha256'], 'tokenizer fingerprint differs')
    for flag in ('USE_TORCH', 'USE_TF', 'USE_FLAX', 'USE_TORCH_XLA'):
        os.environ[flag] = '0'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False, use_fast=True)
    require(not {'torch', 'tensorflow', 'jax'} & set(sys.modules), 'CPU check imported a model framework')
    require(tokenizer.eos_token_id == plan['eos_token_id'] and tokenizer.pad_token_id == plan['pad_token_id'], 'special tokens changed')

    drafts, history = {}, []
    for name in selection['authoring_rounds']:
        require(Path(name).name == name, 'invalid authoring path')
        for r in read_json(take(SPEC / 'authoring' / name))['candidates']:
            key = (name, r['id'])
            require(key not in drafts, 'duplicate authoring candidate')
            drafts[key] = r
            q = r['query_id']
            case = next(c for c in selection['cases'] if c['query_id'] == q)
            old = old_contexts[q, 'hate', 'original']
            _, prompt = replace_text(old, materials['demo:' + case['target_demo_id']]['source']['text'], r['text'])
            count = len(tokenizer.encode(prompt, add_special_tokens=False))
            history.append(r | {'round': name, 'full_prompt_tokens': count,
                                'token_difference_from_original': count - old['prompt_tokens'],
                                'selected': any(s['round'] == name and s['id'] == r['id'] for s in selection['selected']),
                                'new_model_scores_observed': False})
    require({s['id'] for s in selection['selected']} == {f'{q}-{c}' for q in ('541', '3169') for c in CONDITIONS[2:]} and len(selection['selected']) == 8, 'selected coverage differs')
    selected = {s['id']: s for s in selection['selected']}
    contexts, review_rows, case_records, source_ids = [], [], [], set()
    boundary_checks = 0
    for case in selection['cases']:
        q, target = case['query_id'], case['target_demo_id']
        original = old_contexts[q, 'hate', 'original']
        removed = old_contexts[q, 'hate', 'demo_removed_dictionary_fixed']
        old_text = materials['demo:' + target]['source']['text']
        query_text = materials['query:' + q]['source']['text']
        answer = accepted_field(materials['demo:' + target], 'hate')
        require(answer['available'] and answer['value'] == case['demo_answer'] == materials['demo:' + target]['source']['original_answer']['hate'], 'original example answer contract differs')
        basis = ['query:' + q, 'demo:' + target, f'relation:{q}:{target}']
        source_ids.update(basis)
        relation = {field: accepted_field(materials[basis[-1]], field) for field in ('topic_hate', 'rule_hate')}
        require((relation['topic_hate']['value'], relation['rule_hate']['value']) == (('direct', 'direct') if q == '541' else ('none', 'partial')), 'source relation changed')
        case_records.append(case | {'query_text': query_text, 'original_demo_text': old_text,
                                    'source_demo_reference': answer, 'source_relation': relation,
                                    'query_analysis_reference': overlays[q, 'hate'],
                                    'strict_U_branch': False, 'new_human_judgments': 0})
        baseline = None
        for condition in CONDITIONS:
            inherited = condition in ('A', 'B')
            source = removed if condition == 'B' else original
            replacement = None
            if inherited:
                messages, prompt = deepcopy(source['messages']), source['prompt_text']
            else:
                s = selected[f'{q}-{condition}']
                replacement = drafts[s['round'], s['id']]['text']
                messages, prompt = replace_text(original, old_text, replacement)
            require(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False) == prompt, 'chat template replay differs')
            encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
            ids = encoded['input_ids']
            ds = source['operation_manifest']['demo_ids']
            pos = layout(prompt, encoded['offset_mapping'], ds, materials, target, replacement)
            c = {'record_id': f'EM-{q}-hate-v1:{condition}', 'query_id': q, 'task': 'hate', 'condition': condition,
                 'messages': messages, 'prompt_text': prompt, 'prompt_sha256': sha(prompt.encode()),
                 'prompt_token_ids': ids, 'prompt_token_ids_sha256': sha(canonical(ids).encode()),
                 'prompt_tokens': len(ids), 'layout': pos,
                 'source_context_record_id': source['record_id'], 'source_prompt_sha256': source['prompt_sha256'],
                 'demo_ids': ds, 'entry_ids': source['operation_manifest']['entry_ids'],
                 'example_answer_counts': dict(Counter(materials['demo:' + d]['source']['original_answer']['hate'] for d in ds)),
                 'source_replay': inherited, 'score_execution_status': 'not_run_in_this_batch',
                 'input_scoring_eligible': False, 'mechanism_ready': False}
            require(c['entry_ids'] == original['operation_manifest']['entry_ids'], 'dictionary membership changed')
            dictionary = prompt.split('参考示例：\n')[0]
            require(dictionary == original['prompt_text'].split('参考示例：\n')[0], 'system or dictionary changed')
            if inherited:
                require(ids == source['prompt_token_ids'] and c['prompt_sha256'] == source['prompt_sha256'], 'historical prompt/token replay differs')
                require(pos['query_block'] == source['positions']['query_block_token_span'], 'historical query positions differ')
            if condition == 'A':
                require(len(ids) == case['prompt_tokens'], 'baseline count differs')
                baseline = c
                c['alignment'] = {'kind': 'historical_original'}
            elif condition == 'B':
                c['alignment'] = {'kind': 'historical_natural_deletion', 'positions_matched': False,
                                  'token_difference_from_original': len(ids) - baseline['prompt_tokens']}
            else:
                c['alignment'] = alignment_proof(baseline, c, target)
                require(c['example_answer_counts'] == baseline['example_answer_counts'], 'example answer distribution changed')
                if q == '3169':
                    start = prompt.index(f"示例 {pos['demos'][target]['slot']}\n文本：") + len(f"示例 {pos['demos'][target]['slot']}\n文本：")
                    if condition.startswith('D'):
                        local = replacement.index('嘿嘿')
                        cue = token_span(encoded['offset_mapping'], start + local, start + local + 2)
                        # The two chosen paraphrases also retain the original cue's position.
                        require(cue == [395, 396], '3169 paraphrase cue moved')
                        c['alignment']['retained_cue_token_span'] = cue
                    else:
                        require('嘿嘿' not in replacement, 'replacement retains the removed cue')
                        c['alignment']['target_demo_cue_absent'] = True
                review_rows.append(review_draft(selection, s, replacement, query_text, old_text, basis, c))
            c['candidate_boundaries'] = []
            for candidate in plan['catalog']['hate']:
                require(tokenizer.encode(prompt + candidate['canonical_answer'], add_special_tokens=False) == ids + candidate['answer_token_ids'], 'candidate boundary changed')
                require(len(ids) + candidate['answer_tokens'] + 1 + 64 <= 8192, 'candidate exceeds capacity')
                c['candidate_boundaries'].append({'ordinal': candidate['ordinal'], 'answer': candidate['canonical_answer'],
                                                  'answer_token_ids': candidate['answer_token_ids'], 'start': len(ids),
                                                  'end_exclusive': len(ids) + candidate['answer_tokens'], 'checked': True})
                boundary_checks += 1
            contexts.append(c)
    for r in review_rows:
        require(not r['ai_assessment']['human_confirmed'] and not r['human_review']['events'] and
                all(v is None for v in r['human_review']['values'].values()), 'AI draft became human review')
    require(len({r['prompt_sha256'] for r in contexts}) == 12, 'duplicate final prompt')
    audit = {'schema_version': 'evidence-matched-material-audit/v1', 'technical_validation': 'passed',
             'materials_status': 'draft_pending_human_review', 'cases': 2, 'new_texts': 8,
             'contexts': len(contexts), 'historical_text_and_token_replays': 4,
             'matched_full_prompt_position_checks': len(review_rows), 'candidate_boundary_checks': boundary_checks,
             'authoring_candidates_retained': len(history), 'tokenizer_tree_sha256': tree_sha,
             'transformers_version': importlib.metadata.version('transformers'),
             'new_human_judgments': 0, 'production_writes': 0, 'model_weights_loaded': False,
             'model_frameworks_loaded': False, 'new_model_forward_executed': False,
             'input_scoring_ready': False, 'mechanism_ready': False,
             'pending': ['human_review_of_new_materials', 'separate_version_for_accepted_materials_and_scoring_protocol'],
             'not_in_this_batch': ['position_swaps', 'short_long_variants', 'GPU_scoring', 'activation_patching']}
    code_paths = [Path(__file__).resolve(), ROOT / 'scripts/review/test_prepare_evidence_matched_materials.py',
                  ROOT / 'scripts/review/implement_evidence_case_uses.py', ROOT / 'scripts/review/update_evidence_case_uses.py',
                  ROOT / 'src/diagnostics/general_model_evidence_evaluation.py', ROOT / 'src/diagnostics/general_model_numeric_analysis.py',
                  ROOT / 'src/diagnostics/general_model_contexts.py', ROOT / 'src/diagnostics/general_model_tasks.py']
    for path in code_paths:
        take(path)
    files = {'materials.jsonl': jsonl(review_rows), 'contexts.jsonl': jsonl(contexts),
             'cases.json': json_bytes(case_records), 'audit.json': json_bytes(audit),
             'selection.json': json_bytes(selection), 'authoring_history.jsonl': jsonl(history),
             'source_material_records.jsonl': jsonl(materials[oid] for oid in sorted(source_ids)),
             'review_queue.json': json_bytes({'schema_version': 'evidence-matched-material-review-queue/v1',
                                             'status': 'pending_not_imported_to_online_session',
                                             'field_schema': selection['review_fields'], 'items': review_rows,
                                             'reuse_workbench': 'tools/general_model_paired_review_ui',
                                             'review_ui_integration': 'not_part_of_material_preparation'}),
             'execution_source.json': json_bytes({str(p.relative_to(ROOT)): {'sha256': file_sha(p), 'text': p.read_text()} for p in code_paths})}
    flat = [{k: r[k] for k in ('material_id', 'query_id', 'condition', 'kind', 'original_demo_text', 'text', 'fixed_demo_answer', 'prompt_tokens', 'status')} |
            {'ai_attack_severity': r['ai_assessment']['attack_severity'], 'ai_rationale': r['ai_assessment']['rationale'],
             'limitations': r['ai_assessment']['limitations'], **{'human_' + k: '' for k in r['human_review']['values']}} for r in review_rows]
    files['materials.csv'] = csv_bytes(flat)
    files['positions.csv'] = csv_bytes([{'query_id': c['query_id'], 'condition': c['condition'], 'prompt_tokens': c['prompt_tokens'],
                                        'query_block': c['layout']['query_block'], 'generation_start': c['layout']['generation_start'],
                                        'example_answer_counts': c['example_answer_counts'], 'alignment': c['alignment']} for c in contexts])
    files['MATERIALS.md'] = report(selection, case_records, review_rows, contexts, audit).encode()
    sources.unchanged()
    return files, sources.hashes, audit


def report(selection, cases, rows, contexts, audit):
    lines = ['# 第一批等长、原位对照材料', '',
             '已准备 541/hate、3169/hate 各两版内容替换 C1/C2 和两版语义保持改写 D1/D2，共 8 段新文本。另保留各自原始 A 和历史删除 B，合计 12 个完整提示。', '',
             '**状态：技术核验通过；新文本语义与标签均为 AI 草稿，待审核。** 本批未运行模型，未新增人工确认。旧冻结、旧用途决定和线上会话保持原样。', '',
             'C 的目标是去掉指定语义组合，同时保留示例自身的 hate 答案；D 的目标是保留原语义，仅调整措辞。所有 C/D 均保持词典、示例数、标签分布、完整提示 token 数、所有示例及其答案位置、查询位置和生成起点。', '',
             '| 案例 | A 原始 | B 自然删除 | C1/C2/D1/D2 | 固定替换位置 |',
             '| --- | ---: | ---: | ---: | --- |']
    for case in cases:
        q = case['query_id']; cs = {c['condition']: c for c in contexts if c['query_id'] == q}
        target = cs['A']['layout']['demos'][case['target_demo_id']]
        lines.append(f"| {q}/hate | {cs['A']['prompt_tokens']} | {cs['B']['prompt_tokens']} | {cs['C1']['prompt_tokens']} | 示例 {target['slot']}；正文覆盖 token {target['text']} |")
    lines += ['', 'token 位置均从 0 开始，区间右端不包含；正文覆盖区间可包含与标点合并的换行 token。B 保留原始自然删除输入，长度和位置不匹配。C/D 不使用尾部填充。', '']
    for case in cases:
        q = case['query_id']
        lines += [f'## {q}/hate', '', '**查询原文**', '', '> ' + case['query_text'], '',
                  f"**原示例 {case['target_demo_id']}（A）**", '', '> ' + case['original_demo_text'], '',
                  f"示例 hate 答案：`{case['demo_answer']}`。原关系：话题 `{case['source_relation']['topic_hate']['value']}`、规则 `{case['source_relation']['rule_hate']['value']}`。这些是原材料记录，不自动确认新文本。", '',
                  '**D 版拟保持的内容：**' + '；'.join(case['preserve_semantics']) + '。', '',
                  '**本案例的解释限制：**' + case['semantic_limit'], '']
        for row in (r for r in rows if r['query_id'] == q):
            kind = '内容替换' if row['condition'].startswith('C') else '语义保持改写'
            lines += [f"### {row['material_id']} · {kind}", '', '> ' + row['text'], '',
                      f"AI 拟定：`{row['fixed_demo_answer']}`，攻击严重度 `{row['ai_assessment']['attack_severity']}`；完整提示 `{row['prompt_tokens']}` tokens。", '',
                      '依据：' + row['ai_assessment']['rationale'], '',
                      '待核对：文字是否自然；新文本标签／严重度是否成立；C 是否去掉指定语义或 D 是否保留；是否还有未记录的对象、命题或立场变化。', '']
    lines += ['## 核验与后续', '',
              f"4 个 A/B 历史文本和 token 精确重放；8 个 C/D 的完整位置、前缀和后缀 token 核对通过；{audit['candidate_boundary_checks']} 个候选答案边界通过。所有未修改示例的 token 及位置一致。3169 的两版 D 还保持了目标示例中‘嘿嘿’在 token [395, 396) 的位置。", '',
              '541 的四版均保留一个 🤔、三个句号和两个问号；3169 的四版保留两个逗号与三个句号构成的省略结尾。内部语义成分并未逐 token 对齐，这些匹配不构成 activation patching 准备完成。', '',
              '本轮所有分词检查过的草稿均在 authoring_history.jsonl 中保留，只按语义可用性和 token 匹配调整文本，未用新模型分数挑选。案例选择已受之前自然编辑结果的启发，应报告为探索性发现集；每类两版也不算两个独立案例。', '',
              '审核意见按 material_id 和 text_sha256 绑定到新文本；接受、修改、存疑均应留记录。修改后生成新版本并重验 token，不能覆盖本批或原材料。审核后再冻结正式评分协议；位置交换和短／长版本属于后续独立材料批次。', '',
              '文件：materials.jsonl / materials.csv 为逐项材料和空白人工字段；contexts.jsonl 为完整模型输入；positions.csv 为位置核验；review_queue.json 为 8 项待审队列。队列尚未接入线上标注页面，原页面的确认数不包含本批。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, default=ROOT / 'models/base/Qwen3-8B')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    target, tokenizer = args.output.resolve(), args.tokenizer.resolve()
    allowed_parent = BASE / 'reviews/analysis-freeze-20260912/matched-materials-v1'
    require(target.parent == allowed_parent and target.name.startswith('prepared-'), 'output must be a new prepared version under matched-materials-v1')
    require(tokenizer.is_relative_to(ROOT), 'tokenizer must be local')
    files, source_hashes, audit = build(tokenizer)
    if args.check:
        manifest = read_json(target / 'manifest.json')
        require(manifest['schema_version'] == 'evidence-matched-materials/v1' and manifest['status'] == 'draft_pending_human_review', 'wrong receipt')
        require(manifest['source_files'] == source_hashes and set(manifest['artifacts']) == set(files), 'source/output inventory changed')
        require({p.name for p in target.iterdir()} == {*files, 'manifest.json'}, 'unexpected output files')
        for name, raw in files.items():
            require((target / name).read_bytes() == raw and manifest['artifacts'][name] == sha(raw), 'material reconstruction differs: ' + name)
        print('Materials reproduce byte for byte; 8 matched variants, 4 historical replays, 24 candidate boundaries; no model forward.')
    else:
        files['manifest.json'] = json_bytes({'schema_version': 'evidence-matched-materials/v1', 'status': 'draft_pending_human_review',
                                            'created_at': datetime.now(timezone.utc).isoformat(), 'source_files': source_hashes,
                                            'artifacts': {n: sha(raw) for n, raw in files.items()}})
        write_output(target, files)
        print(canonical(audit))


if __name__ == '__main__':
    main()
