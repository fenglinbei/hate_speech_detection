#!/usr/bin/env python3
"""Prepare second-batch position and length materials without model forward."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.general_model_evidence_evaluation import (
    EXPERIMENT, Sources, canonical, csv_bytes, file_sha, json_bytes, jsonl,
    local_path, read_json, read_lines, require, sha, unique, write_output,
)
from diagnostics.general_model_contexts import _render_demo_blocks
from scripts.review.implement_evidence_case_uses import TOKENIZER_FILES
from scripts.review.prepare_evidence_matched_materials import layout
from scripts.review.update_evidence_case_uses import accepted_field

BASE = ROOT / EXPERIMENT
SPEC = BASE / 'position-length-materials-v1'
VARIANTS = ('A', 'C1', 'C2', 'D1', 'D2')
REFERENCES = ('A', 'B', 'C1', 'C2', 'D1', 'D2')


def render(source, order, texts, answers):
    require(len(order) == len(set(order)), 'duplicate demo identity')
    require(set(order) <= set(texts) & set(answers), 'missing demo payload')
    for d in order:
        require(texts[d].strip() == texts[d] and texts[d] and
                not any(s in texts[d] for s in ('\n', '\r', '<|', '|>')), 'invalid demo text')
        require(answers[d] in ('hate', 'non-hate'), 'invalid example answer')
    old_user = source['messages'][1]['content']
    require(old_user.count('参考示例：\n') == old_user.count('\n\n待判断文本（JSON 字符串）：\n') == 1, 'ambiguous source blocks')
    prefix, rest = old_user.split('参考示例：\n')
    _, query = rest.split('\n\n待判断文本（JSON 字符串）：\n')
    demos = _render_demo_blocks('hate', [{'id': d, 'content': texts[d]} for d in order], [answers[d] for d in order], 'Full')
    user = prefix + demos + '\n\n待判断文本（JSON 字符串）：\n' + query
    require(source['prompt_text'].count(old_user) == 1, 'ambiguous user slot')
    messages = deepcopy(source['messages'])
    messages[1]['content'] = user
    return messages, source['prompt_text'].replace(old_user, user)


def slot_geometry(context):
    return {p['slot']: {k: p[k] for k in ('block', 'text', 'answer')}
            for p in context['layout']['demos'].values()}


def swap_proof(before, after, target, partner):
    expected = list(before['demo_ids'])
    i, j = expected.index(target), expected.index(partner)
    expected[i], expected[j] = expected[j], expected[i]
    require(after['demo_ids'] == expected, 'not the registered two-demo swap')
    require(before['texts_by_demo'] == after['texts_by_demo'] and before['answers_by_demo'] == after['answers_by_demo'], 'swap changed a text or answer')
    require(before['answers_by_demo'][target] == before['answers_by_demo'][partner], 'swap changes labels by position')
    left, right = before['prompt_token_ids'], after['prompt_token_ids']
    require(len(left) == len(right), 'swap changed total length')
    require(slot_geometry(before) == slot_geometry(after), 'swap moved a slot boundary or answer position')
    require(before['layout']['query_block'] == after['layout']['query_block'], 'swap moved query')
    editable = set()
    for d in (target, partner):
        s, e = before['layout']['demos'][d]['text']
        editable.update(range(s, e))
    require(all(x == y for k, (x, y) in enumerate(zip(left, right)) if k not in editable), 'swap changed tokens outside the two text slots')
    for d in before['demo_ids']:
        s, e = before['layout']['demos'][d]['text']
        a, b = after['layout']['demos'][d]['text']
        require(left[s:e] == right[a:b], 'moved or retained text tokenization changed')
    return {'passed': True, 'scope': 'two_example_relative_order_in_common_rewritten_background',
            'target_slot_before': i + 1, 'target_slot_after': j + 1,
            'prompt_tokens': len(left), 'slot_boundaries_and_answer_positions_fixed': True,
            'labels_fixed_at_each_slot': True, 'query_position_fixed': True,
            'all_other_tokens_fixed': True, 'absolute_single_example_position_isolated': False}


def length_proof(before, after, changed_demo, minimum=4):
    require(before['demo_ids'] == after['demo_ids'] and before['answers_by_demo'] == after['answers_by_demo'], 'length edit changes order or answers')
    changed = [d for d in before['demo_ids'] if before['texts_by_demo'][d] != after['texts_by_demo'][d]]
    require(changed == [changed_demo], 'length edit changes more than its selected text')
    left, right = before['prompt_token_ids'], after['prompt_token_ids']
    delta = len(right) - len(left)
    require(delta and abs(delta) >= minimum, 'insufficient intended length change')
    old, new = before['layout'], after['layout']
    # Include the newline/output prefix before the unchanged answer. End punctuation
    # may merge with the newline, so the overlapping text span alone is insufficient.
    start = old['demos'][changed_demo]['text'][0]
    end = old['demos'][changed_demo]['answer'][0]
    new_start = new['demos'][changed_demo]['text'][0]
    new_end = new['demos'][changed_demo]['answer'][0]
    require(start == new_start and new_end - end == delta, 'answer boundary shift does not explain full length change')
    require(left[:start] == right[:start] and left[end:] == right[new_end:], 'length edit altered prefix or suffix tokens')
    changed_slot = old['demos'][changed_demo]['slot']
    for d, p in old['demos'].items():
        if d == changed_demo:
            require(new['demos'][d]['answer'] == [v + delta for v in p['answer']], 'edited example answer shift differs')
            continue
        shift = delta if p['slot'] > changed_slot else 0
        for part in ('block', 'text', 'answer'):
            require(new['demos'][d][part] == [v + shift for v in p[part]], 'unexpected other-demo displacement')
    require(new['query_block'] == [v + delta for v in old['query_block']], 'query shift differs from length delta')
    return {'passed': True, 'token_delta': delta, 'prefix_and_suffix_token_ids_preserved': True,
            'suffix_anchor': 'unchanged_example_answer_including_boundary_token',
            'query_token_shift': delta, 'pure_length_effect_isolated': False,
            'scope': 'natural_wording_length_and_downstream_position_change'}


def build(tokenizer_path):
    sources = Sources(ROOT)
    def take(path, expected=None):
        return sources.verify(path, expected or file_sha(path))
    selection = read_json(take(SPEC / 'selection.json'))
    require(selection['status'] == 'draft_pending_human_review', 'new adoption needs a separate version')
    first_pointer = read_json(take(BASE / 'matched-materials-v1/current.json'))
    require(first_pointer['manifest_sha256'] == selection['first_batch_manifest_sha256'], 'first material version changed')
    first_folder = local_path(ROOT, str(EXPERIMENT / first_pointer['materials_path']))
    first_manifest = read_json(take(first_folder / 'manifest.json', first_pointer['manifest_sha256']))
    for path, h in first_manifest['source_files'].items():
        take(local_path(ROOT, path), h)
    for name, h in first_manifest['artifacts'].items():
        require(Path(name).name == name, 'invalid first artifact path')
        take(first_folder / name, h)
    feedback_pointer = read_json(take(BASE / 'matched-materials-review-v1/current.json'))
    require(feedback_pointer['feedback_sha256'] == selection['first_batch_feedback_sha256'], 'feedback version changed')
    feedback = read_json(take(BASE / feedback_pointer['feedback_path'], feedback_pointer['feedback_sha256']))
    require(feedback['source_manifest_sha256'] == first_pointer['manifest_sha256'] and feedback['batch_overall_no_objection'] is True and feedback['individual_field_decisions'] == [], 'wrong feedback scope')
    first_contexts = unique(read_lines(first_folder / 'contexts.jsonl'), lambda r: (r['query_id'], r['condition']))
    first_materials = unique(read_lines(first_folder / 'materials.jsonl'), lambda r: r['material_id'])
    require(feedback['materials'] == [{'material_id': r['material_id'], 'text_sha256': r['text_sha256']} for r in first_materials.values()], 'feedback text binding differs')
    freeze_pointer = read_json(BASE / 'analysis-reference-v1/current.json')
    freeze = BASE / freeze_pointer['freeze_path']
    materials = unique(read_lines(freeze / 'material_reviews.jsonl'), lambda r: r['record_id'])
    first_selection = read_json(first_folder / 'selection.json')
    for key in ('preference_snapshot_id', 'relation_preference_snapshot_id'):
        require(selection[key] == first_selection[key], 'preference snapshots differ')
    plan_ref = read_json(ROOT / 'exps/causal_context/general_model_ld_nolabel_v1/plan_ref.json')
    plan = read_json(Path(plan_ref['target_path']) / 'plan.json')
    inventory = []
    for p in sorted(tokenizer_path.rglob('*')):
        if p.is_file() and p.name in TOKENIZER_FILES:
            take(p)
            inventory.append({'path': str(p.relative_to(tokenizer_path)), 'size': p.stat().st_size, 'sha256': file_sha(p)})
    require(sha(canonical(inventory).encode()) == plan['generation_runtime_identity']['tokenizer_tree_sha256'], 'tokenizer changed')
    for name in ('USE_TORCH', 'USE_TF', 'USE_FLAX', 'USE_TORCH_XLA'):
        os.environ[name] = '0'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False, use_fast=True)
    require(not {'torch', 'tensorflow', 'jax'} & set(sys.modules), 'loaded model framework')
    require(tokenizer.eos_token_id == plan['eos_token_id'] and tokenizer.pad_token_id == plan['pad_token_id'], 'special tokens changed')

    def encode(q, condition, family, source, order, texts, answers):
        messages, prompt = render(source, order, texts, answers)
        require(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False) == prompt, 'chat template changed')
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded['input_ids']
        local = {f'demo:{d}': {'source': {'text': texts[d], 'original_answer': {'hate': answers[d]}}} for d in order}
        positions = layout(prompt, encoded['offset_mapping'], order, local, '')
        require(prompt.split('参考示例：\n')[0] == source['prompt_text'].split('参考示例：\n')[0], 'dictionary or system changed')
        boundaries = []
        for candidate in plan['catalog']['hate']:
            require(tokenizer.encode(prompt + candidate['canonical_answer'], add_special_tokens=False) == ids + candidate['answer_token_ids'], 'answer boundary changed')
            require(len(ids) + candidate['answer_tokens'] + 1 + 64 <= 8192, 'capacity exceeded')
            boundaries.append({'ordinal': candidate['ordinal'], 'answer_token_ids': candidate['answer_token_ids'], 'start': len(ids), 'checked': True})
        return {'record_id': f'EPL-{q}-hate-v1:{condition}', 'query_id': q, 'task': 'hate', 'condition': condition,
                'family': family, 'messages': messages, 'prompt_text': prompt, 'prompt_sha256': sha(prompt.encode()),
                'prompt_token_ids': ids, 'prompt_token_ids_sha256': sha(canonical(ids).encode()), 'prompt_tokens': len(ids),
                'demo_ids': list(order), 'texts_by_demo': {d: texts[d] for d in order},
                'answers_by_demo': {d: answers[d] for d in order}, 'answer_counts': dict(Counter(answers[d] for d in order)),
                'entry_ids': source['entry_ids'], 'layout': positions, 'candidate_boundaries': boundaries,
                'model_forward_executed': False, 'input_scoring_eligible': False, 'mechanism_ready': False}

    drafts, history = {}, []
    for name in selection['authoring_rounds']:
        require(Path(name).name == name, 'invalid authoring file')
        for row in read_json(take(SPEC / 'authoring' / name))['candidates']:
            key = name, row['id']
            require(key not in drafts, 'duplicate authoring draft')
            drafts[key] = row
            source = first_contexts[row['query_id'], 'A']; order = source['demo_ids']
            texts = {d: materials['demo:' + d]['source']['text'] for d in order}
            answers = {d: materials['demo:' + d]['source']['original_answer']['hate'] for d in order}
            texts[row['source_demo_id']] = row['text']
            c = encode(row['query_id'], 'authoring', 'authoring', source, order, texts, answers)
            span = c['layout']['demos'][row['source_demo_id']]['block']
            history.append(row | {'round': name, 'full_prompt_tokens': c['prompt_tokens'], 'block_tokens': span[1] - span[0],
                                  'token_delta': c['prompt_tokens'] - source['prompt_tokens'], 'new_model_scores_observed': False})
    selected = {s['id']: s for s in selection['selected']}
    require(len(selected) == 6 and set(selected) == {f'{q}-{kind}' for q in ('541', '3169') for kind in ('P', 'S', 'L')}, 'unexpected material scope')
    chosen = {oid: drafts[s['round'], oid] for oid, s in selected.items()}
    for row in history:
        row['selected'] = selected[row['id']]['round'] == row['round']
    contexts, review_rows, screening, cases, contrasts = [], [], [], [], []
    def contrast(q, label, kind, terms):
        contrasts.append({'contrast_id': f'EPL-{q}:{label}', 'query_id': q, 'task': 'hate', 'kind': kind,
                          'terms': [{'condition': condition, 'coefficient': weight} for condition, weight in terms],
                          'author': 'assistant', 'status': 'prepared_not_scored', 'human_mechanism_judgment': False})
    for spec in selection['cases']:
        q, target, partner = spec['query_id'], spec['target_demo_id'], spec['partner_demo_id']
        source = first_contexts[q, 'A']; order = source['demo_ids']
        texts = {d: materials['demo:' + d]['source']['text'] for d in order}
        answers = {d: materials['demo:' + d]['source']['original_answer']['hate'] for d in order}
        require(answers[target] == answers[partner] == spec['answer'], 'position label mismatch')
        require([order.index(target) + 1, order.index(partner) + 1] == spec['swap_slots_1_based'], 'wrong swap slots')
        partner_ref = accepted_field(materials['demo:' + partner], 'hate')
        require(partner_ref['available'] and partner_ref['value'] == answers[partner], 'partner has an unresolved or changed answer')
        target_span = source['layout']['demos'][target]['block']; target_size = target_span[1] - target_span[0]
        for d in order:
            r = materials['demo:' + d]; span = source['layout']['demos'][d]['block']; adopted = accepted_field(r, 'hate')
            screening.append({'query_id': q, 'demo_id': d, 'slot': order.index(d) + 1, 'block_tokens': span[1] - span[0],
                              'block_length_difference': span[1] - span[0] - target_size, 'original_answer': answers[d],
                              'adopted_answer': adopted, 'same_original_answer_as_target': answers[d] == answers[target],
                              'already_exact_same_label_available': d != target and span[1] - span[0] == target_size and
                              answers[d] == answers[target] and adopted['available'] and adopted['value'] == answers[d],
                              'chosen_for_rewrite': d == partner})
        require(not any(s['already_exact_same_label_available'] for s in screening if s['query_id'] == q), 'an exact source partner exists; reassess unnecessary rewrite')
        by_condition = {}
        for variant in REFERENCES:
            old = first_contexts[q, variant]
            texts_v = dict(texts)
            if variant not in ('A', 'B'):
                texts_v[target] = first_materials[f'{q}-{variant}']['text']
            c = encode(q, 'R-' + variant, 'first_batch_reference', source, old['demo_ids'], texts_v, answers)
            require(c['prompt_text'] == old['prompt_text'] and c['prompt_token_ids'] == old['prompt_token_ids'], 'first batch replay changed')
            c['first_batch_record_id'] = old['record_id']; c['reference_replay_verified'] = True
            contexts.append(c); by_condition[c['condition']] = c
        for variant in VARIANTS:
            texts_v = dict(by_condition['R-' + variant]['texts_by_demo'])
            texts_v[partner] = chosen[q + '-P']['text']
            p0 = encode(q, 'P0-' + variant, 'position_common_background', source, order, texts_v, answers)
            p0['background_edit_proof'] = length_proof(by_condition['R-' + variant], p0, partner, minimum=1)
            swapped = list(order); i, j = swapped.index(target), swapped.index(partner)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            p1 = encode(q, 'P1-' + variant, 'position_swapped', source, swapped, texts_v, answers)
            p1['position_proof'] = swap_proof(p0, p1, target, partner)
            for c in (p0, p1):
                c['target_variant'] = variant; c['partner_material_id'] = q + '-P'
                contexts.append(c); by_condition[c['condition']] = c
            contrast(q, 'position-' + variant, 'two_example_position_effect', [(p1['condition'], 1), (p0['condition'], -1)])
            contrast(q, 'background-' + variant, 'partner_rewrite_bridge', [(p0['condition'], 1), ('R-' + variant, -1)])
        for k in (1, 2):
            for position in ('P0', 'P1'):
                contrast(q, f'semantic-{position}-{k}', 'paired_content_paraphrase_comparison', [(f'{position}-D{k}', 1), (f'{position}-C{k}', -1)])
            contrast(q, f'position-semantic-{k}', 'position_by_semantic_interaction', [(f'P1-D{k}', 1), (f'P1-C{k}', -1), (f'P0-D{k}', -1), (f'P0-C{k}', 1)])
        for suffix in ('S', 'L'):
            edited = dict(texts); edited[target] = chosen[q + '-' + suffix]['text']
            c = encode(q, suffix, 'length_' + ('short' if suffix == 'S' else 'long'), source, order, edited, answers)
            c['length_proof'] = length_proof(by_condition['R-A'], c, target, selection['length_design']['minimum_absolute_token_change'])
            require((c['length_proof']['token_delta'] < 0) == (suffix == 'S'), 'length direction incorrect')
            contexts.append(c); by_condition[suffix] = c
            contrast(q, 'length-' + suffix, 'length_wording_sensitivity', [(suffix, 1), ('R-A', -1)])
        contrast(q, 'length-L-S', 'length_wording_sensitivity', [('L', 1), ('S', -1)])
        cases.append(spec | {'original_prompt_tokens': source['prompt_tokens'], 'position_prompt_tokens': by_condition['P0-A']['prompt_tokens'],
                             'short_prompt_tokens': by_condition['S']['prompt_tokens'], 'long_prompt_tokens': by_condition['L']['prompt_tokens'],
                             'source_partner_reference': partner_ref, 'source_partner_severity': accepted_field(materials['demo:' + partner], 'attack_severity')})
        for suffix in ('P', 'S', 'L'):
            oid = q + '-' + suffix; row, s = chosen[oid], selected[oid]
            source_id = row['source_demo_id']; c = by_condition['P0-A' if suffix == 'P' else suffix]
            review_rows.append(row | {'material_id': oid, 'text_sha256': sha(row['text'].encode()),
                                      'source_text': texts[source_id], 'query_text': materials['query:' + q]['source']['text'],
                                      'source_record_sha256': sha(canonical(materials['demo:' + source_id]).encode()),
                                      'ai_assessment': {k: s[k] for k in ('ai_hate', 'ai_attack_severity', 'rationale')},
                                      'source_hate_reference': accepted_field(materials['demo:' + source_id], 'hate'),
                                      'prompt_tokens_in_unswapped_context': c['prompt_tokens'],
                                      'token_delta_from_first_A': c['prompt_tokens'] - source['prompt_tokens'],
                                      'status': 'draft_pending_human_review', 'author': 'assistant',
                                      'human_review': {'status': 'pending', 'values': {k: None for k in selection['review_fields']}, 'events': []},
                                      'input_scoring_eligible': False, 'mechanism_ready': False})
    available = {(c['query_id'], c['condition']) for c in contexts}
    require(len(contexts) == len(available) == len({c['prompt_sha256'] for c in contexts}) == 36, 'context coverage or identity differs')
    for r in contrasts:
        require(sum(t['coefficient'] for t in r['terms']) == 0 and
                all((r['query_id'], t['condition']) in available for t in r['terms']), 'invalid contrast geometry')
    require(all(not r['human_review']['events'] and all(v is None for v in r['human_review']['values'].values()) for r in review_rows), 'draft adopted as human fields')
    code_paths = [Path(__file__).resolve(), ROOT / 'scripts/review/test_prepare_evidence_position_length_materials.py']
    for p in code_paths:
        take(p)
    audit = {'schema_version': 'evidence-position-length-material-audit/v1', 'technical_validation': 'passed',
             'materials_status': 'draft_pending_human_review', 'cases': 2, 'new_texts': 6,
             'contexts': 36, 'first_batch_replays': 12, 'new_contexts': 24, 'position_contexts': 20,
             'length_contexts': 4, 'position_pair_checks': 10, 'background_bridge_checks': 10,
             'length_shift_checks': 4, 'candidate_boundary_checks': 72, 'contrast_specs': len(contrasts),
             'source_partner_candidates_screened': len(screening), 'authoring_candidates_retained': len(history),
             'tokenizer_tree_sha256': plan['generation_runtime_identity']['tokenizer_tree_sha256'],
             'transformers_version': importlib.metadata.version('transformers'),
             'first_batch_overall_feedback_bound': True, 'new_individual_human_fields': 0,
             'new_batch_human_adoption': False, 'new_model_forward_executed': False,
             'model_frameworks_loaded': False, 'production_writes': 0, 'review_queue_deployed': False,
             'input_scoring_ready': False, 'mechanism_ready': False}
    files = {'materials.jsonl': jsonl(review_rows), 'contexts.jsonl': jsonl(contexts), 'cases.json': json_bytes(cases),
             'partner_screening.jsonl': jsonl(screening), 'authoring_history.jsonl': jsonl(history),
             'contrasts.jsonl': jsonl(contrasts), 'audit.json': json_bytes(audit), 'selection.json': json_bytes(selection),
             'first_batch_feedback.json': json_bytes(feedback),
             'execution_source.json': json_bytes({str(p.relative_to(ROOT)): {'sha256': file_sha(p), 'text': p.read_text()} for p in code_paths}),
             'review_queue.json': json_bytes({'schema_version': 'evidence-position-length-review-queue/v1',
                                             'status': 'pending_not_imported_to_online_session', 'fields': selection['review_fields'],
                                             'items': review_rows, 'reuse_workbench': 'tools/general_model_paired_review_ui'})}
    files['materials.csv'] = csv_bytes([{k: r[k] for k in ('material_id', 'query_id', 'kind', 'source_demo_id', 'query_text', 'source_text', 'text', 'prompt_tokens_in_unswapped_context', 'token_delta_from_first_A', 'ai_assessment', 'status')} |
                                       {'human_' + k: '' for k in selection['review_fields']} for r in review_rows])
    files['positions.csv'] = csv_bytes([{'query_id': c['query_id'], 'condition': c['condition'], 'family': c['family'],
                                        'prompt_tokens': c['prompt_tokens'], 'demo_order': c['demo_ids'], 'query_span': c['layout']['query_block'],
                                        'proof': c.get('position_proof', c.get('length_proof', c.get('background_edit_proof', 'historical_replay')))} for c in contexts])
    files['MATERIALS.md'] = report(cases, review_rows).encode()
    sources.unchanged()
    return files, sources.hashes, audit


def report(cases, rows):
    lines = ['# 第二批：位置交换与短／长版本', '',
             '第一批“没什么问题”的整体反馈已独立记录，绑定原 8 段正文；原稿与逐字段来源保持不变。第二批新增 6 段文本，均为待审核 AI 稿，没有运行模型。', '',
             'P 是交换用配对材料；S/L 是目标示例的短／长版本。每例位置实验对 A、C1、C2、D1、D2 都准备 P0 原顺序及 P1 交换顺序。共 20 个位置条件、4 个长度条件，另保留 12 个第一批提示作参考。', '',
             '| 案例 | 交换位置 | 第一批 A | P0/P1 共同背景 | 短版 S | 长版 L |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for c in cases:
        lines.append(f"| {c['query_id']}/hate | {c['swap_slots_1_based'][0]} ↔ {c['swap_slots_1_based'][1]} | {c['original_prompt_tokens']} | {c['position_prompt_tokens']} | {c['short_prompt_tokens']} | {c['long_prompt_tokens']} |")
    lines += ['', '表中长度都是完整提示的 token 数。P0/P1 的所有槽位边界、答案位置及每个槽位的标签相同；只交换两条示例的正文及各自答案。编号留在原槽位。短／长版使用原始其他示例，查询的 token 位置按长度差相应移动。', '',
              '**解释时要分开两项比较：** P1−P0 检验同一配对背景中的顺序变化；P0−第一批对应版本检验配对材料改写带来的背景变化。不能用 P1−第一批 A 直接声称位置效应。', '']
    names = {'position_partner': '位置交换用材料', 'length_short': '目标示例短版', 'length_long': '目标示例长版'}
    for case in cases:
        q = case['query_id']; group = [r for r in rows if r['query_id'] == q]
        lines += [f'## {q}/hate', '', '**查询**', '', '> ' + group[0]['query_text'], '',
                  '**配对选择依据：**' + case['partner_choice'], '', '**位置解释范围：**' + case['position_limit'], '']
        for row in group:
            lines += [f"### {row['material_id']} · {names[row['kind']]}", '', f"原示例 #{row['source_demo_id']}：", '', '> ' + row['source_text'], '',
                      '**新文本：**', '', '> ' + row['text'], '',
                      f"AI 拟定：`{row['ai_assessment']['ai_hate']}`，严重度 `{row['ai_assessment']['ai_attack_severity']}`。{row['ai_assessment']['rationale']}", '',
                      f"完整提示：{row['prompt_tokens_in_unswapped_context']} tokens；相对第一批 A：{row['token_delta_from_first_A']:+d}。", '',
                      '待核对：是否自然；对象、命题、立场和词义是否保持；标签与严重度是否适当；有无遗漏或额外引入的信息。', '']
    lines += ['## 技术核验与后续', '',
              '12 个第一批提示逐字与逐 token 重放；10 对位置交换的几何和 token 检查、10 对背景改写、4 个长度变化以及 72 个候选答案边界通过。位置交换对全部五种目标内容采用同一配对和相同交换，不按效果选择材料。', '',
              '位置实验同时移动目标与配对示例，不能单独归因为某一条的绝对位置。3169 只测试相邻顺序；541 的跨度不同，不能据此作跨案例距离效应估计。短／长版同时改变措辞、重复、信息密度和后文位置，只提供长度敏感性证据。', '',
              'contrasts.jsonl 提前列出位置比较、背景改写比较、同位置 C/D 比较、位置与语义交互、长短比较；它们仍为 AI 制定的待评分比较，不能当成人工机制裁决。保留以后出现的无效、反向及评分口径敏感结果。', '',
              '本批审核意见按 material_id 和 text_sha256 绑定；修改另出版本并重验 token。六项新材料确认后，再统一冻结正式评分协议。当前未接入线上页面、未运行 GPU，也未开展激活采集或 patching。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, default=ROOT / 'models/base/Qwen3-8B')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    target, tokenizer = args.output.resolve(), args.tokenizer.resolve()
    require(target.parent == BASE / 'reviews/analysis-freeze-20260912/position-length-materials-v1' and target.name.startswith('prepared-'), 'output must be a new prepared version')
    require(tokenizer.is_relative_to(ROOT), 'tokenizer must be local')
    files, hashes, audit = build(tokenizer)
    if args.check:
        manifest = read_json(target / 'manifest.json')
        require(manifest['schema_version'] == 'evidence-position-length-materials/v1' and manifest['status'] == 'draft_pending_human_review', 'wrong output receipt')
        require(manifest['source_files'] == hashes and set(manifest['artifacts']) == set(files), 'inventory changed')
        require({p.name for p in target.iterdir()} == {*files, 'manifest.json'}, 'unexpected files')
        for name, raw in files.items():
            require((target / name).read_bytes() == raw and manifest['artifacts'][name] == sha(raw), 'reconstruction changed: ' + name)
        print('Second material batch reproduces byte for byte; 36 contexts, 10 swaps, 72 candidate boundaries; no model forward.')
    else:
        files['manifest.json'] = json_bytes({'schema_version': 'evidence-position-length-materials/v1', 'status': 'draft_pending_human_review',
                                            'created_at': datetime.now(timezone.utc).isoformat(), 'source_files': hashes,
                                            'artifacts': {name: sha(raw) for name, raw in files.items()}})
        write_output(target, files)
        print(canonical(audit))


if __name__ == '__main__':
    main()
