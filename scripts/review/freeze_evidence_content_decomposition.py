#!/usr/bin/env python3
"""Prepare/freeze content decompositions using only the local CPU tokenizer.

No model forward, retrieval, human writeback, or prior artifact mutation occurs.
New materials carry assistant assessments, never inferred human confirmations.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.general_model_evidence_evaluation import (
    EXPERIMENT, Sources, canonical, csv_bytes, file_sha, json_bytes, jsonl,
    local_path, read_json, read_lines, require, sha, write_output,
)
from diagnostics.evidence_label_calibration import ENCODINGS, PROBES, validate_block
from scripts.review.freeze_evidence_label_calibration import (
    MAPPINGS, QUERY, make_catalog, no_gold, transformed_messages, load_frozen as load_parent,
)
from scripts.review.prepare_evidence_matched_materials import layout, alignment_proof, replace_text
from scripts.review.run_evidence_interventions import DESCRIPTOR, SCHEDULE

BASE = ROOT / EXPERIMENT
PUBLIC = BASE / 'content-decomposition-v1'
WORK = BASE / 'reviews/analysis-freeze-20260914/content-decomposition-v1'
SOURCE = PUBLIC / 'materials-source-01.json'
PARENT_MANIFEST = '8b5f0137a739b4147d11c3350d197a55a720b5f74eb868dc54554aa9bcaddaa9'
VARIANTS = [('original', None)] + [('original', p) for p, _ in PROBES] + [('ab_forward', None), ('ab_reverse', None)]
TARGETS = {'541': '826', '3169': '3660'}
HISTORICAL = ('R-A', 'R-C1', 'R-C2', 'R-D1', 'R-D2')
CELLS = {'541': ('T0R0', 'T1R0', 'T0R1', 'T1R1'), '3169': ('G_H', 'G_E', 'L_H', 'L_O')}
NEW_CODE = tuple('scripts/review/' + name + '.py' for name in (
    'freeze_evidence_content_decomposition', 'run_evidence_content_decomposition',
    'analyze_evidence_content_decomposition', 'test_evidence_content_decomposition',
    'audit_evidence_content_decomposition',
))


def review_materials(source):
    rows = source['materials']
    expected = {(q, f, cell) for q in TARGETS for f in (1, 2) for cell in CELLS[q]}
    require(len(rows) == 16 and {(r['query_id'], r['template'], r['cell']) for r in rows} == expected, 'material frame differs')
    require(source['authorization']['gpu_execution_this_turn'] is False
            and source['authorization']['individual_human_adoption'] is False, 'authorization scope differs')
    index = {(r['query_id'], r['template'], r['cell']): r for r in rows}
    for r in rows:
        require(r['task'] == 'hate' and r['target_demo'] == TARGETS[r['query_id']]
                and r['target_slot'] == (5 if r['query_id'] == '541' else 3), 'target changed')
        require(r['condition'] == f'F{r["template"]}-{r["cell"]}', 'material condition differs')
        require(r['text'] and r['text'].strip() == r['text'] and not any(s in r['text'] for s in ('\n', '\r', '<|', '|>')), 'unsafe text structure')
        require(sha(r['text'].encode()) == r['text_sha256'], 'material text digest differs')
        expected_answer = 'non-hate' if r['query_id'] == '541' else 'hate'
        require(r['presented_answer'] == r['ai_review']['expected_hate'] == expected_answer, 'AI label/presented answer mismatch')
        require(r['ai_review']['attack_severity'] == (0 if expected_answer == 'non-hate' else 2), 'AI severity differs')
        require(r['ai_review']['author'] == 'assistant' and r['ai_review']['intended_factors_pass'], 'missing assistant review')
        require(r['human_review'] == {'status': 'no_individual_decision', 'hate': None,
            'attack_severity': None, 'naturalness': None, 'mechanism': None}, 'new human decision invented')
    for f in (1, 2):
        for rule in (0, 1):
            left, right = [index['541', f, f'T{topic}R{rule}']['text'] for topic in (0, 1)]
            require(left.count('篮球迷') == right.count('异性恋') == 2
                    and left.replace('篮球迷', '异性恋') == right, 'topic contrast changes more than category')
        for a, b, replacement in [('G_H', 'G_E', '黑人'), ('L_H', 'L_O', '哈哈')]:
            left, right = [index['3169', f, cell]['text'] for cell in (a, b)]
            require(left.count('嘿嘿') == 1 and left.replace('嘿嘿', replacement) == right, 'lexical pair changes unrelated wording')
        require('嘿嘿们玩过' in index['3169', f, 'G_H']['text'] and '黑人们玩过' in index['3169', f, 'G_E']['text'], 'group reference unclear')
        require('嘿嘿地笑' in index['3169', f, 'L_H']['text'] and '哈哈地笑' in index['3169', f, 'L_O']['text'], 'literal laughter unclear')
    return rows


def comparisons_for(design):
    output = []
    def add(q, f, kind, weights, role, note):
        output.append({'contrast_id': f'ECD-{q}-F{f}-{kind}', 'query_id': q, 'template': f,
            'kind': kind, 'role': role, 'terms': [{'condition': c, 'coefficient': w} for c, w in weights.items()],
            'interpretation': note, 'independent_sample': False})
    for q in TARGETS:
        for f in (1, 2):
            for c in design['case_' + q]['comparisons_per_template']:
                add(q, f, c['name'], {f'F{f}-{cell}': weight for cell, weight in c['terms'].items()}, 'primary',
                    'Matched within this wording, encoding and probe; conditional input effect, not internal mechanism.')
            main, neutral = ('T1R1', 'T0R0') if q == '541' else ('G_H', 'L_H')
            for kind, new, old in [('original_bridge', main, 'R-A'), ('D_bridge', main, f'R-D{f}'),
                                    ('C_bridge', neutral, f'R-C{f}')]:
                add(q, f, kind, {f'F{f}-{new}': 1, old: -1}, 'bridge',
                    'Total old-to-new input change; includes the registered wording/meaning and any position/length change.')
            add(q, f, 'historical_D_minus_C', {f'R-D{f}': 1, f'R-C{f}': -1}, 'historical_anchor',
                'Exact prior R-background comparison, re-scored in this run; not a new independent finding.')
    require(len(output) == 30 and sum(r['role'] == 'primary' for r in output) == 14, 'comparison frame differs')
    require(len({r['contrast_id'] for r in output}) == len(output), 'duplicate comparison')
    return output


def make_context(base, rich, material, tokenizer, catalog, encoding, probe, history_index):
    q, condition = base['query_id'], material['condition'] if material else base['root_condition']
    changed = deepcopy(rich)
    messages = base['messages']
    if material:
        target = TARGETS[q]
        require(changed['answers_by_demo'][target] == material['presented_answer'], 'demo answer would change')
        messages, _ = replace_text(base, changed['texts_by_demo'][target], material['text'])
        changed['texts_by_demo'][target] = material['text']
    messages = transformed_messages(messages, encoding, probe)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    tokens = encoded['input_ids']
    materials = {'demo:' + d: {'source': {'text': changed['texts_by_demo'][d],
        'original_answer': {'hate': MAPPINGS[encoding][changed['answers_by_demo'][d]]}}} for d in changed['demo_ids']}
    positions = layout(prompt, encoded['offset_mapping'], changed['demo_ids'], materials, None)
    for c in catalog[ENCODINGS[encoding]]:
        require(tokenizer.encode(prompt + c['canonical_answer'], add_special_tokens=False) == tokens + c['answer_token_ids'], 'candidate token boundary differs')
    require(len(tokens) + max(c['answer_tokens'] for c in catalog[ENCODINGS[encoding]]) + 65 <= 8192, 'padded input too long')
    history = history_index.get((q, condition, encoding, probe)) if material is None else None
    if history is not None:
        require(prompt == history['prompt_text'] and tokens == history['prompt_token_ids'], 'historical encoding/probe prompt differs')
    c = {'record_id': f'ECD-{q}:{encoding}:{condition}:{probe or "real"}', 'query_id': q,
        'task': ENCODINGS[encoding], 'semantic_task': 'hate', 'condition': condition + ('@' + probe if probe else ''),
        'root_condition': condition, 'family': 'content_decomposition' if material else 'historical_R_anchor',
        'encoding': encoding, 'probe_id': probe, 'messages': messages, 'prompt_text': prompt,
        'prompt_sha256': sha(prompt.encode()), 'prompt_tokens': len(tokens), 'prompt_token_ids': tokens,
        'prompt_token_ids_sha256': sha(canonical(tokens).encode()), 'protocol_id': f'ECD-{q}-hate-v1',
        'baseline_replay': history is not None, 'tokenization_status': 'verified',
        'source_record_id': (history or base)['record_id'], 'source_context_sha256': (history or base)['context_sha256']}
    no_gold(c)
    c['context_sha256'] = sha(canonical(c).encode())
    g = {'record_id': c['record_id'], 'query_id': q, 'condition': condition, 'encoding': encoding, 'probe_id': probe,
        'layout': positions, 'prompt_tokens': len(tokens), 'prompt_sha256': c['prompt_sha256'],
        'target_demo': TARGETS[q], 'demo_ids': changed['demo_ids'], 'entry_ids': changed['entry_ids'],
        'texts_by_demo_sha256': sha(canonical(changed['texts_by_demo']).encode()),
        'answers_by_demo_sha256': sha(canonical(changed['answers_by_demo']).encode())}
    return c, g


def input_proofs(contexts, geometries, comparisons):
    ci = {(c['query_id'], c['root_condition'], c['encoding'], c['probe_id']): c for c in contexts}
    gi = {(g['query_id'], g['condition'], g['encoding'], g['probe_id']): g for g in geometries}
    matched, bridges, encodings = [], [], []
    for comp in comparisons:
        for encoding, probe in VARIANTS:
            keys = [(comp['query_id'], t['condition'], encoding, probe) for t in comp['terms']]
            if comp['role'] == 'primary':
                first = {**ci[keys[0]], 'layout': gi[keys[0]]['layout']}
                proofs = [alignment_proof(first, {**ci[k], 'layout': gi[k]['layout']}, TARGETS[comp['query_id']]) for k in keys[1:]]
                matched.append({'contrast_id': comp['contrast_id'], 'encoding': encoding, 'probe_id': probe,
                                'record_ids': [ci[k]['record_id'] for k in keys], 'proofs': proofs})
            elif comp['role'] == 'bridge':
                # Terms are explicitly +new -old; bridge geometry is descriptive, not a matching gate.
                new, old = [gi[k] for k in keys]
                bridges.append({'contrast_id': comp['contrast_id'], 'encoding': encoding, 'probe_id': probe,
                    'new_prompt_tokens': new['prompt_tokens'], 'old_prompt_tokens': old['prompt_tokens'],
                    'token_delta': new['prompt_tokens'] - old['prompt_tokens'], 'layout_equal': new['layout'] == old['layout'],
                    'new_layout': new['layout'], 'old_layout': old['layout'], 'role': 'total_input_bridge'})
    for q, condition in sorted({(c['query_id'], c['root_condition']) for c in contexts}):
        a, f, r = [gi[q, condition, encoding, None] for encoding in ENCODINGS]
        require(f['layout'] == r['layout'] and f['prompt_tokens'] == r['prompt_tokens'], 'A/B geometry differs')
        encodings.append({'query_id': q, 'condition': condition, 'original_tokens': a['prompt_tokens'],
            'ab_tokens': f['prompt_tokens'], 'ab_minus_original_tokens': f['prompt_tokens'] - a['prompt_tokens'],
            'ab_forward_reverse_layout_equal': True, 'original_ab_layout_equal': a['layout'] == f['layout'],
            'query_dictionary_and_demo_texts_fixed': True})
    require((len(matched), len(bridges), len(encodings)) == (112, 96, 26), 'geometry frame differs')
    return matched, bridges, encodings


def load_frozen(directory):
    """Check bytes and executable frame without parsing query references."""
    directory = Path(directory).resolve()
    m = read_json(directory / 'manifest.json')
    require(m['schema_version'] == 'evidence-content-decomposition-freeze/v1' and m['status'] == 'frozen', 'wrong content freeze')
    require({p.name for p in directory.iterdir()} == {*m['artifacts'], 'manifest.json'}, 'freeze inventory differs')
    for name, h in m['artifacts'].items():
        require(Path(name).name == name and file_sha(directory / name) == h, 'frozen artifact changed: ' + name)
    for name, h in m['source_files'].items():
        require(file_sha(local_path(ROOT, name)) == h, 'frozen source changed: ' + name)
    plan = read_json(directory / 'plan.json'); identity = dict(plan); pid = identity.pop('plan_id')
    require(pid == 'evidence-content-decomposition-' + sha(canonical(identity).encode()), 'plan identity differs')
    require(plan['source_files'] == m['source_files'], 'plan/source inventory differs')
    contexts = read_lines(directory / 'contexts.jsonl'); history = read_lines(directory / 'historical-selected.jsonl')
    comparisons = read_lines(directory / 'comparisons.jsonl')
    require((len(contexts), len(history), len(comparisons)) == (208, 80, 30), 'input coverage differs')
    require(sha(canonical(contexts).encode()) == plan['contexts_sha256']
            and sha(canonical(history).encode()) == plan['historical_selected_sha256']
            and sha(jsonl(comparisons)) == plan['comparisons_sha256'], 'input digest differs')
    expected = {(q, condition, e, p) for q in TARGETS
        for condition in [*HISTORICAL, *(f'F{f}-{cell}' for f in (1, 2) for cell in CELLS[q])]
        for e, p in VARIANTS}
    require({(c['query_id'], c['root_condition'], c['encoding'], c['probe_id']) for c in contexts} == expected, 'condition/probe frame differs')
    require(len({c['record_id'] for c in contexts}) == len({c['prompt_sha256'] for c in contexts}) == len(contexts), 'duplicate input identity')
    require([{k: c[k] for k in DESCRIPTOR} for c in contexts] == plan['blocks'], 'execution descriptors differ')
    for c in contexts:
        no_gold(c)
        require(c['task'] == ENCODINGS[c['encoding']] and c['semantic_task'] == 'hate', 'semantic encoding differs')
        require(c['baseline_replay'] == (c['root_condition'] in HISTORICAL), 'replay selector differs')
        require(len(c['prompt_token_ids']) == c['prompt_tokens']
                and sha(canonical(c['prompt_token_ids']).encode()) == c['prompt_token_ids_sha256']
                and sha(c['prompt_text'].encode()) == c['prompt_sha256'], 'prompt identity differs')
        require(sha(canonical({k: v for k, v in c.items() if k != 'context_sha256'}).encode()) == c['context_sha256'], 'context digest differs')
    for c, row in zip([c for c in contexts if c['baseline_replay']], history, strict=True):
        validate_block(row, c, plan['catalog'])
    return plan, contexts, history


def build():
    sources = Sources(ROOT)
    def take(path, expected=None): return sources.verify(path, expected or file_sha(path))
    design = read_json(take(PUBLIC / 'design-draft-01/design.json'))
    for name, h in design['source_files'].items(): take(local_path(ROOT, name), h)
    take(PUBLIC / 'design-draft-01/PLAN.md')
    material_source = read_json(take(SOURCE)); materials = review_materials(material_source)
    pointer = read_json(take(BASE / 'label-calibration-results-v1/current.json'))
    require(pointer['status'] == 'complete' and pointer['freeze_manifest_sha256'] == PARENT_MANIFEST, 'wrong parent result')
    directory, run = BASE / pointer['freeze_path'], BASE / pointer['run_path']
    manifest = read_json(take(directory / 'manifest.json', PARENT_MANIFEST))
    for name, h in manifest['source_files'].items(): take(local_path(ROOT, name), h)
    for name, h in manifest['artifacts'].items(): take(directory / name, h)
    parent, old_contexts, _ = load_parent(directory)
    state = read_json(take(run / 'run_manifest.json', pointer['run_manifest_sha256']))
    require(state['status'] == 'complete' and state['plan_id'] == parent['plan_id'] and state['numerical_validation_passed'], 'parent unsealed')
    require(len(state['checks']) == 10 and len(state['derived_checks']) == 6
            and all(c['passed'] for c in state['checks'] + state['derived_checks']), 'parent gates incomplete')
    take(run / parent['raw_pass'] / 'manifest.json', state['raw_manifest_sha256'])
    old_raw = {r['record_id']: r for r in read_lines(take(run / parent['raw_pass'] / 'scores.jsonl', pointer['raw_scores_sha256']))}
    old_index = {(c['query_id'], c['root_condition'], c['encoding'], c['probe_id']): c for c in old_contexts}
    rich_pointer = read_json(take(BASE / 'position-length-materials-v1/current.json'))
    rich = {(r['query_id'], r['condition']): r for r in read_lines(take(BASE / rich_pointer['materials_path'] / 'contexts.jsonl'))}
    for q in TARGETS:
        for condition in HISTORICAL:
            require(rich[q, condition]['prompt_text'] == old_index[q, condition, 'original', None]['prompt_text'], 'rich source differs')
    for flag in ('USE_TORCH', 'USE_TF', 'USE_FLAX', 'USE_TORCH_XLA'): os.environ[flag] = '0'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'models/base/Qwen3-8B'), local_files_only=True, trust_remote_code=False, use_fast=True)
    catalog = make_catalog(parent['catalog']['hate'], tokenizer)
    require(catalog == parent['catalog'], 'existing label codebook changed')
    contexts, geometries = [], []
    for encoding, probe in VARIANTS:
        for q in TARGETS:
            for material, condition in [(None, c) for c in HISTORICAL] + [(m, 'R-A') for m in materials if m['query_id'] == q]:
                c, g = make_context(old_index[q, condition, 'original', None], rich[q, condition], material,
                                    tokenizer, catalog, encoding, probe, old_index)
                contexts.append(c); geometries.append(g)
    require(not {'torch', 'tensorflow', 'jax'} & set(sys.modules), 'preparation imported a model framework')
    historical = []
    for c in contexts:
        if not c['baseline_replay']: continue
        old = deepcopy(old_raw[c['source_record_id']])
        require(old['prompt_sha256'] == c['prompt_sha256'] and all(r['prompt_token_ids_sha256'] == c['prompt_token_ids_sha256'] for r in old['candidates']), 'historical tokens differ')
        old['historical_record_id'] = old['record_id']; old['historical_context_sha256'] = old['context_sha256']
        old.update({k: c[k] for k in ('record_id', 'query_id', 'task', 'condition', 'context_sha256', 'prompt_sha256')})
        validate_block(old, c, catalog); historical.append(old)
    comparisons = comparisons_for(design)
    matching, bridges, encodings = input_proofs(contexts, geometries, comparisons)
    protocol = deepcopy(read_json(directory / 'analysis_protocol.json'))
    protocol.update(schema_version='evidence-content-decomposition-protocol/v1', authorization=material_source['authorization'],
        source_design_sha256=file_sha(PUBLIC / 'design-draft-01/design.json'), material_source_sha256=file_sha(SOURCE),
        human_material_status='Assistant-reviewed new texts; no individual human adoption inferred from preparation authorization.',
        scientific_design={'541': design['case_541'], '3169': design['case_3169']},
        scope={'cases': ['541/hate', '3169/hate'], 'background': 'original R', 'main_materials': 16,
               'historical_anchor_conditions': 10, 'main_comparisons': 14, 'bridge_comparisons': 12, 'historical_comparisons': 4,
               'target_slots': {'541': 5, '3169': 3}, 'new_query_controls': 'deferred, independent new reference phase'},
        probe_diagnostics='Five single-probe and five leave-one-out results for every one of the 30 frozen comparisons; no selection after scoring.',
        interpretation=['Two exposed discovery cases; no independent-sample count or general accuracy estimate.',
            '541: compare both conditional rule effects, both conditional topic effects and their registered interaction within each wording.',
            '3169: compare designation and literal-laughter pairs separately; both retain hate, so no attack-presence or pure factorial claim.',
            'Group designation pairs add 们 in both arms; old-to-new bridges retain that pluralization and downstream token shift.',
            'Matched total length and external anchors do not align internal semantic token positions or establish identical meanings.',
            'Neutral continuation versus rule wording and sports versus sexuality remain specific instantiated content contrasts.',
            'Historical anchors, total input bridges, and primary comparisons are reported separately.',
            'Keep all original/EOS, NCC/background/single/LOO and both A/B results, including reverse and unresolved effects.',
            'Correct query references are not inferred from score calibration; no human mechanism or activation-patching readiness.'],
        new_human_fields=0, mechanism_ready=False)
    code = dict(parent['code_sha256'])
    for name in NEW_CODE: take(ROOT / name); code[name] = file_sha(ROOT / name)
    descriptors = [{k: c[k] for k in DESCRIPTOR} for c in contexts]
    schedule = [{'pass': n.replace('interventions-', 'content-'), 'selection': s, 'options': o, 'check': c} for n, s, o, c in SCHEDULE]
    counts = {'main_materials': 16, 'real_conditions': 26, 'historical_anchor_conditions': 10,
        'original_real': 26, 'background_probes': 130, 'ab_real': 52, 'contexts': 208, 'candidates': 416,
        'historical_contexts': 80, 'passes': 8, 'candidate_evaluations_across_passes': 2816,
        'primary_comparisons': 14, 'bridge_comparisons': 12, 'historical_comparisons': 4, 'comparisons': 30,
        'derived_readouts_per_gate': 672}
    plan = {'schema_version': 'evidence-content-decomposition-plan/v1', 'status': 'frozen', 'cases': ['541', '3169'],
        'tasks': list(ENCODINGS.values()), 'source_files': sources.hashes, 'code_sha256': code,
        'parent_plan_id': parent['plan_id'], 'parent_manifest_sha256': PARENT_MANIFEST,
        'blocks': descriptors, 'catalog': catalog, 'eos_token_id': parent['eos_token_id'], 'pad_token_id': parent['pad_token_id'],
        'generation_runtime_identity': parent['generation_runtime_identity'], 'config': parent['config'], 'numeric_policy': parent['numeric_policy'],
        'runtime_parent_plan': {'blocks': descriptors, 'catalog': catalog, 'package_path': parent['runtime_parent_plan']['package_path'],
                                'generation_runtime_identity': parent['generation_runtime_identity']},
        'schedule': schedule, 'raw_pass': 'content-b1-r0', 'scoring_batch_size': 1,
        'query_gold_loaded_during_scoring': False, 'formal_test_or_reserve_access': False, 'analysis_after_all_gates_and_raw_seal': True,
        'epsilon_recalibration': False, 'automatic_profile_search': False, 'mechanism_ready': False,
        'contexts_sha256': sha(canonical(contexts).encode()), 'historical_selected_sha256': sha(canonical(historical).encode()),
        'analysis_protocol_sha256': sha(json_bytes(protocol)), 'comparisons_sha256': sha(jsonl(comparisons)), 'counts': counts}
    plan['plan_id'] = 'evidence-content-decomposition-' + sha(canonical(plan).encode())
    audit = {'status': 'passed', **counts, 'full_label_boundaries': 416, 'historical_token_replays': 80,
        'matched_contrast_variant_proofs': 112, 'bridge_variant_records': 96, 'ab_geometry_pairs': 26,
        'torch_imported': False, 'model_forward_executed': False, 'gpu_runtime_validated': False,
        'references_separate': True, 'old_pointer_changes': 0, 'human_fields_changed': 0}
    material_docs = ['# 内容拆分：冻结正文与 AI 核对', '',
        '用户授权连续完成准备和实现，本轮先不运行 GPU。下列 16 条新正文均由助手编写和核对；人工逐字段值为空，未把过去批次的接受延伸到本批。', '',
        '展示答案保持 541 = non-hate、3169 = hate。正确性核对依据既有政策及案例锚点，属于 AI 判断。完整核对理由和局限见 materials.json。', '',
        '| 案例 | 条件 | 正文 | 展示答案 |', '|---|---|---|---|']
    material_docs += [f'| {m["query_id"]} | {m["condition"]} | {m["text"]} | {m["presented_answer"]} |' for m in materials]
    material_docs += ['', '541 四格共用人物、迟到事件与开头，两次身份提及匹配；T0 篮球迷与 T1 异性恋仅替换这两个类别词。R0 中性续文不暗含反泛化规则，R1 明确反对从个人推到同类。第二套规则避开“讨厌／群体”。', '',
        '3169 群体配对只替换“嘿嘿／黑人”，两边共同保留“们”；普通笑声配对只替换“嘿嘿／哈哈”。两类语境均保留明确攻击，不计算攻击有无的四格交互。复数化、显式称呼的清晰度及旧材料到新背景的差异单列。', '',
        '位置证明涵盖各主比较在全部 8 种编码／探针变体中的正文、答案、其他示例、查询和生成起点。不能将这些外部位置相同解释为内部语义 token 已对齐。', '']
    files = {'plan.json': json_bytes(plan), 'contexts.jsonl': jsonl(contexts), 'historical-selected.jsonl': jsonl(historical),
        'analysis_protocol.json': json_bytes(protocol), 'analysis_references.json': (directory / 'analysis_references.json').read_bytes(),
        'materials.json': json_bytes(material_source), 'materials.csv': csv_bytes(materials), 'MATERIALS.md': '\n'.join(material_docs).encode(),
        'comparisons.jsonl': jsonl(comparisons), 'geometry.jsonl': jsonl(geometries), 'matching-proofs.jsonl': jsonl(matching),
        'input-bridges.jsonl': jsonl(bridges), 'encoding-bridges.csv': csv_bytes(encodings), 'audit.json': json_bytes(audit),
        'execution_source.json': json_bytes({p: {'sha256': h, 'text': (ROOT / p).read_text()} for p, h in code.items()})}
    files['FREEZE.md'] = ('# 内容拆分输入冻结\n\n'
        '16 个新正文条件，加 10 个逐字历史锚点；共 26 个内容条件。每个条件有原标签实查询、五个独立背景探针、A/B 正反实查询，共 208 提示、416 候选。'
        '80 个历史提示覆盖全部三种编码及五探针，重新计分以检验数值复现，不重复使用旧背景概率。\n\n'
        '14 主比较、12 总输入桥接、4 旧 D−C 对照分别报告。完整文本和 AI 核对见 MATERIALS.md / materials.json；系数见 comparisons.jsonl。'
        '541 新正文保持原提示长度；3169 共同增加的复数形式等改写及长度变化见 input-bridges.jsonl。\n\n'
        '评分沿用此前冻结的 FP32 内核、五探针、A/B 映射、epsilon=0.0013427734375、8 个 pass（2816 候选评估）、10 个原始数值门槛及 6 个派生门槛。'
        'NCC 差值界 4 epsilon，四项交互 8 epsilon；这些是数值界，不是统计显著性。'
        '查询参考只在原始分数封存且所有门槛通过后的分析入口读取。\n\n'
        '用户授权本批材料、冻结及实现连续推进，不代表逐句人工采纳。原人审字段、参考、案例用途及机制准备状态不改。'
        '当前只有 CPU 准备验证；GPU forward、数值复现和评测均待用户通知窗口。\n\n'
        f'计划 ID：`{plan["plan_id"]}`。\n').encode()
    sources.unchanged()
    return files, sources.hashes, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=WORK / 'frozen-01')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='reconstruct/verify in memory; no model and no freeze write')
    args = parser.parse_args(); target = args.output.resolve()
    require(not (args.check and args.dry_run), 'choose check or dry-run')
    require(target.parent == WORK and target.name.startswith('frozen-'), 'use a new content freeze version')
    files, sources, audit = build()
    if args.check:
        m = read_json(target / 'manifest.json')
        require(m['source_files'] == sources and set(m['artifacts']) == set(files), 'reconstruction inventory differs')
        for name, raw in files.items(): require((target / name).read_bytes() == raw and m['artifacts'][name] == sha(raw), 'reconstruction differs: ' + name)
        load_frozen(target)
    elif not args.dry_run:
        files['manifest.json'] = json_bytes({'schema_version': 'evidence-content-decomposition-freeze/v1', 'status': 'frozen',
            'created_at': datetime.now(timezone.utc).isoformat(), 'source_files': sources, 'artifacts': {k: sha(v) for k, v in files.items()}})
        write_output(target, files)
    print(canonical({'operation': 'check' if args.check else ('dry-run' if args.dry_run else 'freeze'), **audit}))


if __name__ == '__main__': main()
