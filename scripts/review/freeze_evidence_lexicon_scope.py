#!/usr/bin/env python3
"""Freeze the authorized 3169 lexicon-dependency experiment, tokenizer only."""
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
from diagnostics.evidence_label_calibration import ENCODINGS, validate_block
from scripts.review.freeze_evidence_content_execution_v2 import load_frozen as load_parent
from scripts.review.freeze_evidence_content_decomposition import VARIANTS
from scripts.review.freeze_evidence_label_calibration import MAPPINGS, no_gold
from scripts.review.prepare_evidence_matched_materials import alignment_proof, layout, token_span
from scripts.review.run_evidence_interventions import DESCRIPTOR

BASE = ROOT / EXPERIMENT
PUBLIC = BASE / 'lexicon-scope-v1'
WORK = BASE / 'reviews/analysis-freeze-20260914/lexicon-scope-v1'
PARENT_HASH = '78b8aaf87abb7d9cc8c4d591a515b2506080a3971b5930ec5ec0becd09bd6d83'
ARMS = ('O', 'D', 'N1', 'E1', 'N2', 'E2', 'P1', 'X1', 'P2', 'X2')
FORMS = ('H', 'O')
TRANSITIONS = (
    ('O', 'D', 'natural_deletion'),
    ('O', 'N1', 'matched_replacement'), ('O', 'N2', 'matched_replacement'),
    ('D', 'E1', 'length_refill_bridge'), ('D', 'E2', 'length_refill_bridge'),
    ('N1', 'E1', 'filler_position_control'), ('N2', 'E2', 'filler_position_control'),
    ('O', 'P1', 'restatement_bridge'), ('O', 'P2', 'restatement_bridge'),
    ('O', 'X1', 'scope_total_bridge'), ('O', 'X2', 'scope_total_bridge'),
    ('P1', 'X1', 'matched_scope_note'), ('P2', 'X2', 'matched_scope_note'),
)
NEW_CODE = tuple('scripts/review/' + n + '.py' for n in (
    'freeze_evidence_lexicon_scope', 'run_evidence_lexicon_scope',
    'analyze_evidence_lexicon_scope', 'test_evidence_lexicon_scope',
    'audit_evidence_lexicon_scope',
))


def condition(family, arm, form):
    return f'F{family}-{arm}-{form}'


def comparison_id(family, kind, suffix):
    return f'ELS-3169-F{family}-{kind}-{suffix}'


def comparisons_for():
    rows = []
    def add(f, kind, suffix, weights, role, **extra):
        rows.append({'contrast_id': comparison_id(f, kind, suffix), 'query_id': '3169',
            'template': f, 'kind': kind, 'role': role,
            'terms': [{'condition': c, 'coefficient': w} for c, w in weights.items()],
            'independent_sample': False, **extra})
    for f in (1, 2):
        for arm in ARMS:
            add(f, 'word_pair', arm, {condition(f, arm, 'H'): 1, condition(f, arm, 'O'): -1},
                'historical_pair' if arm == 'O' else 'within_arm_pair', lexicon_arm=arm)
        for a, b, role in TRANSITIONS:
            suffix = f'{a}-to-{b}'
            for form in FORMS:
                add(f, 'shift_' + form, suffix,
                    {condition(f, b, form): 1, condition(f, a, form): -1}, role,
                    from_arm=a, to_arm=b)
            add(f, 'interaction', suffix, {
                condition(f, b, 'H'): 1, condition(f, b, 'O'): -1,
                condition(f, a, 'H'): -1, condition(f, a, 'O'): 1}, role,
                from_arm=a, to_arm=b)
            add(f, 'common_shift', suffix, {
                condition(f, b, 'H'): .5, condition(f, b, 'O'): .5,
                condition(f, a, 'H'): -.5, condition(f, a, 'O'): -.5}, role,
                from_arm=a, to_arm=b)
    require(len(rows) == 124 and len({r['contrast_id'] for r in rows}) == 124, 'comparison coverage differs')
    require(all(sum(t['coefficient'] for t in r['terms']) == 0 for r in rows), 'unbalanced comparison')
    return rows


def validate_materials(source):
    require(source['query_id'] == '3169' and source['target_entry'] == 'lex-0419'
            and source['target_demo'] == '3660' and source['target_slot'] == 3, 'target differs')
    require(source['authorization']['material_preparation_and_execution_authorized']
            and not source['authorization']['individual_human_material_adoption']
            and not source['authorization']['authority_record_writeback'], 'authorization differs')
    require(source['new_individual_human_fields'] == 0 and not source['mechanism_ready'], 'human scope differs')
    original = source['original_entry_block']
    require(sha(original.encode()) == source['original_entry_block_sha256'], 'original block hash differs')
    require(original == "词条：嘿嘿\n义项 1：\n定义：'嘿嘿'是'黑黑'的中文谐音，对黑人群体的侮辱性称呼。\n\n", 'original sense changed')
    require([r['arm'] for r in source['filler_entries']] == ['N1', 'N2'], 'filler frame differs')
    require([r['arm'] for r in source['definition_notes']] == ['P1', 'X1', 'P2', 'X2'], 'note frame differs')
    for row in source['filler_entries'] + source['definition_notes']:
        require(row['human_review'] is None and row['ai_review']['author'] == 'assistant', 'invented individual review')
        require(sha(row['block'].encode()) == row['block_sha256'], 'material block hash differs')
        require(not any(t in row['block'] for t in ('<|', '|>', 'hate', '3169', '待判断文本')), 'answer or structural cue in material')
    for row in source['definition_notes']:
        require(row['full_definition'] == source['original_definition'] + row['note'], 'original definition rewritten')
        require(sha(row['note'].encode()) == row['text_sha256'], 'note hash differs')
        require(row['block'] == '词条：嘿嘿\n义项 1：\n定义：' + row['full_definition'] + '\n\n', 'definition layout differs')
        require(('普通笑声' in row['note']) == row['arm'].startswith('X'), 'scope-note frame differs')
    require(source['execution']['device_indices'] == [1, 2], 'allocation differs')
    return source


def dictionary_part(user):
    prefix = '词典参考：\n'; end = '参考示例：\n'
    require(user.startswith(prefix) and user.count(end) == 1, 'dictionary boundary differs')
    return user[len(prefix):user.index(end)]


def mutate_messages(base, arm, source):
    """Edit exactly one frozen dictionary block; keep retrieval and demos fixed."""
    require(arm in ARMS, 'unknown dictionary arm')
    messages = deepcopy(base['messages'])
    require([r['role'] for r in messages] == ['system', 'user'], 'unsupported message structure')
    old = source['original_entry_block']; user = messages[1]['content']
    dictionary = dictionary_part(user)
    require(dictionary.count(old) == user.count(old) == 1, 'ambiguous lexicon target')
    fillers = {r['arm']: r['block'] for r in source['filler_entries']}
    notes = {r['arm']: r['block'] for r in source['definition_notes']}
    if arm == 'O': edited = dictionary
    elif arm == 'D': edited = dictionary.replace(old, '')
    elif arm.startswith('N'): edited = dictionary.replace(old, fillers[arm])
    elif arm.startswith('E'): edited = dictionary.replace(old, '') + fillers['N' + arm[1:]]
    else: edited = dictionary.replace(old, notes[arm])
    messages[1]['content'] = '词典参考：\n' + edited + user[len('词典参考：\n') + len(dictionary):]
    require(messages[0] == base['messages'][0], 'system changed')
    require(messages[1]['content'].split('参考示例：\n')[1] == user.split('参考示例：\n')[1], 'examples or query changed')
    require(edited.count('词条：嘿嘿\n') == (1 if arm in ('O', 'P1', 'P2', 'X1', 'X2') else 0), 'target entry presence differs')
    return messages


def geometry(prompt, encoded, rich, material, encoding):
    texts = {**rich['texts_by_demo'], '3660': material['text']}
    demos = {'demo:' + d: {'source': {'text': texts[d], 'original_answer': {
        'hate': MAPPINGS[encoding][rich['answers_by_demo'][d]]}}} for d in rich['demo_ids']}
    positions = layout(prompt, encoded['offset_mapping'], rich['demo_ids'], demos, None)
    dictionary = prompt[prompt.index('词典参考：\n') + len('词典参考：\n'):prompt.index('参考示例：\n')]
    blocks = re.findall(r'词条：[^\n]+\n义项 1：\n定义：[^\n]+\n\n', dictionary)
    require(''.join(blocks) == dictionary, 'unsupported or lost dictionary content')
    lexicon = {}
    for i, block in enumerate(blocks):
        word = block.split('\n')[0].removeprefix('词条：')
        require(word not in lexicon and prompt.count(block) == 1, 'ambiguous lexicon geometry')
        start = prompt.index(block)
        lexicon[word] = {'slot': i + 1, 'block': token_span(encoded['offset_mapping'], start, start + len(block)),
                        'text_sha256': sha(block.encode())}
    return {'layout': positions, 'lexicon': lexicon,
            'demo_ids': rich['demo_ids'], 'entry_words': list(lexicon), 'prompt_tokens': len(encoded['input_ids'])}


def matched_region(a, b, ga, gb, word_a, word_b):
    require(a['prompt_tokens'] == b['prompt_tokens'] and ga['layout'] == gb['layout'], 'external anchors changed')
    left, right = ga['lexicon'][word_a]['block'], gb['lexicon'][word_b]['block']
    require(left == right, 'replacement span changed')
    start, end = left
    require(a['prompt_token_ids'][:start] == b['prompt_token_ids'][:start]
            and a['prompt_token_ids'][end:] == b['prompt_token_ids'][end:], 'tokens outside replacement changed')
    shared = set(ga['lexicon']) & set(gb['lexicon']) - {word_a, word_b}
    require(all(ga['lexicon'][w] == gb['lexicon'][w] for w in shared), 'untouched entry positions changed')
    return {'passed': True, 'matched_span': left, 'full_prompt_tokens': a['prompt_tokens'],
            'all_external_demo_query_anchors_equal': True, 'outside_region_token_ids_equal': True,
            'internal_semantic_alignment_claimed': False}


def input_proofs(contexts, geometries, source):
    ci = {(c['template'], c['lexicon_arm'], c['form'], c['encoding'], c['probe_id']): c for c in contexts}
    gi = {g['record_id']: g for g in geometries}
    proofs, bridges, encoding_bridges = [], [], []
    for f in (1, 2):
        for encoding, probe in VARIANTS:
            for arm in ARMS:
                a, b = [ci[f, arm, form, encoding, probe] for form in FORMS]
                p = alignment_proof({**a, 'layout': gi[a['record_id']]['layout']},
                                    {**b, 'layout': gi[b['record_id']]['layout']}, '3660')
                proofs.append({'kind': 'within_arm_word_pair', 'record_ids': [a['record_id'], b['record_id']], 'proof': p})
            for form in FORMS:
                original = ci[f, 'O', form, encoding, probe]
                go = gi[original['record_id']]
                for a, b, role in TRANSITIONS:
                    left, right = [ci[f, arm, form, encoding, probe] for arm in (a, b)]
                    ga, gb = [gi[c['record_id']] for c in (left, right)]
                    bridges.append({'template': f, 'form': form, 'encoding': encoding, 'probe_id': probe,
                        'from_arm': a, 'to_arm': b, 'role': role, 'record_ids': [left['record_id'], right['record_id']],
                        'token_delta': right['prompt_tokens'] - left['prompt_tokens'],
                        'all_demo_query_positions_equal': ga['layout'] == gb['layout'],
                        'from_geometry': ga, 'to_geometry': gb})
                    if role in ('matched_replacement', 'matched_scope_note'):
                        word_b = next(r['word'] for r in source['filler_entries'] if r['arm'] == b) if b.startswith('N') else '嘿嘿'
                        p = matched_region(left, right, ga, gb, '嘿嘿', word_b)
                        proofs.append({'kind': role, 'record_ids': [left['record_id'], right['record_id']], 'proof': p})
                    if role == 'filler_position_control':
                        require(left['prompt_tokens'] == right['prompt_tokens'] and ga['layout'] == gb['layout'], 'filler relocation moved demos or query')
                        start = ga['layout']['demos'][ga['demo_ids'][0]]['block'][0]
                        require(left['prompt_token_ids'][start:] == right['prompt_token_ids'][start:], 'filler relocation changed examples/query')
                        word = next(r['word'] for r in source['filler_entries'] if r['arm'] == a)
                        require(ga['lexicon'][word]['text_sha256'] == gb['lexicon'][word]['text_sha256'], 'moved filler changed text')
                        require(ga['lexicon'][word]['slot'] == 3 and gb['lexicon'][word]['slot'] == 6, 'filler slots differ')
                        for w in ('fw', '剑人', 'tm'):
                            require([v - 33 for v in ga['lexicon'][w]['block']] == gb['lexicon'][w]['block'], 'remaining entry shift differs')
                        proofs.append({'kind': role, 'record_ids': [left['record_id'], right['record_id']],
                            'proof': {'passed': True, 'same_filler_bytes': True, 'all_demo_query_positions_equal': True,
                                      'remaining_entry_token_shift': -33, 'isolated_absolute_position_claimed': False}})
                for arm in ARMS:
                    c = ci[f, arm, form, encoding, probe]; g = gi[c['record_id']]
                    delta = -33 if arm == 'D' else (14 if arm in ('P1', 'X1') else (13 if arm in ('P2', 'X2') else 0))
                    require(c['prompt_tokens'] - original['prompt_tokens'] == delta, 'full prompt token delta differs')
                    require([v + delta for v in go['layout']['query_block']] == g['layout']['query_block'], 'query position shift differs')
                    for d in go['demo_ids']:
                        for field in ('block', 'text', 'answer'):
                            require([v + delta for v in go['layout']['demos'][d][field]] == g['layout']['demos'][d][field], 'demo position shift differs')
        for arm in ARMS:
            for form in FORMS:
                o, a, b = [ci[f, arm, form, e, None] for e in ENCODINGS]
                require(gi[a['record_id']]['layout'] == gi[b['record_id']]['layout']
                        and gi[a['record_id']]['lexicon'] == gi[b['record_id']]['lexicon'], 'A/B forward/reverse positions differ')
                encoding_bridges.append({'template': f, 'arm': arm, 'form': form, 'original_tokens': o['prompt_tokens'],
                    'ab_tokens': a['prompt_tokens'], 'ab_forward_reverse_geometry_equal': True,
                    'original_to_ab_is_total_encoding_bridge': True})
    require(len(proofs) == 352 and len(bridges) == 416 and len(encoding_bridges) == 40, 'geometry proof coverage differs')
    return proofs, bridges, encoding_bridges


def load_frozen(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / 'manifest.json')
    require(manifest['schema_version'] == 'evidence-lexicon-scope-freeze/v1' and manifest['status'] == 'frozen', 'wrong lexicon freeze')
    require({p.name for p in directory.iterdir()} == {*manifest['artifacts'], 'manifest.json'}, 'freeze inventory differs')
    for name, h in manifest['artifacts'].items():
        require(Path(name).name == name and file_sha(directory / name) == h, 'frozen artifact changed: ' + name)
    for name, h in manifest['source_files'].items():
        require(file_sha(local_path(ROOT, name)) == h, 'frozen source changed: ' + name)
    plan = read_json(directory / 'plan.json'); identity = dict(plan); pid = identity.pop('plan_id')
    require(pid == 'evidence-lexicon-scope-' + sha(canonical(identity).encode()), 'plan identity differs')
    require(plan['source_files'] == manifest['source_files'], 'source inventory differs')
    contexts = read_lines(directory / 'contexts.jsonl'); history = read_lines(directory / 'historical-selected.jsonl')
    comps = read_lines(directory / 'comparisons.jsonl')
    require((len(contexts), len(history), len(comps)) == (320, 32, 124), 'input coverage differs')
    require(comps == comparisons_for(), 'comparison coefficients differ')
    require(sha(canonical(contexts).encode()) == plan['contexts_sha256']
            and sha(canonical(history).encode()) == plan['historical_selected_sha256']
            and sha(jsonl(comps)) == plan['comparisons_sha256'], 'input digest differs')
    expected = {(f, a, h, e, p) for f in (1, 2) for a in ARMS for h in FORMS for e, p in VARIANTS}
    require({(c['template'], c['lexicon_arm'], c['form'], c['encoding'], c['probe_id']) for c in contexts} == expected, 'input frame differs')
    require(len({c['record_id'] for c in contexts}) == len({c['prompt_sha256'] for c in contexts}) == 320, 'duplicate input identity')
    require([{k: c[k] for k in DESCRIPTOR} for c in contexts] == plan['blocks'], 'execution descriptors differ')
    require(file_sha(directory / 'analysis_protocol.json') == plan['analysis_protocol_sha256'], 'protocol digest differs')
    for c in contexts:
        no_gold(c)
        require(c['query_id'] == '3169' and c['task'] == ENCODINGS[c['encoding']] and c['semantic_task'] == 'hate', 'case/encoding differs')
        require(c['baseline_replay'] == (c['lexicon_arm'] == 'O'), 'replay selector differs')
        require(c['root_condition'] == condition(c['template'], c['lexicon_arm'], c['form']), 'condition identity differs')
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
    source = validate_materials(read_json(take(PUBLIC / 'materials-source-01.json')))
    take(PUBLIC / 'PLAN.md')
    pointer = read_json(take(BASE / 'content-decomposition-results-v1/current.json'))
    require(pointer['status'] == 'complete' and pointer['freeze_manifest_sha256'] == PARENT_HASH, 'wrong parent result')
    parent_dir, parent_run = BASE / pointer['freeze_path'], BASE / pointer['run_path']
    manifest = read_json(take(parent_dir / 'manifest.json', PARENT_HASH))
    for name, h in manifest['source_files'].items(): take(local_path(ROOT, name), h)
    for name, h in manifest['artifacts'].items(): take(parent_dir / name, h)
    parent, old_contexts, _ = load_parent(parent_dir)
    state = read_json(take(parent_run / 'run_manifest.json', pointer['run_manifest_sha256']))
    require(state['status'] == 'complete' and state['plan_id'] == parent['plan_id']
            and state['numerical_validation_passed'] and len(state['checks']) == 10
            and len(state['derived_checks']) == 6 and all(r['passed'] for r in state['checks'] + state['derived_checks']), 'parent gates incomplete')
    take(parent_run / parent['raw_pass'] / 'manifest.json', state['raw_manifest_sha256'])
    old_raw = {r['record_id']: r for r in read_lines(take(parent_run / parent['raw_pass'] / 'scores.jsonl', pointer['raw_scores_sha256']))}
    old_index = {(c['query_id'], c['root_condition'], c['encoding'], c['probe_id']): c for c in old_contexts}
    materials = [m for m in read_json(parent_dir / 'materials.json')['materials'] if m['query_id'] == '3169' and m['cell'] in ('L_H', 'L_O')]
    require(len(materials) == 4, 'parent laughter frame differs')
    mi = {(m['template'], m['cell'][-1]): m for m in materials}
    rich_pointer = read_json(take(BASE / 'position-length-materials-v1/current.json'))
    rich = next(r for r in read_lines(take(BASE / rich_pointer['materials_path'] / 'contexts.jsonl'))
                if r['query_id'] == '3169' and r['condition'] == 'R-A')
    require(rich['entry_ids'] == ['lex-0031', 'lex-0104', 'lex-0419', 'lex-0486', 'lex-0493', 'lex-0663'], 'lexicon background differs')
    prefs = BASE / 'ai_reviews/v2-reannotation-20260911'
    take(prefs / 'preferences/snapshots/cf158ae3d22aadf1bff45ff5398c3de415c16fc02963f18d0fdc7918f3aa5ed1/preferences.json')
    take(prefs / 'resource-reviews-v1/preferences.md')
    for flag in ('USE_TORCH', 'USE_TF', 'USE_FLAX', 'USE_TORCH_XLA'): os.environ[flag] = '0'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'models/base/Qwen3-8B'), local_files_only=True, trust_remote_code=False, use_fast=True)
    contexts, geometries, historical = [], [], []
    for encoding, probe in VARIANTS:
        for f in (1, 2):
            for arm in ARMS:
                for form in FORMS:
                    base = old_index['3169', f'F{f}-L_{form}', encoding, probe]
                    messages = mutate_messages(base, arm, source)
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
                    tokens = encoded['input_ids']; root_condition = condition(f, arm, form)
                    for candidate in parent['catalog'][ENCODINGS[encoding]]:
                        require(tokenizer.encode(prompt + candidate['canonical_answer'], add_special_tokens=False)
                                == tokens + candidate['answer_token_ids'], 'candidate boundary differs')
                    require(len(tokens) + 70 <= 8192, 'input too long')
                    if arm == 'O': require(prompt == base['prompt_text'] and tokens == base['prompt_token_ids'], 'historical replay differs')
                    c = {'record_id': f'ELS-3169:{encoding}:{root_condition}:{probe or "real"}',
                        'query_id': '3169', 'task': ENCODINGS[encoding], 'semantic_task': 'hate',
                        'condition': root_condition + ('@' + probe if probe else ''), 'root_condition': root_condition,
                        'family': 'lexicon_scope', 'template': f, 'lexicon_arm': arm, 'form': form,
                        'encoding': encoding, 'probe_id': probe, 'messages': messages, 'prompt_text': prompt,
                        'prompt_sha256': sha(prompt.encode()), 'prompt_token_ids': tokens, 'prompt_tokens': len(tokens),
                        'prompt_token_ids_sha256': sha(canonical(tokens).encode()), 'protocol_id': 'ELS-3169-hate-v1',
                        'baseline_replay': arm == 'O', 'tokenization_status': 'verified',
                        'source_record_id': base['record_id'], 'source_context_sha256': base['context_sha256']}
                    no_gold(c); c['context_sha256'] = sha(canonical(c).encode())
                    g = geometry(prompt, encoded, rich, mi[f, form], encoding)
                    g.update({k: c[k] for k in ('record_id', 'template', 'lexicon_arm', 'form', 'encoding', 'probe_id', 'prompt_sha256')})
                    contexts.append(c); geometries.append(g)
                    if arm == 'O':
                        old = deepcopy(old_raw[base['record_id']]); old['historical_record_id'] = old['record_id']
                        old['historical_context_sha256'] = old['context_sha256']
                        old.update({k: c[k] for k in ('record_id', 'query_id', 'task', 'condition', 'context_sha256', 'prompt_sha256')})
                        validate_block(old, c, parent['catalog']); historical.append(old)
    require(not {'torch', 'tensorflow', 'jax'} & set(sys.modules), 'freezer imported model framework')
    proofs, bridges, encoding_bridges = input_proofs(contexts, geometries, source)
    comps = comparisons_for()
    counts = {'real_conditions': 40, 'original_real': 40, 'background_probes': 200, 'ab_real': 80,
              'contexts': 320, 'candidates': 640, 'historical_contexts': 32, 'comparisons': 124,
              'within_arm_pairs': 20, 'transitions_per_demo_template': 13, 'demo_templates': 2,
              'passes': 8, 'candidate_evaluations_across_passes': 3968, 'derived_readouts_per_gate': 1968}
    protocol = {'schema_version': 'evidence-lexicon-scope-protocol/v1', 'authorization': source['authorization'],
        'counts': counts, 'target': {'query': '3169', 'demo': '3660', 'entry': 'lex-0419', 'sense': 'sense:01'},
        'arms': list(ARMS), 'transitions': [dict(zip(('from_arm', 'to_arm', 'role'), t)) for t in TRANSITIONS],
        'readout': {'margin': 'non-hate minus hate', 'word_pair': 'H minus O',
            'interaction': '(B_H-B_O)-(A_H-A_O)', 'common_shift': '(B_H+B_O-A_H-A_O)/2',
            'primary_original': 'answer_sum', 'parallel_groups': ['original sum/mean', 'five-probe NCC mean', 'A/B forward and reverse'],
            'auxiliary': 'All four score modes, background, both A/B diagnostics, five single and five leave-one-out probes'},
        'numeric_policy': parent['numeric_policy'], 'bound_rule': 'sum(abs(coefficient) * per_condition_bound); NCC condition=2*epsilon',
        'classification_rule': 'Only descriptive direction/magnitude and numerical unresolved; no statistical or percentage equivalence threshold',
        'ratio_rule': 'Only if original pair is numerically resolved; retain signs, absolute magnitudes and reversals',
        'scientific_plan_sha256': file_sha(PUBLIC / 'PLAN.md'), 'references_loaded_after_all_gates': True,
        'new_individual_human_fields': 0, 'mechanism_ready': False}
    code = dict(parent['code_sha256'])
    for name in NEW_CODE: take(ROOT / name); code[name] = file_sha(ROOT / name)
    descriptors = [{k: c[k] for k in DESCRIPTOR} for c in contexts]
    config = deepcopy(parent['config']); config['execution']['device_indices'] = source['execution']['device_indices']
    schedule = deepcopy(parent['schedule'])
    for spec in schedule: spec['pass'] = spec['pass'].replace('content-', 'lexicon-')
    plan = {'schema_version': 'evidence-lexicon-scope-plan/v1', 'status': 'frozen', 'cases': ['3169'],
        'tasks': list(ENCODINGS.values()), 'source_files': sources.hashes, 'code_sha256': code,
        'parent_plan_id': parent['plan_id'], 'parent_manifest_sha256': PARENT_HASH,
        'blocks': descriptors, 'catalog': parent['catalog'], 'eos_token_id': parent['eos_token_id'], 'pad_token_id': parent['pad_token_id'],
        'generation_runtime_identity': parent['generation_runtime_identity'], 'config': config, 'numeric_policy': parent['numeric_policy'],
        'runtime_parent_plan': {'blocks': descriptors, 'catalog': parent['catalog'], 'package_path': parent['runtime_parent_plan']['package_path'],
                                'generation_runtime_identity': parent['generation_runtime_identity']},
        'execution_amendment': source['execution'], 'schedule': schedule, 'raw_pass': 'lexicon-b1-r0', 'scoring_batch_size': 1,
        'query_gold_loaded_during_scoring': False, 'formal_test_or_reserve_access': False, 'analysis_after_all_gates_and_raw_seal': True,
        'epsilon_recalibration': False, 'automatic_profile_search': False, 'mechanism_ready': False,
        'contexts_sha256': sha(canonical(contexts).encode()), 'historical_selected_sha256': sha(canonical(historical).encode()),
        'analysis_protocol_sha256': sha(json_bytes(protocol)), 'comparisons_sha256': sha(jsonl(comps)), 'counts': counts}
    plan['plan_id'] = 'evidence-lexicon-scope-' + sha(canonical(plan).encode())
    audit = {'status': 'passed', **counts, 'candidate_boundary_checks': 640,
        'word_pair_matching_proofs': 160, 'lexicon_slot_or_note_matching_proofs': 128, 'filler_relocation_proofs': 64,
        'geometry_proofs': len(proofs), 'input_bridge_records': len(bridges), 'ab_geometry_pairs': 40,
        'model_forward_executed': False, 'gpu_runtime_validated': False, 'human_fields_changed': 0, 'old_pointer_changes': 0}
    material_doc = ['# 3169 词条实验输入', '', '新词典文本为助手编写的实验变体，原人审定义和字段保留。', '',
        '## 已冻结的示例正文', '', '| 措辞 | 词形 | 原文 | 展示答案 |', '|---|---|---|---|']
    material_doc += [f'| {m["template"]} | {m["cell"]} | {m["text"]} | {m["presented_answer"]} |' for m in materials]
    material_doc += ['', '## 词典替换／追加文本', '', '| 条件 | 完整定义 |', '|---|---|', f'| O | {source["original_definition"]} |']
    material_doc += [f'| {r["arm"]}／{r["word"]} | {r["definition"]} |' for r in source['filler_entries']]
    material_doc += [f'| {r["arm"]} | {r["full_definition"]} |' for r in source['definition_notes']]
    material_doc += ['', 'D 删除目标完整块；E1/E2 将与 N1/N2 相同的块放在词典末尾。详细比较和解释边界见 PLAN.md。', '']
    files = {'plan.json': json_bytes(plan), 'contexts.jsonl': jsonl(contexts), 'historical-selected.jsonl': jsonl(historical),
        'analysis_protocol.json': json_bytes(protocol), 'analysis_references.json': (parent_dir / 'analysis_references.json').read_bytes(),
        'materials.json': json_bytes(source), 'inherited-demo-materials.json': json_bytes(materials), 'MATERIALS.md': '\n'.join(material_doc).encode(),
        'PLAN.md': (PUBLIC / 'PLAN.md').read_bytes(), 'comparisons.jsonl': jsonl(comps), 'geometry.jsonl': jsonl(geometries),
        'matching-proofs.jsonl': jsonl(proofs), 'input-bridges.jsonl': jsonl(bridges), 'encoding-bridges.csv': csv_bytes(encoding_bridges),
        'audit.json': json_bytes(audit), 'execution_source.json': json_bytes({p: {'sha256': h, 'text': (ROOT / p).read_text()} for p, h in code.items()})}
    sources.unchanged()
    return files, sources.hashes, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=WORK / 'frozen-01')
    parser.add_argument('--check', action='store_true'); parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(); target = args.output.resolve()
    require(target.parent == WORK and target.name.startswith('frozen-'), 'use a new lexicon freeze version')
    require(not (args.check and args.dry_run), 'choose check or dry-run')
    files, sources, audit = build()
    if args.check:
        m = read_json(target / 'manifest.json')
        require(m['source_files'] == sources and set(m['artifacts']) == set(files), 'reconstruction inventory differs')
        for name, raw in files.items(): require((target / name).read_bytes() == raw and m['artifacts'][name] == sha(raw), 'reconstruction differs: ' + name)
        load_frozen(target)
    elif not args.dry_run:
        files['manifest.json'] = json_bytes({'schema_version': 'evidence-lexicon-scope-freeze/v1', 'status': 'frozen',
            'created_at': datetime.now(timezone.utc).isoformat(), 'source_files': sources, 'artifacts': {k: sha(v) for k, v in files.items()}})
        write_output(target, files)
    print(canonical({'operation': 'check' if args.check else ('dry-run' if args.dry_run else 'freeze'), **audit}))


if __name__ == '__main__': main()
