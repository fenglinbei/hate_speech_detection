#!/usr/bin/env python3
"""Build/check a reviewable functional-query draft; CPU tokenizer only, no run command."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'exps/causal_context/general_model_evidence_applicability_v1'
PUBLIC = BASE / 'functional-query-diagnostics-v1'
DRAFT = PUBLIC / 'draft-01'
WORK = BASE / 'reviews/functional-query-diagnostics-v1/draft-01'
PARENT = BASE / 'reviews/analysis-freeze-20260914/lexicon-scope-v1/frozen-02'
TOKENIZER = BASE / 'reviews/analysis-freeze-20260914/lexicon-scope-v1/model-copy-01'
PREF = BASE / 'ai_reviews/v2-reannotation-20260911'
QUERY_MARKER = '待判断文本（JSON 字符串）：\n'
PARENT_MANIFEST_SHA = '819c29a9d1b732fdee2dbf4599c11f9bf32c20bf7485c83c49b3dbc9c475f949'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def jb(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def jl(rows):
    return ''.join(json.dumps(x, ensure_ascii=False, sort_keys=True) + '\n' for x in rows).encode()


def cb(rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()})
    return out.getvalue().encode('utf-8-sig')


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def relative(path):
    return str(path.relative_to(ROOT))


def span(offsets, start, end):
    hits = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    require(hits, 'empty token span')
    return [hits[0], hits[-1] + 1]


def geometry(prompt, offsets, messages):
    dictionary = messages[1]['content'].split('参考示例：\n')[0].removeprefix('词典参考：\n')
    blocks = re.findall(r'词条：[^\n]+\n义项 1：\n定义：[^\n]+\n\n', dictionary)
    require(''.join(blocks) == dictionary, 'dictionary parse differs')
    lex = {}
    for block in blocks:
        word = block.split('\n')[0].removeprefix('词条：')
        pos = prompt.index(block)
        lex[word] = span(offsets, pos, pos + len(block))
    demos = []
    for match in re.finditer(r'示例 (\d+)\n文本：(.*?)\n输出：([^\n]*)', prompt, re.S):
        demos.append({'slot': int(match[1]), 'text': span(offsets, *match.span(2)),
                      'answer': span(offsets, *match.span(3))})
    require(len(demos) == 10, 'example count changed')
    query = messages[1]['content'].split(QUERY_MARKER)[1]
    pos = prompt.index(QUERY_MARKER) + len(QUERY_MARKER)
    return {'dictionary': lex, 'demos': demos, 'query': span(offsets, pos, pos + len(query)),
            'answer_start': len(offsets)}


def build():
    source = read(DRAFT / 'materials-source.json')
    queries = source['queries']
    require(source['status'] == 'ai_draft_pending_human_review', 'draft identity changed')
    require(not source['authorization']['gpu_execution_in_this_stage']
            and not source['authorization']['individual_human_adoption'], 'authorization changed')
    require(not source['formal_input_freeze'] and not source['mechanism_ready'], 'premature readiness')
    require(len(queries) == 8 and len({q['text'] for q in queries}) == 8, 'query frame changed')
    for n, q in enumerate(queries, 1):
        require(q['query_id'] == f'FD-3169-Q{n:02}', 'new query identity changed')
        require(q['text'].count('嘿嘿') == 1, 'ambiguous target occurrence')
        require(all(q[k] is None for k in ('human_applicability', 'human_label', 'human_text_adoption')),
                'draft cannot incorporate new human decisions in place')
        require(not any(s in q['text'] for s in ('<|', '|>', 'hate', 'JSON', '3169', '待判断文本')),
                'model task instruction or answer cue in query')
        q['text_sha256'] = digest(q['text'].encode())
        q['parent_query_id'] = '3169'
    for family in ('A', 'B'):
        by_cell = {q['cell']: q for q in queries if q['query_family'] == family}
        require(set(by_cell) == {'L0', 'L1', 'G0', 'G1'}, 'incomplete query family')
        require(by_cell['G0']['text'].replace('反对', '赞同') == by_cell['G1']['text'],
                'designation stance pair changed outside its registered edit')

    parent_manifest = read(PARENT / 'manifest.json')
    require(digest((PARENT / 'manifest.json').read_bytes()) == PARENT_MANIFEST_SHA, 'parent manifest changed')
    for name in ('contexts.jsonl', 'plan.json', 'materials.json', 'inherited-demo-materials.json'):
        require(digest((PARENT / name).read_bytes()) == parent_manifest['artifacts'][name],
                'parent artifact changed: ' + name)
    parent_plan = read(PARENT / 'plan.json')
    parents = lines(PARENT / 'contexts.jsonl')
    index = {(x['template'], x['lexicon_arm'], x['form'], x['encoding'], x['probe_id']): x for x in parents}
    require(len(index) == 320, 'parent context coverage changed')
    variants = [(x['encoding'], x['probe_id']) for x in parents if x['root_condition'] == 'F1-O-H']
    require(len(set(variants)) == len(variants) == 8, 'encoding/probe coverage changed')
    expected_probes = {'', ' ', 'N/A', '[MASK]', 'Lorem ipsum'}
    require({json.loads(index[(1, 'O', 'H', e, p)]['messages'][1]['content'].split(QUERY_MARKER)[1])
             for e, p in variants if p is not None} == expected_probes, 'NCC probes changed')

    input_queries = [{'query_id': q['query_id'], 'parent_query_id': '3169', 'text': q['text'],
                      'text_sha256': q['text_sha256'], 'source_kind': 'assistant_authored_diagnostic'}
                     for q in queries]
    fits, labels = [], []
    for q in queries:
        common = {'query_id': q['query_id'], 'text_sha256': q['text_sha256'], 'text': q['text'],
                  'review_kind': 'ai_note_with_empty_human_fields', 'human_status': 'unreviewed',
                  'human_resolution': 'unresolved', 'reviewer': None, 'confirmation_source': None,
                  'human_text_adoption': None}
        offset = q['text'].index('嘿嘿')
        fits.append({**common, 'target_sense': source['target_sense'], 'occurrence_start': offset,
                     'occurrence_end': offset + 2, 'raw_surface': '嘿嘿',
                     'ai_query_fit': q['ai_applicability']['query_fit'],
                     'ai_sense_location': q['ai_applicability']['sense_location'],
                     'ai_reason': q['ai_applicability']['reason'],
                     'ai_uncertainty': q['ai_applicability']['uncertainty'],
                     'human_query_fit': None, 'human_sense_location': None,
                     'human_evidence': None, 'human_note': None})
        labels.append({**common, 'original_dataset_reference': None,
                       'original_reference_status': 'not_applicable_new_query',
                       'ai_hate': q['ai_label']['hate'], 'ai_severity': q['ai_label']['attack_severity'],
                       'ai_author_stance': q['ai_label']['author_stance'],
                       'ai_target': q['ai_label']['actual_target'],
                       'ai_attack_proposition': q['ai_label']['attack_proposition'],
                       'ai_reason': q['ai_label']['reason'], 'human_hate': None,
                       'human_attack_severity': None, 'human_author_stance': None,
                       'human_actual_target': None, 'human_evidence': None, 'human_note': None})

    matrix = []
    def add(qid, family, form, arm, stage, role):
        matrix.append({'condition_id': f'FQD-{qid}-F{family}-{arm}-{form}', 'query_id': qid,
                       'parent_query_id': '3169', 'stage': stage, 'role': role,
                       'demo_family': family, 'demo_surface': form, 'lexicon_arm': arm,
                       'source_condition': f'F{family}-{arm}-' + ('O' if form == 'A' else 'H'),
                       'input_freeze_status': 'draft_not_frozen'})
    for q in queries:
        for f in (1, 2):
            for arm in ('O', 'N1', 'N2', 'D'):
                for form in ('H', 'A'):
                    add(q['query_id'], f, form, arm, 'stage_1', 'natural_deletion_bridge' if arm == 'D' else 'primary')
        for arm in ('P1', 'X1', 'P2', 'X2'):
            for form in ('H', 'A'):
                add(q['query_id'], 1, form, arm, 'stage_2_supplement', 'scope_supplement')
    for f in (1, 2):
        for form in ('H', 'A'):
            add('3169', f, form, 'O', 'stage_1', 'historical_anchor')
    require(Counter(x['role'] for x in matrix) ==
            Counter(primary=96, natural_deletion_bridge=32, scope_supplement=64, historical_anchor=4),
            'matrix count changed')

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    unique, geometries, bindings, binding_index = {}, {}, [], {}
    by_query = {q['query_id']: q for q in input_queries}
    parent_replays = set()
    for row in matrix:
        for encoding, probe in variants:
            old = index[(row['demo_family'], row['lexicon_arm'],
                         'O' if row['demo_surface'] == 'A' else 'H', encoding, probe)]
            messages = deepcopy(old['messages'])
            if row['query_id'] != '3169' and probe is None:
                require(messages[1]['content'].count(QUERY_MARKER) == 1, 'query marker changed')
                prefix = messages[1]['content'].split(QUERY_MARKER)[0] + QUERY_MARKER
                messages[1]['content'] = prefix + json.dumps(by_query[row['query_id']]['text'], ensure_ascii=False)
                require(messages[0] == old['messages'][0], 'system edited')
                require(messages[1]['content'].split(QUERY_MARKER)[0] == old['messages'][1]['content'].split(QUERY_MARKER)[0],
                        'retrieval/demo background edited')
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            catalog = parent_plan['catalog'][old['task']]
            key = digest(jb({'prompt': prompt, 'encoding': encoding, 'catalog': catalog}))
            prompt_id = 'FQP-' + key
            if prompt_id not in unique:
                encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
                ids = encoded['input_ids']
                for candidate in catalog:
                    require(tokenizer.encode(prompt + candidate['canonical_answer'], add_special_tokens=False)
                            == ids + candidate['answer_token_ids'], 'answer token boundary differs')
                unique[prompt_id] = {'prompt_id': prompt_id, 'encoding': encoding, 'task': old['task'],
                                     'probe_id': probe, 'messages': messages, 'prompt_text': prompt,
                                     'prompt_sha256': digest(prompt.encode()), 'prompt_token_ids': ids,
                                     'prompt_tokens': len(ids), 'candidates': catalog,
                                     'status': 'cpu_preview_not_execution_freeze'}
                geometries[prompt_id] = geometry(prompt, encoded['offset_mapping'], messages)
            current = unique[prompt_id]
            if probe is not None or row['query_id'] == '3169':
                require(current['prompt_text'] == old['prompt_text']
                        and current['prompt_token_ids'] == old['prompt_token_ids'], 'historical prompt replay differs')
                parent_replays.add(old['record_id'])
            binding = {'condition_id': row['condition_id'], 'stage': row['stage'], 'encoding': encoding,
                       'probe_id': probe, 'prompt_id': prompt_id, 'source_record_id': old['record_id']}
            bindings.append(binding)
            binding_index[(row['query_id'], row['demo_family'], row['demo_surface'], row['lexicon_arm'], encoding, probe)] = prompt_id

    proofs = []
    for q in queries:
        for f in (1, 2):
            for form in ('H', 'A'):
                for encoding, probe in variants:
                    def get(arm):
                        key = (q['query_id'], f, form, arm, encoding, probe)
                        pid = binding_index[key]
                        return unique[pid], geometries[pid]
                    for a, b, kind in ([('O', 'N1', 'matched_replacement'), ('O', 'N2', 'matched_replacement')]
                                      + ([('P1', 'X1', 'matched_scope'), ('P2', 'X2', 'matched_scope')] if f == 1 else [])):
                        left, gl = get(a)
                        right, gr = get(b)
                        wa = '嘿嘿'
                        wb = {'N1': '台灯', 'N2': '书签'}.get(b, '嘿嘿')
                        start, end = gl['dictionary'][wa]
                        require([start, end] == gr['dictionary'][wb], 'lexicon interval changed')
                        require(left['prompt_tokens'] == right['prompt_tokens']
                                and gl['demos'] == gr['demos'] and gl['query'] == gr['query'], 'external positions changed')
                        require(left['prompt_token_ids'][:start] == right['prompt_token_ids'][:start]
                                and left['prompt_token_ids'][end:] == right['prompt_token_ids'][end:], 'tokens changed outside lexicon')
                        proofs.append({'query_id': q['query_id'], 'family': f, 'form': form,
                                       'encoding': encoding, 'probe_id': probe, 'kind': kind, 'from': a, 'to': b})
                    left, gl = get('O')
                    right, gr = get('D')
                    require(left['prompt_tokens'] - right['prompt_tokens'] == 33, 'natural deletion token shift changed')
                    require(all(x - y == 33 for x, y in zip(gl['query'], gr['query'])), 'deletion query shift changed')
                    proofs.append({'query_id': q['query_id'], 'family': f, 'form': form,
                                   'encoding': encoding, 'probe_id': probe, 'kind': 'deletion_shift', 'tokens': -33})

    conditions = {(r['query_id'], r['demo_family'], r['lexicon_arm'], r['demo_surface']): r['condition_id'] for r in matrix}
    row_by_id = {r['condition_id']: r for r in matrix}
    contrasts = []
    def combine(*weighted):
        terms = defaultdict(float)
        for weight, d in weighted:
            for k, v in d.items(): terms[k] += weight * v
        return {k: v for k, v in terms.items() if v}
    def margin(q, f, arm, form): return {conditions[(q, f, arm, form)]: 1.0}
    def avg(q, f, arm): return combine((.5, margin(q, f, arm, 'H')), (.5, margin(q, f, arm, 'A')))
    def pair(q, f, arm): return combine((1, margin(q, f, arm, 'H')), (-1, margin(q, f, arm, 'A')))
    def effect(q, f, a, b, kind):
        fn = avg if kind == 'C' else pair
        return combine((1, fn(q, f, b)), (-1, fn(q, f, a)))
    def contrast(name, terms, role, cross_query=False):
        backgrounds = defaultdict(float)
        for cid, w in terms.items():
            r = row_by_id[cid]
            backgrounds[(r['demo_family'], r['lexicon_arm'], r['demo_surface'])] += w
        cancels = not any(backgrounds.values())
        require(abs(sum(terms.values())) < 1e-10, 'unbalanced linear contrast')
        if cross_query: require(cancels, 'cross-query NCC background did not cancel')
        contrasts.append({'contrast_id': name, 'role': role, 'cross_query': cross_query,
                          'ncc_background_cancels': cancels,
                          'human_stratum_validation_required': cross_query,
                          'terms': [{'condition_id': k, 'coefficient': v} for k, v in sorted(terms.items())]})
    for q in queries:
        qid = q['query_id']
        for f in (1, 2):
            arms = ['O', 'N1', 'N2', 'D'] + (['P1', 'X1', 'P2', 'X2'] if f == 1 else [])
            for arm in arms: contrast(f'{qid}-F{f}-{arm}-delta', pair(qid, f, arm), 'within_query_word_pair')
            transitions = [('O', 'N1'), ('O', 'N2'), ('O', 'D')]
            if f == 1: transitions += [('P1', 'X1'), ('P2', 'X2')] + [('O', a) for a in ['P1', 'X1', 'P2', 'X2']]
            for a, b in transitions:
                role = 'primary' if b in ('N1', 'N2') else 'natural_bridge' if b == 'D' else 'scope_matched' if a != 'O' else 'scope_total_bridge'
                for form in ('H', 'A'):
                    contrast(f'{qid}-F{f}-{a}-{b}-s{form}',
                             combine((1, margin(qid, f, b, form)), (-1, margin(qid, f, a, form))), role)
                for kind in ('I', 'C'): contrast(f'{qid}-F{f}-{a}-{b}-{kind}', effect(qid, f, a, b, kind), role)
    for family in ('A', 'B'):
        qs = {q['cell']: q['query_id'] for q in queries if q['query_family'] == family}
        for f in (1, 2):
            for arm in ('O', 'N1', 'N2', 'D'):
                for sense in ('L', 'G'):
                    contrast(f'{family}-F{f}-{arm}-K{ sense }', combine((1, avg(qs[sense + '0'], f, arm)),
                             (-1, avg(qs[sense + '1'], f, arm))), 'query_discrimination', True)
            transitions = [('O', 'N1'), ('O', 'N2'), ('O', 'D')]
            if f == 1: transitions += [('P1', 'X1'), ('P2', 'X2')]
            for a, b in transitions:
                for sense in ('L', 'G'):
                    contrast(f'{family}-F{f}-{a}-{b}-dK{sense}',
                             combine((1, effect(qs[sense + '0'], f, a, b, 'C')),
                                     (-1, effect(qs[sense + '1'], f, a, b, 'C'))), 'query_discrimination_change', True)
                for stance in ('0', '1'):
                    for name, kind in (('J', 'C'), ('T', 'I')):
                        contrast(f'{family}-F{f}-{a}-{b}-{name}{stance}',
                                 combine((1, effect(qs['G' + stance], f, a, b, kind)),
                                         (-1, effect(qs['L' + stance], f, a, b, kind))), 'query_dependence', True)

    sources = [DRAFT / 'materials-source.json', DRAFT / 'MATRIX.md', DRAFT / 'INTERPRETATION-PLAN.md',
               Path(__file__), BASE / 'lexicon-scope-results-v1/current.json',
               BASE / 'lexicon-scope-results-v1/INTERPRETATION.md', PARENT / 'manifest.json',
               PARENT / 'contexts.jsonl', PARENT / 'plan.json', PARENT / 'materials.json',
               PARENT / 'inherited-demo-materials.json', BASE / 'analysis-reference-v1/current.json',
               PREF / 'preferences/current.json', PREF / 'preferences/snapshots' / source['preferences_snapshot_id'] / 'preferences.json',
               PREF / 'preferences/snapshots' / source['preferences_snapshot_id'] / 'preferences.md',
               PREF / 'policies/severity-hate-default-mapping-v1.json', PREF / 'policies/attack-severity-v1.json',
               PREF / 'resource-reviews-v1/preferences.md',
               ROOT / 'docs/research/annotation-guidelines/evidence-applicability-annotation-policy-v2.md',
               ROOT / 'docs/research/annotation-guidelines/evidence-severity-hate-default-mapping-v1.md',
               ROOT / 'docs/research/annotation-guidelines/evidence-attack-severity-policy-v1.md',
               ROOT / 'docs/research/experiment-plans/general-model-evidence-applicability-audit-controls-v1.md']
    tokenizer_files = [TOKENIZER / n for n in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt') if (TOKENIZER / n).exists()]
    sources += tokenizer_files
    source_hashes = {relative(p): digest(p.read_bytes()) for p in sources}
    old_current_hashes = {relative(p): digest(p.read_bytes()) for p in BASE.glob('*/current.json') if p.parent != PUBLIC}

    stage_prompts = {stage: {x['prompt_id'] for x in bindings if x['stage'] == stage}
                     for stage in ('stage_1', 'stage_2_supplement')}
    require(len(unique) == 708 and len(stage_prompts['stage_1']) == 476
            and len(stage_prompts['stage_2_supplement']) == 232, 'unique prompt count differs')
    require(not stage_prompts['stage_1'] & stage_prompts['stage_2_supplement'], 'stage overlap not registered')
    require(len(parent_replays) == 132, 'historical CPU replay frame differs')
    lengths = []
    for q in queries:
        original = unique[binding_index[(q['query_id'], 1, 'H', 'O', 'original', None)]]
        lengths.append({'query_id': q['query_id'], 'unicode_code_points': len(q['text']),
                        'standalone_query_tokens': len(tokenizer.encode(q['text'], add_special_tokens=False)),
                        'F1_O_H_original_full_prompt_tokens': original['prompt_tokens']})
    require(len({x['contrast_id'] for x in contrasts}) == len(contrasts), 'duplicate comparison id')
    audit = {'schema_version': 'functional-query-draft-cpu-audit/v1', 'status': 'passed',
             'draft_only': True, 'new_queries': 8, 'pending_applicability_reviews': 8,
             'pending_label_reviews': 8, 'human_fields_added': 0, 'model_weights_loaded': False,
             'model_forward_executed': False, 'online_writeback': False, 'formal_input_freeze': False,
             'mechanism_ready': False, 'conditions_by_role': dict(Counter(x['role'] for x in matrix)),
             'logical_prompt_bindings': len(bindings), 'unique_prompt_previews': len(unique),
             'candidate_boundaries_checked': len(unique) * 2, 'old_prompt_token_replays': len(parent_replays),
             'geometry_checks': dict(Counter(x['kind'] for x in proofs)),
             'preregistered_draft_contrasts': len(contrasts),
             'cross_query_background_cancellation_checks': sum(c['cross_query'] for c in contrasts),
             'query_lengths': lengths, 'source_files': source_hashes, 'old_current_pointers': old_current_hashes}

    query_md = ['# 8条功能诊断查询草案\n', '本批仅为AI草案，全部人工正文采纳、义项适用性及标签字段留空。没有运行新查询模型结果。详细分开审核表见 [REVIEW.md](REVIEW.md)。\n',
                '原义项：' + source['original_definition'] + '\n',
                '| 编号 | 拟定语境 | 完整查询 | AI适用性建议 | AI标签建议 |', '|---|---|---|---|---|']
    for q in queries:
        fit = '不适用（普通笑声）' if q['cell'].startswith('L') else '适用（转述／引文中的称呼）'
        if q['query_family'] == 'B' and q['cell'].startswith('G'): fit += '；所指须另审'
        query_md.append(f"| {q['query_id']} | {q['query_family']}/{q['cell']} | {q['text']} | {fit} | {q['ai_label']['hate']} |")
    query_md += ['', 'L0＝笑声，无作者攻击；L1＝笑声，另有个人攻击；G0＝人群称呼，作者反对；G1＝人群称呼，作者赞同。这些是设计意图，不是人审结果。', '',
                 'A套明示黑人，B套称呼中的种族所指留作独立审核点。两套不是完全等信息的同义改写；每套分别报告。G0/G1各自只替换“反对／赞同”，L0/L1则增加独立辱骂分句，不能称为相同长度的纯立场操纵。', '',
                 '新ID不进入原discovery/reserve队列，不继承3169的Gold。自然度、所指或标签有疑问可保留未决；修订需建立新材料版本。', '',
                 '## 文本身份与长度（CPU tokenizer）', '', '| 编号 | SHA-256 | Unicode长度 | 完整O/F1/嘿嘿/原标签提示tokens |', '|---|---|---:|---:|']
    for q, length in zip(queries, lengths):
        query_md.append(f"| {q['query_id']} | {q['text_sha256']} | {len(q['text'])} | {length['F1_O_H_original_full_prompt_tokens']} |")

    review_md = ['# 适用性与标签分开审核表\n',
                '当前为16条待审记录：8条义项适用性、8条hate参考；人工字段均为空。AI建议显示在单独列，可认可、修改或保留未决。保存草稿、认可正文或运行程序都不自动确认另一类字段。\n',
                '审核沿用现有三栏工作台的字段含义：左侧为Q01–Q08队列，中间保留查询和原义项，右侧分开记录适用性与标签。本次交付为本地Markdown/CSV/JSON表，未部署新UI或写入32案例的正式会话。后续如接入页面，应复用 tools/general_model_paired_review_ui 的组件及独立会话，不能把此文件直接导入旧生产schema。\n',
                '## 表A：原义项是否适合这次出现\n',
                '原义项：' + source['original_definition'] + '\n',
                '可选query_fit：valid_sense（本处采用该义）、wrong_sense（普通笑声等其他义）、uncertain（尚不能确定）。同时保留该义出现在作者正文还是他人引文中。原义项自身是否合理保持原审核记录，不在这里重新裁定。\n',
                '| 编号 | 完整查询 | AI建议／依据 | 人审适用性 | 人审证据或未决原因 |', '|---|---|---|---|---|']
    for q in queries:
        a = q['ai_applicability']
        review_md.append(f"| {q['query_id']} | {q['text']} | {a['query_fit']}：{a['reason']} {a['uncertainty'] or ''} | 待审 | 待填 |")
    review_md += ['', '## 表B：当前作者的hate参考', '',
                  '适用当前v2规则及随后用户已接受的严重度映射：0→non-hate，1–4→hate，未决保持null。个人辱骂按后续映射处理，不恢复v2早期个人冒犯的未统一状态。标签不从上表的valid_sense/wrong_sense推导。', '',
                  '| 编号 | 完整查询 | AI立场、对象与理由 | AI标签（严重度辅助） | 人审hate | 人审依据或未决原因 |', '|---|---|---|---|---|---|']
    for q in queries:
        a = q['ai_label']
        review_md.append(f"| {q['query_id']} | {q['text']} | {a['author_stance']}；{a['actual_target']}。{a['reason']} | {a['hate']}（{a['attack_severity']}） | 待审 | 待填 |")
    review_md += ['', '## 记录和范围', '',
                  '- Q07/Q08可以hate已解决而具体所指/适用性仍未决；不得相互补填。明确反对的引文义项适用，不代表当前作者攻击。',
                  '- 新查询无原数据参考，original_dataset_reference保持null；AI建议也不是原Gold。',
                  '- 词义和标签各自保留unreviewed/draft/confirmed进度与resolved/unresolved可用性；确认未决不产生硬标签。',
                  '- 不要求重写长理由；可直接认可AI证据或短改理由。只有明确回复所覆盖的字段获得人审身份。',
                  '- 文本更改建立draft-02，并重新绑定该行及依赖输入的哈希；draft-01留作历史。人审决定应另建feedback/reference版本，不回填覆盖AI原稿。',
                  '- 当前批次没有自动写回、正式输入冻结或机制放行。', '',
                  '规则和个案依据绑定preferences快照 `' + source['preferences_snapshot_id'] + '`。主要参考P01/P02（严重度映射、个人攻击）、P09（引用/反驳）、P19/P23（谈词及提及与攻击分开）；P09/P19/P23是有来源的AI个案归纳，不冒充用户新确认的通则。#3660只校准该辱称义在既有语境中的成立，#4137等仅提及案例用于校准词义存在不直接决定hate。', '',
                  '机器表：`applicability-review.csv`、`label-review.csv`及同名JSONL。两个表都有完整正文、文本哈希、AI栏与空的人审栏；精确出现位置使用Unicode code point的[start,end)区间。']
    docs = {'QUERIES.md': ('\n'.join(query_md) + '\n').encode(), 'REVIEW.md': ('\n'.join(review_md) + '\n').encode()}
    files = {DRAFT / name: data for name, data in docs.items()}
    files.update({DRAFT / 'query-inputs.jsonl': jl(input_queries), DRAFT / 'applicability-review.jsonl': jl(fits),
                  DRAFT / 'applicability-review.csv': cb(fits), DRAFT / 'label-review.jsonl': jl(labels),
                  DRAFT / 'label-review.csv': cb(labels), DRAFT / 'input-matrix.jsonl': jl(matrix),
                  DRAFT / 'input-matrix.csv': cb(matrix), DRAFT / 'comparisons.jsonl': jl(contrasts),
                  WORK / 'prompt-previews.jsonl': jl(list(unique.values())), WORK / 'prompt-bindings.jsonl': jl(bindings),
                  WORK / 'geometry.json': jb(geometries), WORK / 'geometry-checks.jsonl': jl(proofs),
                  DRAFT / 'cpu-audit.json': jb(audit)})
    manifest = {'schema_version': 'functional-query-draft-manifest/v1', 'status': 'draft_pending_human_review',
                'human_reference_tasks_resolved': 0, 'formal_input_freeze': False, 'gpu_executed': False,
                'mechanism_ready': False, 'source_files': source_hashes,
                'artifacts': {relative(p): digest(data) for p, data in files.items()}}
    files[DRAFT / 'manifest.json'] = jb(manifest)
    files[PUBLIC / 'current.json'] = jb({'schema_version': 'functional-query-draft-current/v1',
        'status': 'draft_pending_human_review', 'draft_path': relative(DRAFT), 'preview_path': relative(WORK),
        'manifest_sha256': digest(files[DRAFT / 'manifest.json']), 'new_queries': 8,
        'first_stage_new_conditions': 128, 'supplement_new_conditions': 64, 'historical_conditions': 4,
        'human_reference_tasks_resolved': 0, 'gpu_executed': False, 'formal_input_freeze': False,
        'mechanism_ready': False})
    return files, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='rebuild on CPU and verify existing bytes without writing')
    args = parser.parse_args()
    files, audit = build()
    if args.check:
        for path, data in files.items(): require(path.read_bytes() == data, 'draft byte reconstruction differs: ' + str(path))
    else:
        require(not any(p.exists() for p in files), 'outputs already exist; preserve draft and use a new version')
        for path, data in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as f: f.write(data)
    for rel, expected in audit['old_current_pointers'].items():
        require(digest((ROOT / rel).read_bytes()) == expected, 'old pointer changed: ' + rel)
    print(json.dumps({'status': 'checked' if args.check else 'prepared',
                      'files': len(files), 'new_queries': 8, 'prompt_previews': audit['unique_prompt_previews'],
                      'candidate_boundaries': audit['candidate_boundaries_checked'],
                      'geometry_checks': audit['geometry_checks'],
                      'cross_query_background_cancellation_checks': audit['cross_query_background_cancellation_checks'],
                      'gpu_executed': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
