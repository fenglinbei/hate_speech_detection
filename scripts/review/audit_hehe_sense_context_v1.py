#!/usr/bin/env python3
"""Independently reconstruct a nine-input review draft; never loads model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def audit(directory):
    out = Path(directory).absolute()
    manifest = read(out / 'manifest.json')
    for record in manifest['artifacts'] + manifest['sources']:
        data = Path(record['path']).read_bytes()
        check(len(data) == record['bytes'] and sha(data) == record['sha256'], f'Hash mismatch: {record["path"]}')
    for name in ('USE_TORCH', 'USE_TF', 'USE_FLAX', 'USE_TORCH_XLA'):
        os.environ[name] = '0'
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / 'models/base/Qwen3-8B'), local_files_only=True,
                                        trust_remote_code=False, use_fast=True)
    materials = read(out / 'materials.json')
    system = (out / 'model-task.txt').read_text(encoding='utf-8')
    task = ROOT / 'docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01/model-task.txt'
    check(system.encode() == task.read_bytes(), 'Task changed')
    queries = {q['query_id']: q for q in materials['queries']}
    dictionaries = {d['dictionary_id']: d for d in materials['dictionaries']}
    check(dictionaries['D03']['definition'] == dictionaries['D01']['definition'] + '\n' + dictionaries['D02']['definition'], 'Dual composition differs')
    check(queries['Q02']['text'] in queries['Q03']['text'], 'Quoted source changed')
    check(queries['Q01']['reference'] == '无' and queries['Q02']['reference'] == '有', 'Inherited labels changed')
    check(queries['Q03']['reference'] is None and queries['Q03']['proposed_reference'] == '无', 'AI proposal treated as confirmed')
    source_records = read(out / 'source-records.json')['records']
    for qid, rid in [('Q01', 'query:3169'), ('Q02', 'demo:3660')]:
        check(queries[qid]['text'] == source_records[rid]['source']['text'], 'Real text changed')
        check(source_records[rid]['review']['status'] == 'confirmed' and not source_records[rid]['stale'], 'Reference not confirmed')
    rows = [json.loads(line) for line in (out / 'model-inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    check(len(rows) == 9 and len({r['request_id'] for r in rows}) == 9, 'Wrong number of inputs')
    check({(r['query_id'], r['dictionary_id']) for r in rows} == {(q, d) for q in queries for d in dictionaries}, 'Incomplete cross')
    prefixes, spans_checked, within_query_lengths = set(), 0, {}
    for row in rows:
        check(not any('reference' in key or 'review' in key or 'proposed' in key for key in row), 'References in model input')
        q, d = queries[row['query_id']], dictionaries[row['dictionary_id']]
        entry_blocks = []
        for i, entry in enumerate(materials['base_lexicon'], 1):
            definition = d['definition'] if entry['lexicon_id'] == materials['focal_lexicon_id'] else entry['senses'][0]['definition']
            entry_blocks.append(f'词条{i}：\n词形：{entry["term"]}\n释义：{definition}')
        user = '【参考词典】\n' + '\n\n'.join(entry_blocks) + '\n\n【参考示例】\n（无）\n\n【待判断文本】\n' + q['text']
        expected_messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
        check(row['messages'] == expected_messages, 'Messages not independently reconstructed')
        prompt = tok.apply_chat_template(expected_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        check(prompt == row['prompt_text'], 'Native prompt differs')
        check((out / 'prompts' / (row['request_id'] + '.txt')).read_bytes() == prompt.encode(), 'Prompt file differs')
        check(prompt in (out / 'ALL-PROMPTS.md').read_text(encoding='utf-8'), 'Full prompt absent in review appendix')
        encoded = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded['input_ids'], encoded['offset_mapping']
        check(ids == row['input_ids'] and len(ids) == row['prompt_tokens'], 'Native token IDs differ')
        check([list(pair) for pair in offsets] == row['token_offsets'], 'Offsets differ')
        check(sha(prompt.encode()) == row['prompt_sha256'] and sha(canonical(ids)) == row['input_ids_sha256'], 'Input digest differs')
        check(row['candidate_tokens'] == {'有': 18830, '无': 42192}, 'Candidates differ')
        for label, token in [('有', 18830), ('无', 42192)]:
            check(tok.encode(label, add_special_tokens=False) == [token], 'Non-single-token label')
            check(tok.encode(prompt + label, add_special_tokens=False) == ids + [token], 'Candidate boundary differs')
        check(row['roles']['pre_answer'] == [len(ids)-1] and row['roles']['demos_end'] == [], 'Role mismatch')
        user_start = prompt.index(user)
        query_start = user_start + user.index('【待判断文本】\n') + len('【待判断文本】\n')
        q_tokens = [i for i, (a,b) in enumerate(offsets) if a < query_start + len(q['text']) and b > query_start]
        check(row['roles']['query_all'] == q_tokens and row['roles']['query_end'] == [q_tokens[-1]], 'Query positions differ')
        focal_tokens, cursor = set(), 0
        while (pos := q['text'].find('嘿嘿', cursor)) >= 0:
            start, end = query_start + pos, query_start + pos + 2
            focal_tokens.update(i for i, (a,b) in enumerate(offsets) if a < end and b > start)
            cursor = pos + 2
        check(row['roles']['query_focal'] == sorted(focal_tokens), 'Focal query positions differ')
        cut = user_start + user.index('【参考示例】')
        prefix = tok.encode(prompt[:cut], add_special_tokens=False)
        check(prefix == ids[:len(prefix)] and row['roles']['lexicon_end'] == [len(prefix)-1], 'Dictionary prefix is not exact')
        prefixes.add(tuple(prefix))
        leaves = [s for s in row['spans'] if s['char_start'] is not None and s['kind'] not in ('demo', 'lexicon')]
        expected_owners = []
        for a, b in offsets:
            touching = [s for s in leaves if a < s['char_end'] and b > s['char_start']]
            contained = [s for s in touching if s['char_start'] <= a < b <= s['char_end']]
            expected_owners.append(contained[0]['id'] if len(contained) == len(touching) == 1
                                   else 'boundary' if touching else 'structure')
        check(expected_owners == row['token_owners'], 'Semantic token ownership differs')
        for span in row['spans']:
            if span['char_start'] is not None:
                check(0 <= span['char_start'] < span['char_end'] <= len(prompt), 'Invalid span')
                spans_checked += 1
            positions = ([i for i,(a,b) in enumerate(offsets) if span['char_start'] <= a < b <= span['char_end']]
                         if span['kind'] in ('demo', 'lexicon') else
                         [i for i,owner in enumerate(expected_owners) if owner == span['id']])
            check(positions == span['token_positions'], 'Span token positions differ')
        within_query_lengths.setdefault(row['query_id'], {})[row['dictionary_id']] = len(ids)
    for lengths in within_query_lengths.values():
        check(lengths['D01'] == lengths['D02'] < lengths['D03'], 'Unexpected definition length control')
    rows_by_id = {r['request_id']: r for r in rows}
    for qid in queries:
        original = rows_by_id[f'hsc-{qid}-D01']
        ordinary = rows_by_id[f'hsc-{qid}-D02']
        check(original['roles'] == ordinary['roles'], 'Single-sense role positions changed')
        check(original['token_owners'] == ordinary['token_owners'], 'Single-sense ownership positions changed')
        for i, owner in enumerate(original['token_owners']):
            if owner != 'lex-0419:definition':
                check(original['input_ids'][i] == ordinary['input_ids'][i], 'Changed token outside target definition')
    components = read(out / 'definition-components.json')['records']
    check(len(components) == 9, 'Missing definition component records')
    for component_record in components:
        row = rows_by_id[component_record['request_id']]
        assigned = []
        for part in component_record['components']:
            start, end = part['char_start'], part['char_end']
            check(row['prompt_text'][start:end] == part['text'], 'Component text differs')
            positions = [i for i,(a,b) in enumerate(row['token_offsets'])
                         if start <= a < b <= end and row['token_owners'][i] == 'lex-0419:definition']
            check(positions == part['owned_token_positions'], 'Component owners differ')
            assigned += positions
        check(len(assigned) == len(set(assigned)), 'Double-counted component token')
        parent = next(s for s in row['spans'] if s['id'] == 'lex-0419:definition')
        extra = component_record['separator_or_cross_boundary_tokens']
        check(set(assigned).isdisjoint(extra) and set(assigned) | set(extra) == set(parent['token_positions']), 'Components do not partition parent')
    comparisons = read(out / 'comparisons.json')['comparisons']
    check(len(comparisons) == 9 and len({x['comparison_id'] for x in comparisons}) == 9, 'Comparison count mismatch')
    for expression in comparisons:
        check(expression['value'] is None and expression['bound'] is None, 'Invented model result')
        check(sum(term['coefficient'] for term in expression['terms']) == 0, 'Not a contrast')
        check(all(term['request_id'] in rows_by_id for term in expression['terms']), 'Unknown endpoint')
    pending = read(out / 'feedback-template.json')['items']
    check([i['item_id'] for i in pending] == ['Q02','Q03','D03'], 'Review queue changed')
    check(all(i['human_decision'] is None and i['status'] == 'pending' for i in pending), 'Invented human decision')
    old_rows = {r['request_id']: r for r in [json.loads(line) for line in (ROOT / 'reviews/case-content-replacement-v1/prepared-01/scoring-inputs.jsonl').read_text().splitlines()]}
    for new_id, old_id in [('hsc-Q01-D01', 'ccr-3169-L-base'), ('hsc-Q01-D02', 'ccr-3169-L-L05')]:
        for key in ('messages','prompt_text','input_ids','roles','spans','token_offsets','token_owners'):
            check(rows_by_id[new_id][key] == old_rows[old_id][key], 'Historical bridge differs')
    check('torch' not in sys.modules, 'Unexpected torch import')
    return {'status': 'pass', 'manifest_sha256': sha((out / 'manifest.json').read_bytes()),
            'independently_reconstructed_prompts': 9, 'single_token_candidate_boundaries': 18,
            'exact_dictionary_prefix_records': 9, 'unique_dictionary_prefixes': len(prefixes),
            'span_bounds_checked': spans_checked, 'semantic_token_ownership_reconstructed': True,
            'single_sense_outside_definition_tokens_unchanged': True,
            'definition_component_records': len(components),
            'within_query_lengths': within_query_lengths, 'exact_historical_inputs': 2,
            'comparisons_checked': 9, 'pending_review_items': ['Q02','Q03','D03'],
            'model_weights_loaded': False, 'torch_imported': False, 'GPU_touched': False,
            'human_decisions_created': 0, 'execution_qualification_completed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.draft)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write('\n')
    print(json.dumps(result, ensure_ascii=False))
