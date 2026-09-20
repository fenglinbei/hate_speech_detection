#!/usr/bin/env python3
"""Independent CPU reconstruction of prompt texts, spans and registered positions."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from diagnostics import case_attention_inputs_v1 as c


def audit(prepared):
    tok = c.tokenizer()
    prepared = Path(prepared)
    _, profile, inputs = c.validate(prepared, sealed=False)
    materials = c.read(prepared / 'materials.json')
    cases = {case['query_id']: case for case in materials['cases']}
    system = (c.TASK / 'model-task.txt').read_text(encoding='utf-8')
    spec = importlib.util.spec_from_file_location('current_task_contract', c.TASK / 'validate_contract.py')
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    n_spans = boundaries = prefixes = 0
    for r in inputs:
        case = cases[r['query_id']]
        entries = []
        if r['lexicon_mode'] != 'none':
            for lex in case['lexicon']:
                sense = lex['senses'][0]
                text = '词形：' + lex['term'] + '\n释义：' + sense['definition']
                if r['lexicon_mode'] == 'category':
                    text += '\n类别：' + json.dumps(sense['categories'], ensure_ascii=False, separators=(',', ':'))
                entries.append(text)
        demos = [{'text': d['text'], 'answer': d['answer']} for d in case['demos']] if r['with_demos'] else []
        user = contract.render_user(case['query_text'], entries, demos)
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
        assert messages == r['messages'], 'Independent renderer differs'
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert prompt == r['prompt_text']
        enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        assert enc['input_ids'] == r['input_ids'] and [list(x) for x in enc['offset_mapping']] == r['token_offsets']
        for answer in ['有', '无']:
            assert tok.encode(prompt + answer, add_special_tokens=False) == enc['input_ids'] + [r['candidate_tokens'][answer]]
            boundaries += 1
        leaves = [s for s in r['spans'] if s['kind'] not in ('demo', 'lexicon', 'structure', 'boundary')]
        owner_positions = {s['id']: set(s['token_positions']) for s in r['spans'] if s['kind'] not in ('demo', 'lexicon')}
        all_positions = [i for indices in owner_positions.values() for i in indices]
        assert sorted(all_positions) == list(range(len(enc['input_ids']))), 'Leaf partition double-counts or omits tokens'
        for s in r['spans']:
            if s['char_start'] is not None:
                for i in s['token_positions']:
                    a, b = enc['offset_mapping'][i]
                    assert s['char_start'] <= a < b <= s['char_end']
            n_spans += 1
        for i, (a, b) in enumerate(enc['offset_mapping']):
            touching = [s for s in leaves if a < s['char_end'] and s['char_start'] < b]
            complete = [s for s in touching if s['char_start'] <= a < b <= s['char_end']]
            expected = complete[0]['id'] if len(complete) == len(touching) == 1 else ('boundary' if touching else 'structure')
            assert r['token_owners'][i] == expected
        for proof in r['prefix_proofs']:
            marker = '【参考示例】' if proof['role'] == 'lexicon_end' else '【待判断文本】'
            cut = prompt.index(user) + user.index(marker)
            assert cut == proof['char_cut']
            partial = tok.encode(prompt[:cut], add_special_tokens=False)
            assert partial == r['input_ids'][:len(partial)]
            assert r['roles'][proof['role']] == [len(partial) - 1]
            assert c.digest(partial) == proof['prefix_ids_sha256']
            prefixes += 1
        assert r['roles']['pre_answer'] == [len(r['input_ids']) - 1]
        q = next(s for s in r['spans'] if s['id'] == 'query')
        assert all(enc['offset_mapping'][i][0] < q['char_end'] and enc['offset_mapping'][i][1] > q['char_start']
                   for i in r['roles']['query_all'])
        # Source materials are exact raw texts, regardless of changed labels.
        original = c.read(c.CARDS / ('01-541.json' if r['query_id'] == '541' else '07-3169.json'))
        assert case['query_text'] == original['query']['content']
        assert case['lexicon'] == original['lexicon_entries']
        assert [d['text'] for d in case['demos']] == [d['content'] for d in original['demonstrations']]
    assert len(inputs) == 12 and boundaries == 24 and prefixes == 14
    return {'status': 'pass', 'independent_renderer_reconstructions': len(inputs), 'candidate_boundaries': boundaries,
            'exact_prefix_reconstructions': prefixes, 'span_checks': n_spans, 'source_texts_exact': True,
            'leaf_partition_exact': True, 'display_answer_changes': materials['display_answer_changes'],
            'research_pretrained_weights_loaded': False, 'CUDA_initialized': False, 'GPU_qualification': False,
            'implementation_snapshot': [c.info(p) for p in c.CODE],
            'method': 'Current task normative renderer + independent token/span enumeration; no target model predictions.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, default=c.PREPARED)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = audit(args.prepared)
    if args.output:
        c.write(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
