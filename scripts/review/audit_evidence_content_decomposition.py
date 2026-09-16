#!/usr/bin/env python3
"""Independently replay frozen input bytes against parent prompts, on CPU only."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'exps/causal_context/general_model_evidence_applicability_v1'
DEFAULT = BASE / 'reviews/analysis-freeze-20260914/content-decomposition-v1/frozen-01'


def digest(raw): return hashlib.sha256(raw).hexdigest()
def read(path): return json.loads(path.read_bytes())
def lines(path): return [json.loads(line) for line in path.read_bytes().splitlines() if line]
def canonical(value): return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def need(condition, message):
    if not condition: raise ValueError(message)


def audit(directory):
    m = read(directory / 'manifest.json'); plan = read(directory / 'plan.json')
    for name, h in m['source_files'].items():
        path = (ROOT / name).resolve()
        need(path.is_relative_to(ROOT) and digest(path.read_bytes()) == h, 'source bytes changed: ' + name)
    for name, h in m['artifacts'].items():
        need(Path(name).name == name and digest((directory / name).read_bytes()) == h, 'artifact changed: ' + name)
    need(set(m['artifacts']) | {'manifest.json'} == {p.name for p in directory.iterdir()}, 'inventory differs')
    identity = dict(plan); pid = identity.pop('plan_id')
    need(pid == 'evidence-content-decomposition-' + digest(canonical(identity)), 'plan hash differs')
    pointer = read(BASE / 'label-calibration-results-v1/current.json')
    old_dir = BASE / pointer['freeze_path']; old_plan = read(old_dir / 'plan.json')
    old = {(c['query_id'], c['root_condition'], c['encoding'], c['probe_id']): c for c in lines(old_dir / 'contexts.jsonl')}
    raw = {r['record_id']: r for r in lines(BASE / pointer['run_path'] / old_plan['raw_pass'] / 'scores.jsonl')}
    rich_pointer = read(BASE / 'position-length-materials-v1/current.json')
    rich = {(r['query_id'], r['condition']): r for r in lines(BASE / rich_pointer['materials_path'] / 'contexts.jsonl')}
    material = read(directory / 'materials.json')
    materials = {(r['query_id'], r['condition']): r for r in material['materials']}
    contexts = lines(directory / 'contexts.jsonl'); geometry = {g['record_id']: g for g in lines(directory / 'geometry.jsonl')}
    comps = lines(directory / 'comparisons.jsonl')
    targets = {'541': '826', '3169': '3660'}
    variants = [('original', None), *[('original', p) for p in ('empty', 'space', 'na', 'mask', 'lorem')], ('ab_forward', None), ('ab_reverse', None)]
    historical_conditions = ('R-A', 'R-C1', 'R-C2', 'R-D1', 'R-D2')
    cells = {'541': ('T0R0', 'T1R0', 'T0R1', 'T1R1'), '3169': ('G_H', 'G_E', 'L_H', 'L_O')}
    expected = {(q, c, e, p) for q in targets
        for c in [*historical_conditions, *(f'F{f}-{cell}' for f in (1, 2) for cell in cells[q])]
        for e, p in variants}
    need(len(contexts) == len(expected) == len({c['record_id'] for c in contexts}) == 208, 'duplicate/missing inputs')
    need({(c['query_id'], c['root_condition'], c['encoding'], c['probe_id']) for c in contexts} == expected, 'factor/probe frame differs')
    need(plan['catalog'] == old_plan['catalog'] and plan['numeric_policy'] == old_plan['numeric_policy'], 'scoring baseline changed')
    for flag in ('USE_TORCH', 'USE_TF', 'USE_FLAX', 'USE_TORCH_XLA'): os.environ[flag] = '0'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'models/base/Qwen3-8B'), local_files_only=True, trust_remote_code=False, use_fast=True)
    boundaries = 0
    def span(offsets, start, end):
        covered = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
        need(covered and covered == list(range(covered[0], covered[-1] + 1)), 'noncontiguous span')
        return [covered[0], covered[-1] + 1]
    for c in contexts:
        q, condition, e, p = c['query_id'], c['root_condition'], c['encoding'], c['probe_id']
        main = materials.get((q, condition))
        parent_condition = 'R-A' if main else condition
        parent = old[q, parent_condition, e, p]; background = rich[q, parent_condition]
        expected_messages = deepcopy(parent['messages']); expected_prompt = parent['prompt_text']
        texts = dict(background['texts_by_demo'])
        if main:
            original = texts[targets[q]]; replacement = main['text']
            need(expected_prompt.count(original) == expected_messages[1]['content'].count(original) == 1, 'ambiguous edit')
            expected_prompt = expected_prompt.replace(original, replacement)
            expected_messages[1]['content'] = expected_messages[1]['content'].replace(original, replacement)
            texts[targets[q]] = replacement
        need(expected_prompt == c['prompt_text'] and expected_messages == c['messages'], 'changed non-target prompt bytes')
        need(tokenizer.apply_chat_template(c['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False) == expected_prompt, 'chat serialization differs')
        encoded = tokenizer(expected_prompt, add_special_tokens=False, return_offsets_mapping=True)
        need(encoded['input_ids'] == c['prompt_token_ids'] and len(encoded['input_ids']) == c['prompt_tokens'], 'token replay differs')
        need(digest(expected_prompt.encode()) == c['prompt_sha256'] and digest(canonical(encoded['input_ids'])) == c['prompt_token_ids_sha256'], 'prompt hashes differ')
        need(digest(canonical({k: v for k, v in c.items() if k != 'context_sha256'})) == c['context_sha256'], 'context identity differs')
        positions = {}
        mapping = {'original': {'hate': 'hate', 'non-hate': 'non-hate'}, 'ab_forward': {'hate': 'A', 'non-hate': 'B'}, 'ab_reverse': {'hate': 'B', 'non-hate': 'A'}}[e]
        for slot, demo in enumerate(background['demo_ids'], 1):
            answer = json.dumps(mapping[background['answers_by_demo'][demo]])
            prefix = f'示例 {slot}\n文本：'; block = prefix + texts[demo] + '\n输出：' + answer
            need(expected_prompt.count(block) == 1, 'demo bytes differ')
            start = expected_prompt.index(block); end = start + len(block)
            positions[demo] = {'slot': slot, 'block': span(encoded['offset_mapping'], start, end),
                'text': span(encoded['offset_mapping'], start + len(prefix), start + len(prefix) + len(texts[demo])),
                'answer': span(encoded['offset_mapping'], end - len(answer), end)}
        query_start = expected_prompt.index('待判断文本（JSON 字符串）：\n')
        query_end = expected_prompt.index('<|im_end|>', query_start)
        actual = {'demos': positions, 'query_block': span(encoded['offset_mapping'], query_start, query_end), 'generation_start': len(encoded['input_ids'])}
        need(actual == geometry[c['record_id']]['layout'], 'independent layout differs')
        for candidate in plan['catalog'][c['task']]:
            need(tokenizer.encode(expected_prompt + candidate['canonical_answer'], add_special_tokens=False) == encoded['input_ids'] + candidate['answer_token_ids'], 'answer boundary differs')
            boundaries += 1
    ci = {(c['query_id'], c['root_condition'], c['encoding'], c['probe_id']): c for c in contexts}
    matched = 0
    for comp in comps:
        need(sum(t['coefficient'] for t in comp['terms']) == 0, 'contrast coefficients do not balance')
        for e, p in variants:
            rows = [ci[comp['query_id'], t['condition'], e, p] for t in comp['terms']]
            if comp['role'] != 'primary': continue
            first = rows[0]; a = geometry[first['record_id']]['layout']; start, end = a['demos'][targets[comp['query_id']]]['text']
            for other in rows[1:]:
                need(a == geometry[other['record_id']]['layout'], 'primary layout mismatch')
                need(first['prompt_token_ids'][:start] == other['prompt_token_ids'][:start]
                     and first['prompt_token_ids'][end:] == other['prompt_token_ids'][end:], 'non-target tokens differ')
            matched += 1
    historical = lines(directory / 'historical-selected.jsonl')
    selected = [c for c in contexts if c['baseline_replay']]
    need(len(historical) == len(selected) == 80, 'historical frame differs')
    for c, row in zip(selected, historical):
        prior = raw[c['source_record_id']]
        need(row['record_id'] == c['record_id'] and row['historical_record_id'] == prior['record_id'], 'historical binding differs')
        need(row['candidates'] == prior['candidates'], 'historical candidate payload changed')
    design = read(BASE / 'content-decomposition-v1/design-draft-01/design.json')
    primary_expected = {(q, f, c['name']): {f'F{f}-{cell}': w for cell, w in c['terms'].items()}
        for q in targets for f in (1, 2) for c in design['case_' + q]['comparisons_per_template']}
    need({(c['query_id'], c['template'], c['kind']): {t['condition']: t['coefficient'] for t in c['terms']}
          for c in comps if c['role'] == 'primary'} == primary_expected, 'registered primary coefficients changed')
    evaluations = sum(2 * (len(selected) if s['selection'] == 'baseline' else len(contexts)) for s in plan['schedule'])
    need((boundaries, matched, evaluations) == (416, 112, 2816), 'independent counts differ')
    need(not {'torch', 'tensorflow', 'jax'} & set(sys.modules), 'audit loaded model framework')
    return {'schema_version': 'evidence-content-decomposition-independent-input-audit/v1', 'status': 'passed',
        'plan_id': plan['plan_id'], 'freeze_manifest_sha256': digest((directory / 'manifest.json').read_bytes()),
        'source_files_verified': len(m['source_files']), 'prompt_reconstructions': len(contexts),
        'candidate_boundaries': boundaries, 'primary_variant_proofs': matched, 'historical_payloads_exact': len(historical),
        'planned_candidate_evaluations': evaluations, 'model_forward_executed': False, 'gpu_runtime_validated': False,
        'human_decisions_changed': 0, 'query_reference_labels_parsed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(); result = audit(args.plan)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle: handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__': main()
