#!/usr/bin/env python3
"""CPU tokenizer and independent matrix audit for the six-module input freeze."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.general_model_evidence_evaluation import require, file_sha
from diagnostics.q01_module_package import FREEZE, PARENT, read_json, read_lines, load_frozen


def audit(directory):
    for key in ('USE_TORCH', 'USE_TF', 'USE_FLAX', 'USE_TORCH_XLA'):
        os.environ[key] = '0'
    plan, contexts, positions, requests = load_frozen(directory)
    from transformers import AutoTokenizer
    model = read_json(Path(plan['runtime_parent_plan']['package_path']) / 'models.json')[0]
    inventory = model['tokenizer_inventory']
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / inventory['logical_repo_path']),
        local_files_only=True, trust_remote_code=False, use_fast=True)
    for name in ('contexts.jsonl', 'positions.jsonl', 'positions.csv', 'candidate-boundaries.jsonl',
                 'historical-scores.jsonl', 'analysis-reference.json', 'scoring-spec.json'):
        require((directory / name).read_bytes() == (PARENT / name).read_bytes(), 'inherited bytes differ: ' + name)
    boundaries = 0
    for c in contexts:
        encoded = tokenizer(c['prompt_text'], add_special_tokens=False, return_offsets_mapping=True)
        require(encoded['input_ids'] == c['prompt_token_ids'], 'independent tokenizer replay differs')
        require(tokenizer.apply_chat_template(c['messages'], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) == c['prompt_text'], 'chat template differs')
        for role in positions[c['record_id']]['roles'].values():
            for i, token, span in zip(role['token_positions'], role['token_ids'], role['token_char_spans'], strict=True):
                require(encoded['input_ids'][i] == token and list(encoded['offset_mapping'][i]) == span, 'position binding differs')
        require(positions[c['record_id']]['roles']['pre_answer']['token_positions'] == [len(encoded['input_ids']) - 1], 'answer position differs')
        for a in plan['catalog'][c['task']]:
            require(tokenizer.encode(c['prompt_text'] + a['canonical_answer'], add_special_tokens=False) ==
                    c['prompt_token_ids'] + a['answer_token_ids'], 'full-label boundary differs')
            boundaries += 1
    by_id = {c['record_id']: c for c in contexts}
    pairs = set()
    for r in requests:
        if not r['donor'] or r['donor'] == r['recipient']:
            continue
        left, right = by_id[r['recipient']], by_id[r['donor']]
        require(left['encoding'] == right['encoding'] and left['probe_id'] == right['probe_id'], 'cross-variant donor')
        require(left['prompt_tokens'] == right['prompt_tokens'], 'unequal pair length')
        changed = {i for i, (a, b) in enumerate(zip(left['prompt_token_ids'], right['prompt_token_ids'], strict=True)) if a != b}
        slot = set(positions[left['record_id']]['roles']['lexicon_slot']['token_positions'])
        require(changed and changed <= slot, 'pair changes tokens outside fixed lexicon slot')
        pairs.add(tuple(sorted((r['recipient'], r['donor']))))
    primary = [r for r in requests if r['category'] == 'primary']
    expected = {(layer, module, family, neutral, surface, direction, encoding, probe)
                for layer in (33, 34, 35) for module in ('attention', 'mlp') for family in (1, 2)
                for neutral in ('N1', 'N2') for surface in ('H', 'A') for direction in ('R', 'K')
                for encoding, probe in [('original', None), ('ab_forward', None), ('ab_reverse', None)] +
                [('original', p) for p in ('empty', 'space', 'na', 'mask', 'lorem')]}
    actual = {(r['layer'], r['module'], r['family'], r['group'].split('-')[-1], r['surface'], r['direction'],
               r['encoding'], r['probe_id']) for r in primary}
    require(actual == expected and len(primary) == len(expected) == 768, 'independent primary Cartesian product differs')
    return {'status': 'passed', 'plan_id': plan['plan_id'], 'manifest_sha256': file_sha(directory / 'manifest.json'),
        'source_files_verified': len(plan['source_files']), 'prompt_reconstructions': len(contexts),
        'candidate_boundaries': boundaries, 'matched_pairs': len(pairs), 'primary_requests': len(primary),
        'gpu_forward_executed': False, 'weights_loaded': False, 'analysis_reference_parsed': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, default=FREEZE)
    args = p.parse_args()
    print(json.dumps(audit(args.plan.resolve()), ensure_ascii=False))
