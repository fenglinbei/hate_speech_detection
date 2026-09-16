"""Independent CPU replay. Does not call the Q01 builder, hooks or scorer."""
import hashlib
import json
import os
from pathlib import Path
from collections import Counter

ROOT = Path('/data/liaozijie/hate_speech_detection')
WORK = ROOT / 'exps/causal_context/general_model_evidence_applicability_v1/reviews/q01-local-mechanism-v1'
FROZEN = WORK / 'frozen-01'


def read(path): return json.loads(path.read_text())
def lines(path): return [json.loads(x) for x in path.read_text().splitlines()]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest, plan = read(FROZEN / 'manifest.json'), read(FROZEN / 'plan.json')
    for name, h in manifest['source_files'].items(): assert sha(ROOT / name) == h, name
    for name, h in manifest['artifacts'].items(): assert sha(FROZEN / name) == h, name
    parent = WORK.parent / 'functional-query-diagnostics-v1/execution-01/frozen-stage-1-01'
    original = {r['record_id']: r for r in lines(parent / 'contexts.jsonl')}
    contexts = {r['record_id']: r for r in lines(FROZEN / 'contexts.jsonl')}
    positions = {r['record_id']: r for r in lines(FROZEN / 'positions.jsonl')}
    requests = lines(FROZEN / 'requests.jsonl')
    assert len(contexts) == len(positions) == 96
    assert all(c == original[cid] for cid, c in contexts.items())
    package = Path(plan['runtime_parent_plan']['package_path'])
    model = read(package / 'models.json')[0]
    for key in ('USE_TORCH', 'USE_TF', 'USE_FLAX', 'USE_TORCH_XLA'): os.environ[key] = '0'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / model['tokenizer_inventory']['logical_repo_path']),
        local_files_only=True, trust_remote_code=False, use_fast=True)
    boundary_count, role_count, absent = 0, 0, 0
    for cid, c in contexts.items():
        text = c['prompt_text']; encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        assert encoded['input_ids'] == c['prompt_token_ids']
        assert tokenizer.apply_chat_template(c['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False) == text
        for candidate in plan['catalog'][c['task']]:
            aid = tokenizer.encode(candidate['canonical_answer'], add_special_tokens=False)
            assert aid == candidate['answer_token_ids']
            assert tokenizer.encode(text + candidate['canonical_answer'], add_special_tokens=False) == c['prompt_token_ids'] + aid
            boundary_count += 1
        p = positions[cid]
        assert p['candidate_start'] == c['prompt_tokens'] and p['first_prediction_position'] == c['prompt_tokens'] - 1
        for role, r in p['roles'].items():
            role_count += 1
            if not r['applicable']:
                assert role == 'query_hehe' and c['probe_id'] is not None and not r['token_positions']
                absent += 1; continue
            ids = r['token_positions']
            assert ids == r['position_ids'] and all(0 <= i < c['prompt_tokens'] for i in ids)
            assert r['token_ids'] == [encoded['input_ids'][i] for i in ids]
            assert r['token_char_spans'] == [list(encoded['offset_mapping'][i]) for i in ids]
            if role == 'query_hehe':
                a, b = r['semantic_char_span']; assert text[a:b] == '嘿嘿'
                assert a > text.index('待判断文本（JSON 字符串）：')
    proofs = lines(FROZEN / 'pair-proofs.jsonl')
    for proof in proofs:
        a, b = contexts[proof['left']], contexts[proof['right']]
        diff = [i for i, (x, y) in enumerate(zip(a['prompt_token_ids'], b['prompt_token_ids'], strict=True)) if x != y]
        assert diff == proof['changed_positions'] and diff
        assert set(diff) <= set(positions[a['record_id']]['roles']['lexicon_slot']['token_positions'])
    for r in requests:
        rec = contexts[r['recipient']]
        if r['donor'] is None: continue
        donor = contexts[r['donor']]
        assert donor['encoding'] == rec['encoding'] and donor['probe_id'] == rec['probe_id']
        a, b = (next(x for x in c['bindings'] if x['query_id'] == 'FD-3169-Q01') for c in (rec, donor))
        assert (a['demo_family'], a['demo_surface']) == (b['demo_family'], b['demo_surface'])
        assert donor['prompt_tokens'] == rec['prompt_tokens']
        role = positions[rec['record_id']]['roles'][r['role']]
        assert role['applicable'] and set(r['positions']) <= set(role['token_positions'])
        if r['kind'] == 'embedding_reconstruction':
            changed = [i for i, (x, y) in enumerate(zip(rec['prompt_token_ids'], donor['prompt_token_ids'], strict=True)) if x != y]
            assert changed == r['positions']
        else: assert role['token_positions'] == r['positions']
        if r['kind'] == 'self_patch': assert r['donor'] == r['recipient']
    counts = Counter(r['category'] for r in requests)
    assert counts == {'baseline': 96, 'engineering': 4976, 'neutral_control': 2240,
                      'output_diagnostic': 128, 'primary': 3456, 'site_control': 1024}
    schedule = read(FROZEN / 'acceptance-schedule.json')
    assert len(schedule) == 12
    assert sum(p['candidate_evaluations'] for p in schedule) == 144192
    for phase in ('engineering', 'science'):
        initial = next(p for p in schedule if p['pass_id'] == phase + '-reference')
        replica = next(p for p in schedule if p['pass_id'] == phase + '-replica')
        assert initial['request_ids'] == replica['request_ids']
        assert all(i % 4 != (i + replica['options']['replica_shift']) % 4 for i in range(len(initial['request_ids'])))
    result = {'status': 'passed', 'source_files': len(manifest['source_files']), 'artifact_files': len(manifest['artifacts']),
              'exact_parent_contexts': len(contexts), 'candidate_boundaries': boundary_count,
              'position_roles': role_count, 'absent_query_roles': absent, 'pair_proofs': len(proofs),
              'audited_requests': len(requests), 'candidate_evaluations': 144192,
              'planned_four_gpu_rotation_checked': True, 'GPU_numerics_validated': False,
              'qwen_8b_weights_loaded': False, 'gpu_forward_executed': False}
    print(json.dumps(result, ensure_ascii=False))
    path = WORK / 'audits/independent-input-audit.json'
    with path.open('x') as f: json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')


if __name__ == '__main__': main()
