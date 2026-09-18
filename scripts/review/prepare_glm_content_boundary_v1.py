#!/usr/bin/env python3
"""Separately freeze GLM's native assistant-header/content separator correction.

No task text, candidate, arithmetic, tolerance, model code or existing freeze is
edited. The old generation prefix ended inside the assistant header. Append the
native empty-metadata newline so the next token is the answer content token.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics import cross_model_applicability_execution_v1 as c

OUTPUT = c.WORK / 'prepared-glm-content-boundary-01'
KEY = 'glm4-9b-chat'


def derive():
    _, profiles, inputs = c.check_prepared()
    profile = next(p for p in profiles if p['model_key'] == KEY)
    directory = Path(profile['local_directory'])
    spec = importlib.util.spec_from_file_location('glm_boundary_tokenizer', directory / 'tokenization_chatglm.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tok = module.ChatGLM4Tokenizer.from_pretrained(str(directory), local_files_only=True)
    assert tok.build_single_message('assistant', '', '', tokenize=True) == [151337, 198]
    assert tok.build_single_message('assistant', '', '', tokenize=False) == '<|assistant|>\n'
    native, messages = [], {}
    for frame in ['new-model-inputs', 'legacy-model-inputs']:
        native.extend(c.jsonl(c.SCIENCE / f'tokenized/{KEY}-{frame}.jsonl'))
        messages.update({r['condition_id']: r['messages'] for r in c.jsonl(c.SCIENCE / f'{frame}.jsonl')})
    amended, proofs = [], []
    for old, original in zip(inputs[KEY], native, strict=True):
        cid = old['condition_id']
        assert cid == original['condition_id']
        assert old['input_ids'] == original['input_ids'] and old['input_ids'][-1] == 151337
        assert original['chat_prompt'].endswith('<|assistant|>')
        assert original['chat_prompt'] == tok.apply_chat_template(messages[cid], tokenize=False, add_generation_prompt=True)
        prompt = original['chat_prompt'] + '\n'
        ids = old['input_ids'] + [198]
        assert tok.encode(prompt, add_special_tokens=False) == ids
        for label, token in profile['candidate_tokens'].items():
            complete = messages[cid] + [{'role': 'assistant', 'metadata': '', 'content': label}]
            assert tok.apply_chat_template(complete, tokenize=False, add_generation_prompt=False) == prompt + label
            assert tok.apply_chat_template(complete, tokenize=True, add_generation_prompt=False) == ids + [token]
        row = dict(old, input_ids=ids, prompt_tokens=len(ids),
                   input_ids_sha256=hashlib.sha256(c.canonical(ids)).hexdigest(),
                   prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        amended.append(row)
        proofs.append({'condition_id': cid, 'messages_sha256': original['messages_sha256'],
                       'original_generation_prompt_sha256': old['prompt_sha256'],
                       'amended_content_prompt_sha256': row['prompt_sha256'],
                       'original_input_ids_sha256': old['input_ids_sha256'],
                       'amended_input_ids_sha256': row['input_ids_sha256'],
                       'original_tokens': old['prompt_tokens'], 'amended_tokens': len(ids),
                       'appended_header_token': 198, 'two_native_complete_answer_reconstructions': True})
    assert len(amended) == len(proofs) == 540
    for qid in {r['query_id'] for r in amended if r['query_id']}:
        longest = max([r for r in amended if r['query_id'] == qid], key=lambda r: (r['prompt_tokens'], r['request_id']))
        assert longest['request_id'] in profile['format_probe_ids']
    return profile, amended, proofs


def check(output=OUTPUT):
    output = Path(output)
    plan, profiles, inputs = c.check_prepared(output)
    profile, expected, proofs = derive()
    assert profiles == [profile] and inputs == {KEY: expected}
    assert c.jsonl(output / 'native-boundary-proofs.jsonl') == proofs
    assert (output / 'legacy-analysis.json').read_bytes() == (c.PREPARED / 'legacy-analysis.json').read_bytes()
    original_plan = c.read(c.PREPARED / 'execution-plan.json')
    for key in ['acceptance', 'format', 'runtime', 'runtime_versions', 'scientific_manifest', 'reference_join']:
        assert plan[key] == original_plan[key]
    assert plan['model_keys'] == [KEY]
    assert profile['production_ids'] == next(p for p in c.read(c.PREPARED / 'model-profiles.json')['models'] if p['model_key'] == KEY)['production_ids']
    assert c.read(output / 'amendment.json')['added_tokens_per_prompt'] == [198]
    return {'status': 'pass', 'model_key': KEY, 'amended_prompts': 540,
            'native_complete_answer_reconstructions': 1080, 'task_message_changes': 0,
            'candidate_or_score_rule_changes': 0, 'native_header_tokens_appended_per_prompt': 1,
            'format_probe_selection_unchanged': True, 'original_runs_and_freezes_unchanged': True,
            'extra_GPU_forwards': 0, 'GPU_qualification': False}


def prepare(output=OUTPUT):
    output = Path(output)
    assert not output.exists(), 'New immutable preparation directory required'
    profile, inputs, proofs = derive()
    old_run = c.WORK / 'run-glm4-9b-chat-01'
    state = c.read(old_run / 'state.json')
    assert state['status'] == 'paused' and state['worker_exit_code'] == 0 and state['owned_worker_absent']
    assert not (old_run / 'scores/production').exists()
    output.mkdir(); (output / 'inputs').mkdir()
    c.write_jsonl(output / 'inputs' / f'{KEY}.jsonl', inputs)
    c.write_jsonl(output / 'native-boundary-proofs.jsonl', proofs)
    c.dump(output / 'model-profiles.json', {'models': [profile]})
    (output / 'legacy-analysis.json').write_bytes((c.PREPARED / 'legacy-analysis.json').read_bytes())
    amendment = {'schema_version': 'glm-content-boundary-amendment/v1', 'recorded_at': c.now(),
        'parent_preparation': c.info(c.PREPARED / 'manifest.json'),
        'scientific_text_and_design_freeze': c.info(c.SCIENCE / 'manifest.json'),
        'reason': 'The native generation prefix ends at assistant151337, before empty metadata newline198. All 16 preselected first tokens were newline198. Candidate scoring must occur at the native answer-content boundary.',
        'old_prefix_suffix': '<|assistant|>', 'new_prefix_suffix': '<|assistant|>\n',
        'added_tokens_per_prompt': [198], 'all_540_GLM_inputs_receive_identical_boundary_rule': True,
        'task_messages_materials_labels_relations_candidates_and_scoring_math_changed': False,
        'existing_Qwen_inputs_runs_and_qualifications_changed': False,
        'existing_frozen_GLM_token_records_preserved_as_original_generation_prefixes': True,
        'new_GPU_qualification_required': True, 'old_GLM_scores_eligible_for_science': False,
        'post_engineering_format_exposure': True, 'query_reference_join_performed': False,
        'old_run': str(old_run), 'old_release': state['resource_release'],
        'diagnostic_sources': [c.info(c.WORK / 'audits' / name) for name in
            ['format-first-token-01.json', 'glm-format-prefix-inspection-01.json', 'glm-header-stop-01.json']],
        'native_grammar_sources': [c.info(Path(profile['local_directory']) / name) for name in
                                  ['tokenization_chatglm.py', 'tokenizer_config.json']]}
    c.dump(output / 'amendment.json', amendment)
    plan = copy.deepcopy(c.read(c.PREPARED / 'execution-plan.json'))
    budget = next(r for r in plan['budget']['by_model'] if r['model_key'] == KEY)
    plan.update(schema_version='cross-model-applicability-execution/glm-content-boundary-v1', model_keys=[KEY],
                boundary_amendment=c.info(output / 'amendment.json'))
    plan['budget'] = {'by_model': [budget], 'engineering_forwards': 2700, 'new_scientific_forwards': 540,
                      'prompt_forward_total_without_generation': 3240, 'format_extra_forward_cap': 112,
                      'reused_old_8b_scientific_inputs': 0,
                      'old_GLM_diagnostic_forwards_not_reused': state['new_forwards']}
    c.dump(output / 'execution-plan.json', plan)
    ledger = copy.deepcopy(c.read(c.PREPARED / 'source-ledger.json'))
    ledger['files'] += [c.info(c.PREPARED / 'manifest.json'), c.info(Path(__file__).resolve())]
    ledger['new_target_model_outputs_read'] = True
    ledger['exposure_scope'] = 'Predetermined engineering format probe first tokens only; no task reference or scientific interpretation.'
    c.dump(output / 'source-ledger.json', ledger)
    c.dump(output / 'manifest.json', {'schema_version': 'glm-content-boundary-preparation/v1',
        'status': 'CPU_frozen_new_GPU_qualification_required', 'created_at': c.now(),
        'artifacts': [c.info(p) for p in sorted(output.rglob('*')) if p.is_file()],
        'sources': [c.info(c.PREPARED / 'manifest.json'), c.info(Path(__file__).resolve())]})
    return check(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = prepare(args.output) if args.action == 'prepare' else check(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
