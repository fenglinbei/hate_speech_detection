"""Read-only verification of human adoption, immutable inputs and comparison scope."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import runpy
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PLANS = ROOT / 'docs/research/experiment-plans'
PROTOCOL = PLANS / 'task-applicability-scoring-v1/frozen-01'
OLD = PLANS / 'cross-term-joint-design-20260917-v1'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def verify(entry, relative_base=ROOT):
    path = relative_base / entry['path']
    require(digest(path.read_bytes()) == entry['sha256'], f'hash mismatch: {path}')
    if 'bytes' in entry:
        require(path.stat().st_size == entry['bytes'], f'byte size mismatch: {path}')


def check(tokenize=False):
    manifest = read(HERE / 'manifest.json')
    for e in manifest['artifacts'] + manifest['sources']:
        verify(e)
    old_manifest = read(OLD / 'draft-manifest.json')
    for e in old_manifest['artifacts'] + old_manifest['sources']:
        verify(e)
    protocol_manifest = read(PROTOCOL / 'manifest.json')
    for e in protocol_manifest['artifacts']:
        verify(e, PROTOCOL)
    for e in protocol_manifest['tokenizer_sources']:
        verify(e)
    for folder, script in [('cross-term-materials-v1', 'validate_materials.py'),
                           ('cross-term-demos-v1', 'validate_demo_sources.py')]:
        result = runpy.run_path(str(PLANS / folder / 'frozen-01' / script))['check']()
        require(result['status'] == 'pass', f'source validation failed: {folder}')

    feedback = read(HERE.parent / 'feedback-01.json')
    feedback_ref = {'path': str((HERE.parent / 'feedback-01.json').relative_to(ROOT)),
                    'sha256': digest((HERE.parent / 'feedback-01.json').read_bytes())}
    require(feedback['user_message_verbatim'] == '好的，接受这一重审建议，其他检查了暂时没有疑问，冻结后可进行下一步', 'user reply changed')
    changes = read(HERE / 'change-log.json')
    id_map = changes['relation_id_map']
    old_rows = {r['relation_id']: r for r in read(OLD / 'relations.json')['records']}
    records = read(HERE / 'relations.json')['records']
    decisions = {d['relation_id']: d for d in feedback['relation_decisions']}
    require(len(records) == len(old_rows) == len(decisions) == len(id_map) == 84, 'relation inventory changed')
    validator = Draft202012Validator(read(PROTOCOL / 'relation-record.schema.json'), format_checker=FormatChecker())
    validate = runpy.run_path(str(PROTOCOL / 'validate_contract.py'))['validate_relation']
    revised = {r['parent_relation_id']: r for r in feedback['revisions']}
    expected_changes = {('CTDD-G-B-01', 'CTD-G4-v2'): ('none', 'partial'),
                        ('CTDD-X-A-01', 'CTD-J2-v2'): ('partial', 'none'),
                        ('CTDD-X-A-02', 'CTD-J2-v2'): ('direct', 'none'),
                        ('CTDD-X-A-01', 'CTD-B2-v2'): ('partial', 'none')}
    seen_changes = {}
    for r in records:
        validate(r, validator)
        old = old_rows[r['supersedes']]
        d = decisions[r['relation_id']]
        require(r['relation_id'] == id_map[old['relation_id']], 'ID mapping differs')
        for key in ['source', 'target', 'source_quality', 'sense_fit', 'semantic_reference_fit',
                    'lexical_overlap', 'presentation_refs', 'family_id', 'relation_kind']:
            require(r[key] == old[key], f'unapproved change: {r["relation_id"]}/{key}')
        require(r['introduced_lexicon_relation_ids'] == [id_map[x] for x in old['introduced_lexicon_relation_ids']], 'L-D link mapping differs')
        if r['supersedes'] in revised:
            require(r['rule_fit'] == revised[r['supersedes']]['adopted_rule_fit'], 'accepted revision differs')
            require(old['rule_fit'] == revised[r['supersedes']]['old_rule_fit'], 'previous rule not preserved')
            pair = (r['source']['material_id'], r['target']['material_id'])
            seen_changes[pair] = (old['rule_fit']['value'], r['rule_fit']['value'])
        else:
            require(r['rule_fit'] == old['rule_fit'], 'additional rule change was not adopted')
        for pointer in d['accepted_fields']:
            require(r[pointer[1:]] == d['values'][pointer[1:]], 'feedback field mismatch')
        p = r['provenance']
        require(p['authorship'] == 'ai' and p['review_kind'] == 'human_with_ai' and p['adoption'] == 'bulk', 'review provenance differs')
        require(p['decision_ref'] == feedback_ref and p['accepted_fields'] == d['accepted_fields'], 'adoption binding differs')
        require(p['exposure'] == old['provenance']['exposure'], 'exposure reset')
    require(seen_changes == expected_changes, 'four accepted rule corrections not exact')
    counts = Counter(r['rule_fit']['value'] for r in records if r['rule_fit'])
    require(counts == {'direct': 17, 'partial': 17, 'none': 38}, 'rule counts differ')

    audit = read(HERE / 'rule-review-audit.json')
    require(audit['authorship'] == 'ai' and audit['adoption'] == 'none', 'new AI audit falsely human adopted')
    require({a['relation_id'] for a in audit['records']} == set(decisions), 'scope review coverage differs')
    require(sum(a['scope_observation'] is not None for a in audit['records']) == 11, 'scope observation count differs')
    require(all(not a['adopted_value_changed_by_audit'] and not a['new_human_decision'] for a in audit['records']), 'audit invents decisions')

    design = read(HERE / 'design.json')
    old_design = read(OLD / 'design.json')
    require(design['relations_human_adopted'] and design['scientific_input_design_frozen'], 'review/design state differs')
    require(not design['execution_matrix_frozen'] and not design['gpu_qualification'] and not design['model_forward_performed'], 'unperformed runtime claims')
    require(design['rule_fit_causal_stratification_registered'] is False, 'unregistered rule grouping')
    require(design['queries'] == old_design['queries'], 'query references changed')
    require(design['parent_condition_aliases'] == old_design['parent_condition_aliases'], 'aliases changed')
    require(len(design['conditions']) == 120 and len(design['comparisons']) == 252, 'input/comparison count differs')
    for c, old_c in zip(design['conditions'], old_design['conditions']):
        expected = copy.deepcopy(old_c)
        expected['status'] = 'input_design_frozen_runtime_qualification_pending'
        for field in ['relation_ids', 'candidate_lexicon_demo_relation_ids']:
            expected[field] = [id_map[x] for x in expected[field]]
        require(c == expected, 'condition contents changed')
    for c, old_c in zip(design['comparisons'], old_design['comparison_proposals']):
        expected = copy.deepcopy(old_c)
        expected['status'] = 'registered_development_input_contrast_not_confirmation_test'
        require(c == expected, 'comparison coefficients or orientation changed')

    require((HERE / 'prompt-previews.json').read_bytes() == (OLD / 'prompt-previews.json').read_bytes(), 'visible prompt file changed')
    previews = read(HERE / 'prompt-previews.json')['records']
    require(len(previews) == len({p['prompt_sha256'] for p in previews}) == 120, 'prompt dedup mismatch')
    qdata = read(PLANS / 'cross-term-materials-v1/frozen-01/materials.json')
    ddata = read(PLANS / 'cross-term-demos-v1/frozen-01/materials.json')
    queries = {r['query']['material_id']: r for r in qdata['queries']}
    demos = {r['material']['material_id']: r for r in ddata['demos']}
    lex = {r['material']['material_id']: r['material'] for r in qdata['lexicons'] + qdata['form_controls']}
    conditions = {c['condition_id']: c for c in design['conditions']}
    system = (PROTOCOL / 'model-task.txt').read_text()
    by_id = {p['condition_id']: p for p in previews}
    for p in previews:
        c = conditions[p['condition_id']]
        # Independent visible-message reconstruction: no relation or target-reference fields.
        lex_text = '\n\n'.join(f'词条{i}：\n{lex[mid]["raw_text"]}' for i, mid in enumerate(c['lexicon_slot_material_ids'], 1)) or '（无）'
        demo_text = '\n\n'.join(f'示例{i}：\n文本：{demos[mid]["material"]["raw_text"]}\n答案：{demos[mid]["human_review"]["task_label"]}' for i, mid in enumerate(c['demo_ids'], 1)) or '（无）'
        user = f'【参考词典】\n{lex_text}\n\n【参考示例】\n{demo_text}\n\n【待判断文本】\n{queries[c["query_id"]]["query"]["raw_text"]}'
        require(p['messages'] == [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], 'prompt contains changed/unallowed fields')
        require(digest(p['chat_prompt'].encode()) == p['prompt_sha256'] == c['prompt_sha256'], 'prompt hash differs')
        require(digest(json.dumps(p['input_ids'], separators=(',', ':')).encode()) == p['input_ids_sha256'], 'input IDs hash differs')
        require(len(p['input_ids']) == p['prompt_tokens'] == c['prompt_tokens'], 'token length differs')
        require(p['last_input_token_index'] == len(p['input_ids']) - 1 and p['next_token_position'] == len(p['input_ids']), 'answer position differs')
        require(p['candidate_tokens'] == {'有': 18830, '无': 42192}, 'candidate mapping differs')
    for c in design['comparisons']:
        terms = [conditions[t['condition_id']] for t in c['terms']]
        require(all(t['query_id'] == c['query_id'] for t in terms), 'cross-query comparison')
        require(sum(t['coefficient'] for t in c['terms']) == 0, 'contrast coefficients not centered')
        answer = queries[c['query_id']]['human_review']['task_label']
        require(c['correct_direction_sign'] == (1 if answer == '无' else -1), 'reference alignment differs')
        if c['scope'] == 'within_query_fixed_D':
            require(len(terms) == 2 and terms[0]['demo_ids'] == terms[1]['demo_ids'] and terms[0]['demo_answers'] == terms[1]['demo_answers'], 'fixed-D contrast changes D')
        if c['scope'] == 'fixed_pack_lexicon_interaction':
            require(len(terms) == 4, 'interaction arity differs')
            require(terms[0]['demo_ids'] == terms[1]['demo_ids'] and terms[2]['demo_ids'] == terms[3]['demo_ids'] == [], 'interaction pack/background differs')
    scope_counts = Counter(c['scope'] for c in design['comparisons'])
    require(scope_counts == {'within_query_fixed_D': 108, 'fixed_pack_lexicon_interaction': 72, 'secondary_D_bundle_diagnostic': 72}, 'comparison scopes differ')
    old_apps = copy.deepcopy(read(OLD / 'relation-appearances.json')['records'])
    for a in old_apps:
        a['relation_id'] = id_map.get(a['relation_id'], a['relation_id'])
    apps = read(HERE / 'relation-appearances.json')['records']
    require(apps == old_apps and len(apps) == 432, 'appearance mapping differs')
    edges = {r['relation_id']: r for r in records + read(PLANS / 'cross-term-materials-v1/frozen-01/relations.json')['records']}
    for a in apps:
        c = conditions[a['condition_id']]
        present = set(c['demo_ids'] + c['lexicon_slot_material_ids'] + [c['query_id']])
        r = edges[a['relation_id']]
        require(a['source_present'] == (r['source']['material_id'] in present) and a['target_present'] == (r['target']['material_id'] in present), 'material presence differs')
        require(a['prompt_sha256'] == by_id[a['condition_id']]['prompt_sha256'], 'presence prompt binding differs')

    token_checks = 0
    if tokenize:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['HF_HUB_OFFLINE'] = '1'
        from transformers import AutoTokenizer
        spec = read(PROTOCOL / 'scoring-spec.json')
        tokenizer = AutoTokenizer.from_pretrained(str(ROOT / spec['local_model_path']), local_files_only=True, trust_remote_code=False)
        for p in previews:
            prompt = tokenizer.apply_chat_template(p['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            require(prompt == p['chat_prompt'], 'chat template byte mismatch')
            require(tokenizer.encode(prompt, add_special_tokens=False) == p['input_ids'], 'tokenization differs')
            require(tokenizer.apply_chat_template(p['messages'], tokenize=True, add_generation_prompt=True, enable_thinking=False) == p['input_ids'], 'direct template tokenization differs')
            for label, tid in p['candidate_tokens'].items():
                require(tokenizer.encode(prompt + label, add_special_tokens=False) == p['input_ids'] + [tid], 'answer boundary unstable')
                token_checks += 1

    return {'status': 'pass', 'adopted_relations': 84, 'four_discussed_revisions': 4,
            'other_relation_values_preserved': 80, 'DQ_rule_counts': dict(counts),
            'AI_scope_observations_preserved_not_adjudicated': 11,
            'reused_adopted_LQ_edges': 12, 'identical_visible_prompts': 120,
            'presence_records': 432, 'comparison_scope_counts': dict(scope_counts),
            'tokenizer_rechecked': tokenize, 'candidate_boundary_checks_this_call': token_checks,
            'earlier_freezes_unchanged': True, 'model_forward_performed': False,
            'gpu_qualification': False, 'runtime_frozen': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenize', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check(args.tokenize), ensure_ascii=False, indent=2))
