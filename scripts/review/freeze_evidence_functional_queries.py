#!/usr/bin/env python3
"""Versioned bulk adoption and two-stage, four-replica execution freeze."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.general_model_evidence_evaluation import canonical, file_sha, json_bytes, jsonl, read_json, read_lines, require, sha, write_output
from diagnostics.evidence_label_calibration import validate_block
from scripts.review.run_evidence_interventions import DESCRIPTOR, SCHEDULE
from scripts.review.freeze_evidence_label_calibration import no_gold

BASE = ROOT / 'exps/causal_context/general_model_evidence_applicability_v1'
PUBLIC = BASE / 'functional-query-diagnostics-v1'
DRAFT = PUBLIC / 'draft-01'
PREVIEW = BASE / 'reviews/functional-query-diagnostics-v1/draft-01'
WORK = BASE / 'reviews/functional-query-diagnostics-v1/execution-01'
PARENT = BASE / 'reviews/analysis-freeze-20260914/lexicon-scope-v1/frozen-02'
PARENT_RUN = BASE / 'reviews/analysis-freeze-20260914/lexicon-scope-v1/run-02'
DRAFT_HASH = '90ec91cbb8e18cfc8f19383fd5a719c4cb8d2c348e35a622cdd7148f532b86ad'
AUTHORIZATION = 'AI标签以及建议都没什么问题，可以采纳，下面可以直接帮我实施这一实验，当前GPU四卡可用'
DEVICES = {'0': 'GPU-09b29c25-c372-62f4-3098-9734013e93c0',
           '1': 'GPU-9145ca41-4f0b-6f81-9a0b-118cb4596601',
           '2': 'GPU-278f5973-425f-514a-8a2d-db77f55d481b',
           '3': 'GPU-6e7d956e-0a22-7fac-a093-130cc1190f38'}
NEW_CODE = ('src/diagnostics/evidence_functional_queries.py',) + tuple('scripts/review/' + name + '.py' for name in (
    'freeze_evidence_functional_queries', 'run_evidence_functional_queries',
    'analyze_evidence_functional_queries', 'audit_evidence_functional_queries', 'test_evidence_functional_queries'))


def acceptance():
    require(file_sha(DRAFT / 'manifest.json') == DRAFT_HASH, 'approved draft manifest differs')
    source = read_json(DRAFT / 'materials-source.json')
    records = []
    for q in source['queries']:
        records.append({'query_id': q['query_id'], 'text_sha256': sha(q['text'].encode()),
            'text_adopted': True, 'hate': q['ai_label']['hate'],
            'attack_severity': q['ai_label']['attack_severity'],
            'query_fit': q['ai_applicability']['query_fit'],
            'sense_location': q['ai_applicability']['sense_location'],
            'source_kind': 'explicit_user_bulk_adoption_of_exact_ai_draft',
            'individual_question_answer_session': False, 'ai_reasoning_authorship_preserved': True,
            'ai_referent_caveat_retained': q['ai_applicability']['uncertainty'],
            'new_general_rule': False})
    return {'schema_version': 'functional-query-bulk-adoption/v1', 'status': 'accepted',
            'user_verbatim': AUTHORIZATION, 'confirmation_source': 'explicit_user_message_in_this_task',
            'reviewer': 'liaozijie', 'draft_manifest_sha256': DRAFT_HASH,
            'accepted_ai_labels_and_applicability_suggestions': True, 'records': records,
            'execution_authorized': True, 'device_indices': [0, 1, 2, 3],
            'supplement_kept_separate_from_primary': True, 'online_writeback': False,
            'mechanism_hypotheses_human_confirmed': False, 'old_draft_fields_rewritten': False,
            'scope_note': 'Bulk adoption covers the displayed text, label and applicability proposals; it is not a per-field interview or a new general semantic rule. Q07/Q08 remain synthetic implicit-reference items with their documented limitations.'}


def verify_manifest(directory, expected=None):
    if expected: require(file_sha(directory / 'manifest.json') == expected, 'manifest binding differs')
    m = read_json(directory / 'manifest.json')
    for rel, expected_hash in m['source_files'].items(): require(file_sha(ROOT / rel) == expected_hash, 'source changed: ' + rel)
    for name, expected_hash in m['artifacts'].items():
        p = ROOT / name if name.startswith('exps/') else directory / name
        require(file_sha(p) == expected_hash, 'artifact changed: ' + name)
    return m


def build(stage):
    require(stage in (1, 2), 'unknown stage')
    draft_manifest = verify_manifest(DRAFT, DRAFT_HASH)
    approval = read_json(PUBLIC / 'feedback-01.json')
    require(approval == acceptance(), 'bulk acceptance scope changed')
    parent_manifest = verify_manifest(PARENT, '819c29a9d1b732fdee2dbf4599c11f9bf32c20bf7485c83c49b3dbc9c475f949')
    parent = read_json(PARENT / 'plan.json')
    parent_state = read_json(PARENT_RUN / 'run_manifest.json')
    require(parent_state['status'] == 'complete' and parent_state['numerical_validation_passed'], 'parent not sealed')
    old_raw_path = PARENT_RUN / parent['raw_pass'] / 'scores.jsonl'
    require(file_sha(old_raw_path) == parent_state['raw_scores_sha256'], 'parent raw score seal differs')
    old_raw = {r['record_id']: r for r in read_lines(old_raw_path)}
    draft_matrix = read_lines(DRAFT / 'input-matrix.jsonl')
    bindings = read_lines(PREVIEW / 'prompt-bindings.jsonl')
    previews = read_lines(PREVIEW / 'prompt-previews.jsonl')
    by_condition = {r['condition_id']: r for r in draft_matrix}
    if stage == 1:
        matrix = [r for r in draft_matrix if r['stage'] == 'stage_1']
    else:
        matrix = [r for r in draft_matrix if r['stage'] == 'stage_2_supplement'
                  or r['role'] == 'historical_anchor'
                  or (r['demo_family'] == 1 and r['lexicon_arm'] == 'O')]
        matrix = [dict(r, execution_role='stage_bridge_replay' if r['stage'] == 'stage_1' and r['role'] != 'historical_anchor' else r['role']) for r in matrix]
    condition_ids = {r['condition_id'] for r in matrix}
    wanted_pids = {b['prompt_id'] for b in bindings if b['condition_id'] in condition_ids}
    baseline_pids = {b['prompt_id'] for b in bindings if by_condition[b['condition_id']]['role'] == 'historical_anchor'}
    aliases, sources_by_pid = defaultdict(list), defaultdict(set)
    for b in bindings:
        r = by_condition[b['condition_id']]
        aliases[b['prompt_id']].append({k: r[k] for k in ('condition_id', 'query_id', 'demo_family', 'lexicon_arm', 'demo_surface')})
        sources_by_pid[b['prompt_id']].add(b['source_record_id'])
    contexts, historical = [], []
    for p in previews:
        if p['prompt_id'] not in wanted_pids: continue
        source_ids = sorted(sources_by_pid[p['prompt_id']])
        require(len(source_ids) == 1, 'a shared prompt has conflicting source identities')
        real_qids = {a['query_id'] for a in aliases[p['prompt_id']]}
        require(p['probe_id'] is not None or len(real_qids) == 1, 'real query unexpectedly deduplicated')
        qid = 'shared-background' if p['probe_id'] is not None else next(iter(real_qids))
        c = {k: deepcopy(p[k]) for k in ('encoding', 'task', 'probe_id', 'messages', 'prompt_text', 'prompt_sha256', 'prompt_token_ids', 'prompt_tokens')}
        c.update(record_id=p['prompt_id'], query_id=qid, condition=p['prompt_id'],
                 semantic_task='hate', root_condition=p['prompt_id'],
                 baseline_replay=p['prompt_id'] in baseline_pids,
                 source_record_id=source_ids[0], bindings=aliases[p['prompt_id']],
                 prompt_token_ids_sha256=sha(canonical(p['prompt_token_ids']).encode()))
        c['context_sha256'] = sha(canonical(c).encode())
        no_gold(c); contexts.append(c)
        if c['baseline_replay']:
            old = deepcopy(old_raw[source_ids[0]])
            old.update({k: c[k] for k in DESCRIPTOR})
            old['source_record_id'] = source_ids[0]
            validate_block(old, c, parent['catalog']); historical.append(old)
    require(len(contexts) == (476 if stage == 1 else 312) and len(historical) == 32, 'stage input counts differ')
    comparisons = [c for c in read_lines(DRAFT / 'comparisons.jsonl')
                   if {t['condition_id'] for t in c['terms']} <= condition_ids]
    require(len({c['record_id'] for c in contexts}) == len(contexts), 'duplicate scoring id')
    refs = [{'query_id': r['query_id'], 'task': 'hate', 'original_label': None,
             'original_reference_status': 'not_applicable_new_query', 'adjudicated_label': r['hate'],
             'query_fit': r['query_fit'], 'reference_resolution': 'resolved',
             'applicability_resolution': 'resolved_by_bulk_adoption',
             'adoption_source': 'feedback-01.json', 'text_sha256': r['text_sha256'],
             'ai_referent_caveat_retained': r['ai_referent_caveat_retained'],
             'individual_question_answer_session': False} for r in approval['records']]
    old_ref = next(r for r in read_json(PARENT / 'analysis_references.json') if r['query_id'] == '3169' and r['task'] == 'hate')
    refs.append(old_ref)

    sources = {**parent_manifest['source_files'], **draft_manifest['source_files'], **draft_manifest['artifacts']}
    for name, h in parent_manifest['artifacts'].items(): sources[str((PARENT / name).relative_to(ROOT))] = h
    for p in (DRAFT / 'manifest.json', PARENT / 'manifest.json', PUBLIC / 'feedback-01.json',
              PUBLIC / 'EXECUTION-01.md', PARENT_RUN / 'run_manifest.json', old_raw_path):
        sources[str(p.relative_to(ROOT))] = file_sha(p)
    code = dict(parent['code_sha256'])
    for name in NEW_CODE: sources[name] = code[name] = file_sha(ROOT / name)
    config = deepcopy(parent['config']); config['execution']['device_indices'] = [0, 1, 2, 3]
    descriptors = [{k: c[k] for k in DESCRIPTOR} for c in contexts]
    schedule = [{'pass': n.replace('interventions', 'functional'), 'selection': selection,
                 'options': options, 'check': check} for n, selection, options, check in SCHEDULE]
    counts = {'unique_prompts': len(contexts), 'candidates': len(contexts) * 2,
              'logical_conditions': len(matrix), 'historical_prompts': 32,
              'new_query_conditions': 128 if stage == 1 else 64,
              'stage_bridge_new_query_conditions': 0 if stage == 1 else 16,
              'historical_conditions': 4, 'comparisons': len(comparisons),
              'candidate_evaluations': 128 + len(contexts) * 2 * 6}
    protocol = {'status': 'frozen', 'stage': stage, 'user_verbatim': AUTHORIZATION,
                'draft_interpretation_sha256': file_sha(DRAFT / 'INTERPRETATION-PLAN.md'),
                'supplement_is_separate': True, 'background_scores_recomputed_in_each_stage': True,
                'shared_prompt_aliases_are_not_replicates': True,
                'source_referent_caveats_retained': True, 'raw_score_direction': 'non-hate-minus-hate',
                'no_original_dataset_gold_for_new_queries': True, 'mechanism_ready': False,
                'stage_2_bridge': 'Re-score all F1/O new-query prompts plus four historical O anchors; 80 exact shared inputs compared with sealed stage 1 at unchanged epsilon.',
                'four_replica_scope': 'Four physical replicas; rotation by one gives a different producer for every candidate, not scoring every candidate on all four cards.'}
    plan = {'schema_version': 'evidence-functional-query-plan/v1', 'status': 'frozen', 'stage': stage,
            'parent_plan_id': parent['plan_id'], 'draft_manifest_sha256': DRAFT_HASH,
            'source_files': sources, 'code_sha256': code, 'blocks': descriptors,
            'catalog': parent['catalog'], 'eos_token_id': parent['eos_token_id'], 'pad_token_id': parent['pad_token_id'],
            'generation_runtime_identity': parent['generation_runtime_identity'],
            'config': config, 'numeric_policy': parent['numeric_policy'],
            'runtime_parent_plan': {'blocks': descriptors, 'catalog': parent['catalog'],
                'package_path': parent['runtime_parent_plan']['package_path'],
                'generation_runtime_identity': parent['generation_runtime_identity']},
            'execution_amendment': {'device_indices': [0, 1, 2, 3], 'selected_gpu_uuids': DEVICES},
            'schedule': schedule, 'raw_pass': 'functional-b1-r0', 'counts': counts,
            'query_gold_loaded_during_scoring': False, 'formal_test_or_reserve_access': False,
            'analysis_after_all_gates_and_raw_seal': True, 'epsilon_recalibration': False,
            'automatic_profile_search': False, 'mechanism_ready': False,
            'contexts_sha256': sha(jsonl(contexts)), 'comparisons_sha256': sha(jsonl(comparisons)),
            'historical_selected_sha256': sha(jsonl(historical)), 'references_sha256': sha(json_bytes(refs)),
            'protocol_sha256': sha(json_bytes(protocol))}
    plan['plan_id'] = 'functional-query-' + sha(canonical(plan).encode())
    files = {'plan.json': json_bytes(plan), 'contexts.jsonl': jsonl(contexts),
             'historical-selected.jsonl': jsonl(historical), 'comparisons.jsonl': jsonl(comparisons),
             'input-matrix.jsonl': jsonl(matrix), 'analysis_references.json': json_bytes(refs),
             'analysis_protocol.json': json_bytes(protocol), 'feedback.json': json_bytes(approval),
             'execution-source.json': json_bytes({name: {'sha256': h, 'text': (ROOT / name).read_text()} for name, h in code.items()}),
             'audit.json': json_bytes({'status': 'passed', **counts, 'gpu_forward_executed': False})}
    return files, sources


def load_frozen(directory):
    directory = Path(directory).resolve(); manifest = verify_manifest(directory)
    require(manifest['schema_version'] == 'evidence-functional-query-freeze/v1' and manifest['status'] == 'frozen', 'wrong freeze')
    require({p.name for p in directory.iterdir()} == {*manifest['artifacts'], 'manifest.json'}, 'freeze inventory changed')
    plan = read_json(directory / 'plan.json'); identity = dict(plan); pid = identity.pop('plan_id')
    require(pid == 'functional-query-' + sha(canonical(identity).encode()), 'plan identity differs')
    require(plan['source_files'] == manifest['source_files'], 'source closure differs')
    contexts = read_lines(directory / 'contexts.jsonl'); history = read_lines(directory / 'historical-selected.jsonl')
    for name, field in [('contexts.jsonl', 'contexts_sha256'), ('historical-selected.jsonl', 'historical_selected_sha256'),
                        ('comparisons.jsonl', 'comparisons_sha256'), ('analysis_references.json', 'references_sha256'),
                        ('analysis_protocol.json', 'protocol_sha256')]:
        require(file_sha(directory / name) == plan[field], 'content binding differs: ' + name)
    require([{k: c[k] for k in DESCRIPTOR} for c in contexts] == plan['blocks'], 'scoring descriptors differ')
    require(len(contexts) == plan['counts']['unique_prompts'] and len(history) == 32, 'coverage differs')
    for c in contexts:
        no_gold(c)
        require(c['context_sha256'] == sha(canonical({k: v for k, v in c.items() if k != 'context_sha256'}).encode()), 'context hash differs')
        require(sha(c['prompt_text'].encode()) == c['prompt_sha256']
                and sha(canonical(c['prompt_token_ids']).encode()) == c['prompt_token_ids_sha256']
                and len(c['prompt_token_ids']) == c['prompt_tokens'], 'prompt binding differs')
    for c, r in zip([c for c in contexts if c['baseline_replay']], history, strict=True): validate_block(r, c, plan['catalog'])
    return plan, contexts, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('accept', 'freeze', 'check'))
    parser.add_argument('--stage', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.command == 'accept':
        target = PUBLIC / 'feedback-01.json'; require(not target.exists(), 'acceptance already recorded')
        with target.open('xb') as f: f.write(json_bytes(acceptance()))
        print(canonical({'status': 'bulk_adoption_recorded', 'labels': 8, 'applicability': 8, 'old_draft_unchanged': True})); return
    target = WORK / f'frozen-stage-{args.stage}-01'
    files, sources = build(args.stage)
    if args.command == 'check':
        manifest = read_json(target / 'manifest.json')
        require(manifest['source_files'] == sources and set(manifest['artifacts']) == set(files), 'rebuild inventory differs')
        for name, data in files.items(): require((target / name).read_bytes() == data, 'rebuild differs: ' + name)
        load_frozen(target)
    else:
        files['manifest.json'] = json_bytes({'schema_version': 'evidence-functional-query-freeze/v1', 'status': 'frozen',
            'created_at': datetime.now(timezone.utc).isoformat(), 'source_files': sources,
            'artifacts': {name: sha(data) for name, data in files.items()}})
        write_output(target, files)
    print(canonical({'status': args.command, 'stage': args.stage, 'counts': read_json(target / 'plan.json')['counts']}))


if __name__ == '__main__': main()
