#!/usr/bin/env python3
"""Implement the authorized case-use workflow and prepare explicit input records.

Human records and previous freezes are read-only. This command never loads model
weights; optional local tokenization is a CPU input check, not a scoring run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.general_model_evidence_evaluation import (
    EXPERIMENT, TASKS, CORE, Sources, canonical, csv_bytes, file_sha, json_bytes,
    jsonl, local_path, read_json, read_lines, require, sha, unique, write_output,
)
from diagnostics.general_model_contexts import _render_lexicon, _render_demo_blocks
from scripts.review.update_evidence_case_uses import accepted_field

BASE = ROOT / EXPERIMENT
PRIORITY = ('3169', '5086', '541', '6037', '1128', '4026')
AUTHORIZATION = {
    'user_message': '好的，可以按这四步开始实施',
    'scope': ['task_specific_facts', 'case_use_recommendations', 'material_linkage', 'concrete_intervention_work_records'],
    'individual_annotation_or_mechanism_adoption': False,
}
TOKENIZER_FILES = {
    'config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'vocab.txt',
    'merges.txt', 'special_tokens_map.json', 'added_tokens.json', 'chat_template.jinja',
    'tokenizer.model', 'sentencepiece.bpe.model', 'spiece.model',
}


def source_partition(graph, selected):
    require(len(set(selected)) == len(selected) and set(selected) <= set(graph['demo_order']), 'invalid selected demo set')
    retained = [d for d in graph['demo_order'] if d not in selected]
    lr = set(graph['query_entry_ids'])
    for d in retained:
        lr.update(graph['demo_entry_ids'][d])
    from_selected = {entry for d in selected for entry in graph['demo_entry_ids'][d]}
    lu = from_selected - lr
    require(lr | lu == set(graph['entry_order']) and not lr & lu, 'source partition does not reconstruct the original union')
    return {'selected_demo_ids': [d for d in graph['demo_order'] if d in selected], 'retained_demo_ids': retained,
            'L_R': sorted(lr), 'L_U': sorted(lu), 'shared_selected_entry_ids': sorted(from_selected & lr),
            'dictionary_factor_effective': bool(lu)}


def surviving_entries(graph, excluded_hit_ids):
    edges = graph['hit_edges']
    require(set(excluded_hit_ids) <= {r['record_id'] for r in edges}, 'unknown source edge')
    return sorted({r['entry_id'] for r in edges if r['record_id'] not in excluded_hit_ids})


def resource_graph(q, case, materials):
    ctx = next(c for c in case['comparison']['contexts'] if c['task'] == 'hate' and c['condition'] == 'CLDnewNoCat')
    demos, entries = ctx['trace']['source_demo_ids'], ctx['trace']['source_lexicon_ids']
    graph = {'query_id': q, 'demo_order': demos, 'entry_order': entries,
             'query_entry_ids': [], 'demo_entry_ids': {d: [] for d in demos}, 'hit_edges': []}
    for oid in case['object_ids']:
        r = materials[oid]
        if r['kind'] != 'hit':
            continue
        s = r['source']
        start, end = s['raw_span']
        require(s['query_id'] == q and s['text'][start:end] == s['raw_surface'], 'source span mismatch')
        require(s['entry_id'] in entries and s['source_kind'] in ('query', 'demo'), 'unexpected source edge')
        if s['source_kind'] == 'query':
            require(s['source_id'] == q, 'query source ID mismatch')
            target = graph['query_entry_ids']
        else:
            require(s['source_id'] in demos, 'demo source ID mismatch')
            target = graph['demo_entry_ids'][s['source_id']]
        target.append(s['entry_id'])
        graph['hit_edges'].append({k: s[k] for k in ('entry_id', 'sense_id', 'source_kind', 'source_id', 'raw_span', 'raw_surface')} |
                                 {'record_id': oid, 'source_fit': accepted_field(r, 'source_fit'), 'query_fit': accepted_field(r, 'query_fit')})
    graph['query_entry_ids'] = sorted(set(graph['query_entry_ids']))
    graph['demo_entry_ids'] = {d: sorted(set(v)) for d, v in graph['demo_entry_ids'].items()}
    require({r['entry_id'] for r in graph['hit_edges']} == set(entries), 'source edges do not cover frozen dictionary')
    for d in demos:
        rel = materials[f'relation:{q}:{d}']
        require(sorted(rel['source']['introduced_entry_ids']) == graph['demo_entry_ids'][d], 'relation and occurrence sources disagree')
    graph['shared_entry_ids'] = [e for e in entries if len({(r['source_kind'],r['source_id']) for r in graph['hit_edges'] if r['entry_id']==e}) > 1]
    return graph


def protocol_for(row, screen, graph, materials, baseline):
    q, task = row['query_id'], row['task']
    demos, entries = graph['demo_order'], graph['entry_order']
    base_arm = {'condition': 'original', 'demo_ids': demos, 'entry_ids': entries, 'answer_overrides': {}}
    arms = [base_arm]
    p = {'protocol_id': f'EA-{q}-{task}-v1', 'query_id':q, 'task':task,
         'spec_author':'assistant_under_authorized_workflow', 'spec_status':'prepared',
         'human_mechanism_judgment':False, 'source_condition':'CLDnewNoCat',
         'reference_label': row['reviewed_label'], 'original_reference_label':row['original_label'],
         'kept_fixed':['query', 'system_instruction', 'retained_demo_text_and_relative_order',
                       'all_unspecified_demo_answers', 'all_unspecified_entries_and_definitions', 'explicit_category_fields_absent'],
         'position_control':'natural_edit_only', 'mechanism_ready':False,
         'runtime_pending':['current_model_numeric_replay', 'all_arms_scored_in_one_new_run'],
         'mechanism_pending':['matched_position_controls', 'activation_collection_and_writeback_checks'],
         'interpretation_limits':['no_claim_of_pure_semantic_effect', 'no_internal_mechanism_claim'],
         'group_foil':None, 'candidate_decision':'full_canonical_space_answer_sum',
         'analysis_note':'参考标签只进入分析记录；评分上下文不附带查询答案。'}
    dscreen = {d['demo_id']: d for d in screen['demos']}
    if task == 'group':
        from diagnostics.general_model_numeric_analysis import candidate_catalog, _gold_ordinal
        gold = _gold_ordinal(task, row['reviewed_label'])
        scores = baseline['candidate_scores']
        require(scores is not None and len(scores)==32, 'fixed group foil requires the sealed 32 scores')
        foil = min((i for i in range(32) if i!=gold), key=lambda i: (-scores[i], i))
        p['group_foil'] = {'labels':candidate_catalog()['group'][foil]['labels'], 'ordinal':foil,
                           'selection':'best_non_reviewed_candidate_in_frozen_SD; canonical_tie_order',
                           'source_prompt_sha256':baseline['prompt_sha256'], 'fixed_across_new_arms':True}
    if q == '5086' and task == 'hate':
        chosen = '7248'
        part = source_partition(graph, [chosen])
        require(part['L_U']==['lex-0073'] and dscreen[chosen]['original_answer_matches_reviewed'] is True,
                '5086 targeted source contract changed')
        require(dscreen[chosen]['topic']['value']=='direct' and dscreen[chosen]['rule']['value']=='partial', '5086 relation contract changed')
        arms = [{'condition':f'T{a}_L{b}', 'demo_ids':demos if a else part['retained_demo_ids'],
                 'entry_ids':entries if b else part['L_R'], 'answer_overrides':{}} for a,b in ((0,0),(1,0),(0,1),(1,1))]
        p.update(operation='related_demo_source_factor', selected_demo_ids=[chosen], partition=part,
                 baseline_arm='T1_L1', strict_U_branch=False,
                 hypothesis='关联示例 7248 与其独有 txl 词条的增量作用可能不同，分别检验并报告双向或无效结果。',
                 alternatives=['demo_label_count_change', 'length_and_position', 'shared_topic_and_rule_information'],
                 material_basis=[f'demo:{chosen}',f'relation:{q}:{chosen}','definition:lex-0073:sense:01'])
    elif (q, task) in (('5086','group'), ('541','group')):
        chosen = '127' if q=='5086' else '826'
        d = dscreen[chosen]
        require(d['original_answer_matches_reviewed'] is False and d['reviewed_answer']['available'], 'answer intervention is not a resolved difference')
        revised = d['reviewed_answer']['value']
        arms.append({'condition':'answer_reviewed', 'demo_ids':demos, 'entry_ids':entries,
                     'answer_overrides':{chosen:revised}})
        p.update(operation='single_demo_answer_reference_change', selected_demo_ids=[chosen],
                 baseline_arm='original', strict_U_branch=False,
                 hypothesis='仅替换指定示例的任务答案，测量其对查询候选分数和集合预测的总效应；不预设改善。',
                 alternatives=['label_prior_change', 'answer_length_and_position', 'policy_definition_difference'],
                 before_after={'before':d['original_answer'],'after':revised},
                 material_basis=[f'demo:{chosen}',f'relation:{q}:{chosen}'])
    elif q == '6037':
        hit_id = 'hit:6037:demo:5230:lex-0069:sense:01:32:34'
        hit = next(h for h in graph['hit_edges'] if h['record_id']==hit_id)
        require(hit['source_fit']['value']=='substring_mismatch' and hit['query_fit']['value']=='inapplicable', 'edge adjudication changed')
        remaining = surviving_entries(graph,[hit_id])
        require(set(entries)-set(remaining)=={'lex-0069'}, '6037 source sharing changed')
        arms.append({'condition':'source_edge_masked', 'demo_ids':demos, 'entry_ids':remaining, 'answer_overrides':{}, 'masked_hit_ids':[hit_id]})
        p.update(operation='single_source_edge_mask', selected_demo_ids=[], baseline_arm='original', strict_U_branch=False,
                 hypothesis='屏蔽已确认的法国女性→国女误命中边，检验该词条在原示例包背景下的增量作用。',
                 alternatives=['length_and_position', 'definition_information_amount', 'unchanged_demo_answer_background'],
                 material_basis=[hit_id,'definition:lex-0069:sense:01','demo:5230'],
                 unchanged_background_answer_difference=dscreen['5230']['original_answer_matches_reviewed'] is False)
    else:
        chosen = {'3169':'3660', '541':'826', '1128':'387', '4026':'6334'}[q]
        require(dscreen[chosen]['original_answer_matches_reviewed'] is True, 'demo removal reference is unresolved or differs')
        retained = [d for d in demos if d!=chosen]
        arms.append({'condition':'demo_removed_dictionary_fixed','demo_ids':retained,'entry_ids':entries,'answer_overrides':{}})
        p.update(operation='single_demo_removal_dictionary_fixed', selected_demo_ids=[chosen], baseline_arm='original',
                 strict_U_branch=False, partition_for_audit_only=source_partition(graph,[chosen]),
                 alternatives=['length_and_position', 'demo_label_count_change', 'demo_numbering_change'],
                 material_basis=[f'demo:{chosen}', f'relation:{q}:{chosen}'])
        if q == '3169':
            require('lex-0419' in entries and 'lex-0419' in graph['query_entry_ids'], 'shared query source missing')
            p.update(hypothesis='在词典内容完全固定时移除同形异义示例 3660，检验该示例包对错误方向分数的增量作用。',
                     kept_shared_entry_ids=['lex-0419'],
                     definition_branch_pending='lex-0419 原定义被判为 reasonable；adopted_definition=null，尚无已采纳改写。')
            p['material_basis'].append('definition:lex-0419:sense:01')
        elif q == '541':
            p['hypothesis']='固定词典后移除已确认规则直接相关的示例 826，检验其对 non-hate 支持分数的贡献。'
            p['separate_branch_pending']='显式类别字段及其位置替换对照另设条件，不与本次示例移除合并。'
        else:
            require(materials['demo:'+chosen]['source']['original_answer']['hate']=='hate', 'control no longer matches removed hate answer')
            p.update(hypothesis='对稳定正确案例施加相同的一条示例删除操作，记录一般扰动及方向变化。',
                     control_for=[f'EA-3169-{task}-v1'],
                     control_matching={'same_operation':True,'same_removed_demo_count':True,'same_removed_hate_answer':True,
                                       'same_group_answer':dscreen[chosen]['original_answer']==['Racism','Sexism'] if task=='group' else None,
                                       'token_length_matched':False,'semantic_material_matched':False},
                     control_scope='partial_operation_control; not_a_position_or_semantic_matched_control')
    p['arms'] = arms
    p['material_basis'] = [{'record_id':oid,'frozen_record_sha256':sha(canonical(materials[oid]).encode())} for oid in p['material_basis']]
    p['hypothesis_result'] = 'not_tested'
    p['no_effect_policy'] = 'retain_no_effect_and_reverse_effects; do_not_change_selected_materials_after_scores'
    return p


def messages_for(source, query, entries, materials, arm, task):
    dictionary, _ = _render_lexicon([entries[e] for e in arm['entry_ids']], 'Definition')
    demos = [{'id':d,'content':materials['demo:'+d]['source']['text']} for d in arm['demo_ids']]
    outputs = [arm['answer_overrides'].get(d,materials['demo:'+d]['source']['original_answer'][task]) for d in arm['demo_ids']]
    examples = _render_demo_blocks(task,demos,outputs,'Full')
    # Match the existing renderer's JSON wire exactly.
    import json
    tail = '待判断文本（JSON 字符串）：\n' + json.dumps(query,ensure_ascii=False,separators=(',',':'))
    user = '\n\n'.join([v for v in (dictionary,examples) if v] + [tail])
    messages = deepcopy(source['messages'])
    require(len(messages)==2 and [m['role'] for m in messages]==['system','user'], 'unsupported frozen message shape')
    messages[1]['content']=user
    original = source['messages'][1]['content']
    require(source['prompt_text'].count(original)==1, 'ambiguous frozen user-message slot')
    prefix,suffix=source['prompt_text'].split(original)
    return messages,prefix+user+suffix


def build(tokenizer_path=None):
    src=Sources(ROOT)
    def take(path, expected=None):return src.verify(path,expected or file_sha(path))
    cp=read_json(take(BASE/'case-uses-v1/current.json'))
    previous=local_path(ROOT,str(EXPERIMENT/cp['proposal_path']))
    cm=read_json(take(previous/'manifest.json',cp['manifest_sha256']))
    require(cm['status']=='proposal' and cm['schema_version']=='evidence-case-uses-proposal/v1','invalid proposal source')
    for name,h in cm['artifacts'].items():
        require(Path(name).name==name,'invalid artifact name')
        take(previous/name,h)
    for name,h in cm['source_files'].items():take(local_path(ROOT,name),h)
    ep=read_json(BASE/'dual-reference-v1/current.json')
    ev=local_path(ROOT,str(EXPERIMENT/ep['result_path']))
    rp=read_json(BASE/'analysis-reference-v1/current.json')
    freeze=local_path(ROOT,str(EXPERIMENT/rp['freeze_path']))
    require(ep['manifest_sha256']==cm['source_evaluation_manifest_sha256'] and rp['manifest_sha256']==cm['source_reference_manifest_sha256'],'mixed analysis versions')
    bundle=read_json(BASE/'bundle/evidence_bundle.json')
    rows=read_lines(previous/'case_uses.jsonl')
    screening=unique(read_lines(previous/'resource_screening.jsonl'),lambda r:(r['query_id'],r['task']))
    materials=unique(read_lines(freeze/'material_reviews.jsonl'),lambda r:r['record_id'])
    blocks=unique(read_lines(ev/'blocks.jsonl'),lambda r:(r['query_id'],r['task'],r['condition']))
    lex_path=take(ROOT/'data/lexicon/annotated_lexicon_mechanism_frozen_v1.json','31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385')
    entries={e['lexicon_id']:e for e in read_json(lex_path)['terms']}
    order=list(dict.fromkeys(r['query_id'] for r in rows))
    require(len(rows)==64 and len(order)==32 and set(screening)=={(q,t) for q in order for t in TASKS},'case/task coverage changed')
    require(set(order)==set(bundle['cases']),'scope differs from discovery')
    graphs={q:resource_graph(q,bundle['cases'][q],materials) for q in order}
    options=[]
    for row in rows:
        q,task=row['query_id'],row['task']
        parts=[source_partition(graphs[q],[d]) for d in row['strict_u_candidate_demo_ids']]
        options.append({'query_id':q,'task':task,'strict_U_screen_demo_ids':row['strict_u_candidate_demo_ids'],
                        'single_demo_source_options':parts,'selected_U':None,'selection_status':'candidate_options_only'})
    protocols=[]
    contexts=[]
    for q in PRIORITY:
        for task in TASKS:
            row=next(r for r in rows if (r['query_id'],r['task'])==(q,task))
            graph=graphs[q]
            p=protocol_for(row,screening[q,task],graph,materials,blocks[q,task,'CLDnewNoCat'])
            source=next(c for c in bundle['cases'][q]['comparison']['contexts'] if c['task']==task and c['condition']=='CLDnewNoCat')
            require(sha(source['prompt_text'].encode())==source['prompt_sha256']==blocks[q,task,'CLDnewNoCat']['prompt_sha256'],'source prompt identity mismatch')
            full={'entry_ids':graph['entry_order'],'demo_ids':graph['demo_order'],'answer_overrides':{}}
            query=materials['query:'+q]['source']['text']
            replay_messages,replay_prompt=messages_for(source,query,entries,materials,full,task)
            require(replay_messages==source['messages'] and replay_prompt==source['prompt_text'],'full source renderer replay differs')
            arm_hashes=[]
            for arm in p['arms']:
                messages,prompt=messages_for(source,query,entries,materials,arm,task)
                record={'record_id':p['protocol_id']+':'+arm['condition'],'protocol_id':p['protocol_id'],'query_id':q,'task':task,
                        'condition':arm['condition'],'messages':messages,'prompt_text':prompt,'prompt_sha256':sha(prompt.encode()),
                        'source_prompt_sha256':source['prompt_sha256'],'source_context_sha256':source['context_sha256'],
                        'baseline_replay':arm['condition']==p['baseline_arm'],'operation_manifest':arm,
                        'source_slot_numbers':{d:graph['demo_order'].index(d)+1 for d in arm['demo_ids']},
                        'rendered_slot_numbers':{d:i+1 for i,d in enumerate(arm['demo_ids'])},
                        'tokenization_status':'pending','position_alignment':'not_controlled_natural_edit'}
                if record['baseline_replay']:require(prompt==source['prompt_text'],'baseline arm differs')
                contexts.append(record);arm_hashes.append(record['prompt_sha256'])
            require(len(set(arm_hashes))==len(arm_hashes),'duplicate intervention arms')
            p['input_prompt_status']='text_prepared'
            protocols.append(p)
    token_audit={'status':'not_requested','model_weights_loaded':False}
    if tokenizer_path is not None:
        ref_path=take(ROOT/'exps/causal_context/general_model_ld_nolabel_v1/plan_ref.json')
        ref=read_json(ref_path)
        plan_path=Path(ref['target_path'])/'plan.json'
        plan=read_json(take(plan_path,ref['plan_sha256']))
        inventory=[]
        for path in sorted(tokenizer_path.rglob('*')):
            if path.is_file() and path.name in TOKENIZER_FILES:
                take(path)
                inventory.append({'path':str(path.relative_to(tokenizer_path)),'size':path.stat().st_size,'sha256':file_sha(path)})
        require(sha(canonical(inventory).encode())==plan['generation_runtime_identity']['tokenizer_tree_sha256'],'tokenizer source fingerprint mismatch')
        import os
        for flag in ('USE_TORCH','USE_TF','USE_FLAX','USE_TORCH_XLA'):
            os.environ[flag]='0'
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_path),local_files_only=True,trust_remote_code=False,use_fast=True)
        require(not {'torch','tensorflow','jax'} & set(sys.modules), 'CPU tokenizer check loaded a model framework')
        require(tokenizer.eos_token_id==plan['eos_token_id'] and tokenizer.pad_token_id==plan['pad_token_id'],'tokenizer special tokens differ')
        boundaries=0
        for c in contexts:
            prompt=tokenizer.apply_chat_template(c['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            require(prompt==c['prompt_text'],'tokenizer chat-template replay differs')
            encoded=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)
            ids=encoded['input_ids']
            c.update(prompt_token_ids=ids,prompt_token_ids_sha256=sha(canonical(ids).encode()),prompt_tokens=len(ids),tokenization_status='verified')
            source=next(v for v in bundle['cases'][c['query_id']]['comparison']['contexts'] if v['task']==c['task'] and v['condition']=='CLDnewNoCat')
            if c['baseline_replay']:
                require(c['prompt_tokens']==source['prompt_tokens'] and c['prompt_token_ids_sha256']==source['prompt_token_ids_sha256'],'source token replay differs')
            for candidate in plan['catalog'][c['task']]:
                require(tokenizer.encode(prompt+candidate['canonical_answer'],add_special_tokens=False)==ids+candidate['answer_token_ids'],'answer boundary changed')
                boundaries+=1
            require(len(ids)+max(d['answer_tokens'] for d in plan['catalog'][c['task']])+1+64<=8192,'full candidate sequence exceeds registered capacity')
            query_start=prompt.rindex('待判断文本（JSON 字符串）：\n')
            user_end=prompt.index('<|im_end|>',query_start)
            relevant=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>query_start and a<user_end]
            c['positions']={'query_block_token_span':[min(relevant),max(relevant)+1],'answer_start_token':len(ids),'mapping_source':'full_prompt_fast_tokenizer_offsets'}
        for p in protocols:p['input_prompt_status']='text_and_tokens_verified'
        token_audit={'status':'passed','contexts':len(contexts),'baseline_token_replays':sum(c['baseline_replay'] for c in contexts),
                     'candidate_boundary_checks':boundaries,'tokenizer_tree_sha256':plan['generation_runtime_identity']['tokenizer_tree_sha256'],
                     'transformers_version':importlib.metadata.version('transformers'),'model_weights_loaded':False,
                     'model_frameworks_loaded':False}
    next_rows=[]
    option_index={(o['query_id'],o['task']):o for o in options}
    for row in rows:
        q,t=row['query_id'],row['task'];r=deepcopy(row)
        relevant=[p for p in protocols if (p['query_id'],p['task'])==(q,t)]
        r.update(workflow_status='implemented',workflow_authorization=AUTHORIZATION,
                 protocol_ids=[p['protocol_id'] for p in relevant],source_graph_query_id=q,
                 dictionary_effective_single_demo_options=sum(o['dictionary_factor_effective'] for o in option_index[q,t]['single_demo_source_options']),
                 source_partition_status='verified',human_case_use_adopted=False,
                 next_action='current_model_replay_then_score_fixed_inputs' if relevant and tokenizer_path else 'prepare_tokenization' if relevant else 'select_material_and_contrast_from_recorded_options')
        r['pending_by_phase']={'scoring':list(dict.fromkeys(w for p in relevant for w in p['runtime_pending'])),
                               'mechanism':list(dict.fromkeys(w for p in relevant for w in p['mechanism_pending'])),
                               'material_selection':[] if relevant else row['pending_work']}
        r['pending_work']=list(dict.fromkeys(w for p in relevant for w in p['runtime_pending']+p['mechanism_pending'])) if relevant else row['pending_work']
        if q=='3169':r['pending_work'].append('definition_variant_not_adopted; separate_from_current_demo_removal')
        next_rows.append(r)
    code_files=[Path(__file__).resolve(),ROOT/'src/diagnostics/general_model_contexts.py',ROOT/'src/diagnostics/general_model_tasks.py']
    for p in code_files:take(p)
    audit={'schema_version':'evidence-case-use-implementation-audit/v1','status':'passed','cases':len(order),'task_rows':len(rows),
           'resource_graphs':len(graphs),'unique_material_objects':len({oid for r in rows for oid in r['material_record_ids']}),
           'protocols':len(protocols),'prepared_contexts':len(contexts),'priority_queries':list(PRIORITY),
           'strict_U_tasks_without_candidates':sum(not o['strict_U_screen_demo_ids'] for o in options),
           'strict_U_selected_automatically':0,'source_text_replays':sum(c['baseline_replay'] for c in contexts),
           'tokenization':token_audit,'workflow_authorized':True,'new_human_judgments':0,'production_writes':0,'model_forward_executed':False,
           'input_scoring_ready':False,'mechanism_ready':False,'scoring_readiness_reason':'current_model_numeric_replay_not_executed'}
    files={'case_uses.jsonl':jsonl(next_rows),'resource_graphs.jsonl':jsonl(graphs.values()),'source_options.jsonl':jsonl(options),
           'interventions.jsonl':jsonl(protocols),'contexts.jsonl':jsonl(contexts),'audit.json':json_bytes(audit),'authorization.json':json_bytes(AUTHORIZATION),
           'execution_source.json':json_bytes({str(p.relative_to(ROOT)):{'sha256':file_sha(p),'text':p.read_text()} for p in code_files})}
    flat=[{k:r[k] for k in ('query_id','task','original_label','reviewed_label','original_mask','reviewed_mask','observed_behavior','proposed_roles',
                           'strict_u_candidate_demo_ids','demo_answer_difference_ids','dictionary_effective_single_demo_options','protocol_ids','next_action','pending_work')} for r in next_rows]
    files['case_uses.csv']=csv_bytes(flat)
    files['interventions.csv']=csv_bytes([{k:p[k] for k in ('protocol_id','query_id','task','operation','hypothesis','alternatives','material_basis','reference_label','group_foil','input_prompt_status','position_control','runtime_pending')} for p in protocols])
    report=['# 案例用途四步实施记录','','32 个案例、64 个任务均完成事实、用途建议、材料关联和具体待办的更新。用户授权实施工作流程；用途和假设的作者身份仍保留，不追加人工标注。','',
            '| 案例/任务 | 操作 | 主要问题 | 条件数 | 输入准备 |','| --- | --- | --- | ---: | --- |']
    for p in protocols:report.append(f"| {p['query_id']}/{p['task']} | {p['operation']} | {p['hypothesis']} | {len(p['arms'])} | {p['input_prompt_status']} |")
    report+=['','## 范围与接续','','5086/hate 的 7248 为话题直接、规则部分相关，采用独立的相关示例来源分解，不并入严格 U 分支。group 127 和 541/group 826 的答案修订单独构造；6037 只操作已核验的 5230→lex-0069 边。','',
             '3169 固定 lex-0419 及全部其他词典，只移除示例 3660。原定义合理且没有已采纳改写，释义改写仍为另一个待准备分支。','',
             '1128/4026 只作为删除一条 hate 示例且保持词典的部分操作对照；群体答案、内容及 token 长度未匹配，不称为完整位置或语义控制。','',
             '所有新条件保留无效和反向结果。后续须在当前模型环境重放基线，并在同一次新运行中评分全部条件。历史分数仅用于回归核对。自然编辑未控制位置；激活采集和 patching 尚未开始。','',
             '每个 group 协议的 foil 已从封存 SD 的最佳非审核候选固定，不随新条件重选。完整 32 候选仍是主决策。全部 64 个任务的资源选项与尚待落实的事项均保留。','']
    files['IMPLEMENTATION.md']='\n'.join(report).encode()
    src.unchanged()
    return files,src.hashes,audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    target=args.output.resolve()
    require(target.is_relative_to(BASE/'reviews') and target!=BASE/'reviews','output must be in private reviews')
    # Preserve every existing immutable artifact tree, including nested paths.
    for name,key in (('analysis-reference-v1','freeze_path'),('dual-reference-v1','result_path'),('case-uses-v1','proposal_path')):
        pointer=read_json(BASE/name/'current.json');f=local_path(ROOT,str(EXPERIMENT/pointer[key]))
        require(not target.is_relative_to(f) and not f.is_relative_to(target),'output overlaps a frozen source tree')
    tokenizer=args.tokenizer.resolve() if args.tokenizer else None
    if tokenizer:require(tokenizer.is_relative_to(ROOT),'tokenizer must be local to repository')
    files,sources,audit=build(tokenizer)
    if args.check:
        m=read_json(target/'manifest.json')
        require(m['schema_version']=='evidence-case-use-implementation/v1' and m['status']=='prepared','invalid implementation receipt')
        require(m['source_files']==sources and set(m['artifacts'])==set(files),'source/output inventory changed')
        require({p.name for p in target.iterdir()}=={*files,'manifest.json'},'unexpected output files')
        for name,raw in files.items():require((target/name).read_bytes()==raw and m['artifacts'][name]==sha(raw),'implementation reconstruction mismatch: '+name)
        print('Implementation reproduces byte for byte; no model forward.')
    else:
        files['manifest.json']=json_bytes({'schema_version':'evidence-case-use-implementation/v1','status':'prepared',
                                         'created_at':datetime.now(timezone.utc).isoformat(),'source_files':sources,
                                         'tokenizer_path':str(tokenizer.relative_to(ROOT)) if tokenizer else None,
                                         'artifacts':{n:sha(v) for n,v in files.items()}})
        write_output(target,files);print(target)


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print('Case-use implementation failed: '+str(exc),file=sys.stderr);raise SystemExit(1)
