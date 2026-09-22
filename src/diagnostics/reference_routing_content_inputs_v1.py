"""Immutable material adoption, gold-free inputs and bounded execution registry."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from diagnostics.case_attention_inputs_v1 import ROOT, require, read, lines, canonical, digest, sha, info, verify, write, atomic

WORK=ROOT/'reviews/reference-routing-content-v1'
MATERIALS=ROOT/'reviews/reference-routing-content-materials-v1'
DRAFT=MATERIALS/'draft-01'
ADOPTED=MATERIALS/'adopted-01'
PROTOCOL=ROOT/'docs/research/experiment-plans/reference-routing-content-v1/frozen-01'
OLD=ROOT/'reviews/jingba-demo-donor-v1'
PREPARED=WORK/'prepared-01'
USER_QUOTE='材料我过了一遍，整体没什么问题，可以开始做GPU运行前准备'
DRAFT_SHA='9ec58f3ff86f951250265c13488b48da64708aab5b074c18576c2d0f0bc4e11e'
PROTOCOL_SHA='2dffc823407f2f0dd0a9a33b35449889aba4a492cc4f8cc0f571b7be0dee8892'


def verify_manifest(path):
    m=read(path)
    for section in ('artifacts','sources','protected_selectors'):
        for item in m.get(section,[]):verify(item)
    return m


def adopt():
    require(sha(DRAFT/'manifest.json')==DRAFT_SHA,'Reviewed draft identity differs')
    verify_manifest(DRAFT/'manifest.json')
    require(not ADOPTED.exists(),'Adoption is immutable; use a new version')
    ADOPTED.mkdir(parents=True)
    mats=read(DRAFT/'materials.json');relations=read(DRAFT/'relations-ai.json')['records']
    material_fields=['/raw_text','/text_sha256','/split','/term_family_id','/construction_family_id',
        '/focal_form','/sense','/stratum','/proposed_answer','/answer_rationale',
        '/focal_occurrence_spans','/author_stance_spans','/quotation_spans','/independent_attack_spans']
    relation_fields=['/source','/target','/source_quality/reviewed_answer','/source_quality/value',
        '/source_quality/rationale','/semantic_reference_fit','/rule_fit','/lexical_overlap','/presentation_refs']
    decision={'schema_version':'reference-routing-content-bulk-adoption/v1','user_message':USER_QUOTE,
        'authority':'current_task_explicit_user_material_review_and_preparation_request',
        'recorded_at':datetime.now(timezone.utc).isoformat(),'message_date_local':'2026-09-22',
        'reviewed_manifest':info(DRAFT/'manifest.json'),'kind':'bulk_adoption_of_presented_AI_proposals',
        'materials':[{'material_id':m['material_id'],'text_sha256':m['text_sha256'],'accepted_fields':material_fields}
                     for m in mats['queries']+mats['demos']],
        'relations':[{'relation_id':r['relation_id'],'source_text_sha256':r['source']['text_sha256'],
                      'target_text_sha256':r['target']['text_sha256'],'accepted_fields':relation_fields} for r in relations],
        'interpretation':'User broadly accepted the delivered draft. Record one bulk decision, not 732 fabricated individual UI confirmations.',
        'span_scope':'Adopts presented spans as coarse annotation scope; no new minimal sufficient stance/attack localization is asserted.',
        'not_inferred':['individual_review_times','fine_grained_new_spans','severity_or_group_labels','independence_proof','GPU_execution_permission'],
        'GPU_execution_authorized':False,'real_model_outputs_seen_for_this_batch':False}
    write(ADOPTED/'decision.json',decision)
    ref={k:info(ADOPTED/'decision.json')[k] for k in ('path','sha256')}
    def provenance(original,fields):
        return dict(original,review_kind='human_with_ai',adoption='bulk',accepted_fields=fields,
                    decision_ref=ref,recorded_at=decision['recorded_at'])
    for m in mats['queries']+mats['demos']:
        m['human_reference']=m['proposed_answer'];m['reference']=m['proposed_answer']
        m['adopted_fields']=m['accepted_fields']=material_fields
        m['decision_ref']=ref;m['provenance']=provenance(m['provenance'],material_fields)
        m['eligible_for_model_execution']=True
        m['adoption_note']='整批采纳原AI建议；原 span_note/answer_rationale 保留草案时态，不表示新增逐条人工记录。跨度不声称最小充分定位。'
    mats['status']='BULK_ADOPTED_PRE_GPU';mats['GPU_execution_authorized']=False
    write(ADOPTED/'materials.json',mats)
    for r in relations:
        r['provenance']=provenance(r['provenance'],relation_fields)
        r['source_quality']['provenance']=provenance(r['source_quality']['provenance'],
            ['/source_quality/reviewed_answer','/source_quality/value','/source_quality/rationale'])
    write(ADOPTED/'relations.json',{'status':'BULK_ADOPTED_AI_PROPOSALS','decision_ref':ref,'records':relations})
    write(ADOPTED/'references.json',{'worker_must_not_read':True,'decision_ref':ref,
        'queries':[{'query_id':m['material_id'],'reference':m['reference'],'split':m['split'],
                    'term_family_id':m['term_family_id'],'focal_form':m['focal_form'],'stratum':m['stratum'],
                    'construction_family_id':m['construction_family_id'],'new_term':m['term_family_id'] in ('T04','T05','T06')}
                   for m in mats['queries']]})
    (ADOPTED/'README.md').write_text('''# 材料整批采纳记录

2026-09-22，用户审核已交付 draft-01 后表示：“材料我过了一遍，整体没什么问题，可以开始做GPU运行前准备”。本版本据此整批采纳48条查询、108条示例及576条关系建议，并固定开发／确认划分。decision.json 绑定原草案清单和每个被采纳字段。

这是对 AI 原始建议的整批采纳，没有虚构逐条确认、浏览器导出或审核时间。原始文案及建议不改；部分跨度仍是粗粒度全文范围，不代表已经定位最小充分立场片段。source_quality、semantic_reference_fit 和 rule_fit 分别保留来源，不按示例设计名称推断其他关系。此采纳不提供新的 GPU 执行授权。
''',encoding='utf-8')
    shutil.copyfile(Path(__file__),ADOPTED/'preparation-source-snapshot.py')
    write(ADOPTED/'manifest.json',{'schema_version':'reference-routing-content-adopted/v1',
        'artifacts':[info(p) for p in sorted(ADOPTED.iterdir()) if p.is_file()],
        'sources':[info(DRAFT/'manifest.json'),info(DRAFT/'materials.json'),info(DRAFT/'relations-ai.json')],
        'immutable_after_seal':True,'GPU_execution_authorized':False})
    return info(ADOPTED/'manifest.json')


def stage_b_rows(split):
    queries={q['material_id']:q for q in read(ADOPTED/'materials.json')['queries']}
    result=[]
    for row in lines(DRAFT/'model-inputs-preview.jsonl'):
        if queries[row['query_id']]['split']!=split:continue
        r={k:deepcopy(row[k]) for k in ('request_id','query_id','condition','prompt_text','prompt_sha256',
                'input_ids','input_ids_sha256','prompt_tokens','candidate_tokens')}
        focal=row['query_focal_positions'];query=row['query_positions']
        r.update(roles={'query_all':query,'query_focal':focal,'pre_answer':[row['pre_answer']]},
                 capture_positions=focal, capture_prefix_length=max(focal)+1,
                 patch_position_sets={'focal':focal}, material_adoption_complete=True)
        result.append(r)
    return result


def stage_b_jobs(rows,factors):
    by={(r['query_id'],r['condition']):r for r in rows}
    jobs=[];controls=[]
    for r in rows:
        rid=r['request_id'];base={'recipient':rid,'background':'N','upstream':None,'av_override':None,'qas_factor':1}
        jobs.append(dict(base,job_id=rid+'/N',kind='native'))
        if r['condition']=='M00':continue
        d=by[r['query_id'],'M00']
        upstream={'donor':d['request_id'],'layer':17,'site':'decoder_block_output',
                  'positions':r['patch_position_sets']['focal'],'donor_positions':d['patch_position_sets']['focal'],'strength':1}
        jobs.append(dict(base,job_id=rid+'/U',kind='upstream',background='U',upstream=upstream))
        for factor in factors:
            if factor!=1:jobs.append(dict(base,job_id=rid+f'/Q{factor}',kind='qas',qas_factor=factor))
        controls.append(dict(base,job_id=rid+'/Q1_self',kind='qas_native_self'))
        own=dict(upstream,donor=rid,donor_positions=upstream['positions'])
        controls.append(dict(base,job_id=rid+'/N_focal_self',kind='focal_native_self',upstream=own))
    return jobs,controls


def budget(jobs,rows,controls):
    n=len(jobs);c=len(controls);p=len(rows)
    # Per science configuration: raw, observed, repeat, reversed order, left,
    # right; one fresh production replay; exact label + EOS usually one more.
    return {'science_score_configurations':n,'separate_self_controls':c,'native_true_prefixes':p,
            'engineering_six_passes':6*n,'production_replay':n,'usual_format_continuations':n,
            'maximum_format_continuations':7*n,'usual_total':8*n+p+c,'maximum_total':14*n+p+c,
            'no_CPU_derived_score_counted_as_forward':True}


def prepare(output=PREPARED):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    p=Path(output);require(not p.exists(),'Preparation directory must be new')
    require(sha(PROTOCOL/'manifest.json')==PROTOCOL_SHA,'Normative protocol identity differs')
    verify_manifest(PROTOCOL/'manifest.json');verify_manifest(ADOPTED/'manifest.json')
    p.mkdir(parents=True)
    a=lines(OLD/'prepared-01/scoring-inputs.jsonl')
    aj=read(PROTOCOL/'stage-a-jobs.json')['jobs']
    groups={'stage-a':(a,aj,[])}
    dev=stage_b_rows('development');conf=stage_b_rows('confirmation')
    groups['development']=(dev,*stage_b_jobs(dev,[2,4]))
    for f in (1,2,4):groups[f'confirmation-factor-{f}']=(conf,*stage_b_jobs(conf,[f]))
    allbudgets={}
    for name,(rows,jobs,controls) in groups.items():
        folder=p/name;folder.mkdir()
        with (folder/'inputs.jsonl').open('xb') as out:
            out.write(b''.join(canonical(r)+b'\n' for r in rows))
        write(folder/'jobs.json',{'science':jobs,'controls':controls})
        allbudgets[name]=budget(jobs,rows,controls)
    shutil.copyfile(OLD/'prepared-01/model-profile.json',p/'model-profile.json')
    shutil.copyfile(PROTOCOL/'stage-a-index.json',p/'stage-a-index.json')
    shutil.copyfile(ADOPTED/'references.json',p/'analysis-references.json')
    shutil.copyfile(OLD/'prepared-01/analysis-references.json',p/'stage-a-references.json')
    replay=[]
    for j in aj:
        if j['kind'] not in ('native','upstream','preceding'):continue
        if j['kind']=='native':path=OLD/'run-01/records/native-production'/(j['recipient']+'.json')
        else:
            req=next(r for r in a if r['request_id']==j['recipient'])
            path=OLD/'run-01/records/production'/f'{req["query_id"]}-{req["condition"]}-{j["kind"]}.json'
        r=read(path)
        replay.append({'job_id':j['job_id'],'record':info(path),'vector':r['vector'],'states':r['states'],'trajectory':r['trajectory']})
    write(p/'historical-replay.json',{'fresh_required':True,'old_score_substitution':False,'records':replay})
    plan={'schema_version':'reference-routing-content-execution/v1','status':'CPU_PREPARATION',
        'protocol':info(PROTOCOL/'manifest.json'),'materials':info(ADOPTED/'manifest.json'),
        'user_message':USER_QUOTE,'GPU_execution_authorized':False,'runtime_versions':runtime_versions(),
        'budgets':allbudgets,'stage_order':['stage-a','development','CPU_calibration_lock','confirmation'],
        'confirmation_gate':'accepted production development data + normal GPU release + immutable calibration lock',
        'scientific_progression_is_not_GPU_authorization':True,'query_reference_join_during_worker':False,
        'numerical_caps':{'exact_repeat_reverse_raw_replay':True,'margin_padding_absolute':.001,
            'state_scaled':.0001,'projection_absolute':.001,'reconstruction_scaled':.0001,
            'attention_element':.0001,'attention_row_l1':.001,'attention_rowsum':.000002,'floor':.000001},
        'margin_bound':'per physical scoring job max(floor,2*max padding and self reconstruction margin errors), newly measured',
        'self_reconstruction':'AV11 and N_AV00 include full-vector and trajectory tolerances; focal/Q1 self exact',
        'format':{'exact_label_then_eos':True,'max_new_tokens':8,'applies':'all science configurations',
            'original_pre_answer_unchanged_after_append':True,'failure_policy':'terminal retain all; no silent filtering'},
        'cost':['forwards','valid_and_padded_tokens','same_device_seconds','peak_allocated_reserved_VRAM','donor_first_vs_reuse'],
        'run_policy':{'single_freshly_idle_GPU_min_MiB':44000,'batch':1,'parallel_workers':1,
            'no_automatic_retry':True,'terminal_failed_stopped_complete_never_resume':True,
            'no_fixed_old_deadline_inherited':True,'fresh_execution_decision_required':True}}
    write(p/'execution-plan.json',plan)
    require('torch' not in sys.modules,'CPU preparation imported model backend')
    return allbudgets


def validate(p=PREPARED,sealed=True,weights=False):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    p=Path(p)
    if sealed:verify_manifest(p/'manifest.json')
    plan=read(p/'execution-plan.json');profile=read(p/'model-profile.json')
    require(plan['runtime_versions']==runtime_versions(),'Runtime versions changed')
    require(plan['GPU_execution_authorized'] is False,'Preparation cannot authorize launch')
    verify(plan['protocol']);verify(plan['materials']);verify_manifest(ADOPTED/'manifest.json')
    for x in profile['metadata_sources']:verify(x)
    for x in profile['weight_sources']:
        if weights:verify(x)
        else:
            s=Path(x['path']).stat();require((s.st_size,s.st_mtime_ns)==(x['bytes'],x['mtime_ns']),'Checkpoint stat changed')
    for group,b in plan['budgets'].items():
        rows=lines(p/group/'inputs.jsonl');registry=read(p/group/'jobs.json')
        require(b==budget(registry['science'],rows,registry['controls']),'Budget mismatch')
        by={r['request_id']:r for r in rows};seen=set()
        for r in rows:
            require(not set(r)&{'reference','human_reference','proposed_answer','gold','semantic_reference_fit','rule_fit','stratum'},'Worker input reference leak')
            require(len(r['input_ids'])==r['prompt_tokens'] and r['roles']['pre_answer']==[r['prompt_tokens']-1],'Input boundary changed')
            require(__import__('hashlib').sha256(r['prompt_text'].encode()).hexdigest()==r['prompt_sha256'],'Prompt digest')
        for j in registry['science']+registry['controls']:
            require(j['job_id'] not in seen,'Duplicate job');seen.add(j['job_id']);r=by[j['recipient']]
            if j['upstream']:
                u=j['upstream'];d=by[u['donor']]
                require(r['query_id']==d['query_id'] and u['layer']==17 and u['strength']==1,'Upstream rule changed')
                require([r['input_ids'][i] for i in u['positions']]==[d['input_ids'][i] for i in u['donor_positions']],'Donor token identity changed')
                require(set(u['positions'])<=set(r['roles']['query_all']) and max(u['positions'])<r['prompt_tokens']-1,'Invalid patch position')
            if j['av_override']:
                av=j['av_override'];require(av['layer']==18 and av['position']==r['prompt_tokens']-1,'AV target differs')
                require(all(av[k] in (r['request_id']+'/N',r['request_id']+'/U') for k in ('A_from','V_from')),'AV source crosses recipient prompt')
    return plan,profile
