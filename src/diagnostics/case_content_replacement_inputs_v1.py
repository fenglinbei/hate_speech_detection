"""Adopted 88-input content experiment; old construction and captures stay pinned."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

from diagnostics.case_attention_inputs_v1 import (
    ROOT, ROLES, ROLE_LABELS, require, read, lines, canonical, digest, sha, info,
    verify, write, atomic, tokenizer, build_input, TASK)

WORK = ROOT/'reviews/case-content-replacement-v1'
PUBLIC = ROOT/'docs/research/experiment-plans/case-content-replacement-v1'
PREPARED = WORK/'prepared-01'
PARENT = WORK/'draft-01'
OLD = ROOT/'reviews/case-attention-v1/prepared-01'
CONDITIONS = [('C0',False,'none'),('D',True,'none'),('L',False,'definition'),('LD',True,'definition')]
CODE = [ROOT/'src/diagnostics'/f'case_content_replacement_{n}_v1.py' for n in ('inputs','runtime','report')]
CODE += [ROOT/'scripts/review'/n for n in ('run_case_content_replacement_v1.py','run_case_content_replacement_window_v1.py',
                                         'test_case_content_replacement_v1.py','audit_case_content_replacement_execution_v1.py')]
CODE += [ROOT/'tools/case_content_replacement_viewer_v1/viewer.html']


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj


def prepare(output=PREPARED):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions, framework_sources
    out=Path(output).absolute()
    require(not out.exists(),'New execution preparation required')
    for item in read(PARENT/'manifest.json')['artifacts']+read(PARENT/'manifest.json')['sources']:
        verify(item)
    for item in read(WORK/'revision-01/manifest.json')['artifacts']:
        verify(item)
    revision=read(WORK/'revision-01/proposal.json')
    feedback1=read(WORK/'feedback-01.json')
    feedback2={'schema':'case-content-replacement-feedback/v2','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
               'user_message':'没问题，可以通过','prior_run_authorization':'可以直接把改好的结果询问我，我通过了就可以开始GPU运行了',
               'accepted_item_id':'L06','accepted_revision':2,'revision_manifest':info(WORK/'revision-01/manifest.json'),
               'review_kind':'explicit_acceptance_of_revised_proposal','original_AI_authorship_preserved':True,
               'exact_input_text':revision['replacement_text'],
               'chat_display_text':'用“嘿嘿”这一谐音形式代替“黑黑”，作为侮辱黑人群体的称呼。',
               'quote_normalization':'Chat used typographic quotation marks. Execute the already-token-checked revision-01 ASCII quote form to preserve the approved equal-token control; lexical wording unchanged.',
               'GPU_authorized_after_checks':True,'prior_max_GPU_window_minutes':45}
    write(WORK/'feedback-02.json',feedback2)
    # Reuse the immutable constructor as a library. The sole versioned material
    # override is explicit here; neither the original script nor its draft is edited.
    builder=module('ccr_original_constructor',ROOT/'scripts/review/prepare_case_content_replacement_v1.py')
    builder.DEFINITIONS=[(*d[:-1],revision['replacement_text']) if d[0]=='L06' else d for d in builder.DEFINITIONS]
    builder.build(out)
    timing={'user_message':'超时守护定到1:00前即可','absolute_deadline_local':'2026-09-19T01:00:00+08:00',
            'supersedes_relative_45_minute_guard':True,'STOP_at_local':'2026-09-19T00:58:00+08:00',
            'owned_worker_TERM_at_local':'2026-09-19T00:59:00+08:00','owned_worker_KILL_at_local':'2026-09-19T00:59:40+08:00',
            'permission_source':'explicit latest user steering; no new confirmation required'}
    write(out/'timing-amendment.json',timing)
    old={r['request_id']:r for r in lines(PARENT/'inputs.jsonl')}
    requests=lines(out/'inputs.jsonl')
    changed=[r['request_id'] for r in requests if r!=old[r['request_id']]]
    require(set(changed)=={'ccr-3169-L-L06','ccr-3169-LD-L06'},'Unapproved input changed')
    props=read(out/'material-proposals.json')['items']
    original_props={p['item_id']:p for p in read(PARENT/'material-proposals.json')['items']}
    accepted=[]
    for p in props:
        iid=p['item_id']
        require(iid=='L06' or (iid in feedback1['accepted_item_ids'] and p==original_props[iid]),'Unapproved proposal drift')
        accepted.append({'item_id':iid,'proposal_sha256':digest(p),'decision':'accept','review_kind':'explicit_single_revision_acceptance' if iid=='L06' else 'explicit_bulk_acceptance',
                         'feedback':info(WORK/('feedback-02.json' if iid=='L06' else 'feedback-01.json')),
                         'AI_authorship_retained':True,'new_individual_adjudication':False})
    write(out/'adoption.json',{'status':'all_44_accepted','items':accepted,'B01_edited_text_reference':'有',
                             'B01_adoption_kind':'explicit_bulk_acceptance_with_original_AI_authorship','old_human_records_overwritten':False})
    # The scorer gets only the prompt/capture contract. Historical construction
    # status fields stay in inputs.jsonl; adoption.json is the authoritative review.
    scoring=[{k:v for k,v in r.items() if k not in ('review_status','eligible_for_GPU')} for r in requests]
    (out/'scoring-inputs.jsonl').write_bytes(b''.join(canonical(r)+b'\n' for r in scoring))
    write(out/'input-amendment.json',{'parent_manifest':info(PARENT/'manifest.json'),'changed_requests':changed,'unchanged_input_records':86,
                                    'quote_normalization':feedback2['quote_normalization'],
                                    'before_after':[{'request_id':r['request_id'],'old_prompt_sha256':old[r['request_id']]['prompt_sha256'],
                                                     'new_prompt_sha256':r['prompt_sha256'],'total_tokens':r['prompt_tokens']} for r in requests if r['request_id'] in changed]})
    profile=read(OLD/'model-profile.json');write(out/'model-profile.json',profile)
    plan=read(OLD/'execution-plan.json')
    n=len(scoring);prefix=sum(len(r['prefix_proofs']) for r in scoring)
    plan.update(schema_version='case-content-replacement-execution/v1',conditions=[x[0] for x in CONDITIONS],
                status='adopted_cpu_preparation_gpu_pending',runtime_versions=runtime_versions(),all_materials_adopted=True,
                input_count=n,all_comparisons_reported=True,comparison_count=258,max_GPU_window_minutes=None,
                absolute_GPU_deadline=timing['absolute_deadline_local'],
                budget={'unique_prompts':n,'engineering_full_forwards':6*n,'prefix_forwards':prefix,'production_forwards':n,
                        'format_extra_forward_max':7*n,'total_forward_max':14*n+prefix,'usual_if_label_then_eos':8*n+prefix})
    write(out/'execution-plan.json',plan)
    # Query references are unchanged and are joined only in post-release analysis.
    (out/'analysis-references.json').write_bytes((OLD/'analysis-references.json').read_bytes())
    authorization=read(out/'authorization-context.json')
    authorization.update(material_review_passed=True,latest_acceptance=feedback2,accepted_items=44)
    (out/'authorization-context.json').write_bytes(canonical(authorization)+b'\n')
    (out/'README.md').write_text('# 内容替换实验：已采用并完成 CPU 执行准备\n\n44 项均已获用户接受。adoption.json 为采用记录；material-proposals.json/inputs.jsonl 保留构建时的空人审字段，不能据此重新要求审核。scoring-inputs.jsonl 是工作进程唯一输入清单。\n\n88 个输入、258 个比较；仅 L06 的 L/LD 两条相对 draft-01 改变。以 revision-01 半角引号数据稿执行，聊天的中文引号排版差异有明确记录。原始人审和旧实验全部保留。\n\n新建 run，通过数值／格式资格和正常释放后才可正式阶段；按最新指令在北京时间 2026-09-19 01:00 前安全停止并释放 GPU。资格、实际设备和绝对时间由新的绑定及窗口文件记录。不得继承旧数值误差界或重启旧终态 run。\n',encoding='utf-8')
    doc=(out/'REVIEW.md').read_text(encoding='utf-8')
    doc=doc.replace('**状态：待审核，尚未运行。**','**状态：全部 44 项已通过；尚未运行。**').replace('**决定：待审核**（通过／修改／否决）','**决定：通过**（采用依据见 adoption.json）')
    (out/'REVIEW.md').write_text(doc,encoding='utf-8')
    previous=read(out/'source-ledger.json')
    src=[Path(r['path']) for r in previous['files']]+CODE+framework_sources()
    src += [WORK/'feedback-01.json',WORK/'feedback-02.json',WORK/'revision-01/manifest.json',WORK/'revision-01/proposal.json',
            PARENT/'manifest.json',PARENT/'inputs.jsonl',PARENT/'material-proposals.json',
            ROOT/'src/diagnostics/case_attention_capture_v1.py',ROOT/'src/diagnostics/cross_model_applicability_models_v1.py',
            ROOT/'src/diagnostics/cross_model_applicability_execution_v1.py']
    previous['files']=[info(p) for p in sorted(set(src))]
    (out/'source-ledger.json').write_bytes(canonical(previous)+b'\n')
    return {'directory':str(out),'inputs':n,'accepted_items':44,'changed_inputs':changed,'new_GPU_forwards':0}


def validate(prepared, sealed=True, weights=False):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    out=Path(prepared)
    if sealed:
        m=read(out/'manifest.json')
        require(m['accepted_review_items']==44 and m['schema_version']=='case-content-replacement-prepared/v1','Wrong preparation')
        for r in m['artifacts']+m['sources']:verify(r)
    for r in read(out/'source-ledger.json')['files']:verify(r)
    plan,profile=read(out/'execution-plan.json'),read(out/'model-profile.json')
    require(plan['runtime_versions']==runtime_versions() and plan['all_materials_adopted'],'Unqualified preparation runtime/review')
    for r in profile['metadata_sources']:verify(r)
    for r in profile['weight_sources']:
        if weights:verify(r)
        else:
            st=Path(r['path']).stat();require((st.st_size,st.st_mtime_ns)==(r['bytes'],r['mtime_ns']),'Weight stat changed')
    requests=lines(out/'scoring-inputs.jsonl')
    require(len(requests)==len({r['request_id'] for r in requests})==88,'Input inventory differs')
    require(sum(len(r['prefix_proofs']) for r in requests)==128,'Prefix inventory differs')
    for r in requests:
        require(len(r['input_ids'])==r['prompt_tokens'] and digest(r['input_ids'])==r['input_ids_sha256'],'Input token binding')
        require(hashlib.sha256(r['prompt_text'].encode()).hexdigest()==r['prompt_sha256'],'Prompt binding')
        require(r['roles']['pre_answer']==[len(r['input_ids'])-1],'Answer position')
        require(not {'reference','gold','human_decision','review_status','eligible_for_GPU'}&set(r),'Reference/review fields in scorer input')
    return plan,profile,requests


def seal(prepared):
    out=Path(prepared)
    require(not (out/'manifest.json').exists(),'Already sealed')
    validate(out,sealed=False)
    for name in ('cpu-tests.json','cpu-audit.json','watchdog-cpu-test.json'):
        r=read(out/name);require(r['status'] in ('pass','passed') and not r.get('CUDA_initialized',False),'CPU check failed')
        for item in r.get('implementation_snapshot',[]):verify(item)
    for r in read(out/'source-ledger.json')['files']:verify(r)
    write(out/'manifest.json',{'schema_version':'case-content-replacement-prepared/v1','status':'cpu_complete_gpu_pending',
                              'accepted_review_items':44,'artifacts':[info(p) for p in sorted(out.rglob('*')) if p.is_file()],
                              'sources':read(out/'source-ledger.json')['files'],'GPU_allocation':None,'GPU_qualified':False,'immutable_after_seal':True})
    previous=PUBLIC/'current.json'
    write(WORK/'selector-before-execution-01.json',read(previous))
    atomic(previous,{'schema':'case-content-replacement-selector/v1','status':'adopted_prepared_gpu_pending',
                     'directory':str(out.relative_to(ROOT)),'manifest':info(out/'manifest.json'),'accepted_item_count':44,
                     'pending_item_ids':[],'GPU_execution_authorized':True,'absolute_GPU_deadline':'2026-09-19T01:00:00+08:00'},replace=True)
    return info(out/'manifest.json')
