"""Four unchanged prompts and an explicitly adopted stage-2 intervention ledger."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

from diagnostics.case_attention_inputs_v1 import (
    ROOT, require, read, lines, canonical, digest, sha, info, verify, write, atomic)
from diagnostics import hehe_presentation_inputs_v1 as parent

WORK=ROOT/'reviews/hehe-focal-patching-v1'
PUBLIC=ROOT/'docs/research/experiment-plans/hehe-focal-patching-v1'
PREPARED=WORK/'prepared-01'
PARENT=parent.PREPARED
POSITION_OPTIONS={
    'pre':{'Q01':[795,796],'Q02':[795]},
    'post':{'Q01':[800,801],'Q02':[797]},
}
POSITION_TEXT={
    'pre':{'Q01':['回','个'],'Q02':['被']},
    'post':{'Q01':['感觉','好'],'Q02':['玩']},
}
LABELS={'focal':'查询“嘿嘿”','pre':'前置位置对照','post':'后置位置对照'}
PRIMARY_COUNT=144
STATE_SCALED_CAP=0.0001
CODE=[ROOT/'src/diagnostics'/f'hehe_focal_patch_{name}_v1.py' for name in ['inputs','capture','runtime','report']]
CODE += [ROOT/'scripts/review'/name for name in [
    'run_hehe_focal_patch_v1.py','test_hehe_focal_patch_capture_v1.py','test_hehe_focal_patch_runtime_v1.py',
    'audit_hehe_focal_patch_inputs_v1.py','audit_hehe_focal_patch_results_v1.py','test_hehe_focal_patch_viewer_v1.cjs']]
CODE += [ROOT/'tools/hehe_focal_patch_viewer_v1/viewer.html']


def requests():
    _,profile,old=parent.validate(PARENT,weights=False)
    selected=[]
    for row in old:
        if row['query_id'] not in ('Q01','Q02') or row['dictionary_id'] not in ('D01','D02'):continue
        r=deepcopy(row);r['request_id']=row['request_id'].replace('hpm-','hfp-')
        r['parent_request_id']=row['request_id'];selected.append(r)
    require(len(selected)==4,'Expected four already adopted prompts')
    require('torch' not in sys.modules,'Prompt preparation must not import torch')
    return selected,profile


def position_sets(req,mode):
    require(mode in ('pre','pre_post'),'Unadopted position mode')
    q=req['query_id'];sets={'focal':req['roles']['query_focal']}
    for kind in ['pre']+(['post'] if mode=='pre_post' else []):
        ps=POSITION_OPTIONS[kind][q]
        require([req['token_text'][p] for p in ps]==POSITION_TEXT[kind][q],'Proposed native token text changed')
        require(len(ps)==len(sets['focal']) and set(ps)<=set(req['roles']['query_all']),'Position count/query ownership differs')
        require(not set(ps)&set(sets['focal']),'Control overlaps focus')
        require((max(ps)<min(sets['focal'])) if kind=='pre' else (min(ps)>max(sets['focal'])),'Position order differs')
        sets[kind]=list(ps)
    return sets


def make_jobs(reqs,mode,layers=36):
    byid={r['request_id']:r for r in reqs};cross=[];selfs=[]
    for r in reqs:
        donor=r['request_id'][:-3]+('D02' if r['dictionary_id']=='D01' else 'D01')
        d=byid[donor]
        require(r['prompt_tokens']==d['prompt_tokens'] and r['roles']==d['roles'],'Native donor positions do not align')
        groups=position_sets(r,mode)
        for group,ps in groups.items():
            require([r['input_ids'][p] for p in ps]==[d['input_ids'][p] for p in ps],'Donor changes patch token identities')
            for li in range(layers):
                common={'query_id':r['query_id'],'recipient':r['request_id'],'layer':li,
                        'site':'decoder_block_output','group':group,'group_label':LABELS[group],
                        'positions':ps,'position_text':[r['token_text'][p] for p in ps],
                        'last_layer_expected_zero':li==layers-1}
                cross.append(dict(common,job_id=f"{r['query_id']}-{d['dictionary_id']}-into-{r['dictionary_id']}-{group}-L{li:02d}",
                                  donor=donor,kind='primary' if group=='focal' else 'position_control'))
                selfs.append(dict(common,job_id=f"{r['query_id']}-{r['dictionary_id']}-self-{group}-L{li:02d}",
                                  donor=r['request_id'],kind='self_control'))
    require(len({j['job_id'] for j in cross+selfs})==len(cross)+len(selfs),'Job identifiers overlap')
    return cross,selfs


def proposal(output):
    out=Path(output).absolute();require(not out.exists(),'Use a new proposal directory')
    rows,_=requests();out.mkdir(parents=True)
    document=['# 第二阶段待确定项','','本轮方法已获同意；以下具体位置与GPU时间约束仍待用户答复。',
              '仅使用Q01/Q02 × D01/D02四份现有材料；不改文本、答案标签或旧实验。','',
              '建议最小控制：Q01“回／个”，Q02“被”，均在焦点之前。可选追加：Q01“感觉／好”，Q02“玩”，均在焦点之后。',
              '位置对照不是预期零效应：前置仍可读取词典，后置还可读取焦点。只有自替换及末层焦点结构控制预期零效应。','',
              '## 精确token位置（从0开始）','']
    records=[]
    for r in rows:
        sets=position_sets(r,'pre_post')
        qspan=next(s for s in r['spans'] if s['id']=='query')
        record={'request_id':r['request_id'],'parent_request_id':r['parent_request_id'],
                'prompt_sha256':r['prompt_sha256'],'input_ids_sha256':r['input_ids_sha256'],
                'query':r['prompt_text'][qspan['char_start']:qspan['char_end']],
                'positions':{k:[{'index':p,'id':r['input_ids'][p],'text':r['token_text'][p],'offset':r['token_offsets'][p]} for p in ps] for k,ps in sets.items()}}
        records.append(record);document+=['### '+r['request_id'],'',record['query'],'']
        for group,ps in sets.items():document += ['- '+LABELS[group]+'：'+str(ps)+'，'+repr([r['token_text'][p] for p in ps])]
        document+=['']
    write(out/'positions.json',{'status':'awaiting_user_choices','records':records,'no_new_prompts':True})
    write(out/'pending.json',{'position_mode':None,'GPU_time_constraint':None,'user_request':'没问题，可以按这个规划开展，如果有待确定项先与我确定',
                            'main_method_accepted':True,'run_after_choices_and_qualification':True,'GPU_started':False})
    (out/'REVIEW.md').write_text('\n'.join(document)+'\n',encoding='utf-8')
    for r in rows:
        p=out/'prompts'/(r['request_id']+'.txt');p.parent.mkdir(exist_ok=True);p.write_text(r['prompt_text'],encoding='utf-8')
    write(out/'manifest.json',{'status':'proposal_only','artifacts':[info(p) for p in sorted(out.rglob('*')) if p.is_file()],
        'sources':[info(PARENT/'manifest.json'),info(PARENT/'scoring-inputs.jsonl'),info(PARENT/'patch-plan.json'),info(Path(__file__).resolve())]})
    return {'status':'awaiting_choices','directory':str(out),'new_prompts':0}


def prepare(output,decision):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions,framework_sources
    out=Path(output).absolute();decision=Path(decision).absolute()
    require(not out.exists(),'Never overwrite a prepared experiment')
    accepted=read(decision)
    require(accepted['status']=='accepted' and accepted['user_message'],'Explicit choice acceptance required')
    mode=accepted['position_mode'];require(mode in ('pre','pre_post'),'Pending position choice')
    require('GPU_time_constraint' in accepted and accepted['GPU_time_constraint']['confirmed'],'Pending GPU time constraint')
    rows,profile=requests();cross,selfs=make_jobs(rows,mode,profile['layers'])
    original=read(PARENT/'patch-plan.json')['primary_interventions']
    primary=[j for j in cross if j['kind']=='primary']
    normalized=lambda x:(x['query_id'],x['recipient'].replace('hfp-','hpm-'),x['donor'].replace('hfp-','hpm-'),x['layer'],x['site'],tuple(x['positions']))
    require({normalized(j) for j in primary}=={normalized(j) for j in original},'Previously adopted main interventions changed')
    out.mkdir(parents=True)
    for r in rows:
        r['patch_position_sets']=position_sets(r,mode)
        r['capture_positions']=sorted({p for ps in r['patch_position_sets'].values() for p in ps})
        r['capture_prefix_length']=max(r['capture_positions'])+1
    with (out/'scoring-inputs.jsonl').open('xb') as f:
        for r in rows:f.write(canonical(r)+b'\n')
    write(out/'interventions.json',{'cross':cross,'self':selfs})
    write(out/'adoption.json',dict(accepted,source=info(decision),previous_method=info(PARENT/'patch-plan.json'),
                                  no_new_text_or_labels=True,primary_interventions_unchanged=True))
    write(out/'model-profile.json',profile)
    (out/'model-task.txt').write_bytes((PARENT/'model-task.txt').read_bytes())
    references=read(PARENT/'analysis-references.json')
    references['references']=[r for r in references['references'] if r['query_id'] in ('Q01','Q02')]
    write(out/'analysis-references.json',references)
    n=len(cross);selfn=len(selfs)
    plan={'schema_version':'hehe-focal-patching-execution/v1','status':'accepted_CPU_preparation',
          'runtime_versions':runtime_versions(),'position_mode':mode,'input_count':4,'primary_count':len(primary),
          'cross_count':n,'self_count':selfn,'layers':profile['layers'],'acceptance':read(PARENT/'execution-plan.json')['acceptance'],
          'state_scaled_cap':STATE_SCALED_CAP,'query_reference_join_during_worker':False,'old_scores_substituted':False,
          'donor_source':'fresh same-run native block outputs; never stage1 scores/states',
          'format':{'exact_label_then_eos':True,'max_new_tokens':8,'covers':'four baselines and every cross-patched endpoint','failure_policy':'terminal failure; retain all records, no filtering'},
          'budget':{'native_engineering':24,'true_prefixes':4,'self_controls':selfn,'cross_engineering':5*n,
                    'production_native_replay':4,'production_cross_replay':n,'usual_format_continuations':4+n,
                    'usual_total':36+selfn+7*n,'maximum_total':60+selfn+13*n},
          'all_layers_and_directions_retained':True,'independent_confirmation':False,
          'GPU_qualified':False,'GPU_time_constraint':accepted['GPU_time_constraint']}
    write(out/'execution-plan.json',plan)
    doc=['# 第二阶段完整输入','','四份原材料保持不变；干预改变模型内部表示，不改prompt。','']
    for r in rows:
        p=out/'prompts'/(r['request_id']+'.txt');p.parent.mkdir(exist_ok=True);p.write_text(r['prompt_text'],encoding='utf-8')
        doc+=['## '+r['request_id'],'','```text',r['prompt_text'],'```','']
    (out/'ALL-PROMPTS.md').write_text('\n'.join(doc)+'\n',encoding='utf-8')
    sources=CODE+[PARENT/'manifest.json',PARENT/'scoring-inputs.jsonl',PARENT/'patch-plan.json',PARENT/'execution-plan.json',
                  PARENT/'model-profile.json',PARENT/'analysis-references.json',decision,
                  ROOT/'src/diagnostics/hehe_presentation_inputs_v1.py',ROOT/'src/diagnostics/hehe_presentation_runtime_v1.py',ROOT/'src/diagnostics/case_attention_inputs_v1.py',
                  ROOT/'src/diagnostics/case_attention_capture_v1.py',ROOT/'src/diagnostics/cross_model_applicability_models_v1.py',
                  ROOT/'src/diagnostics/cross_model_applicability_execution_v1.py']+framework_sources()
    write(out/'source-ledger.json',{'files':[info(p) for p in sorted(set(sources))]})
    return {'inputs':4,'cross':n,'primary':len(primary),'self':selfn,'budget':plan['budget']}


def validate(prepared,sealed=True,weights=False):
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
    p=Path(prepared)
    if sealed:
        manifest=read(p/'manifest.json');require(manifest['schema_version']=='hehe-focal-patching-prepared/v1','Wrong freeze')
        for x in manifest['artifacts']+manifest['sources']:verify(x)
    for x in read(p/'source-ledger.json')['files']:verify(x)
    plan,profile=read(p/'execution-plan.json'),read(p/'model-profile.json')
    require(plan['runtime_versions']==runtime_versions(),'Runtime changed')
    require(plan['acceptance']==read(PARENT/'execution-plan.json')['acceptance'] and plan['state_scaled_cap']==STATE_SCALED_CAP,'Numerical gates changed')
    for x in profile['metadata_sources']:verify(x)
    for x in profile['weight_sources']:
        if weights:verify(x)
        else:
            st=Path(x['path']).stat();require((st.st_size,st.st_mtime_ns)==(x['bytes'],x['mtime_ns']),'Checkpoint changed')
    rows=lines(p/'scoring-inputs.jsonl');require(len(rows)==4,'Wrong prompt inventory')
    parents={r['request_id']:r for r in lines(PARENT/'scoring-inputs.jsonl')}
    for r in rows:
        old=parents[r['parent_request_id']]
        actual={k:v for k,v in r.items() if k not in ['parent_request_id','patch_position_sets','capture_positions','capture_prefix_length','request_id']}
        require(actual=={k:v for k,v in old.items() if k!='request_id'},'Adopted prompt changed')
        require(r['patch_position_sets']==position_sets(r,plan['position_mode']),'Position selection changed')
        require(r['capture_positions']==sorted({p for ps in r['patch_position_sets'].values() for p in ps}),'Capture positions changed')
        require(r['capture_prefix_length']==max(r['capture_positions'])+1,'True prefix changed')
    cross,selfs=make_jobs(rows,plan['position_mode'],profile['layers'])
    require(read(p/'interventions.json')=={'cross':cross,'self':selfs},'Intervention ledger changed')
    return plan,profile,rows,cross,selfs


def seal(prepared):
    p=Path(prepared).absolute();require(not (p/'manifest.json').exists(),'Already sealed')
    validate(p,sealed=False)
    for name in ['capture-cpu-tests.json','runtime-cpu-tests.json','cpu-audit.json','viewer-cpu-test.json']:
        r=read(p/name);require(r['status']=='pass' and not r.get('CUDA_initialized',False),'Required CPU check failed')
        for x in r.get('implementation_snapshot',[]):verify(x)
    write(p/'manifest.json',{'schema_version':'hehe-focal-patching-prepared/v1','status':'CPU_complete_GPU_pending',
                            'artifacts':[info(x) for x in sorted(p.rglob('*')) if x.is_file()],
                            'sources':read(p/'source-ledger.json')['files'],'immutable_after_seal':True})
    PUBLIC.mkdir(parents=True,exist_ok=True)
    atomic(PUBLIC/'current.json',{'status':'accepted_prepared_GPU_pending','directory':str(p.relative_to(ROOT)),
                                 'manifest':info(p/'manifest.json'),'GPU_started':False})
    return info(p/'manifest.json')
