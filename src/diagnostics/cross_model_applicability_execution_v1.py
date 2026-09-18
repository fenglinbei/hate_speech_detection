"""CPU contracts, checkpoint checks and arithmetic for cross-model applicability."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
SCIENCE=ROOT/'docs/research/experiment-plans/cross-model-applicability-v1/frozen-01'
WORK=ROOT/'reviews/cross-model-applicability-execution-v1'
PREPARED=WORK/'prepared-01'
PASSES=[('reference','none',False),('repeat','none',False),('left_padding','left',False),
        ('right_padding','right',False),('reverse_order','none',True)]
IMPLEMENTATION=[ROOT/'src/diagnostics'/n for n in ['cross_model_applicability_execution_v1.py','cross_model_applicability_models_v1.py']]
IMPLEMENTATION += [ROOT/'scripts/review'/n for n in ['run_cross_model_applicability_v1.py','test_cross_model_applicability_runtime_v1.py']]
LEGACY_SOURCES=[ROOT/'docs/research/experiment-plans/cross-term-behavior-discrimination-v1/frozen-01'/n
                for n in ['design.json','analysis-plan.json']]
LEGACY_SOURCES.append(ROOT/'docs/research/experiment-plans/cross-term-joint-v1/frozen-01/design.json')


def require(v,message):
    if not v: raise ValueError(message)
def read(p): return json.loads(Path(p).read_text())
def jsonl(p): return [json.loads(s) for s in Path(p).read_text().splitlines()]
def canonical(v): return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while block:=f.read(4*1024*1024): h.update(block)
    return h.hexdigest()
def info(p):
    p=Path(p).resolve()
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def verify(r):
    p=Path(r['path']);p=p if p.is_absolute() else ROOT/p
    require(p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],f'Changed pinned file: {p}')
def atomic(p,v,replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with temp.open('xb') as f:
        f.write(canonical(v)+b'\n');f.flush();os.fsync(f.fileno())
    try:
        if replace: os.replace(temp,p)
        else: os.link(temp,p);temp.unlink()
    finally:
        if temp.exists():temp.unlink()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write_jsonl(p,rows): Path(p).write_bytes(b''.join(canonical(r)+b'\n' for r in rows))


def runtime_versions():
    return {n:importlib.metadata.version(n) for n in ['torch','transformers','accelerate','safetensors','numpy','tokenizers','tiktoken']}


def framework_sources():
    # These are the executed model/loading paths; versions additionally bind the
    # installed distributions. Device kernels are recorded at actual execution.
    root=Path(importlib.metadata.distribution('transformers').locate_file('transformers'))
    paths=[root/p for p in ['modeling_utils.py','configuration_utils.py','modeling_rope_utils.py',
                            'models/qwen3/modeling_qwen3.py','models/qwen3/configuration_qwen3.py']]
    accel=Path(importlib.metadata.distribution('accelerate').locate_file('accelerate'))
    paths += [accel/p for p in ['big_modeling.py','hooks.py','utils/modeling.py']]
    return paths


def prepare(output=PREPARED):
    output=Path(output)
    require(not output.exists(),'New preparation directory required')
    manifest=read(SCIENCE/'manifest.json')
    for section in ['artifacts','sources']:
        for r in manifest[section]: verify(r)
    output.mkdir(parents=True)
    inventory=read(SCIENCE/'cpu-model-inventory.json')
    token_audit=read(SCIENCE/'cpu-tokenizer-audit.json')
    candidates={m['model_key']:m['candidate_tokens'] for m in token_audit['models']}
    conditions={r['condition_id']:r for r in read(SCIENCE/'conditions.json')['conditions']}
    legacy={r['condition_id']:r for r in jsonl(SCIENCE/'legacy-model-inputs.jsonl')}
    profiles=[];budgets=[]
    (output/'inputs').mkdir()
    for model in inventory['models']:
        key=model['model_key'];directory=Path(model['local_directory'])
        cfg=read(directory/'config.json')
        generation=read(directory/'generation_config.json') if (directory/'generation_config.json').exists() else {}
        eos=generation.get('eos_token_id',cfg.get('eos_token_id'));eos=[eos] if isinstance(eos,int) else eos
        require(bool(eos) and all(isinstance(i,int) for i in eos),'EOS identity missing')
        profile={'model_key':key,'local_directory':str(directory),'local_snapshot_sha256':model['local_snapshot_sha256'],
            'architecture':'chatglm' if key.startswith('glm') else 'qwen3','layers':model['layers'],
            'vocab_size':cfg.get('vocab_size',cfg.get('padded_vocab_size')),
            'candidate_tokens':candidates[key],'pad_token_id':151329 if key.startswith('glm') else 151643,
            'eos_token_ids':eos,'metadata_sources':model['metadata_sources'],'weight_sources':model['weight_sources'],
            'fp32_weight_bytes':model['fp32_stored_tensor_bytes_estimate'],
            'compute_dtype':'float32','readout_dtype':'float64','attention':'eager','batch_size':1,'use_cache':False,
            'max_input_tokens':8192,'allocation':None,'GPU_qualified':False}
        inputs=[]
        for frame in ['new-model-inputs','legacy-model-inputs']:
            for t in jsonl(SCIENCE/f'tokenized/{key}-{frame}.jsonl'):
                cid=t['condition_id']
                if frame=='new-model-inputs':
                    source_frame=conditions[cid]['analysis_role'];qid=conditions[cid]['query_id']
                else: source_frame=legacy[cid]['frame'];qid=None
                inputs.append({'request_id':cid,'condition_id':cid,'model_key':key,'source_frame':source_frame,
                    'query_id':qid,'prompt_sha256':t['prompt_sha256'],'input_ids':t['input_ids'],
                    'input_ids_sha256':hashlib.sha256(canonical(t['input_ids'])).hexdigest(),
                    'prompt_tokens':t['prompt_tokens'],'candidate_tokens':candidates[key]})
        require(len(inputs)==540 and len({r['request_id'] for r in inputs})==540,'Unexpected input inventory')
        write_jsonl(output/'inputs'/f'{key}.jsonl',inputs)
        production=([r['request_id'] for r in inputs if r['query_id'] is None] if key!='qwen3-8b' else [])
        production += [r['request_id'] for r in inputs if r['query_id'] is not None]
        probes=[]
        for qid in sorted({r['query_id'] for r in inputs if r['query_id']}):
            subset=[r for r in inputs if r['query_id']==qid]
            probes.append(max(subset,key=lambda r:(r['prompt_tokens'],r['request_id']))['request_id'])
        profile['format_probe_ids']=probes;profile['production_ids']=production
        profiles.append(profile)
        budgets.append({'model_key':key,'engineering_inputs':540,'engineering_forwards':2700,
                         'production_forwards':len(production),'format_probes':16,'format_extra_forward_cap':112})
    dump(output/'model-profiles.json',{'models':profiles})
    dump(output/'execution-plan.json',{'schema_version':'cross-model-applicability-execution/v1',
        'scientific_manifest':info(SCIENCE/'manifest.json'),'runtime_versions':runtime_versions(),
        'model_keys':[p['model_key'] for p in profiles],'passes':[{'pass_id':k,'padding':p,'reverse':r} for k,p,r in PASSES],
        'acceptance':{'repeat_and_order_margin_cap':0.0,'padding_margin_cap':0.001,'logprob_identity_cap':1e-10,
                      'margin_bound_floor':1e-6,'margin_bound_multiplier':2,
                      'policy':'New qualification per model and exact allocation. These preset engineering ceilings do not inherit any historical measured error bound.'},
        'format':{'max_new_tokens':8,'exact_single_candidate_then_native_eos':True,'all_16_must_pass':True,
                  'legal_mass':'Report every value; no unregistered filtering or accuracy gate.',
                  'first_vector':'Reuse engineering-reference full-vocabulary vector; count subsequent forwards separately.'},
        'budget':{'by_model':budgets,'engineering_forwards':8100,'new_scientific_forwards':1464,
                  'prompt_forward_total_without_generation':9564,'format_extra_forward_cap':336,
                  'reused_old_8b_scientific_inputs':156},
        'runtime':{'sampling':False,'seed':0,'deterministic_algorithms':True,'cublas_workspace_config':':4096:8',
                   'tf32':False,'CPU_or_disk_offload':False,'checkpoint_boundary':'one committed request',
                   'failed_run_retry':False,'terminal_run_restart':False},
        'phases':['engineering','full'],'full_requires_all_engineering_and_format_gates':True,
        'legacy_8b':'Its 156 engineering replays are diagnostic bridges only; never replace old scientific endpoints or their bounds.',
        'reference_join':'Only after raw production seal and owned worker exit. Scorer parses input/profile/plan files only.',
        'allocation':None,'GPU_qualification':False})
    prepare_legacy_analysis(output)
    sources=[SCIENCE/'manifest.json',*IMPLEMENTATION,*framework_sources(),*LEGACY_SOURCES]
    for f in manifest['sources']: sources.append(ROOT/f['path'] if not Path(f['path']).is_absolute() else Path(f['path']))
    # Bind metadata and checkpoint identities without rereading 63+ GB in this CPU
    # step. Full hashes are mandatory at actual bind/load and closeout.
    for p in profiles:
        for r in p['metadata_sources']: verify(r)
        for r in p['weight_sources']:
            st=Path(r['path']).stat();require((st.st_size,st.st_mtime_ns)==(r['bytes'],r['mtime_ns']),'Weight stat changed')
    dump(output/'source-ledger.json',{'files':[info(p) for p in sorted(set(sources))],
        'checkpoint_full_hashes_reused_from_prior_inventory':True,'checkpoint_full_hashes_required_at_load':True,
        'new_target_model_outputs_read':False,'reference_fields_sent_to_model':False})
    return {'status':'prepared_unsealed','directory':str(output),'budget':read(output/'execution-plan.json')['budget']}


def prepare_legacy_analysis(output):
    core=read(LEGACY_SOURCES[0])['conditions'];registered=read(LEGACY_SOURCES[1])
    historical=read(LEGACY_SOURCES[2])['conditions']
    conditions={r['condition_id']:r for r in historical}
    conditions.update({r['condition_id']:r for r in core})
    selected=[r['condition_id'] for r in jsonl(SCIENCE/'legacy-model-inputs.jsonl')]
    require(len(selected)==156 and set(selected)<=set(conditions),'Legacy reference mapping incomplete')
    families={r['query_id']:r['family_id'] for r in core}
    refs=[{'query_id':r['query_id'],'family_id':families[r['query_id']],
           'analysis_role':'legacy_exposed_development','adopted_label':r['human_reference']['task_label'],
           'adopted_severity':r['human_reference']['attack_severity'],'human_adoption':r['human_reference'],
           'original_gold':r['original_gold'],'original_correct':r['original_correct']} for r in registered['references']]
    comparisons=[dict(r,analysis_role='legacy_exposed_development') for r in registered['comparisons']]
    dump(Path(output)/'legacy-analysis.json',{'scorer_must_not_parse':True,'references':refs,'comparisons':comparisons,
        'condition_query_ids':{cid:conditions[cid]['query_id'] for cid in selected},
        'condition_frames':{cid:'legacy_core' if cid in {r['condition_id'] for r in core} else 'legacy_N_diagnostic' for cid in selected},
        'original_sources':[info(p) for p in LEGACY_SOURCES],
        'primary_conditions':120,'external_N_conditions':36,'independent_new_families':0})


def check_prepared(prepared=PREPARED,weights=False,sealed=True):
    prepared=Path(prepared)
    if sealed:
        m=read(prepared/'manifest.json')
        for r in m['artifacts']+m['sources']:verify(r)
    ledger=read(prepared/'source-ledger.json')
    for r in ledger['files']:verify(r)
    plan=read(prepared/'execution-plan.json')
    verify(plan['scientific_manifest'])
    scientific=read(plan['scientific_manifest']['path'])
    for section in ['artifacts','sources']:
        for r in scientific[section]:verify(r)
    require(plan['runtime_versions']==runtime_versions(),'Runtime package versions changed')
    profiles=read(prepared/'model-profiles.json')['models'];requests={}
    for p in profiles:
        key=p['model_key'];rs=jsonl(prepared/'inputs'/f'{key}.jsonl');requests[key]=rs
        require(len(rs)==len({r['request_id'] for r in rs})==540,'Input coverage differs')
        for r in rs:
            require(set(r)=={'request_id','condition_id','model_key','source_frame','query_id','prompt_sha256',
                'input_ids','input_ids_sha256','prompt_tokens','candidate_tokens'},'Scoring input field leak or missing field')
            require(r['candidate_tokens']==p['candidate_tokens'] and len(r['input_ids'])==r['prompt_tokens'],'Input identity mismatch')
            require(r['input_ids_sha256']==hashlib.sha256(canonical(r['input_ids'])).hexdigest(),'Token hash mismatch')
        require(len(p['format_probe_ids'])==len(set(p['format_probe_ids']))==16,'Format probe coverage differs')
        require(len(p['production_ids'])==(384 if key=='qwen3-8b' else 540),'Production count differs')
        for r in p['metadata_sources']:verify(r)
        for r in p['weight_sources']:
            if weights:verify(r)
            else:
                st=Path(r['path']).stat();require((st.st_size,st.st_mtime_ns)==(r['bytes'],r['mtime_ns']),'Weight stat changed')
    return plan,profiles,requests


def readout(vector,candidates,bound=None):
    import numpy as np
    v=np.asarray(vector)
    require(v.ndim==1 and v.dtype==np.float32 and np.isfinite(v).all(),'Raw vector must be finite FP32')
    ids=[candidates['有'],candidates['无']]
    require(len(set(ids))==2 and min(ids)>=0 and max(ids)<len(v),'Bad candidate IDs')
    if bound is not None:require(math.isfinite(bound) and bound>=0,'Invalid error bound')
    x=v.astype(np.float64);z_yes,z_no=(float(x[i]) for i in ids)
    top=float(x.max());lnz=top+math.log(float(np.exp(x-top).sum(dtype=np.float64)))
    m=z_no-z_yes;pair_top=max(z_no,z_yes)
    pair_lse=pair_top+math.log(math.exp(z_no-pair_top)+math.exp(z_yes-pair_top))
    return {'m':m,'z_yes':z_yes,'z_no':z_no,'log_p_yes':z_yes-lnz,'log_p_no':z_no-lnz,
        'log_legal_mass':pair_lse-lnz,'legal_mass':math.exp(pair_lse-lnz),
        'pair_support_no':math.exp(z_no-pair_lse),'raw_prediction':'无' if m>0 else '有' if m<0 else None,
        'margin_error_bound':bound,'resolution':resolution(m,bound)}


def resolution(value,bound):
    if bound is None:return 'unqualified'
    return 'positive' if value>bound else 'negative' if value< -bound else 'numerical_unresolved'


def linear_effect(terms,scores):
    combined={};values={}
    for t in terms:
        s=scores[t['condition_id']];key=s['physical_score_id']
        require(s['margin_error_bound'] is not None,'Unqualified endpoint')
        require(math.isfinite(s['m']) and math.isfinite(s['margin_error_bound']) and s['margin_error_bound']>=0,'Invalid endpoint')
        require(math.isfinite(t['coefficient']),'Invalid coefficient')
        val=(s['m'],s['margin_error_bound'],s.get('prompt_sha256'),s.get('qualification_ref'))
        require(key not in values or values[key]==val,'Aliased physical scores disagree')
        values[key]=val;combined[key]=combined.get(key,0)+t['coefficient']
    value=math.fsum(c*values[k][0] for k,c in combined.items() if c)
    bound=math.fsum(abs(c)*values[k][1] for k,c in combined.items() if c)
    return {'value':value,'bound':bound,'resolution':resolution(value,bound),
            'physical_terms':[{'physical_score_id':k,'coefficient':c} for k,c in sorted(combined.items()) if c]}


def qualify(values,rules):
    expected=set(values['reference'])
    require(expected and set(values)=={p[0] for p in PASSES},'Incomplete engineering passes')
    differences={};identity=0.0
    for name,rs in values.items():
        require(set(rs)==expected,'Engineering coverage differs')
        for r in rs.values():
            require(all(math.isfinite(r[k]) for k in ['m','z_no','z_yes','log_p_no','log_p_yes','legal_mass']),'Nonfinite readout')
            identity=max(identity,abs(r['m']-(r['log_p_no']-r['log_p_yes'])))
        if name!='reference':differences[name]=max(abs(rs[k]['m']-values['reference'][k]['m']) for k in expected)
    require(differences['repeat']<=rules['repeat_and_order_margin_cap'] and differences['reverse_order']<=rules['repeat_and_order_margin_cap'],'Repeat/order gate failed')
    require(max(differences['left_padding'],differences['right_padding'])<=rules['padding_margin_cap'],'Padding gate failed')
    require(identity<=rules['logprob_identity_cap'],'Log-probability identity gate failed')
    return {'status':'pass','unique_engineering_inputs':len(expected),'max_margin_differences':differences,
        'logprob_identity_error':identity,'margin_error_bound':max(rules['margin_bound_floor'],rules['margin_bound_multiplier']*max(differences.values())),
        'meaning':'Measured engineering reproducibility envelope for this model/allocation; not a statistical interval.'}


def gpu_inventory():
    cmd=['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits']
    out=subprocess.run(cmd,check=True,text=True,capture_output=True).stdout
    devices=[]
    for line in out.splitlines():
        parts=[v.strip() for v in line.split(',')]
        if len(parts)!=6:continue
        i,uuid,name,total,used,util=parts
        devices.append({'index':int(i),'uuid':uuid,'name':name,'total_mib':int(total),'used_mib':int(used),'utilization':int(util)})
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],check=True,text=True,capture_output=True).stdout
    return {'devices':devices,'compute_processes':[line.strip() for line in processes.splitlines() if line.strip()]}


def bind(prepared,key,gpus,path,authorization_note):
    require(bool(authorization_note),'An explicit GPU window authorization note is required')
    plan,profiles,_=check_prepared(prepared,weights=True)
    require(key in {p['model_key'] for p in profiles},'Unknown model')
    require(len(gpus) in [1,2] and len(set(gpus))==len(gpus),'One or two distinct GPUs required')
    inv=gpu_inventory();lookup={d['index']:d for d in inv['devices']};allocation=[]
    for gpu in gpus:
        require(gpu in lookup,'GPU absent');d=lookup[gpu]
        require(d['used_mib']==0 and d['utilization']==0 and not any(d['uuid'] in r for r in inv['compute_processes']),'GPU is not idle')
        allocation.append({k:d[k] for k in ['index','uuid','name','total_mib']})
    atomic(path,{'prepared_manifest':info(Path(prepared)/'manifest.json'),'model_key':key,'allocation':allocation,
                 'runtime_versions':plan['runtime_versions'],'authorization_note':authorization_note,'bound_at':now()})
    return info(path)


def validate_resume(state,binding,phase,resume):
    require(state['binding']==binding,'Resume binding differs')
    require((resume and state['status']=='paused') or (not resume and state['status']=='qualified' and phase=='full'),
            'Only explicit paused resume or qualified-to-full continuation is allowed; failed/complete runs are terminal')


def production_analysis(records,analysis_plan):
    """Pure analysis; production caller requires raw seals and worker release."""
    refs={r['query_id']:r for r in analysis_plan['references']}
    scores={r['condition_id']:dict(r['readout'],physical_score_id=r['physical_score_id'],
             prompt_sha256=r.get('prompt_sha256'),qualification_ref=r.get('qualification_ref')) for r in records}
    require(len(scores)==len(records),'Duplicate condition score')
    query_ids=dict(analysis_plan.get('condition_query_ids',{}))
    for comp in analysis_plan['comparisons']:
        for term in comp['terms']:
            cid=term['condition_id']
            require(cid not in query_ids or query_ids[cid]==comp['query_id'],'Cross-query endpoint alias')
            query_ids[cid]=comp['query_id']
    def label(score):return '无' if score['m']>0 else '有' if score['m']<0 else None
    def resolved(score):return abs(score['m'])>score['margin_error_bound']
    score_rows=[]
    for cid,s in scores.items():
        ref=refs[query_ids[cid]];answer=ref['adopted_label'];sign=1 if answer=='无' else -1
        require(s['margin_error_bound'] is not None,'Unqualified score in reference join')
        score_rows.append(dict(s,condition_id=cid,query_id=query_ids[cid],family_id=ref['family_id'],
            analysis_role=ref['analysis_role'],adopted_reference=answer,adopted_severity=ref['adopted_severity'],
            original_gold=ref['original_gold'],original_correct=ref['original_correct'],raw_prediction=label(s),
            exact_tie=s['m']==0,raw_correct=label(s)==answer,conservative_correct=resolved(s) and label(s)==answer,
            reference_aligned_margin=sign*s['m'],numeric_resolution=resolution(s['m'],s['margin_error_bound']),
            source_frame=analysis_plan.get('condition_frames',{}).get(cid,ref['analysis_role'])))
    expressions=[]
    for c in analysis_plan['comparisons']:
        e=linear_effect(c['terms'],scores);ref=refs[c['query_id']];sign=1 if ref['adopted_label']=='无' else -1
        transition=None
        if len(c['terms'])==2 and sorted(t['coefficient'] for t in c['terms'])==[-1,1]:
            a=next(scores[t['condition_id']] for t in c['terms'] if t['coefficient']==-1)
            b=next(scores[t['condition_id']] for t in c['terms'] if t['coefficient']==1)
            ca,cb=label(a)==ref['adopted_label'],label(b)==ref['adopted_label']
            transition=('unresolved_transition' if not (resolved(a) and resolved(b)) else
                        'repair' if not ca and cb else 'damage' if ca and not cb else 'stable_correct' if ca else 'stable_wrong')
        expressions.append(dict(c,effect=e,reference_aligned_change=sign*e['value'],
            reference_aligned_resolution=resolution(sign*e['value'],e['bound']),verified_classification_transition=transition))
    # Deliberately no mixed A/B or cross-model raw-margin average.
    summary=[]
    for role in sorted({c['analysis_role'] for c in expressions}):
        for kind in sorted({c['kind'] for c in expressions if c['analysis_role']==role}):
            subset=[c for c in expressions if c['analysis_role']==role and c['kind']==kind]
            families={}
            for c in subset:families.setdefault(refs[c['query_id']]['family_id'],[]).append(c)
            means=[math.fsum(c['reference_aligned_change'] for c in cs)/len(cs) for cs in families.values()]
            summary.append({'analysis_role':role,'kind':kind,'families':sorted(families),
                'family_equal_mean_reference_aligned_change':math.fsum(means)/len(means),
                'directions':dict(Counter(c['reference_aligned_resolution'] for c in subset)),
                'independent_sample_count':None})
    return {'scores':score_rows,'comparisons':expressions,'stratified_descriptive_summaries':summary,
            'confirmation_claim':False,'statistical_thresholds':None}
