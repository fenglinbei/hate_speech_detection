"""Bounded, fail-closed GPU controller. Imports never load weights or inspect CUDA.

The same execute() scheduler is exercised by a synthetic CPU backend. Scientific
records are full vectors; scores/calibration are joined only after worker release.
"""
from __future__ import annotations
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from diagnostics import reference_routing_content_inputs_v1 as c
from diagnostics import reference_routing_content_capture_v1 as cap
from diagnostics.cross_model_applicability_execution_v1 import readout, gpu_inventory, runtime_versions
from diagnostics.hehe_presentation_runtime_v1 import idle

PASSES=('probe','raw','repeat','reverse','left','right')


def decision(prepared,path,stage):
    # This test must precede all hardware inventory, checkpoint verification/loading,
    # output-directory creation and subprocess launch.
    c.require(path is not None,'GPU execution requires a new explicit user authorization; current request is preparation only')
    d=c.read(path)
    c.require(d.get('GPU_execution_authorized') is True and d.get('user_message','').strip(),'Missing explicit GPU authorization')
    c.require(d.get('prepared_manifest')==c.info(Path(prepared)/'manifest.json'),'Authorization preparation identity differs')
    c.require(stage in d.get('allowed_stages',[]),'Stage is not authorized')
    c.require(d.get('window_confirmed') is True,'Execution window must be recorded')
    end=d.get('deadline_unix')
    c.require(end is None or time.time()<end-120,'Authorized window expired')
    return d


def calibration_gate(prepared,path):
    c.require(path is not None,'Confirmation requires a sealed development calibration lock')
    m=c.verify_manifest(path);d=c.read(Path(path).parent/'parameters.json')
    c.require(m['schema_version']=='reference-routing-content-calibration/v1','Wrong calibration manifest')
    c.require(d['prepared_manifest']==c.info(Path(prepared)/'manifest.json') and d['split']=='development','Calibration source/split changed')
    c.require(d['confirmation_outputs_seen'] is False and d['QAS_factor'] in (1,2,4),'Invalid confirmation lock')
    c.verify(d['development_audit']);audit=c.read(d['development_audit']['path'])
    c.require(audit['status']=='PASS' and audit['stage']=='development' and not audit['synthetic'],'Unqualified development data')
    c.verify(d['development_state']);s=c.read(d['development_state']['path'])
    c.require(s['status']=='COMPLETE' and s['owned_worker_absent'],'Development GPU must be released before locking calibration')
    return d


def load_group(prepared,stage,calibration=None):
    group=stage
    if stage=='confirmation':group='confirmation-factor-'+str(calibration_gate(prepared,calibration)['QAS_factor'])
    c.require(group in c.read(Path(prepared)/'execution-plan.json')['budgets'],'Unknown stage')
    p=Path(prepared)/group
    return group,c.lines(p/'inputs.jsonl'),c.read(p/'jobs.json')


def ordered(jobs,reverse=False):
    groups=[[],[],[]]
    for j in jobs:
        groups[0 if j['kind']=='native' else 2 if j.get('av_override') else 1].append(j)
    return [j for group in groups for j in (list(reversed(group)) if reverse else group)]


def save_array(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:np.save(f,value,allow_pickle=False)
    return c.info(p)


def save_bundle(path,value):
    if value is None:return None
    arrays={k:v for k,v in value.items() if isinstance(v,np.ndarray)}
    meta={k:v for k,v in value.items() if not isinstance(v,np.ndarray)}
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:np.savez(f,**arrays)
    return {'array_file':c.info(p),'metadata':meta}


def bundle(item):
    if item is None:return None
    c.verify(item['array_file'])
    with np.load(item['array_file']['path'],allow_pickle=False) as z:result={k:z[k] for k in z.files}
    result.update(item['metadata']);return result


def recpath(run,passname,jid):return Path(run)/'records'/passname/(jid+'.json')


def load_record(run,passname,jid):
    path=recpath(run,passname,jid);r=c.read(path)
    c.require(r['job_id']==jid and r['pass']==passname,'Record identity differs')
    c.verify(r['vector']);v=np.load(r['vector']['path'],allow_pickle=False)
    c.require(v.dtype==np.float32 and v.ndim==1 and np.isfinite(v).all(),'Invalid vector')
    bank=None
    if r['bank']:
        c.verify(r['bank']);bank=np.load(r['bank']['path'],allow_pickle=False)
        c.require(bank.dtype==np.float32 and np.isfinite(bank).all(),'Invalid donor bank')
    return r,v,bank,bundle(r['trajectory']),bundle(r['av'])


class Backend:
    """Model forward only; the scheduler never passes labels or relation records."""
    def __init__(self,model):self.model=model
    def forward(self,request,pad_token,**kw):return cap.forward(self.model,request,pad_token,**kw)
    def memory(self):
        import torch
        return {'allocated_peak_bytes':torch.cuda.max_memory_allocated(),
                'reserved_peak_bytes':torch.cuda.max_memory_reserved()}


def execute(prepared,run,stage,backend,profile,requests,registry,*,checkpoint=lambda:None,synthetic=False):
    """No restart/reuse path. A failure is terminal and all produced bytes remain."""
    run=Path(run);jobs=registry['science'];controls=registry['controls'];by={r['request_id']:r for r in requests}
    counts={'forwards':0,'valid_tokens':0,'tensor_tokens':0,'elapsed_forward_seconds':0.,'donor_captures':0,'donor_reuses':0}
    identity=c.info(Path(prepared)/'manifest.json')
    cache={}
    def fetch(passname,jid):
        key=(passname,jid)
        if key not in cache:cache[key]=load_record(run,passname,jid)
        return cache[key]
    def save(passname,j,append=(),prefix=False):
        checkpoint();req=by[j['recipient']];jid=j['job_id'];native=j['kind']=='native'
        source_pass='production' if passname=='production' else 'probe'
        if passname in ('left','right'):padding=passname
        else:padding='none'
        is_format=bool(append)
        observed=not (passname=='raw' or is_format)
        pos=req['capture_positions'] if native else []
        replacement=None;layer=None;sources=[]
        if j.get('upstream'):
            u=j['upstream'];layer=u['layer'];pos=u['positions']
            dr,_,bank,_,_=fetch(source_pass,u['donor']+'/N')
            d=by[u['donor']];replacement=cap.donor_rows(bank,d['capture_positions'],layer,u['donor_positions'])
            sources.append({'kind':'block_donor','record':c.info(recpath(run,source_pass,u['donor']+'/N')),
                            'bank':dr['bank'],'positions':u['donor_positions']})
            counts['donor_reuses']+=1
        av_sources=None
        if j.get('av_override'):
            av_sources=[]
            for field in ('A_from','V_from'):
                source=j['av_override'][field];sr,_,_,_,data=fetch(source_pass,source)
                c.require(data is not None,'Missing same-prompt A/V bank')
                av_sources.append(data);sources.append({'kind':field,'record':c.info(recpath(run,source_pass,source)),'av':sr['av']})
        kw={'positions':pos,'observe':native and observed,'layer':layer,'replacement':replacement,'padding':padding,
            'prefix_length':req['capture_prefix_length'] if prefix else None,'append_tokens':append,
            'trajectory':stage=='stage-a' and observed and not prefix,
            'capture_av':stage=='stage-a' and observed and not prefix,
            'av_sources':av_sources,'qas_factor':j.get('qas_factor',1)}
        # B self control factor=1 explicitly traverses this adapter's identity path.
        start=time.monotonic();v,bank,proof,trajectory,av,operations=backend.forward(req,profile['pad_token_id'],**kw)
        elapsed=time.monotonic()-start
        c.require(v.shape==(profile['vocab_size'],) and v.dtype==np.float32 and np.isfinite(v).all(),'Backend full-vector contract')
        from diagnostics.cross_model_applicability_models_v1 import prepared_input
        used=req['input_ids'][:req['capture_prefix_length']] if prefix else req['input_ids']+list(append)
        tensor_length=len(prepared_input(used,profile['pad_token_id'],padding)['input_ids'])
        counts['forwards']+=1;counts['valid_tokens']+=len(used);counts['tensor_tokens']+=tensor_length
        counts['elapsed_forward_seconds']+=elapsed;counts['donor_captures']+=int(bank is not None)
        base=run/'arrays'/passname/jid
        r={'job_id':jid,'pass':passname,'job':j,'request_id':req['request_id'],'prepared_manifest':identity,
           'prompt_sha256':req['prompt_sha256'],'input_ids_sha256':req['input_ids_sha256'],
           'synthetic':synthetic,'padding':padding,'append_tokens':list(append),'prefix':prefix,
           'vector':save_array(base.with_suffix('.vector.npy'),v),
           'bank':save_array(base.with_suffix('.bank.npy'),bank) if bank is not None else None,
           'trajectory':save_bundle(base.with_suffix('.trajectory.npz'),trajectory),
           'av':save_bundle(base.with_suffix('.av.npz'),av),'patch_proof':proof,'operations':operations,
           'sources':sources,'readout':None if prefix else readout(v,profile['candidate_tokens']),
           'trajectory_validation':cap.bridge.validate_trajectory(trajectory,req,profile,v) if trajectory is not None else None,
           'elapsed_seconds':elapsed,'valid_tokens':len(used),'tensor_tokens':tensor_length}
        c.write(recpath(run,passname,jid),r)
        if passname in ('probe','production'):cache[passname,jid]=(r,v,bank,trajectory,av)
        c.atomic(run/'progress.json',dict(counts,passname=passname,current_job=jid),replace=True)
        return r,v,bank,trajectory,av
    for passname in PASSES:
        for j in ordered(jobs,reverse=passname=='reverse'):save(passname,j)
    for j in jobs:
        if j['kind']=='native':save('prefix',j,prefix=True)
    for j in controls:save('controls',j)
    q=qualify(prepared,run,stage,profile,requests,registry,synthetic=synthetic)
    c.write(run/'qualification.json',q)
    for j in jobs:
        checkpoint();first=fetch('probe',j['job_id'])[1];tokens=[int(first.argmax())];records=[]
        while len(tokens)<8 and tokens[-1] not in profile['eos_token_ids']:
            r,v,*_=save('format-'+str(len(tokens)),j,append=tokens)
            records.append(c.info(recpath(run,r['pass'],j['job_id'])));tokens.append(int(v.argmax()))
        exact=len(tokens)==2 and tokens[0] in profile['candidate_tokens'].values() and tokens[1] in profile['eos_token_ids']
        path=run/'format'/(j['job_id']+'.json')
        c.write(path,{'job_id':j['job_id'],'tokens':tokens,'steps':records,'exact_label_then_eos':exact})
        c.require(exact,'Format qualification failed; no retry or sample filtering')
    for j in ordered(jobs):
        actual=save('production',j);base=fetch('probe',j['job_id'])
        compare(actual,base,exact=True,profile=profile)
    stats=dict(counts,**backend.memory(),synthetic=synthetic,stage=stage)
    c.write(run/'cost.json',stats)
    records=[c.info(p) for p in sorted((run/'records').rglob('*.json'))]
    c.write(run/'raw-seal.json',{'prepared_manifest':identity,'stage':stage,'synthetic':synthetic,
        'records':records,'formats':[c.info(p) for p in sorted((run/'format').rglob('*.json'))],
        'qualification':c.info(run/'qualification.json'),'cost':c.info(run/'cost.json')})
    return stats


def difference(a,b):return float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)),initial=0))


def compare(actual,base,*,exact,profile):
    r,v,bank,t,av=actual;br,bv,bbank,bt,bav=base
    vector=difference(v,bv);margin=abs(readout(v,profile['candidate_tokens'])['m']-readout(bv,profile['candidate_tokens'])['m'])
    c.require(vector==0 if exact else margin<=.001,'Full vector exact replay or padding margin gate failed')
    result={'vector_abs':vector,'margin_abs':margin}
    if bank is not None and bbank is not None:
        e=cap.bridge.scaled_error(bank,bbank);c.require(e['absolute']==0 if exact else e['scaled']<=.0001,'Bank invariance failed');result['bank']=e
    if t is not None and bt is not None:
        result['trajectory']=cap.bridge.compare_trajectories(t,bt,'repeat' if exact else 'left')
    if av is not None and bav is not None:
        errors={}
        for key in ('Q','K','V','scores','A','head_AV','o_proj'):
            e=cap.bridge.scaled_error(av[key],bav[key]);errors[key]=e
            c.require(e['absolute']==0 if exact else e['scaled']<=.0001,'AV telemetry invariance failed '+key)
        l1=np.abs(av['A'].astype(float)-bav['A'].astype(float)).sum(-1).max()
        c.require(l1<=(0 if exact else .001),'Attention row L1 changed');result['av']=errors
    return result


def qualify(prepared,run,stage,profile,requests,registry,synthetic=False):
    by={r['request_id']:r for r in requests};bounds={};checks=[];structures=[]
    alljobs=registry['science']+registry['controls']
    for j in registry['science']:
        jid=j['job_id'];base=load_record(run,'probe',jid);errors=[]
        for passname in PASSES[1:]:
            actual=load_record(run,passname,jid)
            result=compare(actual,base,exact=passname in ('raw','repeat','reverse'),profile=profile)
            checks.append({'job_id':jid,'pass':passname,**result});errors.append(result['margin_abs'])
        bounds[jid]=max(1e-6,2*max(errors))
        if j['kind']=='native':
            prefix=load_record(run,'prefix',jid)
            error=cap.bridge.scaled_error(prefix[2],base[2]);c.require(error['scaled']<=.0001,'True-prefix donor invariance failed')
            checks.append({'job_id':jid,'pass':'prefix','bank':error})
        if j.get('upstream') and stage=='stage-a':
            n=load_record(run,'probe',j['recipient']+'/N')
            cap.bridge.structural_guard(base[3],n[3],17)
            if j['kind'] in ('upstream','preceding'):
                structures.append({'job_id':jid,**cap.structure(n[4],base[4],j['upstream']['positions'])})
    for j in alljobs:
        kind=j['kind'];target=None;exact=True
        if kind=='av_u_self':target=j['recipient']+'/U';exact=False
        elif kind in ('av_native_self','focal_native_self','qas_native_self'):target=j['recipient']+'/N';exact=kind!='av_native_self'
        if target:
            actual=load_record(run,'controls' if j in registry['controls'] else 'probe',j['job_id'])
            base=load_record(run,'probe',target);v=actual[1];bv=base[1]
            c.require(difference(v,bv)<=(0 if exact else .001),'Self full-vector reconstruction failed')
            if actual[3] is not None:cap.bridge.compare_trajectories(actual[3],base[3],'self' if exact else 'left')
            error=abs(readout(v,profile['candidate_tokens'])['m']-readout(bv,profile['candidate_tokens'])['m'])
            c.require(error<=(0 if exact else .001),'Self margin reconstruction failed')
            bounds[target]=max(bounds[target],2*error)
            if j['job_id'] in bounds:bounds[j['job_id']]=max(bounds[j['job_id']],2*error)
            checks.append({'self_job':j['job_id'],'target':target,'margin_abs':error,'exact':exact})
    if stage=='stage-a' and not synthetic:
        for previous in c.read(Path(prepared)/'historical-replay.json')['records']:
            for k in ('record','vector','states','trajectory'):
                if previous[k]:c.verify(previous[k])
            current=load_record(run,'probe',previous['job_id'])
            c.require(np.array_equal(current[1],np.load(previous['vector']['path'],allow_pickle=False)),'Historical full vocabulary replay failed')
            if previous['states']:c.require(np.array_equal(current[2],np.load(previous['states']['path'],allow_pickle=False)),'Historical bank replay failed')
            with np.load(previous['trajectory']['path'],allow_pickle=False) as z:
                cap.bridge.compare_trajectories(current[3],{k:z[k] for k in z.files},'repeat')
            checks.append({'historical_replay_exact':previous['job_id']})
    return {'status':'PASS','stage':stage,'synthetic':synthetic,'per_job_margin_bounds':bounds,
            'comparisons':checks,'structural_guards':structures,'bounds_are_engineering_not_statistical':True}


def audit_run(prepared,run,stage,calibration=None,synthetic=False,fixture=None):
    run=Path(run)
    if fixture:profile,requests,registry=fixture;group=stage
    else:
        _,profile=c.validate(prepared);group,requests,registry=load_group(prepared,stage,calibration)
    seal=c.read(run/'raw-seal.json');c.require(seal['prepared_manifest']==c.info(Path(prepared)/'manifest.json') and seal['stage']==stage and seal['synthetic']==synthetic,'Raw seal identity changed')
    for item in seal['records']+seal['formats']+[seal['qualification'],seal['cost']]:c.verify(item)
    expected={(pas,j['job_id']) for pas in PASSES+('production',) for j in registry['science']}
    expected|={('prefix',j['job_id']) for j in registry['science'] if j['kind']=='native'}
    expected|={('controls',j['job_id']) for j in registry['controls']}
    for j in registry['science']:
        fmt=c.read(run/'format'/(j['job_id']+'.json'));base=load_record(run,'probe',j['job_id'])
        tokens=[int(base[1].argmax())]
        for item in fmt['steps']:
            c.verify(item);r=c.read(item['path']);expected.add((r['pass'],j['job_id']))
            c.require(r['append_tokens']==tokens,'Continuation token chain changed')
            v=load_record(run,r['pass'],j['job_id'])[1];tokens.append(int(v.argmax()))
        c.require(tokens==fmt['tokens'] and len(tokens)==2 and tokens[0] in profile['candidate_tokens'].values() and tokens[1] in profile['eos_token_ids'] and fmt['exact_label_then_eos'],'Format gate reconstruction failed')
        compare(load_record(run,'production',j['job_id']),base,exact=True,profile=profile)
    actual=[];valid=tensor=0
    by={r['request_id']:r for r in requests};jobs={j['job_id']:j for j in registry['science']+registry['controls']}
    for item in seal['records']:
        r=c.read(item['path']);actual.append((r['pass'],r['job_id']))
        c.require(r['prepared_manifest']==seal['prepared_manifest'] and r['synthetic']==synthetic,'Record preparation/origin changed')
        job=jobs[r['job_id']];req=by[job['recipient']]
        c.require(r['job']==job and r['request_id']==req['request_id'] and r['prompt_sha256']==req['prompt_sha256'] and r['input_ids_sha256']==req['input_ids_sha256'],'Record input/job binding differs')
        loaded=load_record(run,r['pass'],r['job_id']);v=loaded[1]
        c.require(r['readout']==(None if r['prefix'] else readout(v,profile['candidate_tokens'])),'Saved readout differs from full vector')
        if not synthetic:validate_proof(run,r,req,job,by,profile,loaded)
        valid+=r['valid_tokens'];tensor+=r['tensor_tokens']
    c.require(len(actual)==len(set(actual)) and set(actual)==expected,'Missing/extra/duplicated forwards')
    q=qualify(prepared,run,stage,profile,requests,registry,synthetic)
    c.require(q==c.read(run/'qualification.json'),'Qualification reconstruction differs')
    cost=c.read(run/'cost.json')
    c.require((cost['forwards'],cost['valid_tokens'],cost['tensor_tokens'])==(len(actual),valid,tensor),'Cost ledger differs')
    return {'status':'PASS','stage':stage,'group':group,'synthetic':synthetic,'forwards':len(actual),
            'raw_seal':c.info(run/'raw-seal.json'),'physical_score_records':len(registry['science']),
            'query_reference_join_performed':False,'full_matrix_and_format_reconstructed':True}


def validate_proof(run,r,req,job,by,profile,loaded):
    from diagnostics.cross_model_applicability_models_v1 import prepared_input
    source_pass='production' if r['pass']=='production' else 'probe'
    ids=req['input_ids'][:req['capture_prefix_length']] if r['prefix'] else req['input_ids']+r['append_tokens']
    padded=prepared_input(ids,profile['pad_token_id'],r['padding']);valid=[i for i,x in enumerate(padded['attention_mask']) if x]
    c.require((r['valid_tokens'],r['tensor_tokens'])==(len(ids),len(padded['input_ids'])),'Token cost differs from actual geometry')
    sources=[]
    if job['upstream']:
        u=job['upstream'];dr,_,bank,_,_=load_record(run,source_pass,u['donor']+'/N');d=by[u['donor']]
        rows=cap.donor_rows(bank,d['capture_positions'],u['layer'],u['donor_positions'])
        proof=r['patch_proof']
        c.require(proof['layer']==u['layer'] and proof['positions']==u['positions'] and proof['padded_positions']==[valid[i] for i in u['positions']],'Block patch coordinates differ')
        c.require(proof['donor_sha256']==cap.array_sha(rows) and proof['outside_rows_exact'] and proof['donor_rows_exact'] and proof['native_output_unmodified'],'Block patch proof failed')
        c.require(set(proof['changed_rows_in_unpadded_coordinates'])<=set(u['positions']),'Outside block rows changed')
        sources.append({'kind':'block_donor','record':c.info(recpath(run,source_pass,u['donor']+'/N')),'bank':dr['bank'],'positions':u['donor_positions']})
    else:c.require(r['patch_proof'] is None,'Unexpected block patch')
    if job['av_override']:
        av=[]
        for field in ('A_from','V_from'):
            source=job['av_override'][field];sr,_,_,_,data=load_record(run,source_pass,source)
            c.require(data['prompt_sha256']==req['prompt_sha256'] and data['input_ids_sha256']==req['input_ids_sha256'],'AV source prompt mismatch')
            av.append(data);sources.append({'kind':field,'record':c.info(recpath(run,source_pass,source)),'av':sr['av']})
        operations=[x for x in r['operations'] if x['kind']=='AV'];c.require(len(operations)==1,'AV operation count')
        op=operations[0]
        c.require(op['layer']==18 and op['position']==req['prompt_tokens']-1 and op['padded_position']==valid[req['prompt_tokens']-1] and op['outside_rows_exact'],'AV output coordinate proof')
        c.require(op['A_sha256']==cap.array_sha(av[0]['A']) and op['V_sha256']==cap.array_sha(av[1]['V']),'AV content identity proof')
        if loaded[4] is not None:
            c.require(op['installed_sha256']==cap.array_sha(loaded[4]['installed_o_proj']),'Installed attention branch digest')
            if job['av_override']['cell'] in ('00','11'):
                native=av[0]['o_proj'];installed=loaded[4]['installed_o_proj']
                c.require(cap.bridge.scaled_error(native,installed)['scaled']<=.0001 and difference(native,installed)<=.001,'00/11 p branch reconstruction failed')
    else:c.require(not any(x['kind']=='AV' for x in r['operations']),'Unexpected AV override')
    qs=[x for x in r['operations'] if x['kind']=='QAS']
    factor=job.get('qas_factor',1)
    c.require([x['layer'] for x in qs]==(list(range(18,36)) if factor!=1 else []),'QAS layers changed')
    for op in qs:
        c.require(op['factor']==factor and op['position']==req['prompt_tokens']-1 and op['query_keys']==req['roles']['query_all'] and op['outside_scores_exact'] and op['mask_preserved'] and op['V_native'],'QAS geometry/content proof failed')
    c.require(r['sources']==sources,'Intervention provenance differs')
    if loaded[3] is not None:
        c.require(cap.bridge.validate_trajectory(loaded[3],req,profile,loaded[1])==r['trajectory_validation'],'Saved trajectory checks differ')
    if loaded[4] is not None:
        av=loaded[4];n=req['prompt_tokens'];shapes={'Q':(profile['heads'],profile['head_dim']),
            'K':(profile['kv_heads'],n,profile['head_dim']),'V':(profile['kv_heads'],n,profile['head_dim']),
            'scores':(profile['heads'],n),'A':(profile['heads'],n),'head_AV':(profile['heads'],profile['head_dim']),
            'o_proj':(profile['hidden_size'],)}
        for key,shape in shapes.items():c.require(av[key].shape==shape and av[key].dtype==np.float32 and np.isfinite(av[key]).all(),'Invalid AV array '+key)
        c.require(np.max(np.abs(av['A'].sum(-1,dtype=np.float64)-1))<=2e-6 and (av['A']>=0).all(),'AV probability normalization')


def launch(prepared,authorization,run,stage,calibration=None):
    prepared=Path(prepared).absolute()
    d=decision(prepared,authorization,stage)
    c.require(run is not None,'A new run directory is required')
    run=Path(run).absolute()
    plan,profile=c.validate(prepared,weights=True)
    group,requests,registry=load_group(prepared,stage,calibration)
    c.require(run.parent==c.WORK and not run.exists(),'Use a new isolated run; no restart')
    inventory=gpu_inventory()
    available=[x for x in inventory['devices'] if x['index'] in d.get('allowed_gpu_indices',[0,1,2,3]) and x['total_mib']>=44000
        and x['used_mib']==0 and x['utilization']==0 and not any(x['uuid'] in row for row in inventory['compute_processes'])]
    c.require(available,'No freshly idle permitted GPU; no worker launched')
    device=min(available,key=lambda x:x['index']);allocation=[{k:device[k] for k in ('index','uuid','name','total_mib')}]
    lockdir=c.WORK/'.locks';lockdir.mkdir(exist_ok=True)
    with (lockdir/(device['uuid']+'.lock')).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        idle(allocation,gpu_inventory());decision(prepared,authorization,stage)
        run.mkdir()
        binding={'prepared_manifest':c.info(prepared/'manifest.json'),'decision':c.info(authorization),'stage':stage,'group':group,
                 'calibration':c.info(calibration) if calibration else None,'allocation':allocation,'runtime_versions':runtime_versions()}
        c.write(run/'binding.json',binding)
        c.write(run/'state.json',{'status':'LAUNCHING','stage':stage,'controller_pid':os.getpid(),'owned_worker_absent':False})
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=device['uuid'],CUBLAS_WORKSPACE_CONFIG=':4096:8',USE_TORCH='1',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0')
        cmd=[sys.executable,str(c.ROOT/'scripts/review/run_reference_routing_content_v1.py'),'_worker','--prepared',str(prepared),
             '--run',str(run),'--stage',stage]
        with (run/'gpu.log').open('xb',buffering=0) as log:
            try:
                proc=subprocess.Popen(cmd,cwd=c.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            except BaseException as exc:
                state=c.read(run/'state.json');state.update(status='FAILED_TERMINAL',owned_worker_absent=True,
                    worker_created=False,failure=repr(exc))
                c.atomic(run/'state.json',state,replace=True);raise
            end=d.get('deadline_unix');term=False;killed=False
            try:
                while proc.poll() is None:
                    now=time.time()
                    if end and now>=end-120:(run/'STOP').touch(exist_ok=True)
                    if end and now>=end-60 and not term:proc.terminate();term=True
                    if end and now>=end-20 and not killed:proc.kill();killed=True
                    time.sleep(.5)
            except BaseException:
                (run/'STOP').touch(exist_ok=True)
                proc.terminate()
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise
            finally:
                code=proc.wait();inventory=gpu_inventory()
                absent=not Path(f'/proc/{proc.pid}').exists() and not any(row.split(',')[-1].strip()==str(proc.pid) for row in inventory['compute_processes'])
                c.write(run/'release.json',{'worker_pid':proc.pid,'worker_exit_code':code,'owned_worker_absent':absent,
                    'inventory':inventory,'checked_at_unix':time.time()})
                state=c.read(run/'state.json');state.update(owned_worker_absent=absent,worker_exit_code=code,release=c.info(run/'release.json'))
                state['status']='COMPLETE' if code==0 and absent and state['status']=='COMPLETE_RELEASING' else 'FAILED_TERMINAL'
                c.atomic(run/'state.json',state,replace=True)
        c.require(code==0 and absent,'Worker did not complete normally; no automatic retry')
        result=audit_run(prepared,run,stage,calibration)
        c.write(run/'audit.json',result);return result


def worker(prepared,run,stage):
    run=Path(run);binding=c.read(run/'binding.json');c.verify(binding['decision'])
    d=decision(prepared,binding['decision']['path'],stage)
    c.require(binding['prepared_manifest']==c.info(Path(prepared)/'manifest.json') and binding['stage']==stage,'Wrong worker binding')
    c.require(binding['runtime_versions']==runtime_versions(),'Runtime versions changed')
    _,profile=c.validate(prepared,weights=True)
    if binding['calibration']:c.verify(binding['calibration'])
    calibration=binding['calibration']['path'] if binding['calibration'] else None
    group,requests,registry=load_group(prepared,stage,calibration)
    c.require(group==binding['group'] and c.read(run/'state.json')['status']=='LAUNCHING','Worker already started or group changed')
    state=c.read(run/'state.json');state.update(status='LOADING',worker_pid=os.getpid(),stage=stage)
    c.atomic(run/'state.json',state,replace=True)
    stopping=False
    def stop(*args):
        nonlocal stopping
        stopping=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    def checkpoint():
        c.require(not stopping and not (run/'STOP').exists(),'Run stopped; terminal, never restart automatically')
        end=d.get('deadline_unix');c.require(end is None or time.time()<end-120,'Deadline stop boundary reached')
    try:
        checkpoint()
        idle(binding['allocation'],gpu_inventory())
        from diagnostics.cross_model_applicability_models_v1 import load_checkpoint
        model,identity=load_checkpoint(profile,binding['allocation']);c.write(run/'runtime-identity.json',identity)
        state['status']='RUNNING';c.atomic(run/'state.json',state,replace=True)
        leases=[(Path(x['path']),Path(x['path']).stat()) for x in profile['metadata_sources']+profile['weight_sources']]
        def checked_checkpoint():
            checkpoint()
            for path,old in leases:
                now=path.stat()
                c.require((now.st_dev,now.st_ino,now.st_size,now.st_mtime_ns)==
                          (old.st_dev,old.st_ino,old.st_size,old.st_mtime_ns),'Checkpoint changed while loaded')
        execute(prepared,run,stage,Backend(model),profile,requests,registry,checkpoint=checked_checkpoint)
        state['status']='COMPLETE_RELEASING';c.atomic(run/'state.json',state,replace=True)
    except BaseException as exc:
        state.update(status='FAILED_TERMINAL',failure=repr(exc));c.atomic(run/'state.json',state,replace=True);raise
