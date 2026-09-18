#!/usr/bin/env python3
"""UTF-8-only continuation with explicitly verified, immutable engineering imports.

Derived from the frozen v1 runner; numerical/model/scientific code is unchanged.
Failed source runs remain terminal. A new run may reference their fully audited
engineering receipts under a separate source-pinned amendment and binding.
"""
from __future__ import annotations

import argparse
import fcntl
from functools import lru_cache
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from diagnostics import cross_model_applicability_execution_v1 as c

c.require(sys.flags.utf8_mode == 1, 'Launch this amendment with PYTHONUTF8=1')


def validate_amendment(bound):
    b=c.read(bound)
    c.require('runtime_amendment' in b and 'engineering_import' in b,'Explicit I/O amendment and import binding required')
    c.verify(b['runtime_amendment']);m=c.read(b['runtime_amendment']['path'])
    c.require(m['required_python_utf8_mode']==1,'Unexpected I/O amendment')
    for source in m['sources']:c.verify(source)
    c.require(m['entry_point']==c.info(Path(__file__).resolve()),'Amended entry point differs')
    c.verify(b['engineering_import'])
    return b


@lru_cache(maxsize=8)
def imported_engineering(run):
    run=Path(run)
    b=validate_amendment(run/'binding.json');m=c.read(b['engineering_import']['path'])
    for key in ['source_state','source_binding','source_identity','numerical_closeout']:
        c.verify(m[key])
    state=c.read(m['source_state']['path'])
    c.require(state['status']=='failed' and state['error_type']=='UnicodeDecodeError' and
              state['owned_worker_absent'] and state['worker_exit_code']==1,'Import is restricted to the documented released I/O failure')
    c.require(m['prepared_manifest']==b['prepared_manifest'] and m['model_key']==b['model_key']=='glm4-9b-chat','Import science/model differs')
    parent=c.read(m['source_binding']['path'])
    c.require(parent['allocation']==b['allocation'] and parent['runtime_versions']==b['runtime_versions'],'Import allocation/runtime differs')
    c.require(state['run_id']==m['source_run_id'] and state['new_forwards']==2700,'Source invocation coverage differs')
    c.require(c.read(m['numerical_closeout']['path'])['qualification_arithmetic']['status']=='pass','Unqualified imported numerical arithmetic')
    source=Path(m['source_directory']).resolve();entries={}
    for r in m['records']:
        stage,rid=r['stage'],r['request_id']
        c.require(stage in [p[0] for p in c.PASSES] and '/' not in rid and '\\' not in rid,'Illegal imported request')
        c.require(Path(r['receipt']['path'])==source/'scores'/stage/(rid+'.json') and
                  Path(r['vector']['path'])==source/'vectors'/stage/(rid+'.npy'),'Import escaped its original run')
        c.require((stage,rid) not in entries,'Duplicate imported request')
        entries[stage,rid]=r
    c.require(len(entries)==2700 and all(sum(stage==name for stage,_ in entries)==540 for name,_,_ in c.PASSES),'Incomplete engineering import')
    return m,entries


def paths(run,stage,rid):
    c.require('/' not in rid and '\\' not in rid and rid not in ['.','..'],'Unsafe request ID')
    _,entries=imported_engineering(str(Path(run).resolve()))
    if (stage,rid) in entries:
        entry=entries[stage,rid]
        return Path(entry['receipt']['path']),Path(entry['vector']['path'])
    return run/'scores'/stage/(rid+'.json'),run/'vectors'/stage/(rid+'.npy')


def check_record(run,stage,request,binding_sha,profile,bound=None,identity=None):
    import numpy as np
    rp,vp=paths(run,stage,request['request_id']);r=c.read(rp)
    imported,entries=imported_engineering(str(Path(run).resolve()))
    entry=entries.get((stage,request['request_id']))
    expected_binding=imported['source_binding']['sha256'] if entry else binding_sha
    if entry:
        c.verify(entry['receipt']);c.require(r['raw_logits']==entry['vector'],'Imported vector identity differs')
    c.require(r['binding_sha256']==expected_binding and r['model_key']==profile['model_key'],'Receipt binding differs')
    for key in ['request_id','condition_id','prompt_sha256','input_ids_sha256']:
        c.require(r[key]==request[key],'Receipt input identity differs')
    c.require(r['pass_id']==stage and r['candidate_tokens']==profile['candidate_tokens'],'Pass/candidate identity differs')
    from diagnostics.cross_model_applicability_models_v1 import prepared_input
    padding=next((padding for name,padding,_ in c.PASSES if name==stage),'none')
    c.require(r['prepared_input']==prepared_input(request['input_ids'],profile['pad_token_id'],padding),'Padding/answer-position receipt differs')
    producer_run=imported['source_run_id'] if entry else c.read(run/'state.json')['run_id']
    c.require(r['physical_score_id']==f"{producer_run}:{stage}:{request['request_id']}",'Physical score identity differs')
    c.require(Path(r['raw_logits']['path'])==vp.resolve(),'Unexpected vector path')
    c.verify(r['raw_logits']);c.verify(r['producer_identity'])
    if identity is not None:c.require(c.read(r['producer_identity']['path'])==identity,'Producer runtime changed')
    vector=np.load(vp,allow_pickle=False)
    c.require(len(vector)==profile['vocab_size'],'Vector vocabulary differs')
    expected=c.readout(vector,profile['candidate_tokens'],bound)
    c.require(expected==r['readout'],'Readout reconstruction differs')
    return r,vector


def save_vector(path,vector):
    import numpy as np
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        orphan=path.parent/'uncommitted';orphan.mkdir(exist_ok=True)
        os.replace(path,orphan/(path.stem+f'-{time.time_ns()}.npy'))
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('xb') as f:
        np.save(f,vector,allow_pickle=False);f.flush();os.fsync(f.fileno())
    os.link(temp,path);temp.unlink()


def seal_or_check(path,value):
    if path.exists():c.require(c.read(path)==value,'Previously sealed result changed')
    else:c.atomic(path,value)


def engineering_values(run,requests,binding_sha,profile,identity=None):
    values={};sources=[]
    for stage,_,_ in c.PASSES:
        values[stage]={}
        for request in requests:
            record,_=check_record(run,stage,request,binding_sha,profile,identity=identity)
            values[stage][request['request_id']]=record['readout']
            sources.append(c.info(paths(run,stage,request['request_id'])[0]))
    return values,sources


def validate_format_record(record,request,profile):
    c.require(record['prompt_sha256']==request['prompt_sha256'] and record['condition_id']==request['condition_id'],'Format input changed')
    tokens=record['generated_token_ids']
    c.require(record['passed']==(len(tokens)==2 and tokens[0] in profile['candidate_tokens'].values()
              and tokens[1] in profile['eos_token_ids']),'Format verdict differs')
    c.require(1<=len(tokens)<=8 and len(tokens)==len(record['steps']),'Generation length/coverage differs')
    import numpy as np
    for step,token in zip(record['steps'],tokens):
        c.verify(step['raw_logits'])
        v=np.load(step['raw_logits']['path'],allow_pickle=False)
        c.require(v.dtype==np.float32 and v.shape==(profile['vocab_size'],) and np.isfinite(v).all(),'Invalid generation vector')
        c.require(int(v.argmax())==token,'Generation greedy token differs')


def worker(prepared,bound,run,invocation,phase):
    from diagnostics import cross_model_applicability_models_v1 as models
    prepared,run=Path(prepared),Path(run)
    validate_amendment(bound)
    plan,profiles,requests=c.check_prepared(prepared,weights=True)
    b=c.read(bound);c.verify(b['prepared_manifest'])
    c.require(b['prepared_manifest']==c.info(prepared/'manifest.json'),'Bound preparation differs')
    profile=next(p for p in profiles if p['model_key']==b['model_key'])
    requests=requests[b['model_key']];lookup={r['request_id']:r for r in requests}
    state=c.read(run/'state.json');binding_sha=c.sha(run/'binding.json')
    c.require(state['status']=='launching' and state['invocations'][-1]['id']==invocation,'Unregistered worker')
    c.require(c.read(run/'binding.json')==b,'Run binding differs')
    state['worker_pid']=os.getpid();state['invocations'][-1]['worker_pid']=os.getpid()
    model=None;stop=False;new_forwards=0;reused=0
    def update(**values):
        state.update(values,updated_at=c.now());c.atomic(run/'state.json',state,replace=True)
    def on_signal(signum,frame):
        nonlocal stop
        stop=True
    class Paused(Exception):pass
    def checkpoint():
        if stop or (run/'STOP').exists():raise Paused('Completed receipts retained; stop before next request')
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,on_signal)
    try:
        update(status='loading_model');checkpoint()
        model,identity=models.load_checkpoint(profile,b['allocation'])
        if (run/'runtime-identity.json').exists():c.require(c.read(run/'runtime-identity.json')==identity,'Resume runtime/layout changed')
        else:c.atomic(run/'runtime-identity.json',identity)
        ip=run/'invocations'/f'{invocation}.json';c.atomic(ip,identity)
        producer=c.info(ip)
        leases=[(Path(r['path']),Path(r['path']).stat()) for r in profile['metadata_sources']+profile['weight_sources']]
        def lease_check():
            for path,previous in leases:
                current=path.stat()
                c.require((current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns)==
                          (previous.st_dev,previous.st_ino,previous.st_size,previous.st_mtime_ns),'Model file changed while loaded')
        for stage,padding,reverse in c.PASSES:
            checkpoint();lease_check();update(status='running_engineering',current_pass=stage)
            ordered=list(reversed(requests)) if reverse else requests
            for n,request in enumerate(ordered,1):
                checkpoint();rp,vp=paths(run,stage,request['request_id'])
                if rp.exists():check_record(run,stage,request,binding_sha,profile,identity=identity);reused+=1
                else:
                    prep,vector=models.forward_vector(model,profile['architecture'],request['input_ids'],profile['pad_token_id'],padding)
                    save_vector(vp,vector)
                    c.atomic(rp,dict({k:request[k] for k in ['request_id','condition_id','prompt_sha256','input_ids_sha256']},
                        model_key=profile['model_key'],pass_id=stage,binding_sha256=binding_sha,
                        physical_score_id=f"{state['run_id']}:{stage}:{request['request_id']}",candidate_tokens=profile['candidate_tokens'],
                        raw_logits=c.info(vp),producer_identity=producer,prepared_input=prep,
                        readout=c.readout(vector,profile['candidate_tokens']),scored_at=c.now()))
                    new_forwards+=1
                if n%24==0 or n==len(ordered):update(pass_completed=n,new_forwards=new_forwards,reused_requests=reused)
        values,sources=engineering_values(run,requests,binding_sha,profile,identity)
        q=c.qualify(values,plan['acceptance']);q.update(model_key=profile['model_key'],binding_sha256=binding_sha,sources=sources)
        seal_or_check(run/'numerical-qualification.json',q)
        update(status='format_diagnostic')
        format_records=[]
        for rid in profile['format_probe_ids']:
            checkpoint();request=lookup[rid];fp=run/'format'/f'{rid}.json'
            if fp.exists():
                record=c.read(fp);validate_format_record(record,request,profile)
            else:
                _,vector=check_record(run,'reference',request,binding_sha,profile,identity=identity)
                tokens=[];steps=[]
                for step in range(plan['format']['max_new_tokens']):
                    checkpoint()
                    if step==0:vp=paths(run,'reference',rid)[1]
                    else:
                        _,vector=models.forward_vector(model,profile['architecture'],request['input_ids']+tokens,profile['pad_token_id'])
                        vp=run/'format-vectors'/rid/f'{step}.npy';save_vector(vp,vector);new_forwards+=1
                    token=int(vector.argmax());tokens.append(token);steps.append({'raw_logits':c.info(vp)})
                    if token in profile['eos_token_ids']:break
                record={'condition_id':rid,'prompt_sha256':request['prompt_sha256'],'generated_token_ids':tokens,'steps':steps,
                    'passed':len(tokens)==2 and tokens[0] in profile['candidate_tokens'].values() and tokens[1] in profile['eos_token_ids'],
                    'binding_sha256':binding_sha}
                c.atomic(fp,record);validate_format_record(record,request,profile)
            c.require(record['binding_sha256']==binding_sha,'Format binding differs')
            format_records.append(c.info(fp))
        passed=sum(c.read(r['path'])['passed'] for r in format_records)
        format_receipt={'passed':passed,'total':16,'all_pass':passed==16,'records':format_records}
        seal_or_check(run/'format-check.json',format_receipt)
        c.require(passed==16,'Format qualification failed; stop, preserve results and do not silently adapt scoring')
        seal_or_check(run/'qualification.json',{'model_key':profile['model_key'],'binding_sha256':binding_sha,
            'numerical':c.info(run/'numerical-qualification.json'),'format':c.info(run/'format-check.json'),
            'margin_error_bound':q['margin_error_bound'],'status':'pass'})
        if phase=='full':
            bound_value=q['margin_error_bound'];update(status='running_production')
            production=[]
            for n,rid in enumerate(profile['production_ids'],1):
                checkpoint();request=lookup[rid];rp,vp=paths(run,'production',rid)
                if rp.exists():record,_=check_record(run,'production',request,binding_sha,profile,bound_value,identity);reused+=1
                else:
                    prep,vector=models.forward_vector(model,profile['architecture'],request['input_ids'],profile['pad_token_id'])
                    save_vector(vp,vector)
                    record=dict({k:request[k] for k in ['request_id','condition_id','prompt_sha256','input_ids_sha256']},
                        model_key=profile['model_key'],pass_id='production',binding_sha256=binding_sha,
                        physical_score_id=f"{state['run_id']}:production:{rid}",candidate_tokens=profile['candidate_tokens'],
                        raw_logits=c.info(vp),producer_identity=producer,prepared_input=prep,
                        readout=c.readout(vector,profile['candidate_tokens'],bound_value),scored_at=c.now(),
                        qualification_ref=c.info(run/'qualification.json'))
                    c.atomic(rp,record);new_forwards+=1
                c.require(abs(record['readout']['m']-values['reference'][rid]['m'])<=bound_value,'Production replay exceeds registered bound')
                production.append(c.info(rp))
                if n%24==0 or n==len(profile['production_ids']):update(pass_completed=n,new_forwards=new_forwards,reused_requests=reused)
            seal_or_check(run/'production-seal.json',{'model_key':profile['model_key'],'records':production,
                'qualification':c.info(run/'qualification.json'),'binding_sha256':binding_sha})
        lease_check();c.check_prepared(prepared,weights=True);validate_amendment(bound)
        update(status='complete_releasing' if phase=='full' else 'qualified_releasing')
    except Paused as e:update(status='paused',reason=str(e))
    except BaseException as e:
        update(status='failed',error_type=type(e).__name__,error=str(e));raise
    finally:
        model=None;gc.collect()
        import torch
        if torch.cuda.is_initialized():torch.cuda.synchronize();torch.cuda.empty_cache()
        state['invocations'][-1].update(ended_at=c.now(),new_forwards=new_forwards,reused_requests=reused)
        update(new_forwards=new_forwards,reused_requests=reused)
    return state['status']


def supervise(prepared,bound,run,phase,resume):
    prepared,bound,run=Path(prepared).resolve(),Path(bound).resolve(),Path(run).resolve()
    validate_amendment(bound)
    c.require(run.parent==c.WORK and run.name.startswith('run-'),'Dedicated new experiment run directory required')
    c.check_prepared(prepared);b=c.read(bound);c.verify(b['prepared_manifest'])
    c.require(b['prepared_manifest']==c.info(prepared/'manifest.json'),'Binding points to another preparation')
    c.require(b['runtime_versions']==c.runtime_versions() and bool(b['authorization_note']),'Missing or stale binding')
    c.require(not (run/'STOP').exists(),'STOP remains present')
    lockdir=c.WORK/'.locks';lockdir.mkdir(exist_ok=True)
    handles=[]
    try:
        for name in [run.name]+sorted(a['uuid'] for a in b['allocation']):
            h=(lockdir/(name+'.lock')).open('a+');handles.append(h);fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        inv=c.gpu_inventory();devices={d['uuid']:d for d in inv['devices']}
        for a in b['allocation']:
            d=devices[a['uuid']]
            c.require(all(d[k]==v for k,v in a.items()) and d['used_mib']==0 and d['utilization']==0
                and not any(a['uuid'] in r for r in inv['compute_processes']),'Bound GPU is changed or occupied')
        if (run/'state.json').exists():
            state=c.read(run/'state.json');c.validate_resume(state,c.info(bound),phase,resume)
        else:
            c.require(not resume and not run.exists(),'New run directory required')
            run.mkdir();c.atomic(run/'binding.json',b)
            state={'run_id':str(uuid.uuid4()),'status':'created','binding':c.info(bound),'invocations':[],
                   'query_reference_join_performed':False}
        invocation=str(uuid.uuid4());state['invocations'].append({'id':invocation,'controller_pid':os.getpid(),'started_at':c.now()})
        state.update(status='launching',phase=phase,owned_worker_absent=False)
        c.atomic(run/'state.json',state,replace=True)
        env=dict(os.environ,PYTHONUTF8='1',CUDA_VISIBLE_DEVICES=','.join(a['uuid'] for a in b['allocation']),CUBLAS_WORKSPACE_CONFIG=':4096:8',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0')
        cmd=[sys.executable,str(Path(__file__).resolve()),'_worker','--prepared',str(prepared),'--bound',str(bound),
             '--run',str(run),'--phase',phase,'--invocation',invocation]
        with (run/'gpu.log').open('ab',buffering=0) as log:
            process=subprocess.Popen(cmd,env=env,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print(json.dumps({'worker_pid':process.pid,'run':str(run),'model':b['model_key'],'phase':phase}),flush=True)
            code=process.wait()
        inventory=c.gpu_inventory()
        absent=not Path(f'/proc/{process.pid}').exists() and not any(line.split(',')[-1].strip()==str(process.pid) for line in inventory['compute_processes'])
        c.require(absent,'Owned worker still exists after wait')
        release=run/f'release-{invocation}.json'
        c.atomic(release,{'worker_pid':process.pid,'worker_exit_code':code,'owned_worker_absent':True,'inventory':inventory,'checked_at':c.now()})
        state=c.read(run/'state.json');state.update(owned_worker_absent=True,worker_exit_code=code,resource_release=c.info(release))
        if code==0 and state['status'] in ['complete_releasing','qualified_releasing']:state['status']=state['status'].removesuffix('_releasing')
        elif code!=0:state.update(status='failed')
        c.atomic(run/'state.json',state,replace=True)
        c.require(code==0,'Worker failed; preserve artifacts and inspect gpu.log; no automatic retry')
        return state
    finally:
        for h in reversed(handles):h.close()


def check_run(prepared,run):
    prepared,run=Path(prepared),Path(run)
    validate_amendment(run/'binding.json')
    plan,profiles,requests=c.check_prepared(prepared)
    b=c.read(run/'binding.json');state=c.read(run/'state.json')
    c.require(state['status'] in ['qualified','complete'] and state['owned_worker_absent'] and state['worker_exit_code']==0,'Unfinished or unreleased run')
    c.verify(state['binding']);c.verify(state['resource_release'])
    c.require(b==c.read(state['binding']['path']) and b['prepared_manifest']==c.info(prepared/'manifest.json'),'Run preparation/binding differs')
    release=c.read(state['resource_release']['path'])
    c.require(release['owned_worker_absent'] and release['worker_exit_code']==0 and release['worker_pid']==state['worker_pid'],'Release identity differs')
    key=b['model_key'];p=next(p for p in profiles if p['model_key']==key);rs=requests[key]
    binding_sha=c.sha(run/'binding.json');identity=c.read(run/'runtime-identity.json')
    vals,sources=engineering_values(run,rs,binding_sha,p,identity)
    q=c.qualify(vals,plan['acceptance']);q.update(model_key=key,binding_sha256=binding_sha,sources=sources)
    c.require(q==c.read(run/'numerical-qualification.json'),'Qualification reconstruction differs')
    probe_records=[];lookup={r['request_id']:r for r in rs}
    for rid in p['format_probe_ids']:
        record=c.read(run/'format'/f'{rid}.json');validate_format_record(record,lookup[rid],p)
        c.require(record['passed'] and record['binding_sha256']==binding_sha,'Format gate failed')
        probe_records.append(c.info(run/'format'/f'{rid}.json'))
    c.require(c.read(run/'format-check.json')=={'passed':16,'total':16,'all_pass':True,'records':probe_records},'Format seal differs')
    expected={'model_key':key,'binding_sha256':binding_sha,'numerical':c.info(run/'numerical-qualification.json'),
              'format':c.info(run/'format-check.json'),'margin_error_bound':q['margin_error_bound'],'status':'pass'}
    c.require(c.read(run/'qualification.json')==expected,'Qualification seal differs')
    records=[]
    if state['status']=='complete':
        sources=[]
        for rid in p['production_ids']:
            record,_=check_record(run,'production',lookup[rid],binding_sha,p,q['margin_error_bound'],identity)
            c.require(record['qualification_ref']==c.info(run/'qualification.json'),'Production qualification differs')
            c.require(abs(record['readout']['m']-vals['reference'][rid]['m'])<=q['margin_error_bound'],'Production replay failed')
            records.append(record);sources.append(c.info(paths(run,'production',rid)[0]))
        c.require(c.read(run/'production-seal.json')=={'model_key':key,'records':sources,'qualification':c.info(run/'qualification.json'),
                  'binding_sha256':binding_sha},'Production seal differs')
    return {'status':state['status'],'model_key':key,'qualification':q,'production_records':records}


def analyze(prepared,run,output):
    result=check_run(prepared,run)
    c.require(result['status']=='complete','Production must finish before reference join')
    # This is the first query-reference parse in the scoring/results lifecycle.
    plan=c.read(c.SCIENCE/'analysis-plan.json')
    new=[r for r in result['production_records'] if r['condition_id'].startswith('CMAD-')]
    c.require(len(new)==384,'New-query production coverage incomplete')
    values=c.production_analysis(new,plan)
    values['model_key']=result['model_key']
    c.atomic(Path(output)/'new-development-results.json',values)
    legacy=[r for r in result['production_records'] if not r['condition_id'].startswith('CMAD-')]
    if legacy:
        c.require(len(legacy)==156,'Legacy production coverage incomplete')
        legacy_values=c.production_analysis(legacy,c.read(Path(prepared)/'legacy-analysis.json'))
        legacy_values['model_key']=result['model_key']
        c.atomic(Path(output)/'legacy-replication-results.json',legacy_values)
    return {'status':'analyzed','new_comparisons':len(values['comparisons']),
            'legacy_replication_conditions':len(legacy),'legacy_comparisons':168 if legacy else 0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','validate','bind','run','_worker','check','analyze'])
    p.add_argument('--prepared',type=Path,default=c.PREPARED);p.add_argument('--bound',type=Path)
    p.add_argument('--run',type=Path);p.add_argument('--output',type=Path);p.add_argument('--model')
    p.add_argument('--gpus',type=int,nargs='+');p.add_argument('--authorization-note')
    p.add_argument('--phase',choices=['engineering','full'],default='engineering')
    p.add_argument('--resume',action='store_true');p.add_argument('--weights',action='store_true');p.add_argument('--invocation')
    a=p.parse_args()
    if a.action=='prepare':result=c.prepare(a.prepared)
    elif a.action=='validate':
        plan,profiles,rs=c.check_prepared(a.prepared,weights=a.weights)
        result={'status':'pass','models':len(profiles),'CPU_inputs':sum(len(v) for v in rs.values()),'budget':plan['budget'],'GPU_qualified':False}
    elif a.action=='bind':result=c.bind(a.prepared,a.model,a.gpus,a.bound,a.authorization_note)
    elif a.action=='run':result=supervise(a.prepared,a.bound,a.run,a.phase,a.resume)
    elif a.action=='_worker':result={'status':worker(a.prepared,a.bound,a.run,a.invocation,a.phase)}
    elif a.action=='check':
        raw=check_run(a.prepared,a.run);result={k:v for k,v in raw.items() if k!='production_records'}
    else:result=analyze(a.prepared,a.run,a.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
