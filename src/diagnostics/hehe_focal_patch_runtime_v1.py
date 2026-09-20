"""Fresh donor/baseline qualification and terminal, reference-free patch execution."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
import numpy as np

from diagnostics import hehe_focal_patch_inputs_v1 as c
from diagnostics import hehe_focal_patch_capture_v1 as cap
from diagnostics.case_attention_capture_v1 import qualify
from diagnostics.cross_model_applicability_execution_v1 import readout, gpu_inventory, runtime_versions
from diagnostics.hehe_presentation_runtime_v1 import idle, save_array, seal_equal

NATIVE_STAGES=['native-baseline','native-capture','native-repeat','native-reverse','native-left','native-right','native-prefix']
CROSS_STAGES=['cross-probe','cross-repeat','cross-reverse','cross-left','cross-right']


def record_path(run,stage,key):return Path(run)/'records'/stage/(key+'.json')


def bind(prepared,gpu,output,authorization_note):
    c.require(authorization_note.strip(),'Execution authorization must be recorded')
    plan,profile,_,_,_=c.validate(prepared,weights=True)
    constraint=plan['GPU_time_constraint'];c.require(constraint['confirmed'],'GPU timing choice still pending')
    if constraint.get('deadline_unix') is not None:c.require(time.time()<constraint['deadline_unix']-120,'No remaining authorized window')
    inv=gpu_inventory();d=next((d for d in inv['devices'] if d['index']==gpu),None)
    c.require(d is not None and d['total_mib']>=44000,'One idle >=44000 MiB device required')
    allocation=[{k:d[k] for k in ['index','uuid','name','total_mib']}];idle(allocation,inv)
    c.atomic(output,{'prepared_manifest':c.info(Path(prepared)/'manifest.json'),'model_key':profile['model_key'],
                     'allocation':allocation,'runtime_versions':plan['runtime_versions'],'authorization_note':authorization_note,
                     'GPU_time_constraint':constraint,'inventory':inv,'created_at_unix':time.time()})
    return c.info(output)


def load_record(run,stage,req,binding,profile,job=None):
    key=job['job_id'] if job else req['request_id'];r=c.read(record_path(run,stage,key))
    c.require(r['stage']==stage and r['request_id']==req['request_id'] and r['job']==job,'Record/ledger mismatch')
    c.require(r['binding_sha256']==binding and r['input_ids_sha256']==req['input_ids_sha256'] and r['prompt_sha256']==req['prompt_sha256'],'Record input binding mismatch')
    c.verify(r['producer']);c.require(c.read(r['producer']['path'])['runtime_identity']==c.read(Path(run)/'runtime-identity.json'),'Producer mismatch')
    c.verify(r['vector']);v=np.load(r['vector']['path'],allow_pickle=False)
    c.require(v.dtype==np.float32 and v.shape==(profile['vocab_size'],) and np.isfinite(v).all(),'Invalid saved logits')
    c.require(r['readout']==(None if stage=='native-prefix' else readout(v,profile['candidate_tokens'])),'Saved margin differs')
    states=None
    if r['states']:
        c.verify(r['states']);states=np.load(r['states']['path'],allow_pickle=False)
        c.require(states.dtype==np.float32 and states.shape==(profile['layers'],len(req['capture_positions']),profile['hidden_size']) and np.isfinite(states).all(),'Invalid donor states')
    wants_states=stage.startswith('native-') and stage!='native-baseline'
    c.require((states is not None)==wants_states,'State capture availability changed')
    c.require(r['positions']==req['capture_positions'] and r['prefix_length']==(req['capture_prefix_length'] if stage=='native-prefix' else None),'Capture geometry mismatch')
    if job:
        source=r['donor_source'];c.verify(source['states']);c.verify(source['record'])
        expected_stage='native-production' if stage=='production' else 'native-capture'
        c.require(source['record']==c.info(record_path(run,expected_stage,job['donor'])),'Donor is not the registered fresh bank')
        donor=c.read(source['record']['path'])
        c.require(donor['request_id']==job['donor'] and donor['binding_sha256']==binding and donor['states']==source['states'],'Wrong donor provenance')
        c.require(source['positions']==donor['positions'],'Donor position lookup changed')
        bank=np.load(source['states']['path'],allow_pickle=False)
        rows=cap.donor_rows(bank,source['positions'],job['layer'],job['positions'])
        proof=r['patch_proof'];padding=stage.removeprefix('cross-') if stage in ['cross-left','cross-right'] else 'none'
        from diagnostics.cross_model_applicability_models_v1 import prepared_input
        prep=prepared_input(req['input_ids'],profile['pad_token_id'],padding);valid=[i for i,x in enumerate(prep['attention_mask']) if x]
        c.require(proof['layer']==job['layer'] and proof['positions']==job['positions'] and proof['padded_positions']==[valid[i] for i in job['positions']],'Patch coordinate proof mismatch')
        c.require(proof['donor_sha256']==cap.array_sha(rows) and proof['outside_rows_exact'] and proof['donor_rows_exact'] and proof['native_output_unmodified'],'Patch execution proof differs')
        c.require(set(proof['changed_rows_in_unpadded_coordinates'])<=set(job['positions']),'Outside row was changed')
    else:c.require(r['patch_proof'] is None and r['donor_source'] is None,'Native call has intervention metadata')
    return r,v,states


def scaled_error(a,b):
    c.require(a.shape==b.shape,'State comparison shape differs')
    return float(np.abs(a.astype(np.float64)-b).max(initial=0)/max(1.,float(np.abs(a).max(initial=0)),float(np.abs(b).max(initial=0))))


def check_engineering(prepared,run):
    plan,profile,reqs,jobs,selfs=c.validate(prepared)
    byid={r['request_id']:r for r in reqs};binding=c.sha(Path(run)/'binding.json')
    checks=[];state_checks=[];records=[];banks={}
    def compare(kind,va,vb,sa=None,sb=None):
        delta=float(np.abs(va.astype(np.float64)-vb).max()) if va is not None else None
        md=abs(readout(va,profile['candidate_tokens'])['m']-readout(vb,profile['candidate_tokens'])['m']) if va is not None else None
        if kind in ['hook','repeat','reverse','self','last_layer']:c.require(delta==0,'Native/self/repeat/reverse/last-layer full-vector mismatch')
        if kind in ['self','last_layer']:return
        checks.append({'kind':kind,'margin_difference':md,'vector_max_difference':delta})
        if sa is not None:
            error=scaled_error(sa,sb)
            c.require(error<=(0 if kind in ['repeat','reverse'] else plan['state_scaled_cap']),'Donor state invariance failed')
            state_checks.append({'kind':kind,'scaled_error':error})
    for req in reqs:
        loaded={stage:load_record(run,stage,req,binding,profile) for stage in NATIVE_STAGES}
        for stage in NATIVE_STAGES:records.append(c.info(record_path(run,stage,req['request_id'])))
        _,base,_=loaded['native-baseline'];ref,v,s=loaded['native-capture'];banks[req['request_id']]=(ref,v,s)
        compare('hook',base,v)
        for tag in ['repeat','reverse','left','right']:
            _,other,states=loaded['native-'+tag];compare(tag,v,other,s,states)
        _,_,prefix=loaded['native-prefix'];compare('prefix',None,None,s,prefix)
    for job in selfs:
        req=byid[job['recipient']];r,v,_=load_record(run,'self',req,binding,profile,job)
        compare('self',v,banks[req['request_id']][1]);c.require(r['patch_proof']['changed_rows_in_unpadded_coordinates']==[],'Self patch changed native rows')
        records.append(c.info(record_path(run,'self',job['job_id'])))
    for job in jobs:
        req=byid[job['recipient']]
        values={stage:load_record(run,stage,req,binding,profile,job) for stage in CROSS_STAGES}
        for stage in CROSS_STAGES:records.append(c.info(record_path(run,stage,job['job_id'])))
        _,ref,_=values['cross-probe']
        if job['last_layer_expected_zero']:compare('last_layer',ref,banks[req['request_id']][1])
        for tag in ['repeat','reverse','left','right']:compare(tag,ref,values['cross-'+tag][1])
    q=qualify(checks,plan['acceptance'])
    q.update(prepared_manifest=c.info(Path(prepared)/'manifest.json'),binding_sha256=binding,
             comparisons=checks,state_comparisons=state_checks,
             state_scaled_error_bound=max(1e-6,2*max((x['scaled_error'] for x in state_checks),default=0)),
             self_controls=len(selfs),last_layer_controls=sum(j['last_layer_expected_zero'] for j in jobs),
             donor_banks='fresh same-run native capture',record_files=records,
             attention_mass_bound_applicable=False,meaning='Fresh score/state engineering bounds; no statistical confidence or causal-path guarantee.')
    return q


def check_format(run,key,req,profile,binding,job=None):
    path=Path(run)/'format'/(key+'.json');rec=c.read(path)
    c.require(rec['key']==key and rec['job']==job and rec['binding_sha256']==binding and rec['input_ids_sha256']==req['input_ids_sha256'],'Format binding mismatch')
    c.verify(rec['producer']);c.require(c.read(rec['producer']['path'])['runtime_identity']==c.read(Path(run)/'runtime-identity.json'),'Format producer mismatch')
    first_record,first,_=load_record(run,'cross-probe' if job else 'native-capture',req,binding,profile,job)
    c.require(rec['donor_sha256']==(first_record['patch_proof']['donor_sha256'] if job else None),'Format donor differs from scoring donor')
    tokens=[int(first.argmax())]
    for step in rec['steps']:
        c.require(step['prefix_tokens']==tokens and tokens[-1] not in profile['eos_token_ids'],'Invalid format continuation')
        c.verify(step['vector']);v=np.load(step['vector']['path'],allow_pickle=False)
        c.require(v.dtype==np.float32 and v.shape==(profile['vocab_size'],) and np.isfinite(v).all(),'Bad continuation vector')
        if job:
            proof=step['patch_proof'];c.require(proof['layer']==job['layer'] and proof['positions']==proof['padded_positions']==job['positions'] and proof['donor_sha256']==rec['donor_sha256'] and proof['outside_rows_exact'] and proof['donor_rows_exact'] and proof['native_output_unmodified'],'Continuation did not reapply exact patch')
        else:c.require(step['patch_proof'] is None,'Native format contains a patch')
        tokens.append(int(v.argmax()))
    valid=len(tokens)==2 and tokens[0] in profile['candidate_tokens'].values() and tokens[1] in profile['eos_token_ids']
    c.require(tokens==rec['tokens'] and valid==rec['exact_label_then_eos'] and len(tokens)<=8,'Format reconstruction mismatch')
    c.require(valid,'Exact single-label-then-EOS gate failed; retain this endpoint without filtering')
    return c.info(path)


def worker(prepared,bound,run,invocation,phase):
    from diagnostics.cross_model_applicability_models_v1 import load_checkpoint
    prepared,run=Path(prepared),Path(run)
    plan,profile,reqs,jobs,selfs=c.validate(prepared,weights=True);byid={r['request_id']:r for r in reqs}
    b,state=c.read(bound),c.read(run/'state.json')
    c.require(b['prepared_manifest']==c.info(prepared/'manifest.json') and c.read(run/'binding.json')==b,'Wrong bound preparation')
    c.require(state['status']=='launching' and state['invocations'][-1]['id']==invocation,'Worker not registered')
    binding=c.sha(run/'binding.json');stopping=False;forwards=reused=0
    state['worker_pid']=os.getpid()
    def update(**fields):
        state.update(fields,updated_at_unix=time.time(),new_forwards=forwards,reused_records=reused)
        c.atomic(run/'state.json',state,replace=True)
    class Paused(Exception):pass
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    def checkpoint():
        if stopping or (run/'STOP').exists():raise Paused()
        deadline=b['GPU_time_constraint'].get('deadline_unix')
        if deadline is not None and time.time()>=deadline-120:raise Paused()
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
    try:
        checkpoint();update(status='loading_model')
        model,identity=load_checkpoint(profile,b['allocation']);seal_equal(run/'runtime-identity.json',identity)
        producer_path=run/'invocations'/(invocation+'.json')
        c.atomic(producer_path,{'runtime_identity':identity,'worker_pid':os.getpid(),'invocation':invocation,'model_loaded_at_unix':time.time(),'binding_sha256':binding})
        producer=c.info(producer_path)
        leases=[(Path(x['path']),Path(x['path']).stat()) for x in profile['metadata_sources']+profile['weight_sources']]
        def lease_check():
            for p,old in leases:
                now=p.stat();c.require((now.st_ino,now.st_dev,now.st_size,now.st_mtime_ns)==(old.st_ino,old.st_dev,old.st_size,old.st_mtime_ns),'Checkpoint changed while loaded')
        def native_record(stage,req):
            nonlocal forwards,reused
            checkpoint();lease_check();rp=record_path(run,stage,req['request_id'])
            if rp.exists():reused+=1;return load_record(run,stage,req,binding,profile)
            update(status='engineering' if phase=='engineering' else 'production',stage=stage,current_request=req['request_id'])
            begin=time.monotonic();prefix=req['capture_prefix_length'] if stage=='native-prefix' else None
            padding=stage.removeprefix('native-') if stage in ['native-left','native-right'] else 'none'
            v,s,_=cap.forward(model,req,profile['pad_token_id'],positions=req['capture_positions'],observe=stage!='native-baseline',padding=padding,prefix_length=prefix)
            forwards+=1
            value={'request_id':req['request_id'],'stage':stage,'job':None,'binding_sha256':binding,
                   'prompt_sha256':req['prompt_sha256'],'input_ids_sha256':req['input_ids_sha256'],
                   'positions':req['capture_positions'],'prefix_length':prefix,'producer':producer,
                   'vector':save_array(run/'arrays'/stage/(req['request_id']+'.npy'),v),
                   'states':save_array(run/'arrays'/stage/(req['request_id']+'.states.npy'),s) if s is not None else None,
                   'readout':None if prefix else readout(v,profile['candidate_tokens']),
                   'donor_source':None,'patch_proof':None,'elapsed_seconds':time.monotonic()-begin}
            c.atomic(rp,value);return value,v,s
        banks={}
        if phase=='engineering':
            for stage in NATIVE_STAGES:
                for req in (list(reversed(reqs)) if stage=='native-reverse' else reqs):native_record(stage,req)
            banks={r['request_id']:load_record(run,'native-capture',r,binding,profile) for r in reqs}
        else:
            c.require(c.read(run/'qualification.json')==check_engineering(prepared,run),'Fresh engineering qualification differs')
            for req in reqs:
                rec,v,s=native_record('native-production',req);old,ov,os_=load_record(run,'native-capture',req,binding,profile)
                c.require(np.array_equal(v,ov) and np.array_equal(s,os_),'Production donor/native replay differs')
                banks[req['request_id']]=(rec,v,s)
        def patch_record(stage,job):
            nonlocal forwards,reused
            checkpoint();lease_check();req=byid[job['recipient']];rp=record_path(run,stage,job['job_id'])
            if rp.exists():reused+=1;return load_record(run,stage,req,binding,profile,job)
            update(status='engineering' if phase=='engineering' else 'production',stage=stage,current_request=job['job_id'])
            dr,_,ds=banks[job['donor']];dreq=byid[job['donor']]
            rows=cap.donor_rows(ds,dreq['capture_positions'],job['layer'],job['positions'])
            padding=stage.removeprefix('cross-') if stage in ['cross-left','cross-right'] else 'none'
            begin=time.monotonic();v,_,proof=cap.forward(model,req,profile['pad_token_id'],positions=job['positions'],layer=job['layer'],replacement=rows,padding=padding)
            forwards+=1
            donor_stage='native-capture' if phase=='engineering' else 'native-production'
            value={'request_id':req['request_id'],'stage':stage,'job':job,'binding_sha256':binding,
                   'prompt_sha256':req['prompt_sha256'],'input_ids_sha256':req['input_ids_sha256'],
                   'positions':req['capture_positions'],'prefix_length':None,'producer':producer,
                   'vector':save_array(run/'arrays'/stage/(job['job_id']+'.npy'),v),'states':None,
                   'readout':readout(v,profile['candidate_tokens']),'patch_proof':proof,
                   'donor_source':{'states':dr['states'],'record':c.info(record_path(run,donor_stage,dreq['request_id'])),'positions':dreq['capture_positions']},
                   'elapsed_seconds':time.monotonic()-begin}
            c.atomic(rp,value);return value,v,None
        if phase=='engineering':
            for job in selfs:patch_record('self',job)
            for stage in CROSS_STAGES:
                for job in (list(reversed(jobs)) if stage=='cross-reverse' else jobs):patch_record(stage,job)
            q=check_engineering(prepared,run);seal_equal(run/'qualification.json',q)
            format_files=[]
            for job in [None]*len(reqs)+jobs:
                index=len(format_files);req=reqs[index] if job is None else byid[job['recipient']]
                key=req['request_id'] if job is None else job['job_id'];path=run/'format'/(key+'.json')
                checkpoint();lease_check()
                if path.exists():format_files.append(check_format(run,key,req,profile,binding,job));reused+=1;continue
                first=load_record(run,'cross-probe' if job else 'native-capture',req,binding,profile,job)[1]
                tokens=[int(first.argmax())];steps=[];rows=None
                if job:
                    dreq=byid[job['donor']];rows=cap.donor_rows(banks[job['donor']][2],dreq['capture_positions'],job['layer'],job['positions'])
                update(status='engineering',stage='format',current_request=key)
                # A format unit is bounded to seven continuation calls. It commits
                # atomically as one unit; STOP is checked between units.
                while len(tokens)<plan['format']['max_new_tokens'] and tokens[-1] not in profile['eos_token_ids']:
                    v,_,proof=cap.forward(model,req,profile['pad_token_id'],positions=job['positions'] if job else (),
                                          layer=job['layer'] if job else None,replacement=rows,append_tokens=tokens)
                    forwards+=1;steps.append({'prefix_tokens':list(tokens),'vector':save_array(run/'format-arrays'/key/(str(len(tokens))+'.npy'),v),'patch_proof':proof});tokens.append(int(v.argmax()))
                valid=len(tokens)==2 and tokens[0] in profile['candidate_tokens'].values() and tokens[1] in profile['eos_token_ids']
                c.atomic(path,{'key':key,'job':job,'binding_sha256':binding,'input_ids_sha256':req['input_ids_sha256'],
                               'tokens':tokens,'steps':steps,'exact_label_then_eos':valid,'producer':producer,
                               'donor_sha256':cap.array_sha(rows) if rows is not None else None})
                format_files.append(check_format(run,key,req,profile,binding,job))
            seal_equal(run/'engineering-seal.json',{'qualification':c.info(run/'qualification.json'),'format':format_files,'binding_sha256':binding})
        else:
            for job in jobs:
                _,v,_=patch_record('production',job);_,ref,_=load_record(run,'cross-probe',byid[job['recipient']],binding,profile,job)
                c.require(np.array_equal(v,ref),'Full-vocabulary production replay differs')
            seal_equal(run/'raw-seal.json',{'engineering_seal':c.info(run/'engineering-seal.json'),'binding_sha256':binding,
                       'native_records':[c.info(record_path(run,'native-production',r['request_id'])) for r in reqs],
                       'records':[c.info(record_path(run,'production',j['job_id'])) for j in jobs]})
        lease_check();update(status='qualified_releasing' if phase=='engineering' else 'complete_releasing')
    except Paused:update(status='paused')
    except Exception as exc:
        update(status='failed',failure=str(exc));raise


def check_run(prepared,run):
    run=Path(run);plan,profile,reqs,jobs,selfs=c.validate(prepared)
    state,b=c.read(run/'state.json'),c.read(run/'binding.json');binding=c.sha(run/'binding.json')
    c.require(b['prepared_manifest']==c.info(Path(prepared)/'manifest.json'),'Wrong result preparation')
    c.require(state['status'] in ['qualified','complete'] and state['owned_worker_absent'] and state['worker_exit_code']==0,'CPU check requires normal release')
    c.verify(state['resource_release']);release=c.read(state['resource_release']['path'])
    c.require(release['owned_worker_absent'] and release['worker_exit_code']==0,'Invalid release receipt')
    q=check_engineering(prepared,run);c.require(q==c.read(run/'qualification.json'),'Qualification reconstruction differs')
    byid={r['request_id']:r for r in reqs}
    formats=[check_format(run,r['request_id'],r,profile,binding) for r in reqs]
    formats += [check_format(run,j['job_id'],byid[j['recipient']],profile,binding,j) for j in jobs]
    c.require(c.read(run/'engineering-seal.json')=={'qualification':c.info(run/'qualification.json'),'format':formats,'binding_sha256':binding},'Engineering seal mismatch')
    if state['status']=='complete':
        for req in reqs:
            _,v,s=load_record(run,'native-production',req,binding,profile);_,rv,rs=load_record(run,'native-capture',req,binding,profile)
            c.require(np.array_equal(v,rv) and np.array_equal(s,rs),'Native replay mismatch')
        for j in jobs:
            _,v,_=load_record(run,'production',byid[j['recipient']],binding,profile,j)
            _,rv,_=load_record(run,'cross-probe',byid[j['recipient']],binding,profile,j)
            c.require(np.array_equal(v,rv),'Production intervention replay mismatch')
        expected={'engineering_seal':c.info(run/'engineering-seal.json'),'binding_sha256':binding,
                  'native_records':[c.info(record_path(run,'native-production',r['request_id'])) for r in reqs],
                  'records':[c.info(record_path(run,'production',j['job_id'])) for j in jobs]}
        c.require(c.read(run/'raw-seal.json')==expected,'Raw seal differs')
    return {'status':state['status'],'qualification':q,'CPU_reconstructed':True,'query_reference_join_performed':False}


def supervise(prepared,bound,run,phase='engineering',resume=False):
    prepared,bound,run=(Path(p).absolute() for p in [prepared,bound,run])
    c.require(run.parent==c.WORK and run.name.startswith('run-'),'Use an isolated new stage2 run directory')
    c.validate(prepared);b=c.read(bound);c.verify(b['prepared_manifest'])
    c.require(b['prepared_manifest']==c.info(prepared/'manifest.json') and b['runtime_versions']==runtime_versions() and b['authorization_note'],'Stale/missing execution binding')
    c.require(not (run/'STOP').exists(),'STOP remains present')
    lockdir=c.WORK/'.locks';lockdir.mkdir(exist_ok=True);locks=[]
    try:
        for name in [run.name,b['allocation'][0]['uuid']]:
            f=(lockdir/(name+'.lock')).open('a+');locks.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        idle(b['allocation'],gpu_inventory())
        if run.exists():
            state=c.read(run/'state.json');c.require(state['binding']==c.info(bound),'Changed run binding')
            c.require((resume and state['status']=='paused' and state['phase']==phase) or
                      (not resume and state['status']=='qualified' and phase=='full'),'Failed/complete runs are terminal')
            c.require(state['owned_worker_absent'] and state['worker_exit_code']==0,'Previous worker not normally released')
            if phase=='full' and not resume:check_run(prepared,run)
        else:
            c.require(not resume and phase=='engineering','Begin with engineering');run.mkdir();c.atomic(run/'binding.json',b)
            state={'status':'created','binding':c.info(bound),'invocations':[],'query_reference_join_performed':False}
        invocation=str(uuid.uuid4());state['invocations'].append({'id':invocation,'controller_pid':os.getpid(),'started_at_unix':time.time()})
        state.update(status='launching',phase=phase,owned_worker_absent=False);c.atomic(run/'state.json',state,replace=True)
        env=dict(os.environ,PYTHONUTF8='1',CUDA_VISIBLE_DEVICES=b['allocation'][0]['uuid'],CUBLAS_WORKSPACE_CONFIG=':4096:8',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',USE_TORCH='1')
        cli=c.ROOT/'scripts/review/run_hehe_focal_patch_v1.py'
        cmd=[sys.executable,str(cli),'_worker','--prepared',str(prepared),'--bound',str(bound),'--run',str(run),'--phase',phase,'--invocation',invocation]
        with (run/'gpu.log').open('ab',buffering=0) as log:
            proc=subprocess.Popen(cmd,cwd=c.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);code=proc.wait()
        inv=gpu_inventory();absent=not Path(f'/proc/{proc.pid}').exists() and not any(x.split(',')[-1].strip()==str(proc.pid) for x in inv['compute_processes'])
        c.require(absent,'Owned worker remains present');release=run/f'release-{invocation}.json'
        c.atomic(release,{'worker_pid':proc.pid,'worker_exit_code':code,'owned_worker_absent':absent,'inventory':inv,'checked_at_unix':time.time()})
        state=c.read(run/'state.json');state.update(owned_worker_absent=True,worker_exit_code=code,resource_release=c.info(release))
        state['invocations'][-1].update(worker_pid=proc.pid,ended_at_unix=time.time())
        if code==0 and state['status'] in ['complete_releasing','qualified_releasing']:state['status']=state['status'].removesuffix('_releasing')
        elif code!=0:state['status']='failed'
        c.atomic(run/'state.json',state,replace=True);c.require(code==0,'Worker failed; never retry automatically')
        return state
    finally:
        for f in reversed(locks):f.close()
