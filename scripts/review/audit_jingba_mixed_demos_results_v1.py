#!/usr/bin/env python3
"""Independent native vector/probability, state, trajectory, and paired-effect audit."""
import argparse,json,math,sys
from pathlib import Path
from decimal import Decimal,localcontext
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
from audit_dictionary_free_donor_results_v1 import read,info,verify,dec,audit_trajectories

def audit(prepared,run,out):
 p,run,out=map(Path,(prepared,run,out));state=read(run/'state.json');assert state['status']=='complete' and state['owned_worker_absent'] and state['worker_exit_code']==0;verify(state['resource_release'])
 for path in [p/'manifest.json',out/'manifest.json']:
  m=read(path)
  for x in m['artifacts']+m.get('sources',[]):verify(x)
 q=read(run/'qualification.json');data=read(out/'results.json');plan=read(p/'execution-plan.json');profile=read(p/'model-profile.json');bound=dec(q['margin_error_bound'])
 requests={r['request_id']:r for r in [json.loads(line) for line in (p/'scoring-inputs.jsonl').read_text().splitlines()]};refs={r['query_id']:r['reference'] for r in read(p/'analysis-references.json')['references']}
 assert len(requests)==30 and len(data['baselines'])==30 and set(refs)=={f'J{i:02}' for i in range(5,11)}
 assert read(p/'interventions.json')=={'cross':[],'self':[]} and data['effects']==[] and data['self_controls']==[]
 absolute=0;margin_count=0;records={};vecs={};scored={}
 def vector(item):
  nonlocal absolute
  verify(item);v=np.load(item['path'],allow_pickle=False);assert v.dtype==np.float32 and v.shape==(profile['vocab_size'],) and np.isfinite(v).all();absolute+=1;return v
 def score(v,s):
  nonlocal margin_count
  y,n=[dec(v[profile['candidate_tokens'][k]]) for k in ['有','无']];m=n-y
  assert dec(s['m'])==m and dec(s['z_yes'])==y and dec(s['z_no'])==n and s['raw_prediction']==('无' if m>0 else '有' if m<0 else None);margin_count+=1
  vv=v.astype(np.longdouble);top=vv.max();lnz=top+np.log(np.exp(vv-top).sum(dtype=np.longdouble));yy,nn=[vv[profile['candidate_tokens'][k]] for k in ['有','无']];pt=max(yy,nn);lse=pt+np.log(np.exp(yy-pt)+np.exp(nn-pt))
  for k,z in [('log_p_yes',yy-lnz),('log_p_no',nn-lnz),('log_legal_mass',lse-lnz),('legal_mass',np.exp(lse-lnz)),('pair_support_no',np.exp(nn-lse))]:assert abs(np.longdouble(s[k])-z)<np.longdouble('1e-12')
  return m
 for file in sorted((run/'records').glob('*/*.json')):
  r=read(file);assert r['job'] is None and r['patch_proof'] is None;rkey=(r['stage'],r['request_id']);v=vector(r['vector']);records[rkey]=r;vecs[rkey]=v
  if r['readout'] is not None:
   h=r['vector']['sha256']
   if h in scored:assert r['readout']==scored[h];margin_count+=1
   else:score(v,r['readout']);scored[h]=r['readout']
  else:assert r['stage']=='native-prefix'
 for file in sorted((run/'format').glob('*.json')):
  r=read(file);assert r['job'] is None;tokens=[int(vecs['native-capture',r['key']].argmax())]
  for step in r['steps']:assert step['prefix_tokens']==tokens;tokens.append(int(vector(step['vector']).argmax()));assert step['patch_proof'] is None
  assert tokens==r['tokens'] and len(tokens)==2 and tokens[0] in profile['candidate_tokens'].values() and tokens[1] in profile['eos_token_ids'] and r['exact_label_then_eos']
 margins={};states={};maxpad=Decimal(0);state_errors=[]
 for s in data['baselines']:
  rid=s['request_id'];req=requests[rid];margins[rid]=score(vecs['native-production',rid],s)
  assert s['condition']==req['condition'] and s['reference']==refs[req['query_id']] and s['raw_reference_correct']==(s['raw_prediction']==s['reference'])
  assert dec(s['reference_aligned_margin'])==margins[rid]*(1 if s['reference']=='无' else -1)
  base=vecs['native-capture',rid];ref=np.load(records['native-capture',rid]['states']['path'],allow_pickle=False).astype(np.float64)
  for stage in ['native-baseline','native-repeat','native-reverse','native-left','native-right','native-production','native-prefix']:
   v=vecs[stage,rid]
   if stage in ['native-baseline','native-repeat','native-reverse','native-production']:assert np.array_equal(v,base)
   if stage!='native-prefix':maxpad=max(maxpad,abs(dec(v[profile['candidate_tokens']['无']])-dec(v[profile['candidate_tokens']['有']])-margins[rid]))
   if stage!='native-baseline':
    item=records[stage,rid]['states'];verify(item);ss=np.load(item['path'],allow_pickle=False).astype(np.float64);err=abs(ss-ref).max()/max(1.,abs(ss).max(),abs(ref).max());assert err<=plan['state_scaled_cap'];state_errors.append(err)
    if stage in ['native-repeat','native-reverse','native-production']:assert np.array_equal(ss,ref)
 assert bound==max(dec(plan['acceptance']['margin_bound_floor']),2*maxpad) and q['state_scaled_error_bound']==max(1e-6,2*max(state_errors))
 assert absolute==270 and len(records)==240 and len(data['native_contrasts'])==36
 ta=audit_trajectories(records,vecs,data,profile,q)
 byid={r['request_id']:r for r in data['baselines']};seen=set()
 for r in data['native_contrasts']:
  a,b=byid[r['target_request']],byid[r['base_request']];assert a['query_id']==b['query_id']==r['query_id'] and a['condition']==r['target'] and b['condition']==r['base'];seen.add((r['query_id'],r['target'],r['base']))
  delta=margins[a['request_id']]-margins[b['request_id']];assert dec(r['delta_m'])==delta and dec(r['bound'])==2*bound and dec(r['reference_aligned_delta'])==delta*(1 if refs[r['query_id']]=='无' else -1)
  pred,bp,ref=a['raw_prediction'],b['raw_prediction'],refs[r['query_id']];resolved=abs(margins[a['request_id']])>bound and abs(margins[b['request_id']])>bound
  assert r['transition']==(('repair' if pred==ref and bp!=ref else 'damage' if pred!=ref and bp==ref else 'unchanged') if resolved else 'unresolved')
  for k,v in r['trajectory_difference'].items():
   x,y=[np.asarray(data['native_trajectories'][rid][k],np.longdouble) for rid in [a['request_id'],b['request_id']]];assert abs(np.asarray(v,np.longdouble)-(x-y)).max()<1e-10
 assert seen=={(q,a,b) for q in refs for a,b in plan['contrasts']}
 history=Path(plan['prior_comparators']['work']);seal=read(history/'run-01/raw-seal.json');pinned={Path(x['path']).resolve():x for x in seal['native_records']};old_records={}
 for item in seal['native_records']:
  verify(item);old=read(item['path']);old_records[old['request_id']]=old
 replay=0
 for req in requests.values():
  if req['condition']!='M00':continue
  rid=req['request_id'];old=old_records['jctx-'+req['query_id']+'-D00'];fresh=records['native-production',rid]
  assert old['input_ids_sha256']==fresh['input_ids_sha256'] and old['prompt_sha256']==fresh['prompt_sha256']
  for prop in ['vector','states','trajectory']:verify(old[prop]);verify(fresh[prop])
  assert np.array_equal(np.load(old['vector']['path']),vecs['native-production',rid])
  oldbank=np.load(old['states']['path']);newbank=np.load(fresh['states']['path']);assert np.array_equal(oldbank[:,[old['positions'].index(i) for i in fresh['positions']]],newbank)
  with np.load(old['trajectory']['path']) as x,np.load(fresh['trajectory']['path']) as y:
   assert set(x.files)==set(y.files)
   for k in x.files:assert np.array_equal(x[k],y[k])
  replay+=1
 assert replay==6
 return {'status':'pass','Decimal_precision':120,'absolute_vectors':absolute,'margin_records':margin_count,'normalizations':len(scored),'trajectory_audit':ta,'native_inputs':30,'native_contrasts':36,'historical_replay':{'native_vectors':6,'native_banks_at_current_positions':6,'trajectories':6,'all_exact_equal':True},'fresh_margin_bound':float(bound),'all_original_gates_retained':True,'source':info(Path(__file__)),'raw_seal':info(run/'raw-seal.json'),'results':info(out/'manifest.json')}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with localcontext() as ctx:ctx.prec=120;r=audit(a.prepared,a.run,a.results)
 with a.output.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2)
 print(json.dumps(r,ensure_ascii=False))
