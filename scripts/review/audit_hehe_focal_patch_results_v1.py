#!/usr/bin/env python3
"""Independent 120-digit score and all-intervention reconstruction after release."""
import argparse,hashlib,itertools,json,math
from decimal import Decimal,localcontext
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(Path(p).read_text())
def info(p):
    p=Path(p).absolute();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def verify(r):assert all(info(r['path'])[k]==r[k] for k in ('path','bytes','sha256')),r['path']
def dec(x):return Decimal.from_float(float(x))
def close(actual,expected,tol=1e-12):assert abs(dec(actual)-expected)<=Decimal(str(tol)),(actual,str(expected))
def audit(p,run,out):
    p,run,out=map(Path,(p,run,out));state=read(run/'state.json')
    assert state['status']=='complete' and state['worker_exit_code']==0 and state['owned_worker_absent'];verify(state['resource_release'])
    for name in (p/'manifest.json',out/'manifest.json'):
        m=read(name)
        for r in m['artifacts']+m.get('sources',[]):verify(r)
    q=read(run/'qualification.json');data=read(out/'results.json');plan=read(p/'execution-plan.json');profile=read(p/'model-profile.json')
    bound=dec(q['margin_error_bound']);assert dec(data['margin_error_bound'])==bound
    vecs={};records={};absolute=0;margin_count=0
    def vector(item):
        nonlocal absolute
        verify(item);v=np.load(item['path'],allow_pickle=False);assert v.dtype==np.float32 and v.shape==(profile['vocab_size'],) and np.isfinite(v).all();absolute+=1
        return v
    def scorecheck(v,s):
        nonlocal margin_count
        y,n=dec(v[18830]),dec(v[42192]);m=n-y;assert dec(s['m'])==m and dec(s['z_yes'])==y and dec(s['z_no'])==n
        assert s['raw_prediction']==('无' if m>0 else '有' if m<0 else None);margin_count+=1
        # Full-vocabulary normalization uses independent extended precision.
        # Decimal is used for all exact margins, effects, ratios and bounds.
        assert np.finfo(np.longdouble).eps < np.finfo(np.float64).eps
        vv=v.astype(np.longdouble);top=vv.max();lnz=top+np.log(np.exp(vv-top).sum(dtype=np.longdouble))
        yy,nn=vv[18830],vv[42192];pt=max(yy,nn);lse=pt+np.log(np.exp(yy-pt)+np.exp(nn-pt))
        for k,x in [('log_p_yes',yy-lnz),('log_p_no',nn-lnz),('log_legal_mass',lse-lnz),('legal_mass',np.exp(lse-lnz)),('pair_support_no',np.exp(nn-lse))]:
            assert abs(np.longdouble(s[k])-x)<=np.longdouble('1e-12'),(k,s[k],str(x))
        return m
    # All margins and absolute vectors are checked; normalization is reused
    # only for byte-identical full vocabulary vectors.
    scored_hashes={}
    for file in sorted((run/'records').glob('*/*.json')):
        r=read(file);v=vector(r['vector']);key=(r['stage'],r['job']['job_id'] if r['job'] else r['request_id']);vecs[key]=v;records[key]=r
        if r['readout'] is not None:
            h=r['vector']['sha256']
            if h in scored_hashes:assert r['readout']==scored_hashes[h];margin_count+=1
            else:scorecheck(v,r['readout']);scored_hashes[h]=r['readout']
        else:assert r['stage']=='native-prefix'
        if r['states']:verify(r['states'])
    for f in sorted((run/'format').glob('*.json')):
        r=read(f);key=('cross-probe',r['key']) if r['job'] else ('native-capture',r['key']);tokens=[int(vecs[key].argmax())]
        for s in r['steps']:
            assert s['prefix_tokens']==tokens;v=vector(s['vector']);tokens.append(int(v.argmax()))
            if r['job']:
                j=r['job'];proof=s['patch_proof'];assert proof['donor_sha256']==records[key]['patch_proof']['donor_sha256']
                assert proof['layer']==j['layer'] and proof['positions']==proof['padded_positions']==j['positions']
        assert tokens==r['tokens'] and len(tokens)==2 and tokens[0] in [18830,42192] and tokens[1] in profile['eos_token_ids'] and r['exact_label_then_eos']
    margins={};states={};requests={r['request_id']:r for r in [json.loads(s) for s in (p/'scoring-inputs.jsonl').read_text().splitlines()]}
    for s in data['baselines']:
        rid=s['request_id'];v=vecs['native-production',rid];margins[rid]=scorecheck(v,s);states[rid]=np.load(records['native-production',rid]['states']['path'],allow_pickle=False)
        assert np.array_equal(v,vecs['native-capture',rid])
        assert np.array_equal(states[rid],np.load(records['native-capture',rid]['states']['path'],allow_pickle=False))
        sign=1 if s['query_id']=='Q01' else -1;assert dec(s['reference_aligned_margin'])==sign*margins[rid]
    effect_count=0;ratio_count=0;maxpad=Decimal(0);state_errors=[]
    for rid,req in requests.items():
        baseline=vecs['native-baseline',rid];capture=vecs['native-capture',rid]
        assert np.array_equal(baseline,capture)
        for stage in ['native-repeat','native-reverse','native-left','native-right','native-prefix']:
            other=vecs[stage,rid]
            if stage in ['native-repeat','native-reverse']:assert np.array_equal(capture,other)
            if stage!='native-prefix':maxpad=max(maxpad,abs((dec(other[42192])-dec(other[18830]))-margins[rid]))
            ss=np.load(records[stage,rid]['states']['path'],allow_pickle=False).astype(np.float64);ref=states[rid].astype(np.float64)
            error=float(np.abs(ss-ref).max()/max(1.,float(np.abs(ss).max()),float(np.abs(ref).max())))
            assert error<=plan['state_scaled_cap'];state_errors.append(error)
    ledger=read(p/'interventions.json');es={e['job_id']:e for e in data['effects']}
    assert set(es)=={j['job_id'] for j in ledger['cross']} and len(data['effects'])==288
    for j in ledger['cross']:
        e=es[j['job_id']];assert all(e[k]==v for k,v in j.items());key=j['job_id'];ref=vecs['cross-probe',key];v=vecs['production',key];assert np.array_equal(v,ref)
        for stage in ['cross-repeat','cross-reverse','cross-left','cross-right']:
            other=vecs[stage,key]
            if stage in ['cross-repeat','cross-reverse']:assert np.array_equal(ref,other)
            difference=abs((dec(other[42192])-dec(other[18830]))-(dec(ref[42192])-dec(ref[18830])))
            assert difference<=dec(plan['acceptance']['margin_padding_prefix_cap']);maxpad=max(maxpad,difference)
        m=dec(v[42192])-dec(v[18830]);r=margins[j['recipient']];d=margins[j['donor']];delta=m-r;gap=d-r
        assert dec(e['m'])==m and dec(e['delta_m'])==delta and dec(e['donor_gap'])==gap and dec(e['delta_bound'])==2*bound
        assert dec(e['reference_aligned_delta'])==delta*(1 if j['query_id']=='Q01' else -1)
        assert e['direction']==('positive' if delta>2*bound else 'negative' if delta< -2*bound else 'unresolved')
        if abs(gap)<=2*bound:assert e['donor_gap_fraction'] is None and e['donor_gap_fraction_interval'] is None
        else:
            close(e['donor_gap_fraction'],delta/gap)
            vs=[(pv-rv)/(dv-rv) for pv,dv,rv in itertools.product((m-bound,m+bound),(d-bound,d+bound),(r-bound,r+bound))]
            close(e['donor_gap_fraction_interval'][0],min(vs));close(e['donor_gap_fraction_interval'][1],max(vs));ratio_count+=1
        pred=lambda x:'无' if x>0 else '有' if x<0 else None
        resolved=abs(m)>bound and abs(r)>bound
        assert e['prediction']==pred(m) and e['flip']==(pred(m)!=pred(r) if resolved else None)
        reference='无' if j['query_id']=='Q01' else '有'
        transition=('repair' if pred(m)==reference and pred(r)!=reference else 'damage' if pred(r)==reference and pred(m)!=reference else 'unchanged') if resolved else 'unresolved'
        assert e['transition']==transition
        if j['last_layer_expected_zero']:assert np.array_equal(v,vecs['native-production',j['recipient']])
        req=requests[j['recipient']];inds=[req['capture_positions'].index(i) for i in j['positions']]
        dr=states[j['donor']][j['layer'],inds];rr=states[j['recipient']][j['layer'],inds]
        assert hashlib.sha256(dr.tobytes()).hexdigest()==records['production',key]['patch_proof']['donor_sha256']
        diff=dr.astype(np.float64)-rr.astype(np.float64)
        assert math.isclose(e['donor_state_l2'],float(np.sqrt((diff*diff).sum())),rel_tol=1e-14,abs_tol=1e-12)
        effect_count+=1
    assert bound==max(dec(plan['acceptance']['margin_bound_floor']),2*maxpad)
    assert q['state_scaled_error_bound']==max(1e-6,2*max(state_errors))
    for s in data['self_controls']:
        assert s['delta_m']==0 and s['changed_rows']==[]
        assert np.array_equal(vecs['self',s['job_id']],vecs['native-capture',s['recipient']])
    assert len(data['self_controls'])==288 and len(data['position_differences'])==144
    for pair in data['position_differences']:
        value=dec(es[pair['focal_job']]['m'])-dec(es[pair['pre_job']]['m']);assert dec(pair['delta_m'])==value and dec(pair['bound'])==2*bound
        assert dec(pair['reference_aligned_delta'])==value*(1 if pair['query_id']=='Q01' else -1)
    assert absolute==plan['budget']['usual_total']
    return {'status':'pass','Decimal_precision_for_margins_and_expressions':120,'absolute_vectors':absolute,'margin_records':margin_count,'distinct_extended_precision_normalizations':len(scored_hashes),
            'effects':effect_count,'ratios_and_bounds':ratio_count,'position_differences':144,'self_controls':288,'last_layer_controls':8,
            'fresh_margin_bound':float(bound),'all_original_gates_retained':True,'source':info(Path(__file__)),'raw_seal':info(run/'raw-seal.json'),'results':info(out/'manifest.json')}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with localcontext() as context:context.prec=120;result=audit(a.prepared,a.run,a.results)
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
