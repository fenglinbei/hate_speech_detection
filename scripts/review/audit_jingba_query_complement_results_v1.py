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
def audit_trajectories(records,vecs,data,profile,qualification):
    arrays={};projection_errors=[];guard_count=0;restore_guard_count=0;scalar_checks=0
    for key,r in records.items():
        item=r['trajectory']
        if item is None:
            assert r['stage'] in ['native-baseline','native-prefix','cross-unobserved']
            continue
        verify(item)
        with np.load(item['path'],allow_pickle=False) as z:t={k:z[k] for k in z.files}
        arrays[key]=t;h=t['states'].astype(np.float64);branches=t['branches'].astype(np.float64)
        assert h.shape==(36,3,profile['hidden_size']) and branches.shape==(36,2,profile['hidden_size'])
        assert np.array_equal(h[1:,0],h[:-1,2])
        normalized=h/np.sqrt(np.mean(h*h,axis=-1,keepdims=True)+t['norm_eps'][0])*t['norm_weight']
        def scaled(a,b):return abs(a-b).max()/max(1.,abs(a).max(),abs(b).max())
        for a,b in [(h[:,0]+branches[:,0],h[:,1]),(h[:,1]+branches[:,1],h[:,2]),(normalized,t['normalized'])]:
            assert scaled(a,b)<=.0001
        logits=t['normalized'].astype(np.float64)@t['label_weights'].astype(np.float64).T
        err=max(float(abs(logits-t['lens_logits']).max()),float(abs(t['lens_logits'][-1,2].astype(np.float64)-vecs[key][[profile['candidate_tokens']['有'],profile['candidate_tokens']['无']]]).max()))
        assert err==r['trajectory_validation']['projection_absolute_error'] and err<=.001
        projection_errors.append(err)
    for key,t in arrays.items():
        stage,identifier=key;r=records[key]
        if r['job']:
            req=r['request_id'];native_stage='native-'+stage.removeprefix('cross-') if stage in ['cross-repeat','cross-reverse','cross-left','cross-right'] else 'native-production' if stage=='production' else 'native-capture'
            base=arrays[native_stage,req]
            for name in ['states','normalized','lens_logits','branches']:assert np.array_equal(t[name][:18],base[name][:18])
            assert np.array_equal(t['states'][18,0],base['states'][18,0]);guard_count+=1
            if stage=='self':
                baseline=arrays['cross-probe',r['job']['upstream_job_id']] if r['job']['kind']=='conditional_self_control' else base
                for name in t:assert np.array_equal(t[name],baseline[name])
            if r['job']['restoration'] is not None:
                j=r['job']
                for index,spec in enumerate(j['restoration']):
                    li=spec['layer'];bi=0 if spec['branch']=='attention' else 1
                    if j['kind']=='self_control':u=base
                    else:
                        previous=j['upstream_job_id']
                        if index and j['kind']=='primary':previous=j['job_id'].replace('restore-joint','restore-L26-attention')
                        u=arrays['cross-probe' if stage=='self' else stage,previous]
                    for name in ['states','normalized','lens_logits','branches']:assert np.array_equal(t[name][:li],u[name][:li])
                    for name in ['states','normalized','lens_logits']:assert np.array_equal(t[name][li,:bi+1],u[name][li,:bi+1])
                    if bi==1:assert np.array_equal(t['branches'][li,0],u['branches'][li,0])
                    source=arrays['cross-probe',j['upstream_job_id']] if spec['source']=='upstream' else arrays['native-production' if stage=='production' else 'native-capture',req]
                    assert np.array_equal(t['branches'][li,bi],source['branches'][li,bi]);restore_guard_count+=1
        reference=None
        if stage in ['native-repeat','native-reverse','native-left','native-right','native-production']:reference=arrays['native-capture',identifier]
        if stage in ['cross-repeat','cross-reverse','cross-left','cross-right','production']:reference=arrays['cross-probe',identifier]
        if reference is not None:
            exact=stage not in ['native-left','native-right','cross-left','cross-right']
            for name in ['states','normalized','lens_logits','branches']:
                if exact:assert np.array_equal(t[name],reference[name])
                else:assert abs(t[name].astype(np.float64)-reference[name]).max()/max(1.,abs(t[name]).max(),abs(reference[name]).max())<=.0001
            projection_errors.append(float(abs(t['lens_logits'].astype(np.float64)-reference['lens_logits']).max()))
    assert qualification['trajectory']['probe_absolute_error_bound']==max(1e-6,2*max(projection_errors))
    def independent(t):
        h=t['states'].astype(np.longdouble);b=t['branches'].astype(np.longdouble)
        w=(t['label_weights'][1].astype(np.longdouble)-t['label_weights'][0].astype(np.longdouble))*t['norm_weight'].astype(np.longdouble)
        rms=np.sqrt(np.mean(h*h,axis=-1)+t['norm_eps'][0]);num=np.sum(h*w,axis=-1)
        branch=np.stack([np.sum(b[:,0]*w,axis=-1)/rms[:,1],np.sum(b[:,1]*w,axis=-1)/rms[:,2]],axis=1)
        scale=num[:,:2]*(1/rms[:,1:]-1/rms[:,:2])
        probe=t['lens_logits'][...,1].astype(np.float64)-t['lens_logits'][...,0].astype(np.float64)
        return {'probe_pre_mid_post':probe,'increment_attention_mlp':np.diff(probe,axis=1),
            'branch_projection_at_destination_scale':branch,'existing_residual_rescaling':scale,
            'floating_remainder':np.diff(probe,axis=1)-branch-scale,'rms_pre_mid_post':rms,
            'state_l2_pre_mid_post':np.sqrt(np.sum(h*h,axis=-1))}
    native={}
    def compare(saved,values):
        nonlocal scalar_checks
        for name,expected in values.items():
            actual=np.asarray(saved[name],dtype=np.longdouble)
            assert actual.shape==expected.shape and abs(actual-expected).max()<1e-10,(name,float(abs(actual-expected).max()))
            scalar_checks+=actual.size
    for rid,saved in data['native_trajectories'].items():
        native[rid]=independent(arrays['native-production',rid]);compare(saved,native[rid])
    for e in data['effects']:
        t=arrays['production',e['job_id']];values=independent(t);saved=data['patched_trajectories'][e['job_id']]
        compare(saved['values'],values)
        diffs={k:values[k]-native[e['recipient']][k] for k in values};compare(saved['minus_recipient'],diffs)
        assert abs(float(diffs['probe_pre_mid_post'][-1,2])-e['delta_m'])<=4*qualification['trajectory']['probe_absolute_error_bound']+e['delta_bound']
    for pair in data['restoration_contrasts']:
        a=independent(arrays['production',pair['job_id']]);b=independent(arrays['production',pair['upstream_job_id']])
        compare(pair['trajectory_difference'],{k:a[k]-b[k] for k in a})
    for pair in data['joint_contrasts']:
        j,a,b,u=[independent(arrays['production',pair[key]]) for key in ['job_id','A_job_id','B_job_id','upstream_job_id']]
        compare(pair['trajectory_AB_minus_A'],{k:j[k]-a[k] for k in j})
        compare(pair['trajectory_AB_minus_B'],{k:j[k]-b[k] for k in j})
        compare(pair['trajectory_interaction'],{k:j[k]-a[k]-b[k]+u[k] for k in j})
    for pair in data['position_differences']:
        a=independent(arrays['production',pair['focal_job_id']]);b=independent(arrays['production',pair['pre_job_id']])
        compare(pair['trajectory_difference'],{k:a[k]-b[k] for k in a})
    for pair in data['scope_comparisons']:
        a=independent(arrays['production',pair['whole_job_id']]);b=independent(arrays['production',pair['focal_job_id']])
        compare(pair['trajectory_difference'],{k:a[k]-b[k] for k in a})
    for pair in data['query_interactions']:
        w,u,cc=[independent(arrays['production',pair[k]]) for k in ['whole_job_id','focal_job_id','complement_job_id']]
        n=native[pair['recipient']]
        compare(pair['trajectory_interaction'],{k:w[k]-u[k]-cc[k]+n[k] for k in w})
    for gap in data['condition_gaps']:
        a=native[gap['D01_request']];b=native[gap['D02_request']]
        compare(gap['trajectory_difference'],{k:b[k]-a[k] for k in a})
    for gap in data['dictionary_addition_gaps']:
        a=native[gap['base_request']];b=native[gap['target_request']]
        compare(gap['trajectory_difference'],{k:b[k]-a[k] for k in a})
    return {'trajectory_files':len(arrays),'early_position_guards':guard_count,'restoration_boundary_guards':restore_guard_count,
        'independent_extended_precision_summary_values':scalar_checks,
        'fresh_probe_bound':qualification['trajectory']['probe_absolute_error_bound']}


def audit(p,run,out):
    p,run,out=map(Path,(p,run,out));state=read(run/'state.json')
    assert state['status']=='complete' and state['worker_exit_code']==0 and state['owned_worker_absent'];verify(state['resource_release'])
    for name in (p/'manifest.json',out/'manifest.json'):
        m=read(name)
        for r in m['artifacts']+m.get('sources',[]):verify(r)
    q=read(run/'qualification.json');data=read(out/'results.json');plan=read(p/'execution-plan.json');profile=read(p/'model-profile.json')
    refs={r['query_id']:r['reference'] for r in read(p/'analysis-references.json')['references']}
    bound=dec(q['margin_error_bound']);assert dec(data['margin_error_bound'])==bound
    vecs={};records={};absolute=0;margin_count=0
    def vector(item):
        nonlocal absolute
        verify(item);v=np.load(item['path'],allow_pickle=False);assert v.dtype==np.float32 and v.shape==(profile['vocab_size'],) and np.isfinite(v).all();absolute+=1
        return v
    def scorecheck(v,s):
        nonlocal margin_count
        y,n=dec(v[profile['candidate_tokens']['有']]),dec(v[profile['candidate_tokens']['无']]);m=n-y;assert dec(s['m'])==m and dec(s['z_yes'])==y and dec(s['z_no'])==n
        assert s['raw_prediction']==('无' if m>0 else '有' if m<0 else None);margin_count+=1
        # Full-vocabulary normalization uses independent extended precision.
        # Decimal is used for all exact margins, effects, ratios and bounds.
        assert np.finfo(np.longdouble).eps < np.finfo(np.float64).eps
        vv=v.astype(np.longdouble);top=vv.max();lnz=top+np.log(np.exp(vv-top).sum(dtype=np.longdouble))
        yy,nn=vv[profile['candidate_tokens']['有']],vv[profile['candidate_tokens']['无']];pt=max(yy,nn);lse=pt+np.log(np.exp(yy-pt)+np.exp(nn-pt))
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
        assert tokens==r['tokens'] and len(tokens)==2 and tokens[0] in [profile['candidate_tokens']['有'],profile['candidate_tokens']['无']] and tokens[1] in profile['eos_token_ids'] and r['exact_label_then_eos']
    margins={};states={};requests={r['request_id']:r for r in [json.loads(s) for s in (p/'scoring-inputs.jsonl').read_text().splitlines()]}
    for s in data['baselines']:
        rid=s['request_id'];v=vecs['native-production',rid];margins[rid]=scorecheck(v,s);states[rid]=np.load(records['native-production',rid]['states']['path'],allow_pickle=False)
        assert np.array_equal(v,vecs['native-capture',rid])
        assert np.array_equal(states[rid],np.load(records['native-capture',rid]['states']['path'],allow_pickle=False))
        sign=1 if refs[s['query_id']]=='无' else -1;assert dec(s['reference_aligned_margin'])==sign*margins[rid]
    effect_count=0;ratio_count=0;maxpad=Decimal(0);state_errors=[]
    for rid,req in requests.items():
        baseline=vecs['native-baseline',rid];capture=vecs['native-capture',rid]
        assert np.array_equal(baseline,capture)
        for stage in ['native-repeat','native-reverse','native-left','native-right','native-prefix']:
            other=vecs[stage,rid]
            if stage in ['native-repeat','native-reverse']:assert np.array_equal(capture,other)
            if stage!='native-prefix':maxpad=max(maxpad,abs((dec(other[profile['candidate_tokens']['无']])-dec(other[profile['candidate_tokens']['有']]))-margins[rid]))
            ss=np.load(records[stage,rid]['states']['path'],allow_pickle=False).astype(np.float64);ref=states[rid].astype(np.float64)
            error=float(np.abs(ss-ref).max()/max(1.,float(np.abs(ss).max()),float(np.abs(ref).max())))
            assert error<=plan['state_scaled_cap'];state_errors.append(error)
    ledger=read(p/'interventions.json');es={e['job_id']:e for e in data['effects']}
    assert set(es)=={j['job_id'] for j in ledger['cross']} and len(data['effects'])==48
    for j in ledger['cross']:
        e=es[j['job_id']];assert all(e[k]==v for k,v in j.items());key=j['job_id'];ref=vecs['cross-probe',key];v=vecs['production',key];assert np.array_equal(v,ref)
        for stage in ['cross-unobserved','cross-repeat','cross-reverse','cross-left','cross-right']:
            other=vecs[stage,key]
            if stage in ['cross-unobserved','cross-repeat','cross-reverse']:assert np.array_equal(ref,other)
            difference=abs((dec(other[profile['candidate_tokens']['无']])-dec(other[profile['candidate_tokens']['有']]))-(dec(ref[profile['candidate_tokens']['无']])-dec(ref[profile['candidate_tokens']['有']])))
            assert difference<=dec(plan['acceptance']['margin_padding_prefix_cap']);maxpad=max(maxpad,difference)
        m=dec(v[profile['candidate_tokens']['无']])-dec(v[profile['candidate_tokens']['有']]);r=margins[j['recipient']];d=margins[j['donor']];delta=m-r;gap=d-r
        assert dec(e['m'])==m and dec(e['delta_m'])==delta and dec(e['donor_gap'])==gap and dec(e['delta_bound'])==2*bound
        assert dec(e['reference_aligned_delta'])==delta*(1 if refs[j['query_id']]=='无' else -1)
        assert e['direction']==('positive' if delta>2*bound else 'negative' if delta< -2*bound else 'unresolved')
        if abs(gap)<=2*bound:assert e['donor_gap_fraction'] is None and e['donor_gap_fraction_interval'] is None
        else:
            close(e['donor_gap_fraction'],delta/gap)
            vs=[(pv-rv)/(dv-rv) for pv,dv,rv in itertools.product((m-bound,m+bound),(d-bound,d+bound),(r-bound,r+bound))]
            close(e['donor_gap_fraction_interval'][0],min(vs));close(e['donor_gap_fraction_interval'][1],max(vs));ratio_count+=1
        pred=lambda x:'无' if x>0 else '有' if x<0 else None
        resolved=abs(m)>bound and abs(r)>bound
        assert e['prediction']==pred(m) and e['flip']==(pred(m)!=pred(r) if resolved else None)
        reference='无' if refs[j['query_id']]=='无' else '有'
        transition=('repair' if pred(m)==reference and pred(r)!=reference else 'damage' if pred(r)==reference and pred(m)!=reference else 'unchanged') if resolved else 'unresolved'
        assert e['transition']==transition
        if j['last_layer_expected_zero']:assert np.array_equal(v,vecs['native-production',j['recipient']])
        req=requests[j['recipient']];inds=[req['capture_positions'].index(i) for i in j['positions']]
        dreq=requests[j['donor']];dinds=[dreq['capture_positions'].index(i) for i in j['donor_positions']]
        dr=states[j['donor']][j['layer'],dinds];rr=states[j['recipient']][j['layer'],inds]
        assert hashlib.sha256(dr.tobytes()).hexdigest()==records['production',key]['patch_proof']['donor_sha256']
        diff=dr.astype(np.float64)-rr.astype(np.float64)
        assert math.isclose(e['donor_state_l2'],float(np.sqrt((diff*diff).sum())),rel_tol=1e-14,abs_tol=1e-12)
        assert e['replacement_token_count']==len(inds)
        assert np.allclose(e['donor_state_l2_per_token'],np.sqrt(np.sum(diff.astype(np.longdouble)**2,axis=-1)),rtol=0,atol=1e-10)
        assert abs(np.longdouble(e['donor_state_rms_per_coordinate'])-np.sqrt(np.mean(diff.astype(np.longdouble)**2)))<1e-10
        effect_count+=1
    assert bound==max(dec(plan['acceptance']['margin_bound_floor']),2*maxpad)
    assert q['state_scaled_error_bound']==max(1e-6,2*max(state_errors))
    for row in data['self_controls']:
        assert row['delta_m']==0
        key=('cross-probe',row['upstream_job_id']) if row['kind']=='conditional_self_control' else ('native-capture',row['recipient'])
        assert np.array_equal(vecs['self',row['job_id']],vecs[key])
    assert len(data['self_controls'])==48 and len(data['restoration_contrasts'])==0 and len(data['joint_contrasts'])==0
    for pair in data['restoration_contrasts']:
        n=margins[pair['recipient']];u=dec(es[pair['upstream_job_id']]['m']);r=dec(es[pair['job_id']]['m'])
        assert all(dec(pair[k])==v for k,v in [('native_m',n),('upstream_m',u),('restored_m',r),('upstream_delta_m',u-n),('remaining_delta_m',r-n),('removed_delta_m',u-r),('difference_bound',2*bound)])
        assert pair['denominator_safe']==(abs(u-n)>2*bound)
        assert pair['absolute_effect_reduced']==(abs(r-n)<abs(u-n)-4*bound)
        assert pair['absolute_effect_increased']==(abs(r-n)>abs(u-n)+4*bound)
        if abs(u-n)>2*bound:
            close(pair['removed_fraction'],(u-r)/(u-n))
            corners=[(uu-rr)/(uu-nn) for nn,uu,rr in itertools.product([n-bound,n+bound],[u-bound,u+bound],[r-bound,r+bound])]
            close(pair['removed_fraction_interval'][0],min(corners));close(pair['removed_fraction_interval'][1],max(corners))
        else:assert pair['removed_fraction'] is None and pair['removed_fraction_interval'] is None
        assert dec(pair['reference_aligned_restore_minus_upstream'])==(r-u)*(1 if refs[pair['query_id']]=='无' else -1)
    for pair in data['joint_contrasts']:
        n=margins[pair['recipient']]
        u,a,b,j=[dec(es[pair[key]]['m']) for key in ['upstream_job_id','A_job_id','B_job_id','job_id']]
        interaction=j-a-b+u
        for key,value in [('native_m',n),('upstream_m',u),('A_m',a),('B_m',b),('AB_m',j),
            ('AB_minus_A',j-a),('AB_minus_B',j-b),('pair_difference_bound',2*bound),
            ('interaction_m',interaction),('interaction_bound',4*bound),
            ('additional_removed_after_A',a-j),('additional_removed_after_B',b-j)]:assert dec(pair[key])==value
        assert pair['interaction_resolved']==(abs(interaction)>4*bound)
        assert pair['absolute_reduced_vs_A']==(abs(j-n)<abs(a-n)-4*bound)
        assert pair['absolute_reduced_vs_B']==(abs(j-n)<abs(b-n)-4*bound)
        assert pair['denominator_safe']==(abs(u-n)>2*bound)
        for key,expression in [('interaction_fraction',lambda nn,uu,aa,bb,jj:(jj-aa-bb+uu)/(uu-nn)),
            ('additional_fraction_after_A',lambda nn,uu,aa,bb,jj:(aa-jj)/(uu-nn)),
            ('additional_fraction_after_B',lambda nn,uu,aa,bb,jj:(bb-jj)/(uu-nn))]:
            if abs(u-n)>2*bound:
                close(pair[key],expression(n,u,a,b,j))
                corners=[expression(*values) for values in itertools.product(*[(v-bound,v+bound) for v in [n,u,a,b,j]])]
                close(pair[key+'_interval'][0],min(corners));close(pair[key+'_interval'][1],max(corners))
            else:assert pair[key] is None and pair[key+'_interval'] is None
    assert absolute==plan['budget']['usual_total']
    trajectory_audit=audit_trajectories(records,vecs,data,profile,q)
    branch_proofs=audit_branch_sources(run,records,requests,profile)
    replay=audit_historical_replay(p,plan,records)
    additional=audit_new_case_comparisons(data,margins,es,refs,bound)
    prefix_audit=audit_shared_prefix(data,records,requests)
    scope_count=audit_scope(data,margins,es,refs,bound,plan)
    interaction_count=audit_query_interactions(data,margins,es,bound)


    return {'status':'pass','Decimal_precision_for_margins_and_expressions':120,'absolute_vectors':absolute,'margin_records':margin_count,'distinct_extended_precision_normalizations':len(scored_hashes),
            'trajectory_audit':trajectory_audit,'effects':effect_count,'ratios_and_bounds':ratio_count,'restoration_contrasts':len(data['restoration_contrasts']),'joint_contrasts':len(data['joint_contrasts']),'self_controls':len(data['self_controls']),'new_case_comparisons':additional,'branch_proofs':branch_proofs,'shared_prefix_checks':prefix_audit,'historical_replay':replay,'scope_comparisons':scope_count,'query_interactions':interaction_count,'last_layer_controls':0,
            'fresh_margin_bound':float(bound),'all_original_gates_retained':True,'source':info(Path(__file__)),'raw_seal':info(run/'raw-seal.json'),'results':info(out/'manifest.json')}
def audit_new_case_comparisons(data,margins,effects,refs,bound):
    assert refs=={f'J{i:02}':('无' if i<=8 else '有') for i in range(5,11)}
    assert len(data['baselines'])==18 and len(data['condition_gaps'])==6 and len(data['position_differences'])==12 and len(data['dictionary_addition_gaps'])==12
    for gap in data['condition_gaps']:
        value=margins[gap['D02_request']]-margins[gap['D01_request']]
        assert dec(gap['delta_m_D02_minus_D01'])==value and dec(gap['bound'])==2*bound
        assert dec(gap['reference_aligned_delta'])==value*(1 if refs[gap['query_id']]=='无' else -1)
    for pair in data['position_differences']:
        a,b=effects[pair['focal_job_id']],effects[pair['pre_job_id']]
        assert a['recipient']==b['recipient']==pair['recipient'] and a['condition']=='upstream' and b['condition']=='preceding'
        value=dec(a['m'])-dec(b['m'])
        assert dec(pair['delta_m'])==value and dec(pair['bound'])==2*bound
        assert dec(pair['reference_aligned_delta'])==value*(1 if refs[pair['query_id']]=='无' else -1)
    provenance={r['query_id']:r for r in data['reference_provenance']['references']}
    assert all(r['reference_basis']=='user_bulk_adoption_of_proposed_current_task_reference' and r['source_labels'] is None for r in provenance.values())
    for gap in data['dictionary_addition_gaps']:
        value=margins[gap['target_request']]-margins[gap['base_request']]
        assert dec(gap['delta_m'])==value and dec(gap['bound'])==2*bound
        assert dec(gap['reference_aligned_delta'])==value*(1 if refs[gap['query_id']]=='无' else -1)
    return {'six_condition_gaps':True,'twelve_dictionary_addition_gaps':True,'twelve_focal_minus_preceding':True,'six_accepted_references_preserved':True}


def audit_branch_sources(run,records,requests,profile):
    count=0
    arrays={}
    for key,r in records.items():
        if r['trajectory']:
            with np.load(r['trajectory']['path'],allow_pickle=False) as z:arrays[key]=z['branches'].copy()
    def proof_check(stage,j,proof,source,req):
        nonlocal count
        specs=j['restoration']
        if specs is None:
            assert proof['restoration'] is None and source is None;return
        assert len(specs)==len(source)==len(proof['restoration'])
        for index,(spec,source) in enumerate(zip(specs,source)):
            li=spec['layer'];bi=0 if spec['branch']=='attention' else 1
            source_key=('cross-probe',j['upstream_job_id']) if spec['source']=='upstream' else ('native-production' if stage=='production' else 'native-capture',j['recipient'])
            sr=records[source_key];verify(source['record']);verify(source['trajectory'])
            assert source['record']==info(run/'records'/source_key[0]/(source_key[1]+'.json')) and source['trajectory']==sr['trajectory']
            original=arrays[source_key][li,bi]
            for k in ['layer','branch','position','source']:assert source[k]==spec[k]
            h=hashlib.sha256(original.tobytes()).hexdigest();rp=proof['restoration'][index]
            assert source['replacement_sha256']==h==rp['replacement_sha256']
            if j['kind']=='self_control':before_key=('native-capture',j['recipient'])
            else:
                previous=j['upstream_job_id']
                if index and j['kind']=='primary':previous=j['job_id'].replace('restore-joint','restore-L26-attention')
                before_key=('cross-probe' if stage in ['self','format','cross-unobserved'] else stage,previous)
            before=arrays[before_key][li,bi]
            assert rp['before_sha256']==hashlib.sha256(before.tobytes()).hexdigest()
            assert rp['layer']==li and rp['branch']==spec['branch'] and rp['position']==req['prompt_tokens']-1
            # Native padding helper is independently checked in input/tiny tests;
            # here enforce actual offsets used by its fixed 3-token padding contract.
            if stage=='cross-left':assert rp['padded_position']-rp['position']==proof['padded_positions'][0]-proof['positions'][0]>0
            else:assert rp['padded_position']==rp['position']
            assert rp['outside_rows_exact'] and rp['replacement_exact'] and rp['native_output_unmodified']
            assert set(rp['changed_rows_in_unpadded_coordinates'])<={spec['position']}
            for name,value in [('replacement_l2',original),('before_l2',before),('difference_l2',original.astype(np.float64)-before)]:
                expected=np.sqrt(np.sum(value.astype(np.longdouble)**2,dtype=np.longdouble))
                assert abs(np.longdouble(rp[name])-expected)<1e-10
            count+=1
    for (stage,key),r in records.items():
        if r['job']:proof_check(stage,r['job'],r['patch_proof'],r['restoration_source'],requests[r['request_id']])
    for file in sorted((run/'format').glob('*.json')):
        f=read(file)
        if f['job']:
            for step in f['steps']:proof_check('format',f['job'],step['patch_proof'],f['restoration_source'],requests[f['job']['recipient']])
    return count



def audit_shared_prefix(data,records,requests):
    assert len(data['shared_prefix_comparison'])==3
    for item in data['shared_prefix_comparison']:
        a,b=[next(r for r in requests.values() if r['query_id']==q and r['dictionary_id']==item['dictionary_id']) for q in ['J08','J10']]
        assert a['input_ids'][:a['focal_prefix_length']]==b['input_ids'][:b['focal_prefix_length']]
        assert a['patch_position_sets']['focal']==b['patch_position_sets']['focal'] and item['token_prefix_equal']
        for stage,key in [('native-prefix','query_prefix'),('native-production','full_prompt')]:
            aa,bb=[np.load(records[stage,r['request_id']]['states']['path'],allow_pickle=False).astype(np.float64) for r in [a,b]]
            for name,inds in [('focal',[a['capture_positions'].index(i) for i in a['patch_position_sets']['focal']]),('preceding',[a['capture_positions'].index(i) for i in a['patch_position_sets']['pre']])]:
                binds=[b['capture_positions'].index(i) for i in b['patch_position_sets']['focal' if name=='focal' else 'pre']]
                x,y=aa[:,inds],bb[:,binds];saved=item[key][name]
                diff=abs(x-y).max(axis=(1,2));scale=np.maximum(1.,np.maximum(abs(x).max(axis=(1,2)),abs(y).max(axis=(1,2))))
                assert np.array_equal(diff,saved['max_abs_by_layer'])
                assert np.array_equal(diff/scale,saved['scaled_by_layer'])
                assert saved['byte_equal']==np.array_equal(x,y)
        assert item['query_prefix']['focal']['byte_equal'] and item['query_prefix']['preceding']['byte_equal']
    return 3


def audit_scope(data,margins,es,refs,bound,plan):
    assert len(data['scope_comparisons'])==12
    assert {(r['query_id'],r['recipient']) for r in data['scope_comparisons']}=={(e['query_id'],e['recipient']) for e in es.values() if e['condition']=='whole-query'}
    for r in data['scope_comparisons']:
        u,w=es[r['focal_job_id']],es[r['whole_job_id']]
        assert u['condition']=='upstream' and w['condition']=='whole-query' and u['recipient']==w['recipient']==r['recipient'] and u['donor']==w['donor']==r['donor']
        n,d=margins[r['recipient']],margins[r['donor']];a,b=dec(u['m']),dec(w['m']);gain=abs(d-a)-abs(d-b)
        for k,v in [('native_m',n),('donor_m',d),('focal_m',a),('whole_m',b),('focal_delta_m',a-n),('whole_delta_m',b-n),('whole_minus_focal',b-a),('pair_difference_bound',2*bound),('donor_minus_focal',d-a),('donor_minus_whole',d-b),('donor_distance_improvement',gain),('distance_improvement_bound',4*bound)]:assert dec(r[k])==v,(k,r[k],v)
        assert r['closer_to_donor']==(gain>4*bound) and r['farther_from_donor']==(gain< -4*bound)
        for prefix,e in [('focal',u),('whole',w)]:
            assert r[prefix+'_tokens']==e['replacement_token_count'] and r[prefix+'_l2']==e['donor_state_l2'] and r[prefix+'_prediction']==e['prediction']
        assert r['whole_transition']==w['transition']
    shift=data['CPU_score_shift'];assert shift['offset']==plan['CPU_score_shift']['offset']==7.0
    assert shift['fit_on_current_results'] is False and shift['new_model_forwards']==0 and shift['independent_validation'] is False
    assert len(shift['records'])==6 and {r['query_id'] for r in shift['records']}==set(refs)
    for r in shift['records']:
        n=margins[r['request_id']];m=n+dec(7.0)
        assert r['request_id'].endswith('D01') and dec(r['native_m'])==n and dec(r['shifted_m'])==m
        assert r['shifted_prediction']==('无' if m>0 else '有' if m<0 else None)
        assert r['correct']==(r['shifted_prediction']==refs[r['query_id']]) and r['reference']==refs[r['query_id']]
    assert shift['correct']==sum(r['correct'] for r in shift['records']) and shift['total']==6
    return 12


def audit_query_interactions(data,margins,es,bound):
    assert len(data['query_interactions'])==12
    assert {(r['query_id'],r['recipient']) for r in data['query_interactions']}=={(e['query_id'],e['recipient']) for e in es.values() if e['condition']=='query-complement'}
    for r in data['query_interactions']:
        u,cc,w=[es[r[k]] for k in ['focal_job_id','complement_job_id','whole_job_id']]
        assert [e['condition'] for e in [u,cc,w]]==['upstream','query-complement','whole-query']
        assert all(e['recipient']==r['recipient'] and e['donor']==r['donor'] and e['query_id']==r['query_id'] for e in [u,cc,w])
        n=margins[r['recipient']];uu,cv,ww=[dec(e['m']) for e in [u,cc,w]];ii=ww-uu-cv+n
        for k,v in [('native_m',n),('focal_m',uu),('complement_m',cv),('whole_m',ww),('focal_delta_m',uu-n),('complement_delta_m',cv-n),('whole_delta_m',ww-n),('conditional_complement_delta_m',ww-uu),('conditional_focal_delta_m',ww-cv),('additive_prediction_delta_m',uu+cv-2*n),('interaction_m',ii),('interaction_bound',4*bound),('pair_difference_bound',2*bound)]:assert dec(r[k])==v,(k,r[k],v)
        assert r['interaction_resolved']==(abs(ii)>4*bound)
        assert r['interaction_direction']==('positive' if ii>4*bound else 'negative' if ii< -4*bound else 'unresolved')
        for name,e in [('focal',u),('complement',cc),('whole',w)]:assert r[name+'_tokens']==e['replacement_token_count'] and r[name+'_l2']==e['donor_state_l2']
        assert r['focal_tokens']+r['complement_tokens']==r['whole_tokens']
        assert abs(dec(r['whole_l2'])**2-dec(r['focal_l2'])**2-dec(r['complement_l2'])**2)<Decimal('1e-8')
        assert r['complement_prediction']==cc['prediction'] and r['complement_transition']==cc['transition'] and r['whole_prediction']==w['prediction']
        # This independent contrast check also protects the separately exported curves.
        ts=[data['patched_trajectories'][e['job_id']] for e in [w,u,cc]]
        for k in ['probe_pre_mid_post','increment_attention_mlp','branch_projection_at_destination_scale','existing_residual_rescaling','floating_remainder','rms_pre_mid_post','state_l2_pre_mid_post']:
            expected=np.asarray(ts[0]['values'][k],dtype=np.longdouble)-np.asarray(ts[1]['values'][k],dtype=np.longdouble)-np.asarray(ts[2]['minus_recipient'][k],dtype=np.longdouble)
            assert np.max(np.abs(np.asarray(r['trajectory_interaction'][k],dtype=np.longdouble)-expected))<1e-10
    return 12


def audit_historical_replay(prepared,plan,records):
    historic=Path(plan['prior_comparators']['work']);oldrun=historic/'run-01'
    # The real input audit/seal pins the previous raw seal before GPU execution.
    oldseal=read(oldrun/'raw-seal.json')
    pinned={Path(x['path']).resolve():x for x in oldseal['native_records']+oldseal['records']}
    count=0;native_banks=0;trajectories=0
    for (stage,key),rec in records.items():
        if stage!='native-production' and not (stage=='production' and rec['job']['condition'] in ['upstream','preceding','whole-query']):continue
        path=oldrun/'records'/stage/(key+'.json');assert path.resolve() in pinned
        verify(pinned[path.resolve()]);old=read(path)
        assert rec['job']==old['job'] and rec['request_id']==old['request_id']
        assert rec['input_ids_sha256']==old['input_ids_sha256'] and rec['prompt_sha256']==old['prompt_sha256']
        for prop in ['vector','states']:
            a,b=rec[prop],old[prop];assert (a is None)==(b is None)
            if a is None:continue
            verify(a);verify(b)
            av,bv=np.load(a['path'],allow_pickle=False),np.load(b['path'],allow_pickle=False)
            if prop=='states':
                assert old['positions']==rec['positions']
                av=av[:,[rec['positions'].index(i) for i in old['positions']]]
            assert np.array_equal(av,bv),(key,prop)
            if prop=='states':native_banks+=1
        a,b=rec['trajectory'],old['trajectory'];verify(a);verify(b)
        with np.load(a['path'],allow_pickle=False) as x,np.load(b['path'],allow_pickle=False) as y:
            assert set(x.files)==set(y.files)
            for field in x.files:assert np.array_equal(x[field],y[field]),(key,field)
        count+=1;trajectories+=1
    assert count==54 and native_banks==18 and trajectories==54
    return {'applicable':True,'native_vectors':18,'upstream_preceding_and_whole_vectors':36,'trajectories':trajectories,
        'native_state_banks_compared_at_old_positions':native_banks,'all_exact_equal':True,'old_scores_substituted':False,'previous_raw_seal':info(oldrun/'raw-seal.json')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with localcontext() as context:context.prec=120;result=audit(a.prepared,a.run,a.results)
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
