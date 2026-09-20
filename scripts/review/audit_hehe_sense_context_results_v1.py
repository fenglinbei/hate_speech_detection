#!/usr/bin/env python3
"""Independent post-release Decimal and raw-array audit; never loads a model."""
from __future__ import annotations
import argparse
import base64
from decimal import Decimal, localcontext
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from diagnostics import hehe_sense_context_inputs_v1 as c


def D(value):
    return Decimal.from_float(float(value))


def audit(prepared,run,results):
    prepared,run,results=map(Path,(prepared,run,results))
    state=c.read(run/'state.json')
    c.require(state['status']=='complete' and state['owned_worker_absent'] and state['worker_exit_code']==0,'Only complete normally released runs')
    c.verify(state['resource_release'])
    for record in c.read(results/'manifest.json')['artifacts']:c.verify(record)
    plan,profile,requests=c.validate(prepared)
    # Reconstruct component ownership from frozen metadata, independently of the report adapter.
    components={r['request_id']:r for r in c.read(prepared/'definition-components.json')['records']}
    for req in requests:
        item=components[req['request_id']]
        for part in item['components']:
            req['spans'].append({'id':'lex-0419:component:'+part['component'],
                'token_positions':part['owned_token_positions']})
        if item['separator_or_cross_boundary_tokens']:
            req['spans'].append({'id':'lex-0419:component:boundary',
                'token_positions':item['separator_or_cross_boundary_tokens']})
    refs={r['query_id']:r['reference'] for r in c.read(prepared/'analysis-references.json')['references']}
    by_id={r['request_id']:r for r in requests}
    q=c.read(run/'qualification.json')
    bound=D(q['margin_error_bound'])
    scores={s['request_id']:s for s in c.read(results/'scores.json')}
    expressions=c.read(results/'comparisons.json')
    registered={x['comparison_id']:x for x in c.read(prepared/'comparisons.json')['comparisons']}
    c.require(set(scores)==set(by_id) and set(x['comparison_id'] for x in expressions)==set(registered),'Result inventories differ')
    vectors=record_margins=decimal_values=aggregate_values=display_values=summary_values=0
    decimal_error=Decimal(0);aggregate_error=0.0
    production={};physical={};releases=0
    compact={(x['request_id'],x['role'],x['span_id']):x for x in c.read(results/'attention-summary.json')}
    with localcontext() as ctx:
        # The unchanged 1e-6 engineering floor needs more than 60 decimal digits
        # to represent its binary64 value exactly. Avoid rounding the independent
        # bound sum before exact comparison; no scientific tolerance is changed.
        ctx.prec=120
        for path in sorted((run/'records').glob('*/*.json')):
            rec=c.read(path);c.verify(rec['vector']);c.verify(rec['producer'])
            vec=np.load(rec['vector']['path'],allow_pickle=False)
            c.require(vec.dtype==np.float32 and vec.shape==(profile['vocab_size'],) and np.isfinite(vec).all(),'Bad absolute logits')
            yes,no=D(vec[18830]),D(vec[42192]);m=no-yes;vectors+=1
            if rec['readout'] is not None:
                saved=rec['readout']
                c.require(D(saved['m'])==m and D(saved['z_yes'])==yes and D(saved['z_no'])==no,'Decimal raw margin differs')
                c.require(abs((saved['log_p_no']-saved['log_p_yes'])-float(m))<=1e-12,'Log-probability difference identity')
                record_margins+=1
            if rec['stage']=='production':
                production[rec['request_id']]=m;physical[rec['request_id']]=rec['vector']['path']
                s=scores[rec['request_id']]
                resolution='positive' if m>bound else 'negative' if m < -bound else 'numerical_unresolved'
                pred='无' if m>0 else '有' if m<0 else None
                c.require(D(s['m'])==m and D(s['margin_error_bound'])==bound and s['raw_prediction']==pred and s['resolution']==resolution,'Qualified prediction differs')
                reference=refs[by_id[rec['request_id']]['query_id']]
                c.require(s['reference']==reference and s['raw_reference_correct']==(pred==reference),'Reference correctness differs')
                c.require(D(s['reference_aligned_margin'])==(m if reference=='无' else -m),'Reference-aligned margin sign differs')
                c.require(s['resolved_reference_correct']==((pred==reference) if resolution!='numerical_unresolved' else None),'Resolved correctness differs')
        for req in requests:
            f=c.read(run/'format'/(req['request_id']+'.json'))
            capture=c.read(run/'records/capture'/(req['request_id']+'.json'))
            first=np.load(capture['vector']['path'],allow_pickle=False)
            tokens=[int(first.argmax())]
            for step in f['steps']:
                c.verify(step['vector']);v=np.load(step['vector']['path'],allow_pickle=False)
                c.require(v.dtype==np.float32 and v.shape==(profile['vocab_size'],) and np.isfinite(v).all(),'Bad format logits')
                c.require(step['prefix_tokens']==tokens,'Format conditioning differs')
                tokens.append(int(v.argmax()));vectors+=1
            c.require(tokens==f['tokens'] and len(tokens)==2 and tokens[0] in (18830,42192) and tokens[1] in profile['eos_token_ids'],'Format gate differs')
        c.require(len(production)==9 and len(set(physical.values()))==9,'Production physical identity collision')
        for e in expressions:
            reg=registered[e['comparison_id']]
            c.require(e['terms']==reg['terms'],'Expression terms differ from freeze')
            coefficients={}
            for t in e['terms']:
                k=physical[t['request_id']]
                coefficients[k]=coefficients.get(k,0)+t['coefficient']
            value=sum((Decimal(t['coefficient'])*production[t['request_id']] for t in e['terms']),Decimal(0))
            error=sum((abs(Decimal(v))*bound for v in coefficients.values()),Decimal(0))
            c.require(D(e['delta_m'])==value==D(e['value']) and D(e['bound'])==error==D(e['numerical_bound']),'Decimal expression/bound differs')
            queries={by_id[t['request_id']]['query_id'] for t in e['terms']}
            c.require(len(queries)==1,'Cross-query contrast changed')
            reference=refs[next(iter(queries))]
            c.require(D(e['reference_aligned_delta'])==(value if reference=='无' else -value),'Reference-aligned contrast sign differs')
            direction='positive' if value>error else 'negative' if value < -error else 'unresolved'
            c.require(e['direction']==direction and e['direction_resolved']==(abs(value)>error),'Expression direction differs')
            if len(e['terms'])==2 and sorted(t['coefficient'] for t in e['terms'])==[-1,1]:
                plus=scores[next(t['request_id'] for t in e['terms'] if t['coefficient']==1)]
                minus=scores[next(t['request_id'] for t in e['terms'] if t['coefficient']==-1)]
                expected='unresolved' if 'numerical_unresolved' in (plus['resolution'],minus['resolution']) else (
                    'repair' if plus['raw_reference_correct'] and not minus['raw_reference_correct'] else
                    'damage' if minus['raw_reference_correct'] and not plus['raw_reference_correct'] else 'unchanged')
                c.require(e['transition']==expected,'Classification transition differs')
        for req in requests:
            rec=c.read(run/'records/production'/(req['request_id']+'.json'));c.verify(rec['attention'])
            attention=np.load(rec['attention']['path'],allow_pickle=False)
            display=c.read(results/(req['request_id']+'.view.json'))
            c.require(display['request_id']==req['request_id'] and display['prompt_sha256']==req['prompt_sha256']
                      and display['source']==rec['attention'] and display['score']==scores[req['request_id']], 'Display source binding differs')
            visual=np.frombuffer(base64.b64decode(display['data']),dtype='<f4').reshape(display['shape'])
            c.require(visual.shape==attention.shape and np.array_equal(visual,attention.astype('<f4')),'Visualization copy differs from native raw attention')
            display_values+=visual.size
            artifact=c.read(results/(req['request_id']+'-aggregates.json'))
            c.require(artifact['source']==rec['attention'],'Aggregate raw source differs')
            aggregates=artifact['aggregates'];spans={s['id']:s for s in req['spans']}
            c.require(display['aggregates']==aggregates,'Viewer aggregates differ from FP64 analysis')
            c.require(len(aggregates)==len(c.ROLES)*len(spans),'Aggregate inventory')
            for a in aggregates:
                role=a['role'];ri=c.ROLES.index(role);targets=req['roles'][role];keys=spans[a['span_id']]['token_positions']
                count=sum(sum(k<=pos for k in keys) for pos in targets)
                visible=bool(targets and count)
                c.require(a['visible']==visible,'Visibility/NA differs')
                if not visible:
                    c.require(a['mass'] is None and a['density'] is None,'Absent material reported as zero measurement');continue
                denominator=Decimal(count)/Decimal(len(targets))
                c.require(abs(D(a['mean_visible_tokens'])-denominator)<=Decimal('1e-12'),'Visible-token denominator differs')
                independent=np.take(attention[:,:,ri,:],keys,axis=2).sum(axis=2,dtype=np.float64)
                mass=np.asarray(a['mass'],dtype=np.float64);density=np.asarray(a['density'],dtype=np.float64)
                delta=max(float(np.max(np.abs(independent-mass))),float(np.max(np.abs(independent/float(denominator)-density))))
                aggregate_error=max(aggregate_error,delta);aggregate_values+=2*mass.size
                c.require(delta<=2e-12,'Independent FP64 aggregate differs')
                short=compact[(req['request_id'],role,a['span_id'])]
                for metric,values in (('mass',mass),('density',density)):
                    c.require(short['all_layer_head_mean_'+metric]==float(values.mean())
                        and short['late_18_35_mean_'+metric]==float(values[18:36].mean()),'Layer/head summary differs')
                    summary_values+=2
                for layer,head in ((0,0),(18,0),(35,31)):
                    exact=sum((D(attention[layer,head,ri,k]) for k in keys),Decimal(0))
                    err=max(abs(D(mass[layer,head])-exact),abs(D(density[layer,head])-exact/denominator))
                    decimal_error=max(decimal_error,err);decimal_values+=2
                    c.require(err<=Decimal('2e-12'),'Decimal aggregate/density differs')
            del attention,visual,display
        for entry in state['invocations']:
            c.require(entry.get('ended_at_unix') is not None,'Unreleased invocation')
            c.require(not Path('/proc/'+str(entry['worker_pid'])).exists(),'Owned worker still present')
            releases+=1
    return {'schema':'hehe-sense-context-result-audit/v1','status':'pass','CPU_only':True,
            'absolute_logit_vectors':vectors,'raw_margins_decimal_checked':record_margins,'predictions':9,'registered_expressions':9,
            'expression_values_bounds_directions_transitions':'all checked; exact Decimal equality',
            'Decimal_precision':120,
            'aggregate_elements_reconstructed':aggregate_values,'max_independent_FP64_aggregate_error':aggregate_error,
            'FP32_visualization_values_checked':display_values,'FP64_summary_values_checked':summary_values,
            'Decimal_aggregate_values_checked':decimal_values,'max_Decimal_aggregate_error':str(decimal_error),
            'normal_worker_releases':releases,'query_reference_join_after_normal_release':True,
            'sources':[c.info(prepared/'manifest.json'),c.info(run/'raw-seal.json'),c.info(results/'manifest.json'),c.info(Path(__file__))],
            'new_GPU_forwards':0,'query_count':3,'independent_cases_claimed':False,'confirmation_claimed':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,default=c.PREPARED);p.add_argument('--run',type=Path,default=c.WORK/'run-01')
    p.add_argument('--results',type=Path,default=c.WORK/'results-01');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.prepared,a.run,a.results);c.write(a.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
