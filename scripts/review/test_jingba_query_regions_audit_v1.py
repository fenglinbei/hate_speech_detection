#!/usr/bin/env python3
"""Independent-audit regressions: expanded banks, corrupt comparators, scope scores."""
import argparse,copy,json,sys,tempfile,unittest
from pathlib import Path
from decimal import Decimal,localcontext
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from diagnostics import jingba_query_regions_inputs_v1 as c
from diagnostics.jingba_query_regions_report_v1 import scope_comparisons,score_shift,query_interactions,region_factorial,FIELDS
from audit_jingba_query_regions_results_v1 import audit_historical_replay,audit_scope,audit_query_interactions,audit_regions
from test_jingba_query_regions_runtime_v1 import make_synthetic_history


def historical_fixture(root):
    records={}
    for i in range(66):
        stage='native-production' if i<18 else 'production';key=f'fixture-{i}';folder=root/'fresh'/key;folder.mkdir(parents=True)
        np.save(folder/'v.npy',np.asarray([i,1,-1],np.float32))
        np.save(folder/'s.npy',np.arange(36*5*2,dtype=np.float32).reshape(36,5,2)+i)
        np.savez(folder/'t.npz',states=np.full((36,3,2),i,np.float32))
        job=None if i<18 else {'job_id':key,'condition':['upstream','preceding','whole-query','query-complement'][i%4]}
        records[stage,key]={'stage':stage,'job':job,'request_id':key,'positions':[5,6,7,8,9],'input_ids_sha256':'synthetic-'+key,
            'prompt_sha256':'synthetic-only','vector':c.info(folder/'v.npy'),'states':c.info(folder/'s.npy') if i<18 else None,'trajectory':c.info(folder/'t.npz')}
    hist=root/'history';make_synthetic_history(hist,records)
    return {'prior_comparators':{'work':str(hist)}},records


def scope_fixture():
    data={'effects':[],'baselines':[],'patched_trajectories':{},'margin_error_bound':1e-6};ms={};refs={q:'无' if int(q[1:])<=8 else '有' for q in c.IDS}
    for i,q in enumerate(c.IDS):
        for d,n in [('D01',-8.+i),('D02',4.+i)]:
            rid=q+'-'+d;ms[rid]=Decimal(n)
            data['baselines'].append(dict(request_id=rid,query_id=q,dictionary_id=d,m=n,reference=refs[q],raw_prediction='无' if n>0 else '有'))
            for k,delta in [('upstream',1.),('whole-query',3.),('query-complement',1.25)]:
                jid=rid+'-'+k;value=n+delta
                e=dict(job_id=jid,query_id=q,recipient=rid,donor=q+('-D02' if d=='D01' else '-D01'),condition=k,m=value,
                    prediction='无' if value>0 else '有' if value<0 else None,transition='unchanged',replacement_token_count={'upstream':2,'whole-query':5,'query-complement':3}[k],donor_state_l2={'upstream':1.,'whole-query':2.,'query-complement':3.**.5}[k])
                data['effects'].append(e);data['patched_trajectories'][jid]={'values':{f:np.full((36,3 if f in ['probe_pre_mid_post','rms_pre_mid_post','state_l2_pre_mid_post'] else 2),value).tolist() for f in FIELDS}}
    for e in data['effects']:
        t=data['patched_trajectories'][e['job_id']];t['minus_recipient']={k:(np.asarray(v)-float(ms[e['recipient']])).tolist() for k,v in t['values'].items()}
    data['query_interactions']=query_interactions(data)
    plan={'CPU_score_shift':{'offset':7.0}}
    data['scope_comparisons']=scope_comparisons(data);data['CPU_score_shift']=score_shift(data,plan)
    return data,ms,{e['job_id']:e for e in data['effects']},refs,plan


class Checks(unittest.TestCase):
    def test_exact_expanded_banks_use_old_position_subset(self):
        with tempfile.TemporaryDirectory() as td:
            plan,records=historical_fixture(Path(td));r=audit_historical_replay(None,plan,records)
            self.assertTrue(r['all_exact_equal']);self.assertEqual(r['native_state_banks_compared_at_old_positions'],18)
    def test_wrong_expanded_position_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            plan,records=historical_fixture(Path(td));records['native-production','fixture-0']['positions']=[6,5,7,8,9]
            with self.assertRaises(AssertionError):audit_historical_replay(None,plan,records)
    def test_wrong_fresh_vector_or_trajectory_rejected_even_if_repinned(self):
        for prop in ['vector','trajectory','states']:
            with self.subTest(prop=prop),tempfile.TemporaryDirectory() as td:
                plan,records=historical_fixture(Path(td));r=records['native-production','fixture-0'];p=Path(r[prop]['path'])
                if prop=='trajectory':np.savez(p,states=np.full((36,3,2),9,np.float32))
                else:
                    a=np.load(p);a.flat[0]+=1;np.save(p,a)
                r[prop]=c.info(p)
                with self.assertRaises(AssertionError):audit_historical_replay(None,plan,records)
    def test_missing_prior_endpoint_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            plan,records=historical_fixture(Path(td));del records['production','fixture-65']
            with self.assertRaises(AssertionError):audit_historical_replay(None,plan,records)
    def test_scope_scores_and_fixed_offset(self):
        data,ms,es,refs,plan=scope_fixture()
        with localcontext() as ctx:
            ctx.prec=120;self.assertEqual(audit_scope(data,ms,es,refs,Decimal.from_float(1e-6),plan),12)
            for key,value in [('whole_minus_focal',-2.),('pair_difference_bound',4e-6),('donor_distance_improvement',-2.)]:
                bad=copy.deepcopy(data);bad['scope_comparisons'][0][key]=value
                with self.assertRaises(AssertionError):audit_scope(bad,ms,es,refs,Decimal.from_float(1e-6),plan)
    def test_duplicate_pair_or_fitted_offset_rejected(self):
        data,ms,es,refs,plan=scope_fixture()
        for kind in ['duplicate','offset','prediction']:
            bad=copy.deepcopy(data)
            if kind=='duplicate':bad['scope_comparisons'][1]=bad['scope_comparisons'][0]
            elif kind=='offset':bad['CPU_score_shift']['offset']=8.
            else:bad['CPU_score_shift']['records'][0]['shifted_prediction']='无'
            with self.assertRaises(AssertionError):audit_scope(bad,ms,es,refs,Decimal.from_float(1e-6),plan)

    def test_interaction_keeps_shared_native_once_and_correct_bound(self):
        data,ms,es,refs,plan=scope_fixture()
        self.assertTrue(all(r['interaction_m']==.75 for r in data['query_interactions']))
        with localcontext() as ctx:
            ctx.prec=120;self.assertEqual(audit_query_interactions(data,ms,es,Decimal.from_float(1e-6)),12)
            for key,value in [('interaction_m',-.75),('interaction_bound',6e-6),('conditional_complement_delta_m',1.25)]:
                bad=copy.deepcopy(data);bad['query_interactions'][0][key]=value
                with self.assertRaises(AssertionError):audit_query_interactions(bad,ms,es,Decimal.from_float(1e-6))
    def test_interaction_wrong_curve_or_duplicate_rejected(self):
        data,ms,es,refs,plan=scope_fixture()
        for kind in ['curve','duplicate','geometry']:
            bad=copy.deepcopy(data)
            if kind=='curve':bad['query_interactions'][0]['trajectory_interaction']['probe_pre_mid_post'][29][2]+=.01
            elif kind=='duplicate':bad['query_interactions'][1]=bad['query_interactions'][0]
            else:bad['query_interactions'][0]['complement_tokens']=4
            with self.assertRaises(AssertionError):audit_query_interactions(bad,ms,es,Decimal.from_float(1e-6))
    def test_zero_interaction_is_allowed_not_assumed_resolved(self):
        data,ms,es,refs,plan=scope_fixture()
        for e in data['effects']:
            if e['condition']!='query-complement':continue
            e['m']=float(ms[e['recipient']])+2.
            t=data['patched_trajectories'][e['job_id']]
            for k in t['values']:
                shape=np.asarray(t['values'][k]).shape;t['values'][k]=np.full(shape,e['m']).tolist();t['minus_recipient'][k]=np.full(shape,2.).tolist()
        data['query_interactions']=query_interactions(data)
        self.assertTrue(all(r['interaction_m']==0 and not r['interaction_resolved'] for r in data['query_interactions']))
        with localcontext() as ctx:
            ctx.prec=120;self.assertEqual(audit_query_interactions(data,ms,es,Decimal.from_float(1e-6)),12)


    def test_region_factorial_closure_and_corrupt_score_bound_curve(self):
        data,ms,es,refs,plan=scope_fixture();data['native_trajectories']={}
        for n in data['baselines']:
            rid=n['request_id'];data['native_trajectories'][rid]={f:np.full((36,3 if f in ['probe_pre_mid_post','rms_pre_mid_post','state_l2_pre_mid_post'] else 2),n['m']).tolist() for f in FIELDS}
            for condition,delta,count in [('query-before',.5,2),('query-after',.25,1),('focal-before',1.75,4),('focal-after',1.5,3),('preceding',.5,2)]:
                e=copy.deepcopy(next(e for e in data['effects'] if e['recipient']==rid and e['condition']=='upstream'))
                e.update(job_id=rid+'-'+condition,condition=condition,m=n['m']+delta,replacement_token_count=count)
                data['effects'].append(e);data['patched_trajectories'][e['job_id']]={'values':{f:(np.asarray(v)+delta).tolist() for f,v in data['native_trajectories'][rid].items()}}
        es={e['job_id']:e for e in data['effects']};data['region_factorial']=region_factorial(data)
        with localcontext() as ctx:
            ctx.prec=120;self.assertEqual(audit_regions(data,ms,es,Decimal.from_float(1e-6)),12)
            for kind in ['third_order','bound','curve','duplicate','job']:
                bad=copy.deepcopy(data);row=bad['region_factorial'][0]
                if kind=='third_order':row['I_UBA']+=.125
                elif kind=='bound':row['bound_UBA']=4e-6
                elif kind=='curve':row['trajectories']['UA']['increment_attention_mlp'][26][0]+=.01
                elif kind=='duplicate':bad['region_factorial'][1]=row
                else:row['jobs']['B']=row['jobs']['A']
                with self.assertRaises(AssertionError):audit_regions(bad,ms,es,Decimal.from_float(1e-6))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    c.write(a.output,{'status':'pass' if r.wasSuccessful() else 'fail','tests':r.testsRun,'CUDA_initialized':False,'failures':[str(e) for _,e in r.failures+r.errors]})
    sys.exit(0 if r.wasSuccessful() else 1)
