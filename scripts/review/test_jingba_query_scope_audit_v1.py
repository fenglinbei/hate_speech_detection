#!/usr/bin/env python3
"""Independent-audit regressions: expanded banks, corrupt comparators, scope scores."""
import argparse,copy,json,sys,tempfile,unittest
from pathlib import Path
from decimal import Decimal,localcontext
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from diagnostics import jingba_query_scope_inputs_v1 as c
from diagnostics.jingba_query_scope_report_v1 import scope_comparisons,score_shift,FIELDS
from audit_jingba_query_scope_results_v1 import audit_historical_replay,audit_scope
from test_jingba_query_scope_runtime_v1 import make_synthetic_history


def historical_fixture(root):
    records={}
    for i in range(42):
        stage='native-production' if i<18 else 'production';key=f'fixture-{i}';folder=root/'fresh'/key;folder.mkdir(parents=True)
        np.save(folder/'v.npy',np.asarray([i,1,-1],np.float32))
        np.save(folder/'s.npy',np.arange(36*5*2,dtype=np.float32).reshape(36,5,2)+i)
        np.savez(folder/'t.npz',states=np.full((36,3,2),i,np.float32))
        job=None if i<18 else {'job_id':key,'condition':'upstream' if i%2 else 'preceding'}
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
            for k,delta in [('upstream',1.),('whole-query',3.)]:
                jid=rid+'-'+k;value=n+delta
                e=dict(job_id=jid,query_id=q,recipient=rid,donor=q+('-D02' if d=='D01' else '-D01'),condition=k,m=value,
                    prediction='无' if value>0 else '有' if value<0 else None,transition='unchanged',replacement_token_count=2 if k=='upstream' else 5,donor_state_l2=1. if k=='upstream' else 2.)
                data['effects'].append(e);data['patched_trajectories'][jid]={'values':{f:np.full((36,3 if f in ['probe_pre_mid_post','rms_pre_mid_post','state_l2_pre_mid_post'] else 2),value).tolist() for f in FIELDS}}
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
            plan,records=historical_fixture(Path(td));del records['production','fixture-41']
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

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    c.write(a.output,{'status':'pass' if r.wasSuccessful() else 'fail','tests':r.testsRun,'CUDA_initialized':False,'failures':[str(e) for _,e in r.failures+r.errors]})
    sys.exit(0 if r.wasSuccessful() else 1)
