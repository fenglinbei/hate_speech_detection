#!/usr/bin/env python3
"""CPU synthetic regression checks for paired-context contrasts and exact replay."""
import argparse,copy,json,sys,tempfile,unittest
from pathlib import Path
from decimal import Decimal,localcontext
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from diagnostics import jingba_attn_restore_inputs_v1 as c
from diagnostics.jingba_attn_restore_report_v1 import context_restoration,FIELDS
from audit_jingba_attn_restore_results_v1 import audit_context_restoration,audit_historical_replay
from test_jingba_attn_restore_runtime_v1 import make_synthetic_history


def paired_fixture():
    data={'effects':[],'patched_trajectories':{},'margin_error_bound':1e-6};margins={}
    # First direction narrows an existing response gap; reverse enlarges it.
    for d,values in [('D01',(4.,2.,3.,2.5)),('D02',(-4.,-2.,-3.,1.))]:
        for (q,k),v in zip([('J08','upstream'),('J10','upstream'),('J08','restore-L26-attention'),('J10','restore-L26-attention')],values):
            rid=q+'-'+d;jid=rid+'-'+k;n=10. if q=='J08' else -10.
            margins[rid]=Decimal(n);e=dict(job_id=jid,query_id=q,condition=k,recipient=rid,donor=q+('-D02' if d=='D01' else '-D01'),m=n+v,delta_m=v)
            data['effects'].append(e)
            data['patched_trajectories'][jid]={'minus_recipient':{f:np.full((36,3 if f in ['probe_pre_mid_post','rms_pre_mid_post','state_l2_pre_mid_post'] else 2),v).tolist() for f in FIELDS}}
    data['context_restoration']=context_restoration(data)
    return data,{e['job_id']:e for e in data['effects']},margins


def historical_fixture(root):
    records={}
    for i in range(42):
        stage='native-production' if i<18 else 'production';key=f'fixture-{i}';folder=root/'fresh'/key;folder.mkdir(parents=True)
        np.save(folder/'v.npy',np.asarray([i,1,-1],np.float32))
        np.save(folder/'s.npy',np.asarray([i,2],np.float32))
        np.savez(folder/'t.npz',states=np.full((36,3,2),i,np.float32),probe=np.full((36,3),i,np.float64))
        job=None if i<18 else {'job_id':key,'condition':'upstream' if i%2 else 'preceding'}
        records[stage,key]={'stage':stage,'job':job,'request_id':key,'input_ids_sha256':'synthetic-'+key,
            'prompt_sha256':'synthetic-only','vector':c.info(folder/'v.npy'),'states':c.info(folder/'s.npy') if i<18 else None,'trajectory':c.info(folder/'t.npz')}
    hist=root/'history';make_synthetic_history(hist,records)
    return {'prior_comparators':{'work':str(hist)}},records


def repin_old(plan,index,mutator):
    root=Path(plan['prior_comparators']['work'])/'run-01';seal=c.read(root/'raw-seal.json');group='native_records' if index<18 else 'records'
    item=seal[group][index if index<18 else index-18];record=c.read(item['path']);mutator(record)
    Path(item['path']).write_bytes(c.canonical(record));seal[group][index if index<18 else index-18]=c.info(item['path'])
    (root/'raw-seal.json').write_bytes(c.canonical(seal))


class Checks(unittest.TestCase):
    def test_shared_native_cancels_and_both_directions(self):
        data,es,ms=paired_fixture();a,b=data['context_restoration']
        self.assertEqual((a['upstream_context_difference'],a['remaining_context_difference'],a['change_due_to_restoration']),(2.,.5,-1.5))
        self.assertTrue(a['absolute_context_difference_reduced']);self.assertTrue(b['absolute_context_difference_increased'])
        self.assertEqual(a['change_bound'],4e-6)
        with localcontext() as ctx:
            ctx.prec=120;self.assertEqual(audit_context_restoration(data,es,ms,Decimal.from_float(1e-6)),2)
            for key,value in [('change_due_to_restoration',1.5),('change_bound',8e-6),('absolute_context_difference_reduced',False)]:
                bad=copy.deepcopy(data);bad['context_restoration'][0][key]=value
                with self.assertRaises(AssertionError):audit_context_restoration(bad,es,ms,Decimal.from_float(1e-6))
    def test_duplicate_direction_or_wrong_trajectory_rejected(self):
        data,es,ms=paired_fixture()
        for kind in ['duplicate','trajectory']:
            bad=copy.deepcopy(data)
            if kind=='duplicate':bad['context_restoration'][1]=bad['context_restoration'][0]
            else:bad['context_restoration'][0]['change_trajectory_difference']['probe_pre_mid_post'][27][2]+=1
            with self.assertRaises(AssertionError):audit_context_restoration(bad,es,ms,Decimal.from_float(1e-6))
    def test_saved_replay_exact_vectors_states_and_trajectories(self):
        with tempfile.TemporaryDirectory() as td:
            plan,records=historical_fixture(Path(td));r=audit_historical_replay(None,plan,records)
            self.assertEqual((r['native_vectors'],r['upstream_and_preceding_vectors'],r['trajectories']),(18,24,42));self.assertTrue(r['all_exact_equal'])
    def test_resealed_wrong_old_vector_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            plan,records=historical_fixture(Path(td))
            def mutate(r):
                p=Path(r['vector']['path']);v=np.load(p);v[0]+=1;np.save(p,v);r['vector']=c.info(p)
            repin_old(plan,19,mutate)
            with self.assertRaises(AssertionError):audit_historical_replay(None,plan,records)
    def test_resealed_wrong_old_trajectory_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            plan,records=historical_fixture(Path(td))
            def mutate(r):
                p=Path(r['trajectory']['path'])
                with np.load(p) as a:v={k:a[k] for k in a.files}
                v['states'][30,2,1]+=1;np.savez(p,**v);r['trajectory']=c.info(p)
            repin_old(plan,0,mutate)
            with self.assertRaises(AssertionError):audit_historical_replay(None,plan,records)
    def test_missing_old_endpoint_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            plan,records=historical_fixture(Path(td));root=Path(plan['prior_comparators']['work'])/'run-01';seal=c.read(root/'raw-seal.json');seal['records'].pop();(root/'raw-seal.json').write_bytes(c.canonical(seal))
            with self.assertRaises(AssertionError):audit_historical_replay(None,plan,records)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    receipt={'status':'pass' if result.wasSuccessful() else 'fail','tests':result.testsRun,'CUDA_initialized':False,'research_checkpoint_loaded':False,
        'synthetic_only':True,'implementation_snapshot':[c.info(p) for p in c.CODE], 'failures':[str(e) for _,e in result.failures+result.errors]}
    c.write(a.output,receipt);print(json.dumps(receipt));sys.exit(0 if result.wasSuccessful() else 1)
