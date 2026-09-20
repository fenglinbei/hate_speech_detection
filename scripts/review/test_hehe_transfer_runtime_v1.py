#!/usr/bin/env python3
"""Actual new controller/worker lifecycle on reference-free synthetic telemetry."""
import argparse,json,sys,tempfile,unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from diagnostics import hehe_transfer_inputs_v1 as c
from diagnostics import hehe_transfer_runtime_v1 as rt
from diagnostics import hehe_joint_restore_capture_v1 as cap
from diagnostics.cross_model_applicability_models_v1 import prepared_input
from diagnostics.cross_model_applicability_execution_v1 import runtime_versions

RULES={'margin_repeat_order_hook_cap':0.,'margin_padding_prefix_cap':.001,'attention_repeat_order_cap':0.,
 'attention_element_cap':.0001,'attention_row_l1_cap':.001,'attention_row_sum_cap':.000002,
 'margin_bound_floor':.000001,'attention_mass_bound_floor':.0000001}


def fixture():
    rows=[]
    for q in ['T01','T02','T03','T04']:
        for d in ['D01','D02']:
            rid=f'synthetic-{q}-{d}';ids=[6 if d=='D01' else 7,8,9,10,11,12,13,14]
            rows.append({'request_id':rid,'query_id':q,'dictionary_id':d,'input_ids':ids,'prompt_tokens':8,
                'input_ids_sha256':c.digest(ids),'prompt_sha256':c.digest(rid),'prompt_text':'abcdefgh','spans':[{'id':'query','char_start':3,'char_end':7}],
                'roles':{'query_all':[3,4,5,6],'pre_answer':[7]},'capture_positions':[3,4],
                'capture_prefix_length':5,'candidate_tokens':{'有':3,'无':4},
                'patch_position_sets':{'focal':[4],'pre':[3]}})
    jobs,selfs=c.make_jobs(rows)
    return rows,jobs,selfs


def synthetic(model,req,pad,*,positions=(),observe=False,layer=None,replacement=None,padding='none',prefix_length=None,append_tokens=(),trajectory=False,restoration=None):
    value=-4. if req['dictionary_id']=='D01' else 4.
    bank=np.full((36,len(positions),8),value,np.float32) if observe else None
    h=np.full(8,value,np.float32);hs=[];bs=[];restore_proof=[] if restoration is not None else None
    p=prepared_input(req['input_ids']+list(append_tokens),pad,padding);valid=[i for i,x in enumerate(p['attention_mask']) if x]
    for li in range(36):
        pre=h.copy();a=np.zeros(8,np.float32)
        if layer is not None and li==layer+1:
            a[0]=(.75 if positions==[4] else .1)*(float(replacement[0,0])-value)
        if li==26:a[0]=np.float32(.4)*pre[0]
        mid=pre+a;mlp=np.zeros(8,np.float32)
        if li==28:mlp[0]=np.float32(.3)*mid[0]
        spec=next((ss for ss in (restoration or []) if ss['layer']==li),None)
        if spec is not None:
            branch=spec['branch'];before=(a if branch=='attention' else mlp).copy();replacement_branch=spec['value']
            if branch=='attention':a=replacement_branch.copy();mid=pre+a
            else:mlp=replacement_branch.copy()
            pos=spec['position']
            restore_proof.append({'layer':li,'branch':branch,'position':pos,'padded_position':valid[pos],
                'replacement_sha256':cap.array_sha(replacement_branch),'before_sha256':cap.array_sha(before),
                'replacement_l2':float(np.linalg.norm(replacement_branch.astype(np.float64))),
                'before_l2':float(np.linalg.norm(before.astype(np.float64))),
                'difference_l2':float(np.linalg.norm(replacement_branch.astype(np.float64)-before)),
                'changed_rows_in_unpadded_coordinates':[] if np.array_equal(before,replacement_branch) else [pos],
                'outside_rows_exact':True,'replacement_exact':True,'native_output_unmodified':True,
                'other_attention_outputs_preserved':branch=='attention'})
        h=mid+mlp
        hs.append(np.stack([pre,mid,h]));bs.append(np.stack([a,mlp]))
    states=np.stack(hs);normalized=states/np.sqrt(np.mean(states*states,axis=-1,keepdims=True)+np.float32(1e-6))
    weights=np.zeros((2,8),np.float32);weights[0,0]=-1;weights[1,0]=1
    lens=normalized@weights.T
    t={'states':states,'normalized':normalized,'lens_logits':lens,'branches':np.stack(bs),
        'label_weights':weights,'norm_weight':np.ones(8,np.float32),'norm_eps':np.asarray([1e-6],np.float64),
        'position':np.asarray([7],np.int64),'candidate_ids':np.asarray([3,4],np.int64)}
    v=np.full(16,-20,np.float32);v[[3,4]]=lens[-1,2]
    if append_tokens:v[2]=20
    proof=None
    if layer is not None:
        p=prepared_input(req['input_ids']+list(append_tokens),pad,padding);valid=[i for i,x in enumerate(p['attention_mask']) if x]
        before=np.full_like(replacement,value)
        proof={'layer':layer,'positions':list(positions),'padded_positions':[valid[i] for i in positions],
            'donor_sha256':cap.array_sha(replacement),'recipient_before_sha256':cap.array_sha(before),
            'changed_rows_in_unpadded_coordinates':[] if np.array_equal(before,replacement) else list(positions),
            'outside_rows_exact':True,'donor_rows_exact':True,'native_output_unmodified':True,'restoration':restore_proof}
    return v,bank,proof,t if trajectory else None


class Checks(unittest.TestCase):
    def test_lifecycle_and_corruption(self):
        rows,jobs,selfs=fixture()
        real_popen=rt.subprocess.Popen
        profile={'layers':36,'hidden_size':8,'vocab_size':16,'candidate_tokens':{'有':3,'无':4},
            'eos_token_ids':[2],'pad_token_id':0,'rms_norm_eps':1e-6,'metadata_sources':[],'weight_sources':[]}
        plan={'acceptance':RULES,'state_scaled_cap':.0001,'format':{'max_new_tokens':8}}
        with tempfile.TemporaryDirectory(prefix='hehe-joint-restore-runtime-') as td, ExitStack() as stack:
            tmp=Path(td);prepared=tmp/'prepared';prepared.mkdir();c.write(prepared/'manifest.json',{'synthetic':True})
            c.write(prepared/'analysis-references.json',{'references':[{'query_id':q,'reference':'无' if q in ['T01','T02'] else '有'} for q in ['T01','T02','T03','T04']]})
            allocation=[{'index':0,'uuid':'synthetic-only','name':'synthetic','total_mib':48000}]
            inventory={'devices':[dict(allocation[0],used_mib=0,utilization=0)],'compute_processes':[],'synthetic':True}
            bound=tmp/'bound.json';c.write(bound,{'prepared_manifest':c.info(prepared/'manifest.json'),'allocation':allocation,
                'runtime_versions':runtime_versions(),'authorization_note':'CPU SYNTHETIC ONLY','GPU_time_constraint':{'deadline_unix':None}})
            run=tmp/'run-synthetic';calls=[]
            stack.enter_context(patch.object(c,'WORK',tmp));stack.enter_context(patch.object(c,'validate',return_value=(plan,profile,rows,jobs,selfs)))
            stack.enter_context(patch.object(rt,'gpu_inventory',return_value=inventory));stack.enter_context(patch.object(rt.signal,'signal'))
            stack.enter_context(patch('diagnostics.cross_model_applicability_models_v1.load_checkpoint',return_value=(object(),{'synthetic':True})))
            def counted(*a,**kw):
                v=synthetic(*a,**kw);calls.append(kw)
                if len(calls)==2:(run/'STOP').touch()
                return v
            stack.enter_context(patch.object(cap,'forward',side_effect=counted))
            class Process:
                pid=987654321
                def __init__(self,cmd,**kw):
                    get=lambda name:cmd[cmd.index(name)+1]
                    rt.worker(get('--prepared'),get('--bound'),get('--run'),get('--invocation'),get('--phase'))
                def wait(self):return 0
            stack.enter_context(patch.object(rt.subprocess,'Popen',Process))
            self.assertEqual(rt.supervise(prepared,bound,run)['status'],'paused');self.assertEqual(len(calls),2)
            path=rt.record_path(run,'native-baseline',rows[0]['request_id']);original=path.read_bytes();(run/'STOP').unlink()
            self.assertEqual(rt.supervise(prepared,bound,run,resume=True)['status'],'qualified')
            self.assertEqual(path.read_bytes(),original);self.assertEqual(len(calls),408)
            self.assertFalse(rt.check_run(prepared,run)['query_reference_join_performed'])
            from diagnostics.hehe_transfer_report_v1 import analyze
            with self.assertRaisesRegex(ValueError,'Full run'):analyze(prepared,run,tmp/'premature')
            self.assertFalse((tmp/'premature').exists())
            self.assertEqual(rt.supervise(prepared,bound,run,phase='full')['status'],'complete')
            self.assertEqual(len(calls),456);checked=rt.check_run(prepared,run)
            self.assertEqual(len(checked['qualification']['trajectory']['structural_guards']),264)
            self.assertEqual(checked['qualification']['margin_error_bound'],1e-6)
            from audit_hehe_transfer_results_v1 import audit_branch_sources
            records={}
            for file in (run/'records').glob('*/*.json'):
                rec=c.read(file);records[rec['stage'],rec['job']['job_id'] if rec['job'] else rec['request_id']]=rec
            request_map={r['request_id']:r for r in rows}
            self.assertEqual(audit_branch_sources(run,records,request_map,profile),320)
            joint=next(j for j in jobs if j['condition']=='restore-joint')
            proof=records['production',joint['job_id']]['patch_proof']['restoration'][1]
            correct=proof['before_sha256'];proof['before_sha256']='wrong-later-source'
            with self.assertRaises(AssertionError):audit_branch_sources(run,records,request_map,profile)
            proof['before_sha256']=correct
            # Rendering may invoke fc-list. Only worker launches are simulated;
            # do not route font discovery through the fake GPU worker process.
            with patch.object(rt.subprocess,'Popen',real_popen):
                result=analyze(prepared,run,tmp/'analysis')
            self.assertEqual(result['interventions'],40)
            self.assertEqual(result['restoration_contrasts'],24)
            self.assertEqual(len(checked['qualification']['trajectory']['restoration_guards']),224)
            results=c.read(tmp/'analysis/results.json')
            self.assertTrue(all(r['removed_fraction']>0 for r in results['restoration_contrasts']))
            self.assertTrue(all(s['delta_m']==0 for s in results['self_controls']))
            self.assertEqual(len(results['condition_gaps']),4)
            self.assertEqual(len(results['position_differences']),8)
            self.assertEqual({s['query_id']:s['reference'] for s in results['baselines']},{'T01':'无','T02':'无','T03':'有','T04':'有'})
            with self.assertRaisesRegex(ValueError,'terminal'):rt.supervise(prepared,bound,run,phase='full')
            rec=c.read(rt.record_path(run,'production',jobs[0]['job_id']));p=Path(rec['trajectory']['path'])
            original=p.read_bytes();bad=bytearray(original);bad[-1]^=1;p.write_bytes(bad)
            with self.assertRaisesRegex(ValueError,'Pinned bytes'):rt.check_run(prepared,run)
            p.write_bytes(original)
            p=run/'format'/(jobs[0]['job_id']+'.json');bad=c.read(p);bad['steps'][0]['patch_proof']['donor_sha256']='wrong';p.write_bytes(c.canonical(bad))
            with self.assertRaisesRegex(ValueError,'exact patch'):rt.check_run(prepared,run)

    def test_contrast_dependency_and_NA(self):
        from diagnostics.hehe_transfer_report_v1 import contrast
        a=contrast(1.,3.,2.,.1)
        self.assertEqual(a['removed_fraction'],.5)
        self.assertTrue(a['absolute_effect_reduced'])
        self.assertEqual(contrast(0.,0.000001,0.,.000001)['removed_fraction'],None)
        self.assertLess(contrast(0.,1.,2.,.000001)['removed_fraction'],0)
        self.assertGreater(contrast(0.,1.,-2.,.000001)['removed_fraction'],1)
        self.assertTrue(contrast(0.,1.,-2.,.000001)['absolute_effect_increased'])

    def test_joint_interaction_and_shared_score_bounds(self):
        from diagnostics.hehe_transfer_report_v1 import joint_contrast
        r=joint_contrast(0.,10.,7.,8.,5.,.000001)
        self.assertEqual(r['interaction_m'],0.)
        self.assertEqual(r['additional_removed_after_A'],2.)
        self.assertTrue(r['absolute_reduced_vs_A'] and r['absolute_reduced_vs_B'])
        self.assertFalse(r['interaction_resolved'])
        r=joint_contrast(0.,10.,7.,8.,6.,.1)
        self.assertAlmostEqual(r['interaction_fraction'],.1)
        self.assertTrue(r['interaction_resolved'])
        self.assertLess(r['interaction_fraction_interval'][0],.1)
        self.assertGreater(r['interaction_fraction_interval'][1],.1)
        self.assertIsNone(joint_contrast(0.,.000001,1.,2.,3.,.000001)['interaction_fraction'])
        self.assertFalse(joint_contrast(0.,10.,7.,8.,-9.,.000001)['absolute_reduced_vs_A'])

    def test_original_score_gates(self):
        self.assertEqual(rt.qualify([{'kind':'repeat','margin_difference':0.}],RULES)['margin_error_bound'],1e-6)
        for v in [{'kind':'repeat','margin_difference':1e-12},{'kind':'left','margin_difference':.001001}]:
            with self.assertRaises(ValueError):rt.qualify([v],RULES)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    receipt={'status':'pass' if result.wasSuccessful() else 'fail','tests':result.testsRun,
        'CUDA_initialized':False,'research_checkpoint_loaded':False,'GPU_inventory':'synthetic mock only',
        'synthetic_forwards':456,'failures':[str(e) for _,e in result.failures+result.errors]}
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);c.write(a.output,receipt)
    print(json.dumps(receipt));sys.exit(0 if result.wasSuccessful() else 1)
