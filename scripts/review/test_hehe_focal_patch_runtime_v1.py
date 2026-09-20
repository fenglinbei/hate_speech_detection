#!/usr/bin/env python3
"""Actual lifecycle with synthetic telemetry; native patching tested separately."""
import argparse
from contextlib import ExitStack
from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from diagnostics import hehe_focal_patch_inputs_v1 as c
from diagnostics import hehe_focal_patch_runtime_v1 as rt
from diagnostics import hehe_focal_patch_capture_v1 as cap
from diagnostics.hehe_focal_patch_report_v1 import effect,ratio_interval,analyze
from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
from diagnostics.cross_model_applicability_models_v1 import prepared_input

RULES={'margin_repeat_order_hook_cap':0.,'margin_padding_prefix_cap':.001,'attention_repeat_order_cap':0.,
       'attention_element_cap':.0001,'attention_row_l1_cap':.001,'attention_row_sum_cap':.000002,
       'margin_bound_floor':.000001,'attention_mass_bound_floor':.0000001}

def fixture():
    reqs=[];jobs=[];selfs=[]
    for q in ('Q01','Q02'):
        for d in ('D01','D02'):
            rid=f'synthetic-{q}-{d}';ids=[6 if d=='D01' else 7,8,9,10,11,12,13,14]
            r={'request_id':rid,'query_id':q,'dictionary_id':d,'prompt_text':'明确的CPU合成材料',
               'input_ids':ids,'input_ids_sha256':c.digest(ids),'prompt_sha256':c.digest(rid),'prompt_tokens':8,
               'roles':{'query_all':[3,4,5,6],'pre_answer':[7]},'capture_positions':[3,4],'capture_prefix_length':5,
               'patch_position_sets':{'focal':[4],'pre':[3]},'token_text':['x']*8}
            reqs.append(r)
            for group,ps in [('focal',[4]),('pre',[3])]:
                for li in range(2):
                    j={'query_id':q,'recipient':rid,'donor':rid[:-3]+('D02' if d=='D01' else 'D01'),
                       'layer':li,'positions':ps,'position_text':['x'],'site':'decoder_block_output','group':group,
                       'group_label':group,'kind':'primary' if group=='focal' else 'position_control','last_layer_expected_zero':li==1}
                    jobs.append(dict(j,job_id=f'{q}-{d}-{group}-{li}'))
                    selfs.append(dict(j,donor=rid,kind='self_control',job_id=f'{q}-{d}-{group}-{li}-self'))
    return reqs,jobs,selfs

def synthetic_forward(model,req,pad,*,positions=(),observe=False,layer=None,replacement=None,padding='none',prefix_length=None,append_tokens=()):
    value=-4. if req['dictionary_id']=='D01' else 4.
    states=np.full((2,len(positions),8),value,np.float32) if observe else None
    v=np.full(16,-20.,np.float32);m=value*2;proof=None
    if replacement is not None:
        if layer==0:m+=(float(replacement[0,0])-value)*(1. if positions==[4] else .25)
        valid=[i for i,a in enumerate(prepared_input(req['input_ids'],pad,padding)['attention_mask']) if a]
        before=np.full_like(replacement,value)
        proof={'layer':layer,'positions':list(positions),'padded_positions':[valid[i] for i in positions],
               'donor_sha256':cap.array_sha(replacement),'recipient_before_sha256':cap.array_sha(before),
               'changed_rows_in_unpadded_coordinates':list(positions) if np.any(before!=replacement) else [],
               'outside_rows_exact':True,'donor_rows_exact':True,'native_output_unmodified':True}
    v[3]=10-m/2;v[4]=10+m/2
    if append_tokens:v[2]=30
    return v,states,proof

class Tests(unittest.TestCase):
    def test_metrics_bound_shared_baseline_and_unclipped_ratio(self):
        for p in (-12.,-1.,2.,20.):
            for r,d in [(-8.,8.),(8.,-8.)]:
                ratio,interval=ratio_interval(p,d,r,1e-6)
                self.assertEqual(ratio,float((Fraction(p)-Fraction(r))/(Fraction(d)-Fraction(r))))
                self.assertLessEqual(interval[0],ratio);self.assertGreaterEqual(interval[1],ratio)
                for reference in ('有','无'):
                    score=lambda m:dict(m=m,raw_prediction='无' if m>0 else '有')
                    e=effect({},score(p),score(r),score(d),1e-6,reference)
                    self.assertEqual(e['reference_aligned_delta'],(p-r)*(1 if reference=='无' else -1))
                    self.assertEqual(e['delta_bound'],2e-6)
        self.assertEqual(ratio_interval(1.,1e-6,0.,1e-6),(None,None))
        self.assertGreater(ratio_interval(20.,8.,-8.,1e-6)[0],1)
        self.assertLess(ratio_interval(-12.,8.,-8.,1e-6)[0],0)
        e=effect({},dict(m=0.,raw_prediction='无'),dict(m=-8.,raw_prediction='有'),dict(m=8.,raw_prediction='无'),1e-6,'无')
        self.assertIsNone(e['flip']);self.assertEqual(e['transition'],'unresolved')

    def test_lifecycle_pause_reuse_production_release_format_tamper_and_terminal(self):
        with tempfile.TemporaryDirectory(prefix='focal-patch-test-') as td,ExitStack() as stack:
            tmp=Path(td);prepared=tmp/'prepared';prepared.mkdir();c.write(prepared/'manifest.json',{'synthetic':True})
            reqs,jobs,selfs=fixture();plan={'runtime_versions':runtime_versions(),'acceptance':RULES,'state_scaled_cap':1e-4,'format':{'max_new_tokens':8}}
            profile={'vocab_size':16,'layers':2,'hidden_size':8,'pad_token_id':0,'candidate_tokens':{'有':3,'无':4},'eos_token_ids':[2], 'metadata_sources':[],'weight_sources':[]}
            c.write(prepared/'analysis-references.json',{'references':[{'query_id':q,'reference':'无' if q=='Q01' else '有'} for q in ('Q01','Q02')]})
            allocation=[{'index':0,'uuid':'synthetic-only','name':'synthetic','total_mib':48000}]
            inventory={'devices':[dict(allocation[0],used_mib=0,utilization=0)],'compute_processes':[],'synthetic':True}
            bound=tmp/'bound.json';c.write(bound,{'prepared_manifest':c.info(prepared/'manifest.json'),'allocation':allocation,'runtime_versions':runtime_versions(),
                                               'authorization_note':'CPU SYNTHETIC ONLY','GPU_time_constraint':{'deadline_unix':None}})
            run=tmp/'run-synthetic';calls=[]
            stack.enter_context(patch.object(c,'WORK',tmp));stack.enter_context(patch.object(c,'validate',return_value=(plan,profile,reqs,jobs,selfs)))
            stack.enter_context(patch.object(rt,'gpu_inventory',return_value=inventory));stack.enter_context(patch.object(rt.signal,'signal'))
            stack.enter_context(patch('diagnostics.cross_model_applicability_models_v1.load_checkpoint',return_value=(object(),{'synthetic':True})))
            def counted(*a,**kw):
                result=synthetic_forward(*a,**kw);calls.append((a[1]['request_id'],kw))
                if len(calls)==2:(run/'STOP').touch()
                return result
            stack.enter_context(patch.object(cap,'forward',side_effect=counted))
            class Process:
                pid=987654321
                def __init__(self,cmd,**kw):
                    get=lambda name:cmd[cmd.index(name)+1]
                    rt.worker(get('--prepared'),get('--bound'),get('--run'),get('--invocation'),get('--phase'))
                def wait(self):return 0
            stack.enter_context(patch.object(rt.subprocess,'Popen',Process))
            paused=rt.supervise(prepared,bound,run);self.assertEqual(paused['status'],'paused');self.assertEqual(len(calls),2)
            path=rt.record_path(run,'native-baseline',reqs[0]['request_id']);before=path.read_bytes();(run/'STOP').unlink()
            self.assertEqual(rt.supervise(prepared,bound,run,resume=True)['status'],'qualified');self.assertEqual(path.read_bytes(),before)
            self.assertEqual(len(calls),28+len(selfs)+5*len(jobs)+4+len(jobs))
            self.assertFalse(rt.check_run(prepared,run)['query_reference_join_performed'])
            with self.assertRaisesRegex(ValueError,'Full run'):analyze(prepared,run,tmp/'premature')
            self.assertFalse((tmp/'premature').exists())
            self.assertEqual(rt.supervise(prepared,bound,run,phase='full')['status'],'complete');self.assertEqual(len(calls),36+len(selfs)+7*len(jobs))
            rt.check_run(prepared,run);result=analyze(prepared,run,tmp/'analysis')
            self.assertEqual((result['native_inputs'],result['interventions'],result['self_controls'],result['position_differences']),(4,16,16,8))
            data=c.read(tmp/'analysis/results.json');self.assertTrue(all(x['delta_m']==0 for x in data['self_controls']))
            for e in data['effects']:
                expected=(1 if e['recipient'].endswith('D01') else -1)*(8 if e['group']=='focal' else 2) if e['layer']==0 else 0
                self.assertEqual(e['delta_m'],expected)
            for pair in data['position_differences']:
                expected=(6 if pair['recipient'].endswith('D01') else -6) if pair['layer']==0 else 0
                self.assertEqual(pair['delta_m'],expected);self.assertEqual(pair['bound'],2e-6)
            with self.assertRaisesRegex(ValueError,'terminal'):rt.supervise(prepared,bound,run,phase='full')
            fmt=run/'format'/(jobs[0]['job_id']+'.json');saved=fmt.read_bytes();value=c.read(fmt);value['steps'][0]['patch_proof']['donor_sha256']='wrong';fmt.write_bytes(c.canonical(value))
            with self.assertRaisesRegex(ValueError,'exact patch'):rt.check_run(prepared,run)
            fmt.write_bytes(saved)
            prod=c.read(rt.record_path(run,'production',jobs[0]['job_id']));source=Path(prod['vector']['path']);corrupt=bytearray(source.read_bytes());corrupt[-1]^=1;source.write_bytes(corrupt)
            with self.assertRaisesRegex(ValueError,'Pinned bytes'):rt.check_run(prepared,run)

    def test_original_numerical_gates(self):
        self.assertEqual(rt.qualify([{'kind':'repeat','margin_difference':0.}],RULES)['margin_error_bound'],1e-6)
        for row in [{'kind':'repeat','margin_difference':1e-12},{'kind':'left','margin_difference':.001001}]:
            with self.assertRaises(ValueError):rt.qualify([row],RULES)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    receipt={'status':'pass' if result.wasSuccessful() else 'fail','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
             'CUDA_initialized':False,'research_pretrained_weights_loaded':False,'GPU_inventory':'mock only','synthetic_forwards':164,
             'coverage':'actual controller/worker pause/reuse, fresh donor provenance, patch format continuation, production replay, normal release, terminal refusal, corruption rejection, post-release report and signed metrics',
             'implementation_snapshot':[c.info(p) for p in c.CODE]}
    if a.output:c.write(a.output,receipt)
    print(json.dumps(receipt,ensure_ascii=False));sys.exit(0 if result.wasSuccessful() else 1)
