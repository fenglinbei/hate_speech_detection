#!/usr/bin/env python3
"""Actual native-only controller lifecycle, output audit, and terminal-state guards."""
import argparse,json,sys,tempfile,unittest,shutil,copy
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
from decimal import localcontext
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from diagnostics import jingba_mixed_demos_inputs_v1 as c
from diagnostics import jingba_mixed_demos_runtime_v1 as rt
from diagnostics import hehe_joint_restore_capture_v1 as cap
from diagnostics.cross_model_applicability_execution_v1 import runtime_versions
from test_dictionary_free_donor_runtime_v1 import synthetic,RULES
ARTIFACTS=None

def fixture():
 rows=[]
 for q in c.IDS:
  for k,d,length in [('M00','D00',3),('MP','D01',5),('MS','D02',6),('MPS','D00',7),('MSP','D01',7)]:
   rid=f'jmix-{q}-{k}';ids=[6]*length+[8,9,10,11,12,13];focal=[length+2,length+3];pre=[length,length+1]
   rows.append({'request_id':rid,'query_id':q,'dictionary_id':d,'condition':k,'input_ids':ids,'prompt_tokens':len(ids),'input_ids_sha256':c.digest(ids),'prompt_sha256':c.digest(rid),'prompt_text':'CPU-SYNTHETIC','spans':[{'id':'query','char_start':0,'char_end':9}],
    'roles':{'query_all':list(range(length,length+5)),'pre_answer':[len(ids)-1]},'capture_positions':pre+focal,'capture_prefix_length':focal[-1]+1,'candidate_tokens':{'有':3,'无':4},'patch_position_sets':{'focal':focal,'pre':pre}})
 return rows

def make_history(hist,run,rows):
 folder=hist/'run-01/records/native-production';folder.mkdir(parents=True);items=[]
 for r in rows:
  if r['condition']!='M00':continue
  old=c.read(rt.record_path(run,'native-production',r['request_id']));old['request_id']='jctx-'+r['query_id']+'-D00';path=folder/(old['request_id']+'.json');c.write(path,old);items.append(c.info(path))
 c.write(hist/'run-01/raw-seal.json',{'native_records':items,'records':[],'CPU_SYNTHETIC':True})

class Checks(unittest.TestCase):
 def test_complete_lifecycle_independent_audit_and_corruption(self):
  rows=fixture();profile={'layers':36,'hidden_size':8,'vocab_size':16,'candidate_tokens':{'有':3,'无':4},'eos_token_ids':[2],'pad_token_id':0,'rms_norm_eps':1e-6,'metadata_sources':[],'weight_sources':[]}
  plan={'acceptance':RULES,'state_scaled_cap':.0001,'format':{'max_new_tokens':8},'budget':{'usual_total':270},'contrasts':[['MP','M00'],['MS','M00'],['MPS','M00'],['MSP','M00'],['MS','MP'],['MSP','MPS']]}
  with tempfile.TemporaryDirectory(prefix='mixed-native-') as td,ExitStack() as stack:
   root=Path(td);p=root/'prepared';p.mkdir();c.write(p/'manifest.json',{'artifacts':[],'sources':[],'CPU_SYNTHETIC':True});c.write(p/'interventions.json',{'cross':[],'self':[]});c.write(p/'model-profile.json',profile)
   refs=[{'query_id':q,'reference':'无' if int(q[1:])<=8 else '有'} for q in c.IDS];c.write(p/'analysis-references.json',{'references':refs});plan['prior_comparators']={'work':str(root/'history')};c.write(p/'execution-plan.json',plan)
   (p/'scoring-inputs.jsonl').write_bytes(b''.join(c.canonical(r)+b'\n' for r in rows))
   decision=root/'decision.json';window={'confirmed':True,'deadline_unix':None};c.write(decision,{'GPU_execution_authorized':True,'user_message':'CPU SYNTHETIC','prepared_manifest':c.info(p/'manifest.json'),'GPU_time_constraint':window})
   allocation=[{'index':0,'uuid':'CPU-SYNTHETIC','name':'CPU','total_mib':48000}];inventory={'devices':[dict(allocation[0],used_mib=0,utilization=0)],'compute_processes':[]};bound=root/'bound.json'
   c.write(bound,{'prepared_manifest':c.info(p/'manifest.json'),'allocation':allocation,'runtime_versions':runtime_versions(),'authorization_note':'CPU SYNTHETIC','GPU_time_constraint':window,'execution_decision':c.info(decision)})
   run=root/'run-test';calls=[];real=rt.subprocess.Popen
   stack.enter_context(patch.object(c,'WORK',root));stack.enter_context(patch.object(c,'validate',return_value=(plan,profile,rows,[],[])));stack.enter_context(patch.object(rt,'gpu_inventory',return_value=inventory));stack.enter_context(patch.object(rt.signal,'signal'))
   stack.enter_context(patch('diagnostics.cross_model_applicability_models_v1.load_checkpoint',return_value=(object(),{'CPU_SYNTHETIC':True})))
   def counted(*a,**kw):
    calls.append(kw)
    if len(calls)==2:(run/'STOP').touch()
    return synthetic(*a,**kw)
   stack.enter_context(patch.object(cap,'forward',side_effect=counted))
   class Process:
    pid=987654321
    def __init__(self,cmd,**kw):
     get=lambda n:cmd[cmd.index(n)+1];rt.worker(get('--prepared'),get('--bound'),get('--run'),get('--invocation'),get('--phase'))
    def wait(self):return 0
   stack.enter_context(patch.object(rt.subprocess,'Popen',Process))
   self.assertEqual(rt.supervise(p,bound,run)['status'],'paused');self.assertEqual(len(calls),2);before=rt.record_path(run,'native-baseline',rows[0]['request_id']).read_bytes();(run/'STOP').unlink()
   self.assertEqual(rt.supervise(p,bound,run,resume=True)['status'],'qualified');self.assertEqual(len(calls),240);self.assertEqual(rt.record_path(run,'native-baseline',rows[0]['request_id']).read_bytes(),before)
   from diagnostics.jingba_mixed_demos_report_v1 import analyze
   with self.assertRaisesRegex(ValueError,'Full run'):analyze(p,run,root/'premature')
   self.assertEqual(rt.supervise(p,bound,run,phase='full')['status'],'complete');self.assertEqual(len(calls),270)
   make_history(root/'history',run,rows)
   with patch.object(rt.subprocess,'Popen',real):analyze(p,run,root/'results')
   from audit_jingba_mixed_demos_results_v1 import audit
   with localcontext() as ctx:ctx.prec=120;result=audit(p,run,root/'results')
   self.assertEqual(result['absolute_vectors'],270);self.assertEqual(result['native_contrasts'],36);self.assertTrue(result['historical_replay']['all_exact_equal'])
   if ARTIFACTS:
    ARTIFACTS.mkdir(parents=True);shutil.copytree(root/'results',ARTIFACTS/'synthetic-results');c.write(ARTIFACTS/'audit.json',result);(ARTIFACTS/'README.md').write_text('CPU合成数据，仅测试流程，不是科学结果，禁止部署。')
   with self.assertRaisesRegex(ValueError,'terminal'):rt.supervise(p,bound,run,phase='full')
   # Repin a wrong published comparison: numerical audit must still reject it.
   path=root/'results/results.json';good=c.read(path);bad=copy.deepcopy(good);bad['native_contrasts'][0]['delta_m']+=.01;c.atomic(path,bad,replace=True)
   manifest=c.read(root/'results/manifest.json');manifest['artifacts']=[c.info(x['path']) for x in manifest['artifacts']];c.atomic(root/'results/manifest.json',manifest,replace=True)
   with localcontext() as ctx:
    ctx.prec=120
    with self.assertRaises(AssertionError):audit(p,run,root/'results')
 def test_no_jobs_and_refusal_before_model(self):
  self.assertEqual(c.make_jobs(fixture()),([],[]))
  with patch.object(rt,'gpu_inventory') as inventory,patch.object(c,'validate') as validate:
   with self.assertRaisesRegex(ValueError,'not authorized'):rt.bind('/missing',0,'/unused','none')
   inventory.assert_not_called();validate.assert_not_called()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--artifacts',type=Path);a=p.parse_args();ARTIFACTS=a.artifacts
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));c.write(a.output,{'status':'pass' if r.wasSuccessful() else 'fail','tests':r.testsRun,'synthetic_forwards':270,'CUDA_initialized':False,'failures':[str(e) for _,e in r.failures+r.errors]});sys.exit(0 if r.wasSuccessful() else 1)
