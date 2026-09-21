#!/usr/bin/env python3
"""CPU mock checks of the actual launch gate, ordering, cancellation and no retry."""
import argparse,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from diagnostics import dictionary_free_donor_inputs_v1 as c
import launch_dictionary_free_donor_v1 as launcher

class Checks(unittest.TestCase):
    def run_fixture(self,mode):
        with tempfile.TemporaryDirectory() as td,ExitStack() as stack:
            p=Path(td);prepared=p/'prepared';prepared.mkdir();c.write(prepared/'manifest.json',{})
            decision=p/'decision.json';here=p/'launch';run=p/'run'
            c.write(decision,{'GPU_execution_authorized':mode!='denied','user_message':'CPU SYNTHETIC ONLY',
                'prepared_manifest':c.info(prepared/'manifest.json'),'GPU_time_constraint':{'confirmed':True,'deadline_unix':None}})
            seen=[]
            stack.enter_context(patch.object(c,'validate',return_value=None))
            def inventory():
                seen.append('inventory')
                if mode=='cancel':(here/'CANCEL').touch();return {'devices':[],'compute_processes':[]}
                return {'devices':[{'index':0,'uuid':'SYNTHETIC','total_mib':48000,'used_mib':0,'utilization':0}],'compute_processes':[]}
            stack.enter_context(patch.object(launcher.rt,'gpu_inventory',side_effect=inventory))
            stack.enter_context(patch.object(launcher.time,'sleep',side_effect=lambda x:seen.append('wait')))
            def bind(prepared,gpu,bound,note,decision):seen.append('bind');c.write(bound,{'synthetic':True})
            stack.enter_context(patch.object(launcher.rt,'bind',side_effect=bind))
            def supervise(prepared,bound,run,phase):
                seen.append(phase)
                if mode=='fail':raise ValueError('deliberate phase failure')
                run.mkdir(exist_ok=True);c.atomic(run/'state.json',{'resource_release':{'synthetic':True}},replace=True)
                if phase=='full':c.write(run/'raw-seal.json',{'synthetic':True})
            stack.enter_context(patch.object(launcher.rt,'supervise',side_effect=supervise))
            def check(*args):seen.append('check');return {'status':'complete' if 'full' in seen else 'qualified'}
            stack.enter_context(patch.object(launcher.rt,'check_run',side_effect=check))
            if mode in ['fail','denied']:
                with self.assertRaises(ValueError):launcher.launch(prepared,decision,here,run)
            else:launcher.launch(prepared,decision,here,run)
            if mode=='denied':self.assertEqual(seen,[]);self.assertFalse(here.exists());self.assertFalse(run.exists())
            elif mode=='cancel':self.assertEqual(seen,['inventory','wait']);self.assertFalse(run.exists())
            elif mode=='fail':self.assertEqual(seen,['inventory','bind','engineering']);self.assertEqual(c.read(here/'state.json')['event'],'failed_no_automatic_retry')
            else:
                self.assertEqual(seen,['inventory','bind','engineering','check','full','check'])
                self.assertEqual(c.read(here/'state.json')['event'],'complete_GPU_released')
                with self.assertRaisesRegex(ValueError,'duplicate'):launcher.launch(prepared,decision,here,run)
    def test_denied_before_any_inventory_or_creation(self):self.run_fixture('denied')
    def test_ordered_pipeline_no_duplicate(self):self.run_fixture('success')
    def test_cancel_wait_without_binding(self):self.run_fixture('cancel')
    def test_failure_terminal_no_retry(self):self.run_fixture('fail')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    c.write(a.output,{'status':'pass' if result.wasSuccessful() else 'fail','tests':result.testsRun,
        'CUDA_initialized':False,'GPU_inventory':'synthetic mock only','research_checkpoint_loaded':False,
        'failures':[str(e) for _,e in result.failures+result.errors]})
    sys.exit(0 if result.wasSuccessful() else 1)
