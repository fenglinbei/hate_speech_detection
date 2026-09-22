#!/usr/bin/env python3
"""CPU report/audit/controller integration, including terminal launch failures."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from diagnostics import reference_routing_content_inputs_v1 as c
from diagnostics import reference_routing_content_runtime_v1 as rt
from analyze_reference_routing_content_v1 import build_report
from audit_reference_routing_content_results_v1 import independent
from test_reference_routing_content_analysis_v1 import records
HERE=None


class Checks(unittest.TestCase):
    def test_full_dev_report_matrix_and_relation_groups(self):
        qs=[q for q in c.read(c.PREPARED/'analysis-references.json')['queries'] if q['split']=='development']
        r=records(qs);p={'bN':'0','bU':'0','QAS_factor':1,'Bstar':'N'}
        conditions=['BG','BA','BI','MPS','MSP']+[f'{o}_replace_{k}' for o in ('MPS','MSP') for k in range(1,5)]
        report=build_report(r,qs,p,conditions,{'synthetic':True,'research_forwards':0})
        self.assertEqual(len(report['per_endpoint']),24*13*8)
        self.assertEqual(len(report['bilateral']['rows']),24*8)
        self.assertEqual(report['primary']['OFFSET7']['balanced_G']['exact_value'],'0')
        self.assertEqual(report['method_cost']['U']['distinct_production_forwards_with_shared_M00'],336)
        self.assertEqual(report['method_cost']['N']['distinct_production_forwards_with_shared_M00'],312)
        self.assertEqual(set(report['bilateral']['by_human_relation']['semantic_reference_fit']),{'direct','none'})
        c.write(HERE/'synthetic-report.json',dict(report,synthetic=True,not_scientific=True))

    def test_independent_120_digit_audit_and_corruption_refusal(self):
        run=c.WORK/'checks/synthetic-01/stage-a';profile={'vocab_size':8,'candidate_tokens':{'有':3,'无':4}}
        receipt=independent(run,profile)
        self.assertEqual(receipt['vectors'],930);self.assertEqual(receipt['exact_Fraction_margins'],912)
        c.write(HERE/'synthetic-numerical-audit.json',dict(receipt,synthetic=True))
        source=c.read(run/'raw-seal.json')['records'][0];r=c.read(source['path'])
        self.assertFalse(r['prefix']);r['readout']['m']+=1
        bad=HERE/'bad-margin';bad.mkdir();c.write(bad/'tampered-record.json',r)
        c.write(bad/'raw-seal.json',{'records':[c.info(bad/'tampered-record.json')]})
        with self.assertRaisesRegex(ValueError,'exact FP32 subtraction'):independent(bad,profile)

    def controller(self,root,spawn):
        prepared=root/'prepared';prepared.mkdir(parents=True);c.write(prepared/'manifest.json',{'synthetic':True})
        auth=root/'decision.json';c.write(auth,{'GPU_execution_authorized':True,'user_message':'SYNTHETIC CONTROL FIXTURE; NOT USER GPU AUTHORIZATION',
            'prepared_manifest':c.info(prepared/'manifest.json'),'allowed_stages':['stage-a'],'window_confirmed':True,
            'allowed_gpu_indices':[0],'deadline_unix':None})
        device={'index':0,'uuid':'GPU-CPU-FIXTURE','name':'CPU MOCK','total_mib':48000,'used_mib':0,'utilization':0}
        inv={'devices':[device],'compute_processes':[]}
        with patch.object(c,'WORK',root),patch.object(c,'validate',return_value=({},{})),patch.object(rt,'load_group',return_value=('stage-a',[],{})),\
             patch.object(rt,'gpu_inventory',return_value=inv),patch.object(rt,'idle'),patch.object(rt.subprocess,'Popen',side_effect=spawn),\
             patch.object(rt,'audit_run',return_value={'status':'PASS','synthetic':True}):
            return rt.launch(prepared,auth,root/'run-01','stage-a')

    def test_controller_normal_release_and_terminal_duplicate(self):
        root=HERE/'controller-success'
        class Process:
            pid=2147483600
            def __init__(self,cmd,**kw):
                run=Path(cmd[cmd.index('--run')+1]);s=c.read(run/'state.json');s['status']='COMPLETE_RELEASING';c.atomic(run/'state.json',s,replace=True)
                self.env=kw['env']
            def poll(self):return 0
            def wait(self,**kw):return 0
        result=self.controller(root,Process);self.assertTrue(result['synthetic'])
        s=c.read(root/'run-01/state.json');self.assertEqual(s['status'],'COMPLETE');self.assertTrue(s['owned_worker_absent'])
        with patch.object(c,'WORK',root),patch.object(c,'validate',return_value=({},{})),patch.object(rt,'load_group',return_value=('stage-a',[],{})),patch.object(rt,'gpu_inventory',side_effect=AssertionError('queried after duplicate')):
            with self.assertRaisesRegex(ValueError,'no restart'):
                rt.launch(root/'prepared',root/'decision.json',root/'run-01','stage-a')

    def test_spawn_failure_is_terminal_without_worker(self):
        root=HERE/'controller-failure'
        def fail(*args,**kw):raise OSError('deliberate CPU spawn failure')
        with self.assertRaises(OSError):self.controller(root,fail)
        s=c.read(root/'run-01/state.json');self.assertEqual(s['status'],'FAILED_TERMINAL')
        self.assertFalse(s['worker_created']);self.assertTrue(s['owned_worker_absent'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);x=p.parse_args()
    HERE=x.directory.absolute();c.require(not HERE.exists(),'New test directory required');HERE.mkdir(parents=True)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    receipt={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,'CPU_only':True,
        'hardware_and_process_launch_mocked':True,'real_model_forwards':0,'failures':[str(e) for _,e in result.errors+result.failures]}
    c.write(x.output,receipt);print(json.dumps(receipt));sys.exit(not result.wasSuccessful())
