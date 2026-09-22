#!/usr/bin/env python3
"""CPU synthetic execution of the actual scheduler plus launch refusal controls."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import reference_routing_content_inputs_v1 as c
from diagnostics import reference_routing_content_runtime_v1 as rt

HERE=None


class Synthetic:
    """Algebra fixture, no tokenizer, torch, trained weights or CUDA."""
    def __init__(self,profile,fail_at=None):self.profile=profile;self.calls=0;self.fail_at=fail_at
    def memory(self):return {'allocated_peak_bytes':0,'reserved_peak_bytes':0}
    def forward(self,r,pad,**kw):
        self.calls+=1
        if self.calls==self.fail_at:raise ValueError('deliberate synthetic backend failure')
        p=self.profile;L,D=p['layers'],p['hidden_size'];n=r['prompt_tokens'];H=2;KV=1;hd=2
        h=np.zeros(D,np.float32);h[0]=1
        norm=h/np.float32(np.sqrt(np.mean(h*h)+p['rms_norm_eps']))
        weight=np.zeros((2,D),np.float32);weight[1,0]=1
        vector=np.zeros(p['vocab_size'],np.float32);vector[4]=norm[0]
        if kw['append_tokens']:vector[2]=10
        bank=np.broadcast_to(h,(L,len(kw['positions']),D)).copy() if kw['observe'] else None
        t=None
        if kw['trajectory']:
            t={'states':np.broadcast_to(h,(L,3,D)).copy(),'normalized':np.broadcast_to(norm,(L,3,D)).copy(),
               'lens_logits':np.broadcast_to(norm@weight.T,(L,3,2)).copy(),'branches':np.zeros((L,2,D),np.float32),
               'position':np.array([n-1],np.int64),'candidate_ids':np.array([3,4],np.int64),
               'label_weights':weight,'norm_weight':np.ones(D,np.float32),'norm_eps':np.array([p['rms_norm_eps']],np.float64)}
        av=None
        if kw['capture_av']:
            av={'prompt_sha256':r['prompt_sha256'],'input_ids_sha256':r['input_ids_sha256'],'position':n-1,'layer':18,
                'Q':np.zeros((H,hd),np.float32),'K':np.zeros((KV,n,hd),np.float32),'V':np.zeros((KV,n,hd),np.float32),
                'A':np.full((H,n),1/n,np.float32),'scores':np.zeros((H,n),np.float32),
                'head_AV':np.zeros((H,hd),np.float32),'o_proj':np.zeros(D,np.float32)}
            if kw['av_sources'] is not None:av['installed_o_proj']=np.zeros(D,np.float32)
        proof=None
        if kw['replacement'] is not None:
            proof={'layer':17,'positions':kw['positions'],'outside_rows_exact':True,'donor_rows_exact':True,
                   'native_output_unmodified':True,'synthetic':True}
        return vector,bank,proof,t,av,[]


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile={'vocab_size':8,'layers':36,'hidden_size':4,'rms_norm_eps':1e-6,
                     'candidate_tokens':{'有':3,'无':4},'pad_token_id':0,'eos_token_ids':[2]}
        cls.prepared=HERE/'fixture-prepared';cls.prepared.mkdir(parents=True)
        c.write(cls.prepared/'manifest.json',{'synthetic':True,'not_scientific_inputs':True})
        cls.receipts=[]

    def matrix(self,group,one_query=False):
        rows=c.lines(c.PREPARED/group/'inputs.jsonl');jobs=c.read(c.PREPARED/group/'jobs.json')
        if one_query:
            qid=rows[0]['query_id'];rows=[r for r in rows if r['query_id']==qid];ids={r['request_id'] for r in rows}
            jobs={k:[j for j in values if j['recipient'] in ids] for k,values in jobs.items()}
        rows=deepcopy(rows)
        for r in rows:r['candidate_tokens']=self.profile['candidate_tokens']
        return rows,jobs

    def test_all_114_A_configs_actual_scheduler_and_audit(self):
        rows,registry=self.matrix('stage-a');run=HERE/'stage-a';run.mkdir()
        cost=rt.execute(self.prepared,run,'stage-a',Synthetic(self.profile),self.profile,rows,registry,synthetic=True)
        self.assertEqual(cost['forwards'],930)
        receipt=rt.audit_run(self.prepared,run,'stage-a',synthetic=True,fixture=(self.profile,rows,registry))
        self.receipts.append(receipt)

    def test_B_dev_and_both_confirmation_schedules(self):
        for group in ('development','confirmation-factor-1','confirmation-factor-4'):
            rows,registry=self.matrix(group,True);run=HERE/group;run.mkdir()
            cost=rt.execute(self.prepared,run,group,Synthetic(self.profile),self.profile,rows,registry,synthetic=True)
            self.assertEqual(cost['forwards'],c.budget(registry['science'],rows,registry['controls'])['usual_total'])
            receipt=rt.audit_run(self.prepared,run,group,synthetic=True,fixture=(self.profile,rows,registry))
            self.receipts.append(receipt)

    def test_failure_keeps_partial_records_no_raw_seal_or_retry(self):
        rows,registry=self.matrix('development',True);run=HERE/'deliberate-failure';run.mkdir()
        b=Synthetic(self.profile,fail_at=3)
        with self.assertRaisesRegex(ValueError,'deliberate'):
            rt.execute(self.prepared,run,'development',b,self.profile,rows,registry,synthetic=True)
        self.assertEqual(b.calls,3);self.assertFalse((run/'raw-seal.json').exists())
        self.assertEqual(len(list((run/'records').rglob('*.json'))),2)
        # Existing record exclusive writes reject an accidental rerun.
        with self.assertRaises(FileExistsError):
            rt.execute(self.prepared,run,'development',Synthetic(self.profile),self.profile,rows,registry,synthetic=True)

    def test_launch_refuses_before_weight_hardware_and_output(self):
        out=HERE/'must-not-create'
        with patch.object(rt,'gpu_inventory',side_effect=AssertionError('hardware queried')),patch.object(c,'validate',side_effect=AssertionError('weights checked')):
            with self.assertRaisesRegex(ValueError,'new explicit user authorization'):
                rt.launch(self.prepared,None,out,'stage-a')
        self.assertFalse(out.exists())
        self.assertNotIn('torch',sys.modules)

    def test_confirmation_without_calibration_is_blocked(self):
        with self.assertRaisesRegex(ValueError,'sealed development'):
            rt.calibration_gate(self.prepared,None)
        bad=HERE/'bad-lock';bad.mkdir()
        c.write(bad/'parameters.json',{'prepared_manifest':c.info(self.prepared/'manifest.json'),
            'split':'confirmation','confirmation_outputs_seen':True,'QAS_factor':4})
        c.write(bad/'manifest.json',{'schema_version':'reference-routing-content-calibration/v1',
            'artifacts':[c.info(bad/'parameters.json')]})
        with self.assertRaisesRegex(ValueError,'source/split'):
            rt.calibration_gate(self.prepared,bad/'manifest.json')

    def test_expired_or_wrong_stage_authorization_fails(self):
        for name,fields,stage in [('wrong',{},'development'),('expired',{'deadline_unix':1},'stage-a')]:
            path=HERE/(name+'.json')
            c.write(path,dict(GPU_execution_authorized=True,user_message='SYNTHETIC TEST ONLY',
                prepared_manifest=c.info(self.prepared/'manifest.json'),allowed_stages=['stage-a'],window_confirmed=True,**fields))
            with self.assertRaises(ValueError):rt.decision(self.prepared,path,stage)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    HERE=a.directory;c.require(not HERE.exists(),'New synthetic check directory required');HERE.mkdir(parents=True)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    receipt={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
        'synthetic_scheduler_runs':Checks.receipts,'CPU_only':True,'research_model_forwards':0,
        'failures':[str(e) for _,e in result.errors+result.failures]}
    c.write(a.output,receipt);print(json.dumps(receipt));sys.exit(not result.wasSuccessful())
