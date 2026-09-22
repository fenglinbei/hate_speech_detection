#!/usr/bin/env python3
"""Random tiny native CPU model only; never research/model evidence."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from test_hehe_bridge_capture_v1 import Checks as Base, request as old_request
from diagnostics import reference_routing_content_capture_v1 as cap


def request(donor=False):
    r=old_request(donor)
    r['prompt_sha256']=hashlib.sha256(str(r['input_ids']).encode()).hexdigest()
    r['input_ids_sha256']=r['prompt_sha256']
    return r


class Checks(Base):
    def forward(self,r,**kw):
        return cap.forward(self.model,r,0,av_layer=2,qas_layers=(2,3),**kw)

    def banks(self):
        r=request();d=request(True)
        n=self.forward(r,positions=[5,6],observe=True,trajectory=True,capture_av=True)
        donor=self.forward(d,positions=[5,6],observe=True)
        u=self.forward(r,positions=[5,6],layer=1,replacement=donor[1][1],trajectory=True,capture_av=True)
        return r,n,u,donor[1][1]

    def test_av_native_GQA_shapes_and_actual_reconstruction(self):
        r,n,u,rows=self.banks()
        raw=self.forward(r,positions=[5,6],observe=True,trajectory=True)
        np.testing.assert_array_equal(n[0],raw[0]);np.testing.assert_array_equal(n[1],raw[1])
        cap.bridge.compare_trajectories(n[3],raw[3],'repeat')
        t=n[4]
        self.assertEqual(t['Q'].shape,(4,8));self.assertEqual(t['K'].shape,(2,10,8))
        self.assertEqual(t['A'].shape,(4,10));self.assertEqual(t['head_AV'].shape,(4,8))
        manual=np.einsum('hs,hsd->hd',t['A'].astype(float),np.repeat(t['V'],2,axis=0).astype(float))
        np.testing.assert_allclose(t['head_AV'],manual,atol=1e-7,rtol=0)
        proof=cap.structure(n[4],u[4],[5,6]);self.assertEqual(proof['outside_key_count'],8)
        self.assertFalse(np.array_equal(n[4]['K'][:,[5,6]],u[4]['K'][:,[5,6]]))

    def test_factorial_all_cells_independent_o_projection(self):
        r,n,u,rows=self.banks();torch=self.torch
        for a,b in ((0,0),(0,1),(1,0),(1,1)):
            sources=((n,u)[a][4],(n,u)[b][4])
            actual=self.forward(r,positions=[5,6],layer=1,replacement=rows,
                                trajectory=True,capture_av=True,av_sources=sources)
            # Independent per-head numpy multiplication + projection, acceptable
            # tiny numerical difference from native full-shaped reduction.
            heads=np.einsum('hs,hsd->hd',sources[0]['A'].astype(float),np.repeat(sources[1]['V'],2,axis=0).astype(float))
            expected=heads.reshape(-1) @ self.model.model.layers[2].self_attn.o_proj.weight.detach().numpy().astype(float).T
            np.testing.assert_allclose(actual[4]['installed_o_proj'],expected,atol=2e-7,rtol=0)
            self.assertTrue(actual[5][0]['outside_rows_exact'])
            cap.bridge.validate_trajectory(actual[3],r,self.profile,actual[0])
            cap.bridge.structural_guard(actual[3],n[3],1)
            if (a,b)==(1,1):
                np.testing.assert_array_equal(actual[0],u[0]);cap.bridge.compare_trajectories(actual[3],u[3],'self')
            if (a,b)==(0,0):np.testing.assert_array_equal(actual[4]['installed_o_proj'],n[4]['o_proj'])
        native_self=self.forward(r,av_sources=(n[4],n[4]),capture_av=True,trajectory=True)
        np.testing.assert_array_equal(native_self[0],n[0])

    def test_padding_answer_continuation_and_causal_mask(self):
        r,n,u,rows=self.banks()
        for padding in ('left','right'):
            out=self.forward(r,positions=[5,6],layer=1,replacement=rows,padding=padding,
                             capture_av=True,av_sources=(n[4],u[4]))
            base=self.forward(r,positions=[5,6],layer=1,replacement=rows,
                              capture_av=True,av_sources=(n[4],u[4]))
            np.testing.assert_allclose(out[0],base[0],atol=2e-6,rtol=0)
            self.assertEqual(out[4]['A'].shape,(4,10))
        continuation=self.forward(r,positions=[5,6],layer=1,replacement=rows,
                                 append_tokens=[3],av_sources=(u[4],u[4]),capture_av=True)
        expected=self.forward(r,positions=[5,6],layer=1,replacement=rows,append_tokens=[3])
        np.testing.assert_array_equal(continuation[0],expected[0])
        self.assertEqual(continuation[5][0]['position'],9)
        np.testing.assert_array_equal(continuation[4]['A'],u[4]['A'])

    def test_QAS_factor_one_and_independent_probability_reweighting(self):
        r=request();n=self.forward(r,capture_av=True)
        same=self.forward(r,qas_factor=1,capture_av=True)
        np.testing.assert_array_equal(n[0],same[0])
        for f in (2,4):
            q=self.forward(r,qas_factor=f,capture_av=True)
            expected=n[4]['A'].astype(float).copy();expected[:,r['roles']['query_all']]*=f
            expected/=expected.sum(-1,keepdims=True)
            np.testing.assert_allclose(q[4]['A'],expected,rtol=0,atol=1e-7)
            np.testing.assert_array_equal(q[4]['V'],n[4]['V'])
            self.assertEqual([x['layer'] for x in q[5]],[2,3])
            for p in ('left','right'):
                qp=self.forward(r,qas_factor=f,padding=p)
                np.testing.assert_allclose(qp[0],q[0],rtol=0,atol=2e-6)
            # Original p intervention still affects downstream answer tokens.
            qc=self.forward(r,qas_factor=f,append_tokens=[3],capture_av=True)
            np.testing.assert_allclose(qc[4]['A'],q[4]['A'],atol=2e-7,rtol=0)

    def test_wrong_recipient_source_and_structural_corruption_rejected(self):
        r,n,u,rows=self.banks();other=deepcopy(n[4]);other['prompt_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'same recipient'):
            self.forward(r,av_sources=(other,n[4]))
        for field,index in [('Q',(0,0)),('K',(0,0,0)),('V',(0,0,0)),('scores',(0,0))]:
            bad=deepcopy(u[4]);bad[field][index]+=1
            with self.assertRaises(ValueError):cap.structure(n[4],bad,[5,6])
        bad=deepcopy(n[4]);bad['A']*=2
        with self.assertRaisesRegex(ValueError,'probability'):self.forward(r,av_sources=(bad,n[4]))

    def test_cleanup_after_forward_exception(self):
        from transformers.models.qwen3 import modeling_qwen3 as native
        old=native.eager_attention_forward
        def fail(*args):raise RuntimeError('deliberate CPU fixture failure')
        h=self.model.model.layers[3].register_forward_pre_hook(fail)
        try:
            with self.assertRaises(RuntimeError):self.forward(request(),capture_av=True)
        finally:h.remove()
        self.assertIs(old,native.eager_attention_forward)
        self.assertFalse(self.model.model.layers[2].self_attn.o_proj._forward_hooks)
        self.forward(request(),capture_av=True)
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_fixed17_18_actual_native_provenance_audit(self):
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from diagnostics import reference_routing_content_runtime_v1 as rt
        from diagnostics import reference_routing_content_inputs_v1 as inputs
        cfg=Qwen3Config(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=36,
            num_attention_heads=4,num_key_value_heads=2,head_dim=4,attention_dropout=0)
        cfg._attn_implementation='eager';model=Qwen3ForCausalLM(cfg).float().eval()
        profile={'vocab_size':64,'layers':36,'hidden_size':16,'heads':4,'kv_heads':2,'head_dim':4,
                 'candidate_tokens':{'有':3,'无':4},'rms_norm_eps':cfg.rms_norm_eps,'pad_token_id':0}
        r,d=request(),request(True);r['request_id']='recipient';d['request_id']='donor'
        for x in (r,d):x['capture_positions']=[5,6];x['capture_prefix_length']=7
        n=cap.forward(model,r,0,positions=[5,6],observe=True,trajectory=True,capture_av=True)
        donor=cap.forward(model,d,0,positions=[5,6],observe=True)
        up={'donor':'donor','layer':17,'positions':[5,6],'donor_positions':[5,6],'strength':1}
        with tempfile.TemporaryDirectory(prefix='rrc-proof-cpu-') as tmp:
            run=Path(tmp)
            def save(jid,req,out,job):
                v,bank,proof,t,av,ops=out;base=run/'arrays'/jid
                record={'job_id':jid,'pass':'probe','job':job,'padding':'none','prefix':False,'append_tokens':[],
                    'vector':rt.save_array(base.with_suffix('.npy'),v),
                    'bank':rt.save_array(base.with_suffix('.bank.npy'),bank) if bank is not None else None,
                    'trajectory':rt.save_bundle(base.with_suffix('.trajectory.npz'),t),'av':rt.save_bundle(base.with_suffix('.av.npz'),av),
                    'patch_proof':proof,'operations':ops,'sources':[], 'valid_tokens':10,'tensor_tokens':10,
                    'trajectory_validation':cap.bridge.validate_trajectory(t,req,profile,v) if t is not None else None}
                inputs.write(rt.recpath(run,'probe',jid),record);return record
            nr=save('recipient/N',r,n,{});dr=save('donor/N',d,donor,{})
            sources=[{'kind':'block_donor','record':inputs.info(rt.recpath(run,'probe','donor/N')),
                      'bank':dr['bank'],'positions':[5,6]}]
            job={'job_id':'recipient/U','recipient':'recipient','upstream':up,'av_override':None,'qas_factor':1}
            u=cap.forward(model,r,0,positions=[5,6],layer=17,replacement=donor[1][17],trajectory=True,capture_av=True)
            ur=save('recipient/U',r,u,job);ur['sources']=sources
            rt.validate_proof(run,ur,r,job,{'recipient':r,'donor':d},profile,(ur,u[0],u[1],u[3],u[4]))
            cap.structure(n[4],u[4],[5,6])
            avj=dict(job,job_id='recipient/AV00',av_override={'cell':'00','A_from':'recipient/N','V_from':'recipient/N'})
            av=cap.forward(model,r,0,positions=[5,6],layer=17,replacement=donor[1][17],
                           trajectory=True,capture_av=True,av_sources=(n[4],n[4]))
            ar=save('recipient/AV00',r,av,avj);ar['sources']=sources+[
                {'kind':key,'record':inputs.info(rt.recpath(run,'probe','recipient/N')),'av':nr['av']} for key in ('A_from','V_from')]
            rt.validate_proof(run,ar,r,avj,{'recipient':r,'donor':d},profile,(ar,av[0],av[1],av[3],av[4]))
            bad=deepcopy(ar);bad['operations'][0]['outside_rows_exact']=False
            with self.assertRaises(ValueError):rt.validate_proof(run,bad,r,avj,{'recipient':r,'donor':d},profile,(ar,av[0],av[1],av[3],av[4]))
            qj=dict(job,job_id='recipient/Q2',upstream=None,qas_factor=2)
            q=cap.forward(model,r,0,qas_factor=2);qr=save('recipient/Q2',r,q,qj)
            rt.validate_proof(run,qr,r,qj,{'recipient':r,'donor':d},profile,(qr,q[0],q[1],q[3],q[4]))
        self.assertFalse(self.torch.cuda.is_initialized())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args()
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    import torch
    receipt={'status':'PASS' if r.wasSuccessful() else 'FAIL','tests':r.testsRun,
             'random_CPU_fixture_only':True,'research_checkpoint_loaded':False,'CUDA_initialized':torch.cuda.is_initialized(),
             'failures':[str(e) for _,e in r.errors+r.failures]}
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
    print(json.dumps(receipt));sys.exit(not r.wasSuccessful())
