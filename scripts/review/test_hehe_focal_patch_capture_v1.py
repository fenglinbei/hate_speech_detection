#!/usr/bin/env python3
"""Independent small random CPU Qwen3 tests, never research weights or CUDA."""
from pathlib import Path
from unittest.mock import patch
import argparse, json, sys, unittest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from diagnostics import hehe_focal_patch_capture_v1 as cap


def request(ids=None):
    ids=list(range(6,16)) if ids is None else ids
    return {'input_ids':ids,'prompt_tokens':len(ids),'roles':{'query_all':[5,6,7,8]},'candidate_tokens':{'有':3,'无':4}}


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        with patch.object(torch.cuda,'is_available',return_value=False):
            from transformers import Qwen3Config,Qwen3ForCausalLM
        torch.manual_seed(43);torch.set_num_threads(2)
        cfg=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=3,
                       num_attention_heads=4,num_key_value_heads=2,head_dim=8,attention_dropout=0,
                       bos_token_id=1,eos_token_id=2,pad_token_id=0)
        cfg._attn_implementation='eager';cls.torch=torch;cls.model=Qwen3ForCausalLM(cfg).float().eval()

    def test_capture_native_state_and_logit_equality(self):
        req=request();ps=[5,6,8,9]
        v,s,_=cap.forward(self.model,req,0,observe=True,positions=ps)
        raw,_,_=cap.forward(self.model,req,0)
        np.testing.assert_array_equal(v,raw)
        observed={};handles=[]
        def hook(li):
            def save(m,a,o):observed[li]=o.detach().cpu().numpy().copy()
            return save
        for li,b in enumerate(self.model.model.layers):handles.append(b.register_forward_hook(hook(li)))
        try:cap.forward(self.model,req,0)
        finally:
            for h in handles:h.remove()
        for li in range(3):np.testing.assert_array_equal(s[li],observed[li][0,ps])
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_two_tokens_self_cross_padding_and_last_layer(self):
        r=request();d=request([20,21,22]+r['input_ids'][3:]);ps=[5,6]
        vr,sr,_=cap.forward(self.model,r,0,observe=True,positions=ps)
        vd,sd,_=cap.forward(self.model,d,0,observe=True,positions=ps)
        self.assertFalse(np.array_equal(sr,sd))
        for li in range(3):
            vs,_,pr=cap.forward(self.model,r,0,positions=ps,layer=li,replacement=sr[li])
            np.testing.assert_array_equal(vs,vr);self.assertEqual(pr['changed_rows_in_unpadded_coordinates'],[])
            vp,_,pr=cap.forward(self.model,r,0,positions=ps,layer=li,replacement=sd[li])
            self.assertTrue(pr['outside_rows_exact'] and pr['native_output_unmodified'])
            if li==2:np.testing.assert_array_equal(vp,vr)
            else:self.assertFalse(np.array_equal(vp,vr))
            for padding in ['left','right']:
                vv,_,proof=cap.forward(self.model,r,0,positions=ps,layer=li,replacement=sd[li],padding=padding)
                np.testing.assert_allclose(vv,vp,rtol=0,atol=2e-6)
                self.assertEqual(proof['positions'],ps)

    def test_independent_patch_matches_and_continuation_is_repatched(self):
        r=request();ps=[5,6];_,sd,_=cap.forward(self.model,request([20]+r['input_ids'][1:]),0,observe=True,positions=ps)
        for append in [[],[3]]:
            actual,_,_=cap.forward(self.model,r,0,positions=ps,layer=0,replacement=sd[0],append_tokens=append)
            def independent(m,a,o):
                out=o.clone();out[:,ps,:]=self.torch.from_numpy(sd[0]);return out
            h=self.model.model.layers[0].register_forward_hook(independent)
            try:
                rr=request(r['input_ids']+append);expected,_,_=cap.forward(self.model,rr,0)
            finally:h.remove()
            np.testing.assert_array_equal(actual,expected)

    def test_prefix_and_future_invariance(self):
        r=request();ps=[5,6];_,s,_=cap.forward(self.model,r,0,positions=ps,observe=True)
        _,p,_=cap.forward(self.model,r,0,positions=ps,observe=True,prefix_length=7)
        np.testing.assert_allclose(s,p,rtol=0,atol=2e-6)
        r2=request(r['input_ids'][:7]+[30,31,32]);_,future,_=cap.forward(self.model,r2,0,positions=ps,observe=True)
        np.testing.assert_array_equal(s,future)

    def test_reject_geometry_and_remove_hook_on_failure(self):
        r=request();x=np.zeros((1,32),np.float32)
        for ps in [[9],[3],[5,5]]:
            with self.assertRaises(Exception):cap.forward(self.model,r,0,positions=ps,layer=0,replacement=x)
        with self.assertRaises(Exception):cap.forward(self.model,r,0,positions=[5],layer=0,replacement=x.astype(np.float64))
        with patch.object(cap.native,'forward',side_effect=RuntimeError('deliberate CPU failure')):
            with self.assertRaises(RuntimeError):cap.forward(self.model,r,0,positions=[5],layer=0,replacement=x)
        self.assertTrue(all(not b._forward_hooks for b in self.model.model.layers))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    import torch
    receipt={'status':'pass' if result.wasSuccessful() else 'fail','tests':result.testsRun,
             'CUDA_initialized':torch.cuda.is_initialized(),'research_checkpoint_loaded':False,
             'model':'tiny random CPU Qwen3:3 layers,4 heads,GQA','failures':[str(e) for _,e in result.failures+result.errors]}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
    print(json.dumps(receipt));sys.exit(0 if result.wasSuccessful() else 1)
