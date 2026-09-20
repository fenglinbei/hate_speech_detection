#!/usr/bin/env python3
"""Tiny random native CPU Qwen3: trajectory hooks and causal placement checks."""
import argparse,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from diagnostics import hehe_bridge_capture_v1 as cap


def request(donor=False):
    ids=[20,21,22] if donor else [6,7,8]
    ids += [9,10,11,12,13,14,15]
    return {'input_ids':ids,'prompt_tokens':10,'roles':{'query_all':[5,6,7,8],'pre_answer':[9]},'candidate_tokens':{'有':3,'无':4}}


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        with patch.object(torch.cuda,'is_available',return_value=False):
            from transformers import Qwen3Config,Qwen3ForCausalLM
        torch.manual_seed(71);torch.set_num_threads(2)
        cfg=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,attention_dropout=0,bos_token_id=1,eos_token_id=2,pad_token_id=0)
        cfg._attn_implementation='eager';cls.model=Qwen3ForCausalLM(cfg).float().eval();cls.torch=torch
        cls.profile={'layers':4,'hidden_size':32,'candidate_tokens':{'有':3,'无':4},'rms_norm_eps':cfg.rms_norm_eps}

    def test_readonly_native_and_independent_states(self):
        r=request();v,s,_,t=cap.forward(self.model,r,0,positions=[5,6],observe=True,trajectory=True)
        raw,ss,_,none=cap.forward(self.model,r,0,positions=[5,6],observe=True)
        np.testing.assert_array_equal(v,raw);np.testing.assert_array_equal(s,ss);self.assertIsNone(none)
        cap.validate_trajectory(t,r,self.profile,v)
        observed={}
        handles=[b.register_forward_hook(lambda m,a,o,i=i:observed.update({i:o[0,9].detach().numpy().copy()})) for i,b in enumerate(self.model.model.layers)]
        try:cap.forward(self.model,r,0)
        finally:
            for h in handles:h.remove()
        for i in range(4):np.testing.assert_array_equal(t['states'][i,2],observed[i])

    def test_self_cross_structural_and_independent_patch(self):
        r,d=request(),request(True);ps=[5,6]
        v,s,_,t=cap.forward(self.model,r,0,positions=ps,observe=True,trajectory=True)
        _,ds,_,_=cap.forward(self.model,d,0,positions=ps,observe=True)
        own,_,proof,selft=cap.forward(self.model,r,0,positions=ps,layer=1,replacement=s[1],trajectory=True)
        np.testing.assert_array_equal(v,own);self.assertEqual(proof['changed_rows_in_unpadded_coordinates'],[])
        cap.compare_trajectories(t,selft,'self')
        vp,_,_,tp=cap.forward(self.model,r,0,positions=ps,layer=1,replacement=ds[1],trajectory=True)
        cap.validate_trajectory(tp,r,self.profile,vp);cap.structural_guard(tp,t,1)
        self.assertFalse(np.array_equal(tp['states'][2,1],t['states'][2,1]))
        def independent(m,a,o):
            value=o.clone();value[0,ps]=self.torch.from_numpy(ds[1]);return value
        h=self.model.model.layers[1].register_forward_hook(independent)
        try:expected,_,_,et=cap.forward(self.model,r,0,trajectory=True)
        finally:h.remove()
        np.testing.assert_array_equal(vp,expected);cap.compare_trajectories(tp,et,'repeat')

    def test_padding_prefix_and_final_layer(self):
        r=request();ps=[5,6]
        v,s,_,t=cap.forward(self.model,r,0,positions=ps,observe=True,trajectory=True)
        _,ds,_,_=cap.forward(self.model,request(True),0,positions=ps,observe=True)
        ref,_,_,pt=cap.forward(self.model,r,0,positions=ps,layer=1,replacement=ds[1],trajectory=True)
        for padding in ['left','right']:
            vv,_,_,tt=cap.forward(self.model,r,0,positions=ps,layer=1,replacement=ds[1],padding=padding,trajectory=True)
            np.testing.assert_allclose(vv,ref,rtol=0,atol=2e-6);cap.compare_trajectories(tt,pt,padding)
        end,_,_,end_t=cap.forward(self.model,r,0,positions=ps,layer=3,replacement=ds[3],trajectory=True)
        np.testing.assert_array_equal(v,end);cap.compare_trajectories(t,end_t,'self')
        _,prefix,_,missing=cap.forward(self.model,r,0,positions=ps,observe=True,prefix_length=7)
        np.testing.assert_allclose(s,prefix,rtol=0,atol=2e-6);self.assertIsNone(missing)

    def test_reject_leakage_and_cleanup(self):
        r=request();_,_,_,t=cap.forward(self.model,r,0,trajectory=True)
        corrupt={k:v.copy() for k,v in t.items()};corrupt['states'][0,0,0]+=1
        with self.assertRaisesRegex(ValueError,'before cross-position'):cap.structural_guard(corrupt,t,1)
        with patch.object(cap.patcher,'forward',side_effect=RuntimeError('deliberate test failure')):
            with self.assertRaises(RuntimeError):cap.forward(self.model,r,0,trajectory=True)
        for b in self.model.model.layers:
            for m in [b,b.mlp,b.self_attn,b.post_attention_layernorm]:
                self.assertFalse(m._forward_hooks);self.assertFalse(m._forward_pre_hooks)
        self.assertFalse(self.torch.cuda.is_initialized())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    import torch
    receipt={'status':'pass' if result.wasSuccessful() else 'fail','tests':result.testsRun,
        'CUDA_initialized':torch.cuda.is_initialized(),'research_checkpoint_loaded':False,
        'failures':[str(e) for _,e in result.failures+result.errors]}
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
    print(json.dumps(receipt));sys.exit(0 if result.wasSuccessful() else 1)
