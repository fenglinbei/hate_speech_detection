#!/usr/bin/env python3
"""Reuse sealed native-hook checks and add actual single-token focus/pre controls."""
import argparse,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from test_hehe_joint_restore_capture_v1 import Checks as ParentChecks
from test_hehe_bridge_capture_v1 import request
from diagnostics import hehe_joint_restore_capture_v1 as cap


class Checks(ParentChecks):
    def test_single_token_focal_and_preceding_against_independent_patch(self):
        r,d=request(),request(True)
        for ps in [[6],[5]]:
            v,s,_,n=cap.forward(self.model,r,0,positions=ps,observe=True,trajectory=True)
            _,ds,_,_=cap.forward(self.model,d,0,positions=ps,observe=True)
            own,_,proof,st=cap.forward(self.model,r,0,positions=ps,layer=1,replacement=s[1],trajectory=True)
            np.testing.assert_array_equal(own,v);cap.compare_trajectories(st,n,'self')
            actual,_,proof,t=cap.forward(self.model,r,0,positions=ps,layer=1,replacement=ds[1],trajectory=True)
            cap.structural_guard(t,n,1)
            self.assertEqual(proof['positions'],ps)
            def independent(module,args,out):
                result=out.clone();result[0,ps[0]]=self.torch.from_numpy(ds[1,0]);return result
            handle=self.model.model.layers[1].register_forward_hook(independent)
            try:expected,_,_,et=cap.bridge.forward(self.model,r,0,trajectory=True)
            finally:handle.remove()
            np.testing.assert_array_equal(actual,expected);cap.compare_trajectories(t,et,'repeat')
            self.assertFalse(self.model.model.layers[1]._forward_hooks)

    def test_unequal_prefix_lengths_with_mapped_focal_and_whole_query_donors(self):
        from copy import deepcopy
        r=request();d=deepcopy(request(True))
        d['input_ids']=[23,24,25]+d['input_ids'];d['prompt_tokens']+=3
        d['roles']={k:[i+3 for i in value] for k,value in d['roles'].items()}
        for ps in [[6],[5,6],[5,6,7,8],[5,8]]:
            dp=[i+3 for i in ps]
            self.assertEqual([r['input_ids'][i] for i in ps],[d['input_ids'][i] for i in dp])
            _,bank,_,_=cap.forward(self.model,d,0,positions=dp,observe=True)
            replacement=cap.donor_rows(bank,dp,1,dp)
            actual,_,proof,t=cap.forward(self.model,r,0,positions=ps,layer=1,replacement=replacement,trajectory=True)
            def independent(module,args,out):
                result=out.clone();result[0,ps]=self.torch.from_numpy(bank[1]);return result
            handle=self.model.model.layers[1].register_forward_hook(independent)
            try:expected,_,_,et=cap.bridge.forward(self.model,r,0,trajectory=True)
            finally:handle.remove()
            np.testing.assert_array_equal(actual,expected);cap.compare_trajectories(t,et,'repeat')
            self.assertEqual(proof['positions'],ps)
            with self.assertRaises((ValueError,KeyError)):cap.donor_rows(bank,dp,1,ps)
            for padding in ['left','right']:
                padded,_,p,_=cap.forward(self.model,r,0,positions=ps,layer=1,replacement=replacement,padding=padding)
                np.testing.assert_allclose(padded,actual,atol=2e-6,rtol=0)
                self.assertEqual(p['donor_sha256'],proof['donor_sha256'])
            _,prefix,_,_=cap.forward(self.model,d,0,positions=dp,observe=True,prefix_length=max(dp)+1)
            np.testing.assert_allclose(prefix,bank,atol=2e-6,rtol=0)
        self.assertFalse(self.torch.cuda.is_initialized())


    def test_disjoint_focal_complement_compose_exact_whole_replacement(self):
        r,d=request(),request(True);whole=r['roles']['query_all'];focal=[6,7];rest=[i for i in whole if i not in focal]
        _,bank,_,_=cap.forward(self.model,d,0,positions=whole,observe=True)
        actual,_,_,at=cap.forward(self.model,r,0,positions=whole,layer=1,replacement=bank[1],trajectory=True)
        def composed(module,args,out):
            z=out.clone()
            for group in [focal,rest]:z[0,group]=self.torch.from_numpy(bank[1,[whole.index(i) for i in group]])
            return z
        h=self.model.model.layers[1].register_forward_hook(composed)
        try:expected,_,_,et=cap.bridge.forward(self.model,r,0,trajectory=True)
        finally:h.remove()
        np.testing.assert_array_equal(actual,expected);cap.compare_trajectories(at,et,'repeat')
        self.assertFalse(self.model.model.layers[1]._forward_hooks)


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
