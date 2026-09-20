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
