#!/usr/bin/env python3
"""Tiny native Qwen3/GQA: independently check both intervention boundaries."""
import argparse,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from diagnostics import hehe_joint_restore_capture_v1 as cap
from test_hehe_bridge_capture_v1 import Checks as BaseChecks,request


class Checks(BaseChecks):
    def banks(self):
        r=request();d=request(True);ps=[5,6]
        v,s,_,n=cap.forward(self.model,r,0,positions=ps,observe=True,trajectory=True)
        _,ds,_,_=cap.forward(self.model,d,0,positions=ps,observe=True)
        uv,_,_,u=cap.forward(self.model,r,0,positions=ps,layer=1,replacement=ds[1],trajectory=True)
        return r,ps,v,s,n,ds,uv,u

    def test_native_and_conditional_self(self):
        r,ps,v,s,n,ds,uv,u=self.banks()
        for li,branch,bi in [(2,'attention',0),(3,'mlp',1)]:
            for bank,source,expected,t in [(s,n,v,n),(ds,u,uv,u)]:
                spec={'layer':li,'branch':branch,'position':9,'source':'recipient_native','value':source['branches'][li,bi]}
                vv,_,proof,tt=cap.forward(self.model,r,0,positions=ps,layer=1,replacement=bank[1],restoration=[spec],trajectory=True)
                np.testing.assert_array_equal(vv,expected);cap.compare_trajectories(tt,t,'self')
                self.assertEqual(proof['restoration'][0]['changed_rows_in_unpadded_coordinates'],[])

    def test_both_branches_against_independent_hooks(self):
        r,ps,v,s,n,ds,uv,u=self.banks();torch=self.torch
        for li,branch,bi in [(2,'attention',0),(3,'mlp',1)]:
            value=n['branches'][li,bi]
            spec={'layer':li,'branch':branch,'position':9,'source':'recipient_native','value':value}
            vv,_,p,t=cap.forward(self.model,r,0,positions=ps,layer=1,replacement=ds[1],restoration=[spec],trajectory=True)
            cap.validate_trajectory(t,r,self.profile,vv);cap.structural_guard(t,n,1)
            cap.restoration_guard(t,u,n,spec)
            self.assertGreater(p['restoration'][0]['difference_l2'],0)
            raw,_,_,missing=cap.forward(self.model,r,0,positions=ps,layer=1,replacement=ds[1],restoration=[spec])
            np.testing.assert_array_equal(raw,vv);self.assertIsNone(missing)
            def upstream(m,a,o):
                q=o.clone();q[0,ps]=torch.from_numpy(ds[1]);return q
            def restore(m,a,o):
                original=o[0] if branch=='attention' else o
                q=original.clone();q[0,9]=torch.from_numpy(value)
                return (q,*o[1:]) if branch=='attention' else q
            block=self.model.model.layers[li]
            hh=[self.model.model.layers[1].register_forward_hook(upstream),
                (block.self_attn if branch=='attention' else block.mlp).register_forward_hook(restore)]
            try:expected,_,_,et=cap.bridge.forward(self.model,r,0,trajectory=True)
            finally:
                for h in reversed(hh):h.remove()
            np.testing.assert_array_equal(vv,expected);cap.compare_trajectories(t,et,'repeat')

    def test_padding_and_continuation_keep_original_coordinate(self):
        r,ps,v,s,n,ds,uv,u=self.banks()
        for li,branch,bi in [(2,'attention',0),(3,'mlp',1)]:
            spec={'layer':li,'branch':branch,'position':9,'source':'recipient_native','value':n['branches'][li,bi]}
            for tokens in [(),(3,)]:
                ref,_,pr,_=cap.forward(self.model,r,0,positions=ps,layer=1,replacement=ds[1],restoration=[spec],append_tokens=tokens)
                self.assertEqual(pr['restoration'][0]['position'],9)
                for padding in ['left','right']:
                    vv,_,pp,_=cap.forward(self.model,r,0,positions=ps,layer=1,replacement=ds[1],restoration=[spec],padding=padding,append_tokens=tokens)
                    np.testing.assert_allclose(vv,ref,atol=2e-6,rtol=0)
                    self.assertEqual(pp['restoration'][0]['replacement_sha256'],pr['restoration'][0]['replacement_sha256'])

    def test_restoration_rejection_and_cleanup(self):
        r,ps,v,s,n,ds,uv,u=self.banks()
        spec={'layer':2,'branch':'attention','position':9,'source':'recipient_native','value':n['branches'][2,0]}
        with patch.object(cap.bridge,'forward',side_effect=RuntimeError('deliberate')):
            with self.assertRaises(RuntimeError):cap.forward(self.model,r,0,positions=ps,layer=1,replacement=ds[1],restoration=[spec])
        self.assertFalse(self.model.model.layers[2].self_attn._forward_hooks)
        with self.assertRaisesRegex(ValueError,'registered pre-answer'):
            cap.forward(self.model,r,0,positions=ps,layer=1,replacement=ds[1],restoration=[dict(spec,position=8)])
        bad={k:x.copy() for k,x in u.items()};bad['states'][2,0,0]+=1
        with self.assertRaisesRegex(ValueError,'earlier same-layer'):cap.restoration_guard(bad,u,n,spec)
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_joint_actual_two_hooks_and_conditional_boundary(self):
        r,ps,v,s,n,ds,uv,u=self.banks();torch=self.torch
        def specs(source):return [{'layer':li,'branch':branch,'position':9,'source':'recipient_native','value':source['branches'][li,bi]} for li,branch,bi in [(2,'attention',0),(3,'mlp',1)]]
        ss=specs(n)
        av,_,_,a=cap.forward(self.model,r,0,positions=ps,layer=1,replacement=ds[1],restoration=ss[:1],trajectory=True)
        vv,_,p,t=cap.forward(self.model,r,0,positions=ps,layer=1,replacement=ds[1],restoration=ss,trajectory=True)
        cap.validate_trajectory(t,r,self.profile,vv);cap.structural_guard(t,n,1)
        cap.restoration_guard(t,u,n,ss[0]);cap.restoration_guard(t,a,n,ss[1])
        self.assertEqual(p['restoration'][1]['before_sha256'],cap.array_sha(a['branches'][3,1]))
        self.assertFalse(np.array_equal(a['branches'][3,1],u['branches'][3,1]))
        with self.assertRaisesRegex(ValueError,'earlier layers'):cap.restoration_guard(t,u,n,ss[1])
        def upstream(m,args,out):
            q=out.clone();q[0,ps]=torch.from_numpy(ds[1]);return q
        def attention(m,args,out):
            q=out[0].clone();q[0,9]=torch.from_numpy(n['branches'][2,0]);return (q,*out[1:])
        def mlp(m,args,out):
            q=out.clone();q[0,9]=torch.from_numpy(n['branches'][3,1]);return q
        hh=[self.model.model.layers[1].register_forward_hook(upstream),self.model.model.layers[2].self_attn.register_forward_hook(attention),self.model.model.layers[3].mlp.register_forward_hook(mlp)]
        try:expected,_,_,et=cap.bridge.forward(self.model,r,0,trajectory=True)
        finally:
            for h in reversed(hh):h.remove()
        np.testing.assert_array_equal(vv,expected);cap.compare_trajectories(t,et,'repeat')
        for bank,source,expected,tt in [(s,n,v,n),(ds,u,uv,u)]:
            x,_,proof,xt=cap.forward(self.model,r,0,positions=ps,layer=1,replacement=bank[1],restoration=specs(source),trajectory=True)
            np.testing.assert_array_equal(x,expected);cap.compare_trajectories(xt,tt,'self')
            self.assertTrue(all(pp['changed_rows_in_unpadded_coordinates']==[] for pp in proof['restoration']))
        for tokens in [(),(3,)]:
            ref,_,_,_=cap.forward(self.model,r,0,positions=ps,layer=1,replacement=ds[1],restoration=ss,append_tokens=tokens)
            for padding in ['none','left','right']:
                x,_,proof,_=cap.forward(self.model,r,0,positions=ps,layer=1,replacement=ds[1],restoration=ss,padding=padding,append_tokens=tokens)
                np.testing.assert_allclose(x,ref,atol=2e-6,rtol=0)
                self.assertEqual([pp['position'] for pp in proof['restoration']],[9,9])
        for bad in [ss[::-1],[ss[0],ss[0]],[ss[0],dict(ss[1],position=8)]]:
            with self.assertRaises(ValueError):cap.forward(self.model,r,0,positions=ps,layer=1,replacement=ds[1],restoration=bad)
        with patch.object(cap.bridge,'forward',side_effect=RuntimeError('deliberate')):
            with self.assertRaises(RuntimeError):cap.forward(self.model,r,0,positions=ps,layer=1,replacement=ds[1],restoration=ss)
        self.assertFalse(self.model.model.layers[2].self_attn._forward_hooks)
        self.assertFalse(self.model.model.layers[3].mlp._forward_hooks)


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
