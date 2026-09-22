#!/usr/bin/env python3
"""Independent full-vector Fraction/Decimal120/longdouble numerical audit."""
import argparse
from decimal import Decimal,localcontext
from fractions import Fraction
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import reference_routing_content_inputs_v1 as c
from diagnostics import reference_routing_content_runtime_v1 as rt


def independent(run,profile):
    seal=c.read(Path(run)/'raw-seal.json');vectors=margins=0;maxprob=maxlog=0.
    for item in seal['records']:
        c.verify(item);r=c.read(item['path']);c.verify(r['vector'])
        v=np.load(r['vector']['path'],allow_pickle=False)
        c.require(v.dtype==np.float32 and v.shape==(profile['vocab_size'],) and np.isfinite(v).all(),'Invalid stored vector')
        vectors+=1
        if r['prefix']:continue
        yes=float(v[profile['candidate_tokens']['有']]);no=float(v[profile['candidate_tokens']['无']])
        exact=Fraction(no)-Fraction(yes)
        c.require(Fraction(r['readout']['m'])==exact,'Margin differs from exact FP32 subtraction')
        margins+=1
        with localcontext() as ctx:
            ctx.prec=120;d=Decimal(no)-Decimal(yes)
            prob=1/(1+(-d).exp())
        # Full-vocabulary normalization independently in extended precision.
        x=v.astype(np.longdouble);top=x.max();logz=top+np.log(np.exp(x-top).sum(dtype=np.longdouble))
        ro=r['readout']
        err=abs(float(prob)-ro['pair_support_no']);maxprob=max(maxprob,err);c.require(err<1e-12,'Pair probability mismatch')
        pairtop=np.longdouble(max(no,yes));pairz=pairtop+np.log(np.exp(np.longdouble(no)-pairtop)+np.exp(np.longdouble(yes)-pairtop))
        for key,expected in [('log_p_no',np.longdouble(no)-logz),('log_p_yes',np.longdouble(yes)-logz),('log_legal_mass',pairz-logz)]:
            err=abs(float(expected)-ro[key]);maxlog=max(maxlog,err);c.require(err<1e-10,'Full-vocabulary normalizer mismatch')
        c.require(abs(float(np.exp(pairz-logz))-ro['legal_mass'])<1e-12,'Legal mass mismatch')
    return {'vectors':vectors,'exact_Fraction_margins':margins,'Decimal_precision':120,
            'longdouble_mantissa_bits':np.finfo(np.longdouble).nmant,'maximum_pair_probability_error':maxprob,
            'maximum_log_probability_error':maxlog}


def audit(prepared,run,stage,calibration=None):
    state=c.read(Path(run)/'state.json')
    c.require(state['status']=='COMPLETE' and state['owned_worker_absent'] and state['worker_exit_code']==0,'Independent audit requires normal worker release')
    c.verify(state['release']);release=c.read(state['release']['path'])
    c.require(release['owned_worker_absent'] and release['worker_exit_code']==0,'Resource release proof failed')
    reconstruction=rt.audit_run(prepared,run,stage,calibration)
    _,profile=c.validate(prepared)
    return dict(reconstruction,numerical=independent(run,profile),independent_from_runtime_readout=True,
                development_eligible_for_calibration=stage=='development',release=state['release'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,default=c.PREPARED)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['stage-a','development','confirmation'],required=True)
    p.add_argument('--calibration',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=audit(a.prepared,a.run,a.stage,a.calibration);c.write(a.output,r);print(json.dumps(r,indent=2))
