#!/usr/bin/env python3
"""Versioned CLI for the adopted nine-input experiment; never restarts old runs."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from diagnostics import hehe_sense_context_inputs_v1 as c

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','validate','seal','bind','run','_worker','check','analyze'])
    p.add_argument('--prepared',type=Path,default=c.PREPARED)
    p.add_argument('--output',type=Path);p.add_argument('--run',type=Path);p.add_argument('--bound',type=Path)
    p.add_argument('--gpu',type=int);p.add_argument('--authorization-note');p.add_argument('--phase',choices=['engineering','full'],default='engineering')
    p.add_argument('--resume',action='store_true');p.add_argument('--invocation');p.add_argument('--unsealed',action='store_true')
    a=p.parse_args()
    if a.command=='prepare': result=c.prepare(a.prepared)
    elif a.command=='seal': result=c.seal(a.prepared)
    elif a.command=='validate':
        plan,_,rows=c.validate(a.prepared,sealed=not a.unsealed);result={'status':'CPU_valid','inputs':len(rows),'budget':plan['budget']}
    else:
        from diagnostics import hehe_sense_context_runtime_v1 as rt
        if a.command=='bind':
            c.require(a.gpu is not None and a.output and a.authorization_note,'Explicit binding fields required')
            result=rt.bind(a.prepared,a.gpu,a.output,a.authorization_note)
        elif a.command in ('run','_worker'):
            c.require(a.bound and a.run,'Run binding required')
            if a.command=='_worker':
                c.require(a.invocation,'Worker invocation required');result=rt.worker(a.prepared,a.bound,a.run,a.invocation,a.phase)
            else:result=rt.supervise(a.prepared,a.bound,a.run,a.phase,a.resume)
        elif a.command=='check':result=rt.check_run(a.prepared,a.run)
        else:
            from diagnostics.hehe_sense_context_report_v1 import analyze
            result=analyze(a.prepared,a.run,a.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
