#!/usr/bin/env python3
"""Explicit CPU preparation; GPU entry requires a separate later execution decision."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import cross_term_mechanism_inputs_v1 as c

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','validate','seal','bind','run','_worker','check','analyze'])
    p.add_argument('--prepared',type=Path,default=c.PREPARED);p.add_argument('--output',type=Path);p.add_argument('--decision',type=Path)
    p.add_argument('--run',type=Path);p.add_argument('--bound',type=Path);p.add_argument('--gpu',type=int);p.add_argument('--authorization-note')
    p.add_argument('--phase',choices=['engineering','full'],default='engineering');p.add_argument('--resume',action='store_true')
    p.add_argument('--invocation');p.add_argument('--unsealed',action='store_true');a=p.parse_args()
    if a.command=='prepare':c.require(a.decision,'Accepted decisions required');result=c.prepare(a.prepared,a.decision)
    elif a.command=='seal':result=c.seal(a.prepared)
    elif a.command=='validate':
        plan,_,rows,jobs,selfs=c.validate(a.prepared,sealed=not a.unsealed)
        result={'status':'CPU_valid','prompts':len(rows),'cross':len(jobs),'self':len(selfs),'budget':plan['budget']}
    else:
        from diagnostics import cross_term_mechanism_runtime_v1 as rt
        if a.command=='bind':
            c.require(a.gpu is not None and a.output and a.authorization_note,'Explicit GPU binding required');result=rt.bind(a.prepared,a.gpu,a.output,a.authorization_note,a.decision)
        elif a.command in ['run','_worker']:
            c.require(a.bound and a.run,'Bound run required')
            if a.command=='_worker':c.require(a.invocation,'Registered invocation required');result=rt.worker(a.prepared,a.bound,a.run,a.invocation,a.phase)
            else:result=rt.supervise(a.prepared,a.bound,a.run,a.phase,a.resume)
        elif a.command=='check':c.require(a.run,'Run required');result=rt.check_run(a.prepared,a.run)
        else:
            from diagnostics.cross_term_mechanism_report_v1 import analyze
            c.require(a.run and a.output,'Analysis paths required');result=analyze(a.prepared,a.run,a.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
