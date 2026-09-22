#!/usr/bin/env python3
"""CPU audit or separately authorized new GPU execution; never launches by default."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import reference_routing_content_inputs_v1 as c
from diagnostics import reference_routing_content_runtime_v1 as rt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['validate','audit','launch','_worker'])
    p.add_argument('--prepared',type=Path,default=c.PREPARED)
    p.add_argument('--run',type=Path)
    p.add_argument('--stage',choices=['stage-a','development','confirmation'],default='stage-a')
    p.add_argument('--decision',type=Path)
    p.add_argument('--calibration',type=Path)
    a=p.parse_args()
    if a.command=='validate':
        plan,_=c.validate(a.prepared);print(json.dumps({'status':'PASS','GPU_launched':False,'budgets':plan['budgets']},indent=2))
    elif a.command=='launch':print(json.dumps(rt.launch(a.prepared,a.decision,a.run,a.stage,a.calibration),indent=2))
    elif a.command=='_worker':rt.worker(a.prepared,a.run,a.stage)
    else:print(json.dumps(rt.audit_run(a.prepared,a.run,a.stage,a.calibration),indent=2))


if __name__=='__main__':main()
