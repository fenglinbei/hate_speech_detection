#!/usr/bin/env python3
"""User-authorized eight-input fixed-layer transfer run; one freshly idle GPU, no new deadline."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from diagnostics import hehe_transfer_inputs_v1 as c
from diagnostics import hehe_transfer_runtime_v1 as rt

HERE=Path(__file__).resolve().parent
RUN=c.WORK/'run-01'
BOUND=HERE/'bound-01.json'
def event(kind,**fields):
    value={'event':kind,'at_UTC':datetime.now(timezone.utc).isoformat(),**fields}
    print(json.dumps(value,ensure_ascii=False),flush=True)
    c.atomic(HERE/'state.json',value,replace=True)

def main():
    c.require(not RUN.exists() and not BOUND.exists(),'Never restart or overwrite a run/binding')
    c.validate(c.PREPARED)
    c.verify(c.read(HERE/'source-pin.json')['launcher'])
    c.require(c.read(HERE/'authorization.json')['deadline'] is None,'Unexpected deadline; use an explicitly bounded launcher')
    event('waiting_for_idle',controller_pid=os.getpid())
    i=1
    while True:
        if (HERE/'CANCEL').exists():
            event('cancelled_before_binding');return
        inv=rt.gpu_inventory()
        c.write(HERE/f'inventory-{i:03d}.json',dict(inv,checked_at_unix=time.time()));i+=1
        idle=[d for d in inv['devices'] if d['total_mib']>=44000 and d['used_mib']==0 and d['utilization']==0
              and not any(d['uuid'] in p for p in inv['compute_processes'])]
        if idle:
            device=min(idle,key=lambda d:d['index']);break
        event('waiting_for_idle',controller_pid=os.getpid(),occupied_devices=[d['index'] for d in inv['devices']],inventory_count=i-1)
        time.sleep(45)
    event('binding',gpu=device['index'],uuid=device['uuid'])
    rt.bind(c.PREPARED,device['index'],BOUND,'User explicitly said 可以接入并启动运行 after adopting T01-T04, with T03 inheriting its original label. Eight adopted prompts; layer17 focal/preceding, layer26 attention/layer28 MLP single/joint restoration. Prior four-GPU permission persists; use one freshly idle L20, no new cutoff. See launch-01/authorization.json.')
    c.atomic(c.PUBLIC/'execution-01.json',{'status':'authorized','prepared_manifest':c.info(c.PREPARED/'manifest.json'),
        'binding':c.info(BOUND),'authorization':c.info(HERE/'authorization.json'),'launcher':c.info(Path(__file__)),
        'run':str(RUN.relative_to(ROOT)),'authorized_new_case_transfer_run':True})
    event('engineering_started',gpu=device['index'],uuid=device['uuid'],controller_pid=os.getpid())
    rt.supervise(c.PREPARED,BOUND,RUN,phase='engineering')
    check=rt.check_run(c.PREPARED,RUN)
    c.write(HERE/'engineering-check.json',check)
    c.require(check['status']=='qualified','Engineering did not qualify')
    event('engineering_qualified_and_released',qualification=c.info(RUN/'qualification.json'))
    if (HERE/'CANCEL').exists():
        event('cancelled_before_production');return
    rt.supervise(c.PREPARED,BOUND,RUN,phase='full')
    check=rt.check_run(c.PREPARED,RUN)
    c.write(HERE/'complete-check.json',check)
    c.require(check['status']=='complete','Production not complete')
    event('complete_GPU_released',resource_release=c.read(RUN/'state.json')['resource_release'],raw_seal=c.info(RUN/'raw-seal.json'))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        event('failed_no_automatic_retry',error=repr(exc));raise
