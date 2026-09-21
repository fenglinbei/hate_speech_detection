#!/usr/bin/env python3
"""One future authorized GPU run; this file never schedules or launches on import."""
from datetime import datetime,timezone
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import jingba_query_regions_inputs_v1 as c
from diagnostics import jingba_query_regions_runtime_v1 as rt


def launch(prepared,decision,here,run):
    prepared,decision,here,run=map(Path,(prepared,decision,here,run))
    # Refuse before creating state, querying hardware or waiting.
    authorization=rt.execution_decision(prepared,decision)
    c.validate(prepared)
    c.require(not here.exists() and not run.exists(),'Never duplicate or restart a launch/run')
    here.mkdir(parents=True)
    bound=here/'bound-01.json'
    c.write(here/'authorization.json',dict(authorization,source=c.info(decision)))
    c.write(here/'source-pin.json',{'launcher':c.info(Path(__file__)),'prepared_manifest':c.info(prepared/'manifest.json'),
                                  'decision':c.info(decision)})
    selectors=sorted((ROOT/'docs/research/experiment-plans').glob('*/results-current.json'))
    selectors+=[ROOT/'deploy/case_attention/digitalocean-sgp/current.json']
    c.write(here/'parent-selectors.json',{'files':[c.info(p) for p in selectors if p.exists()]})
    def event(kind,**fields):
        value={'event':kind,'at_UTC':datetime.now(timezone.utc).isoformat(),'controller_pid':os.getpid(),**fields}
        print(json.dumps(value,ensure_ascii=False),flush=True);c.atomic(here/'state.json',value,replace=True)
    try:
        event('waiting_for_idle');i=1
        while True:
            if (here/'CANCEL').exists():event('cancelled_before_binding');return
            rt.execution_decision(prepared,decision)
            inv=rt.gpu_inventory();c.write(here/f'inventory-{i:03d}.json',dict(inv,checked_at_unix=time.time()));i+=1
            idle=[d for d in inv['devices'] if d['index'] in authorization.get('allowed_gpu_indices',[0,1,2,3])
                  and d['total_mib']>=44000 and d['used_mib']==0 and d['utilization']==0
                  and not any(d['uuid'] in p for p in inv['compute_processes'])]
            if idle:device=min(idle,key=lambda d:d['index']);break
            event('waiting_for_idle',inventory_count=i-1);time.sleep(45)
        rt.bind(prepared,device['index'],bound,authorization['user_message'],decision)
        event('engineering_started',binding=c.info(bound))
        rt.supervise(prepared,bound,run,phase='engineering')
        checked=rt.check_run(prepared,run);c.write(here/'engineering-check.json',checked)
        c.require(checked['status']=='qualified','Engineering did not qualify')
        event('engineering_qualified_and_released')
        if (here/'CANCEL').exists():event('cancelled_before_production');return
        rt.supervise(prepared,bound,run,phase='full')
        checked=rt.check_run(prepared,run);c.write(here/'complete-check.json',checked)
        c.require(checked['status']=='complete','Production not complete')
        event('complete_GPU_released',resource_release=c.read(run/'state.json')['resource_release'],raw_seal=c.info(run/'raw-seal.json'))
    except BaseException as exc:
        event('failed_no_automatic_retry',error=repr(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,default=c.PREPARED)
    p.add_argument('--decision',type=Path,required=True,help='Separate explicit future authorization bound to this preparation manifest')
    p.add_argument('--launch-dir',type=Path,default=c.WORK/'launch-01');p.add_argument('--run',type=Path,default=c.WORK/'run-01')
    a=p.parse_args();launch(a.prepared,a.decision,a.launch_dir,a.run)

if __name__=='__main__':main()
