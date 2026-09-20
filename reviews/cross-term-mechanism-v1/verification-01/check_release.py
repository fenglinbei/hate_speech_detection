"""Read-only host verification of this run's normal release; never sends signals."""
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import cross_term_mechanism_inputs_v1 as c
from diagnostics.cross_model_applicability_execution_v1 import gpu_inventory

work=c.WORK;state=c.read(work/'run-01/state.json');launch=c.read(work/'launch-01/state.json')
c.require(state['status']=='complete' and state['owned_worker_absent'] and state['worker_exit_code']==0,'Normal final release required')
c.require(launch['event']=='complete_GPU_released','Controller must finish first')
for r in state['invocations']:c.require('worker_pid' in r and r['ended_at_unix']>=r['started_at_unix'],'Incomplete phase')
pids=sorted({r['controller_pid'] for r in state['invocations']}|{r['worker_pid'] for r in state['invocations']})
processes=[{'pid':pid,'proc_exists':Path(f'/proc/{pid}').exists()} for pid in pids]
c.require(not any(r['proc_exists'] for r in processes),'Owned process remains in host namespace')
inv=gpu_inventory()
c.require(not any(any(row.split(',')[-1].strip()==str(pid) for row in inv['compute_processes']) for pid in pids),'Owned GPU process remains')
for row in c.read(work/'launch-01/parent-selectors.json')['files']:c.verify(row)
c.verify(state['resource_release'])
result={'status':'pass','checked_at_unix':time.time(),'host_pid_context':True,'processes':processes,
        'GPU_inventory':inv,'all_owned_processes_absent':True,'signals_sent':False,'worker_exit_code':0,
        'controller_exit_code':0,'controller_exec_session':65692,
        'initial_ownership':c.info(work/'launch-01/ownership-01.json'),'source':c.info(Path(__file__)),
        'run_state':c.info(work/'run-01/state.json'),'final_release':state['resource_release'],
        'old_scientific_and_website_selectors_unchanged':True}
c.write(work/'process-release-check.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
