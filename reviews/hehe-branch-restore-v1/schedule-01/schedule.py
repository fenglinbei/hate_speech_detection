#!/usr/bin/env python3
"""One-shot local timer, then the authorized frozen pipeline; no retries."""
from __future__ import annotations
from datetime import datetime,timezone
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from diagnostics import hehe_branch_restore_inputs_v1 as c
HERE=Path(__file__).resolve().parent
LAUNCH=c.WORK/'launch-02'
RUN=c.WORK/'run-01'
PYTHON=ROOT/'.conda/stage1-p0/bin/python'
stopping=False
claimed=False


def wait_until(target,stopped,now=time.time,sleep=time.sleep):
    while True:
        if stopped():return False
        remaining=target-now()
        if remaining<=0:return True
        sleep(min(30.,remaining))


def pipeline_commands():
    scripts=ROOT/'scripts/review';common=['--prepared',str(c.PREPARED),'--run',str(RUN)]
    return [
        ('GPU', [str(PYTHON),str(LAUNCH/'launch.py')]),
        ('analysis',[str(PYTHON),str(scripts/'run_hehe_branch_restore_v1.py'),'analyze',*common,'--output',str(c.WORK/'results-01')]),
        ('independent_audit',[str(PYTHON),str(scripts/'audit_hehe_branch_restore_results_v1.py'),*common,'--results',str(c.WORK/'results-01'),'--output',str(c.WORK/'result-audit-01.json')]),
        ('readable_report',[str(PYTHON),str(scripts/'report_hehe_branch_restore_v1.py'),'--output',str(c.WORK/'report-01')]),
        ('closeout',[str(PYTHON),str(scripts/'closeout_hehe_branch_restore_v1.py')])]


def run_pipeline(stage,stopped):
    for name,cmd in pipeline_commands():
        if stopped():return False
        stage(name,cmd)
    return True


def identity(pid):
    proc=Path(f'/proc/{pid}')
    return {'pid':pid,'start_ticks':int((proc/'stat').read_text().rsplit(') ',1)[1].split()[19]),
        'argv':(proc/'cmdline').read_bytes().split(b'\0')[:-1],
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def event(status,**extra):
    value={'status':status,'at_UTC':datetime.now(timezone.utc).isoformat(),**extra}
    c.atomic(HERE/'state.json',value,replace=True)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def cancelled():return stopping or (HERE/'CANCEL').exists()


def request_safe_stop():
    # Only this task's cancellation files. Never signal an unrelated process.
    if not RUN.exists():(LAUNCH/'CANCEL').touch()
    else:
        state=c.read(RUN/'state.json')
        if state['status'] not in ['complete','failed']:(RUN/'STOP').touch()


def stage(name,cmd):
    env=dict(os.environ,USE_TORCH='0',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
        MPLCONFIGDIR='/tmp/hehe-branch-restore-scheduled-mpl')
    with (HERE/(name+'.log')).open('xb',buffering=0) as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record=identity(proc.pid);record['argv']=[x.decode() for x in record['argv']]
        c.write(HERE/(name+'-process.json'),dict(record,command=cmd,started_at_unix=time.time()))
        event('running_'+name,child_pid=proc.pid)
        while True:
            if cancelled() and name=='GPU':request_safe_stop()
            try:code=proc.wait(timeout=15);break
            except subprocess.TimeoutExpired:continue
    c.write(HERE/(name+'-exit.json'),{'pid':proc.pid,'returncode':code,
        'absent_after_reap':not Path(f'/proc/{proc.pid}').exists(),'ended_at_unix':time.time()})
    c.require(code==0,f'{name} failed; stop without automatic retry. See {name}.log')


def main():
    global stopping,claimed
    def stop(signum,frame):
        global stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (HERE/'schedule.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c.require(not (HERE/'owner.json').exists(),'One-shot timer already started; do not restart')
        for source in c.read(HERE/'source-pins.json')['files']:c.verify(source)
        c.validate(c.PREPARED)
        config=c.read(HERE/'config.json');target=datetime.fromisoformat(config['not_before']).timestamp()
        record=identity(os.getpid());record['argv']=[x.decode() for x in record['argv']]
        c.write(HERE/'owner.json',dict(record,started_at_unix=time.time(),not_before=config['not_before']))
        claimed=True
        c.require(not RUN.exists() and not (LAUNCH/'bound-01.json').exists(),'Unexpected preexisting scientific run')
        event('scheduled_waiting',not_before=config['not_before'],pid=os.getpid(),GPU_touched=False)
        if not wait_until(target,cancelled):event('cancelled_before_due',GPU_touched=False);return
        c.require(time.time()>=target,'Early execution guard')
        for source in c.read(HERE/'source-pins.json')['files']:c.verify(source)
        c.require(not RUN.exists() and not (LAUNCH/'bound-01.json').exists(),'Another run started; refuse duplication')
        event('due_waiting_for_idle_GPU',not_before=config['not_before'])
        if not run_pipeline(stage,cancelled):event('cancelled_between_stages');return
        result=c.read(c.PUBLIC/'results-current.json')
        event('complete',report=result['report'],closeout=result['closeout_manifest'],terminal_do_not_restart=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if claimed:event('cancelled_or_paused' if cancelled() else 'failed_no_automatic_retry',error=repr(exc))
        else:print(json.dumps({'not_started':True,'error':repr(exc)}),file=sys.stderr,flush=True)
        raise
