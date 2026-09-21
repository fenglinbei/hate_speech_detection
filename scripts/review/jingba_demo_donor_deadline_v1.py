#!/usr/bin/env python3
"""Bounded worker wait with exact PID ownership and inherited tested deadline policy."""
import argparse,importlib.util,json,os,signal,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'scripts/review/run_hehe_sense_context_window_v1.py'
spec=importlib.util.spec_from_file_location('inherited_owned_window',source);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.CLI=ROOT/'scripts/review/run_jingba_demo_donor_v1.py'


def wait_owned(proc,run,bound,deadline,phase):
    folder=Path(run)/('deadline-guard-'+phase);folder.mkdir(exist_ok=False)
    window=base.Window({'run':str(run),'bound':str(bound),'user_window_end_unix':float(deadline)},folder)
    controller=base.identity(os.getpid());controller['kind']='controller'
    # The actual child returned by Popen must match this run/CLI; do not infer ownership from a GPU index.
    owner=base.identity(proc.pid)
    if owner is not None:
        cmd=owner['command']
        assert owner['ppid']==os.getpid() and cmd[1:3]==[str(base.CLI),'_worker']
        assert cmd[cmd.index('--run')+1]==str(run) and cmd[cmd.index('--bound')+1]==str(bound)
        owner['kind']='worker';window.owners.append(owner);window.event('owned_worker_registered',owner=owner)
    window.save()
    try:
        while proc.poll() is None:
            # Window.tick manages workers; leave the live controller to reap and record release.
            window.tick();time.sleep(.2)
    except BaseException:
        window.cancel(signal.SIGTERM,None)
        while proc.poll() is None:window.tick();time.sleep(.2)
        raise
    finally:
        window.status.update(status='finished',worker_exit_code=proc.poll(),ended_at_unix=time.time(),
                             owned_processes_absent=all(not base.same_process(o) for o in window.owners))
        window.save()
    assert window.status['owned_processes_absent']
    return proc.wait()


def tests():
    base.self_test()
    with tempfile.TemporaryDirectory(prefix='regions-deadline-test-') as td:
        root=Path(td);run=root/'run';run.mkdir();bound=root/'bound.json';bound.write_text('{}')
        cli=root/'cpu_worker.py';ready=root/'ready'
        cli.write_text('import signal,time,pathlib\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\npathlib.Path('+repr(str(ready))+').write_text("ready")\ntime.sleep(30)\n')
        old=base.CLI;base.CLI=cli
        other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
        try:
            proc=subprocess.Popen([sys.executable,str(cli),'_worker','--run',str(run),'--bound',str(bound)])
            until=time.time()+3
            while not ready.exists() and proc.poll() is None and time.time()<until:time.sleep(.01)
            assert ready.exists(), 'CPU child did not install its signal handler'
            code=wait_owned(proc,run,bound,time.time()+19,'test')
            assert code==-signal.SIGKILL and not Path('/proc/'+str(proc.pid)).exists()
            assert other.poll() is None and (run/'STOP').exists()
            receipt=base.read(run/'deadline-guard-test/window-state.json');assert receipt['owned_processes_absent']
        finally:
            base.CLI=old;other.terminate();other.wait()
    return {'status':'pass','tests':5,'CUDA_initialized':False,'real_CPU_processes':True,'policy':'STOP -120s; owned TERM -60s; owned KILL -20s; controller reaps','stale_PID_rejected':True,'unrelated_survives':True}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=tests()
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
