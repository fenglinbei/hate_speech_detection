#!/usr/bin/env python3
"""One explicitly authorized, bounded GPU window around the frozen case runner.

This wrapper changes no model, scientific input, or numerical acceptance rule.
STOP is requested 120 s before the absolute limit; only verified owned processes
may receive TERM (60 s) or emergency KILL (20 s). Controllers remain to reap them.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / 'scripts/review/run_case_content_replacement_v1.py'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def identity(pid):
    try:
        base = Path('/proc') / str(pid)
        fields = (base / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'ppid': int(fields[1]), 'start_ticks': int(fields[19]),
                'command': (base / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(owner):
    now = identity(owner['pid'])
    return now is not None and (now['start_ticks'], now['command']) == (owner['start_ticks'], owner['command'])


def send_owned(owner, sig):
    if not same_process(owner):
        return False
    try:
        # Prefer pidfd when available. The host's older kernel may return ENOSYS;
        # then recheck exact start ticks and command immediately before os.kill.
        libc = ctypes.CDLL(None, use_errno=True)
        if hasattr(os, 'pidfd_open'):
            fd = os.pidfd_open(owner['pid'])
        else:
            if os.uname().machine != 'x86_64':
                raise RuntimeError('This runtime needs a reviewed pidfd syscall mapping')
            fd = libc.syscall(434, ctypes.c_int(owner['pid']), ctypes.c_uint(0))
            if fd < 0:
                error = ctypes.get_errno()
                if error == errno.ENOSYS:
                    if not same_process(owner):
                        return False
                    os.kill(owner['pid'], sig)
                    return True
                raise OSError(error, os.strerror(error))
        try:
            if not same_process(owner):
                return False
            if hasattr(signal, 'pidfd_send_signal'):
                signal.pidfd_send_signal(fd, sig)
            else:
                result = libc.syscall(424, ctypes.c_int(fd), ctypes.c_int(sig), ctypes.c_void_p(), ctypes.c_uint(0))
                if result < 0:
                    error = ctypes.get_errno()
                    raise OSError(error, os.strerror(error))
        finally:
            os.close(fd)
        return True
    except OSError as error:
        if error.errno == errno.ESRCH:
            return False
        raise


class Window:
    def __init__(self, config, directory):
        self.config, self.directory = config, Path(directory)
        self.run = Path(config['run'])
        self.deadline = float(config['user_window_end_unix'])
        self.owners, self.sent = [], set()
        self.cancelled = False
        self.status = {'status': 'starting', 'pid': os.getpid(), 'started_at_unix': time.time(),
                       'user_window_end_unix': self.deadline, 'commands': []}

    def event(self, kind, **values):
        with (self.directory / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(at_unix=time.time(), event=kind, **values), ensure_ascii=False) + '\n')

    def save(self):
        write(self.directory / 'window-state.json', self.status)
        write(self.directory / 'owned-processes.json', self.owners)

    def cancel(self, signum, frame):
        self.cancelled = True
        self.deadline = min(self.deadline, time.time() + 90)

    def discover(self, controller):
        if not same_process(controller):
            return
        path = Path(f'/proc/{controller["pid"]}/task/{controller["pid"]}/children')
        try:
            children = path.read_text().split()
        except FileNotFoundError:
            return
        for pid in children:
            owner = identity(int(pid))
            if not owner or owner['ppid'] != controller['pid']:
                continue
            cmd = owner['command']
            if len(cmd) < 3 or cmd[1:3] != [str(CLI), '_worker']:
                continue
            if '--run' not in cmd or cmd[cmd.index('--run') + 1] != str(self.run):
                continue
            if '--bound' not in cmd or cmd[cmd.index('--bound') + 1] != self.config['bound']:
                continue
            if not any(x['pid'] == owner['pid'] and x['start_ticks'] == owner['start_ticks'] for x in self.owners):
                owner['kind'] = 'worker'
                self.owners.append(owner)
                self.event('owned_worker_registered', owner=owner)
                self.save()

    def tick(self, controller=None):
        if controller:
            self.discover(controller)
        now = time.time()
        if now >= self.deadline - 120 or self.cancelled:
            if self.run.exists() and not (self.run / 'STOP').exists():
                (self.run / 'STOP').write_text('Authorized window ending; pause at committed request boundary.\n', encoding='utf-8')
                self.event('STOP_written')
        for owner in self.owners:
            if owner['kind'] != 'worker':
                continue
            for before, sig in ((60, signal.SIGTERM), (20, signal.SIGKILL)):
                key = (owner['pid'], owner['start_ticks'], int(sig))
                if now >= self.deadline - before and key not in self.sent:
                    self.sent.add(key)
                    if send_owned(owner, sig):
                        self.event('owned_worker_signal', owner=owner, signal=int(sig), emergency=sig == signal.SIGKILL)
        if controller and now >= self.deadline - 10:
            if not any(x['kind'] == 'worker' and same_process(x) for x in self.owners):
                if send_owned(controller, signal.SIGTERM):
                    self.event('owned_controller_signal', owner=controller, signal=int(signal.SIGTERM))
        if controller and now >= self.deadline - 5:
            if not any(x['kind'] == 'worker' and same_process(x) for x in self.owners):
                if send_owned(controller, signal.SIGKILL):
                    self.event('owned_controller_signal', owner=controller, signal=int(signal.SIGKILL))

    def command(self, name, arguments, gpu=False):
        if gpu and (time.time() >= self.deadline - 180 or self.cancelled):
            raise RuntimeError('Window too short for another GPU phase; no launch')
        command = [sys.executable, str(CLI)] + arguments
        self.status.update(status=name)
        with (self.directory / (name + '.log')).open('ab', buffering=0) as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, PYTHONUTF8='1'),
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            owner = identity(proc.pid)
            if owner is None:
                raise RuntimeError('Child exited before ownership registration')
            owner['kind'] = 'controller' if gpu else 'cpu_check'
            self.owners.append(owner)
            receipt = {'name': name, 'command': command, 'pid': proc.pid, 'started_at_unix': time.time()}
            self.status['commands'].append(receipt)
            self.event('command_started', **receipt)
            self.save()
            try:
                while proc.poll() is None:
                    self.tick(owner if gpu else None)
                    time.sleep(0.5)
            except BaseException:
                self.cancel(signal.SIGTERM, None)
                while proc.poll() is None:
                    self.tick(owner if gpu else None)
                    time.sleep(0.5)
                raise
            receipt.update(exit_code=proc.returncode, ended_at_unix=time.time())
            self.save()
            self.event('command_ended', **receipt)
            if proc.returncode != 0:
                raise RuntimeError(f'{name} failed; no retry; inspect {name}.log')

    def execute(self):
        args = ['--prepared', self.config['prepared'], '--run', str(self.run)]
        try:
            self.command('engineering', ['run', '--bound', self.config['bound']] + args, gpu=True)
            state = read(self.run / 'state.json')
            if state['status'] == 'paused':
                self.status['status'] = 'paused'
                return
            self.command('engineering-check', ['check'] + args)
            self.command('full', ['run', '--bound', self.config['bound'], '--phase', 'full'] + args, gpu=True)
            state = read(self.run / 'state.json')
            if state['status'] == 'paused':
                self.status['status'] = 'paused'
                return
            self.command('final-check', ['check'] + args)
            self.status['status'] = 'complete'
        except Exception as exc:
            self.status.update(status='stopped', error=str(exc))
        finally:
            self.status.update(ended_at_unix=time.time(), run_state=read(self.run / 'state.json') if (self.run / 'state.json').exists() else None,
                               owned_processes_absent=all(not same_process(x) for x in self.owners))
            self.save()
            self.event('window_ended', status=self.status['status'], owned_processes_absent=self.status['owned_processes_absent'])
        if not self.status['owned_processes_absent']:
            raise RuntimeError('Owned process still present; preserve guard until released')


def self_test():
    # Real CPU subprocesses: STOP checkpoint, emergency termination, stale identity
    # rejection, and survival of an unrelated process. No model or CUDA imports.
    with tempfile.TemporaryDirectory(prefix='case-window-test-') as temp:
        directory = Path(temp)
        run = directory / 'run'
        run.mkdir()
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])
        try:
            graceful = subprocess.Popen([sys.executable, '-c',
                'import pathlib,time,sys; p=pathlib.Path(sys.argv[1]); '
                '\nwhile not (p/"STOP").exists(): time.sleep(.02)\n(p/"committed").write_text("saved")', str(run)])
            w = Window({'run': str(run), 'bound': '', 'user_window_end_unix': time.time() + 119}, directory)
            w.tick()
            assert graceful.wait(timeout=3) == 0 and (run / 'committed').read_text() == 'saved'
            hung = subprocess.Popen([sys.executable, '-c',
                'import signal,time,pathlib,sys; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                'pathlib.Path(sys.argv[1]).write_text("ready"); time.sleep(20)', str(directory / 'ready')])
            while not (directory / 'ready').exists():
                time.sleep(.01)
            owner = identity(hung.pid)
            owner['kind'] = 'worker'
            stale = dict(owner, start_ticks=owner['start_ticks'] + 1)
            assert not send_owned(stale, signal.SIGKILL)
            w.owners = [owner]
            w.deadline = time.time() + 19
            w.tick()
            assert hung.wait(timeout=3) == -signal.SIGKILL and unrelated.poll() is None
        finally:
            if 'hung' in locals() and hung.poll() is None:
                hung.kill()
                hung.wait()
            unrelated.terminate()
            unrelated.wait()
    print(json.dumps({'status': 'passed', 'CPU_only': True, 'checks': ['STOP_checkpoint', 'owned_termination', 'stale_identity_rejected', 'unrelated_process_preserved']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    config = read(args.config)
    directory = args.config.resolve().parent
    authorization=read(config['authorization']['path'])
    assert authorization['user_window_end_unix']==config['user_window_end_unix']
    assert authorization['latest_user_message']=='超时守护定到1:00前即可'
    assert authorization['absolute_deadline_local']=='2026-09-19T01:00:00+08:00'
    assert 0 < config['user_window_end_unix'] - config['user_window_start_unix'] <= 24*3600
    assert config['script_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert not Path(config['run']).exists(), 'Existing runs may not be restarted by this window'
    assert not (directory / 'window-state.json').exists(), 'Window already started'
    with (directory / '.window.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        window = Window(config, directory)
        signal.signal(signal.SIGTERM, window.cancel)
        signal.signal(signal.SIGINT, window.cancel)
        window.execute()


if __name__ == '__main__':
    main()
