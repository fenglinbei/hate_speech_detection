"""One authorized stop, exact lj DeepSpeed process tree; preserve checkpoint files."""
import hashlib
import json
import os
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
TRAINING = Path('/data/lj/Coding/G-LLaVA')
CHECKPOINT_ROOT = TRAINING / 'checkpoints/llava1.5_7b_qa_full'
MAIN = 3844994
WORKERS = [3845199, 3845200, 3845201, 3845202]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def checkpoint_snapshot():
    checkpoints = sorted(CHECKPOINT_ROOT.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
    assert checkpoints, 'No checkpoint; refuse stop'
    checkpoint = checkpoints[-1]
    step = int(checkpoint.name.split('-')[-1])
    trainer = json.loads((checkpoint / 'trainer_state.json').read_text())
    assert trainer['global_step'] == step
    assert (checkpoint / 'latest').read_text().strip() == f'global_step{step}'
    for rank in range(4):
        for name in [f'global_step{step}/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt',
                     f'global_step{step}/zero_pp_rank_{rank}_mp_rank_00_model_states.pt',
                     f'rng_state_{rank}.pth']:
            assert (checkpoint / name).stat().st_size > 0
    index = json.loads((checkpoint / 'pytorch_model.bin.index.json').read_text())
    for shard in set(index['weight_map'].values()):
        assert (checkpoint / shard).stat().st_size > 0
    files = []
    for path in sorted(checkpoint.rglob('*')):
        if not path.is_file():
            continue
        state = path.stat()
        row = {'path': str(path.relative_to(checkpoint)), 'bytes': state.st_size, 'mtime_ns': state.st_mtime_ns}
        if state.st_size < 1048576:
            row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix in ['.pt', '.pth', '.bin']:
            with zipfile.ZipFile(path) as archive:
                row['torch_zip_entries'] = len(archive.infolist())
                assert row['torch_zip_entries'] > 0
        files.append(row)
    return {'checkpoint': str(checkpoint), 'global_step': step, 'max_steps': trainer['max_steps'],
            'files': files, 'torch_pickle_executed': False,
            'validation': 'stable sizes/mtime; small-file hashes; archive central directories; expected four-rank state inventory'}


def process_identity(pid):
    proc = Path('/proc') / str(pid)
    status = dict(s.split(':', 1) for s in (proc / 'status').read_text().splitlines() if ':' in s)
    cmdline = (proc / 'cmdline').read_bytes()
    uid = int(status['Uid'].split()[0])
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    assert uid == 1003, 'Owner changed'
    assert b'/data/lj/envs/gllava/bin/python3.10' in cmdline and b'gllava/train/train.py' in cmdline
    if pid == MAIN:
        assert b'deepspeed.launcher.launch' in cmdline
    else:
        assert int(status['PPid']) == MAIN
    return {'pid': pid, 'uid': uid, 'ppid': int(status['PPid']), 'start_ticks': fields[19],
            'cmdline_sha256': hashlib.sha256(cmdline).hexdigest(), 'process_group': os.getpgid(pid)}


assert not (OUT / 'stop-before.json').exists(), 'One-shot stop already attempted'
authorization = {'user_instruction': '可以启动GPU运行了，现有进程先帮我安全停掉',
    'specific_ownership_confirmation': '是，授权停止这组 lj 训练', 'confirmed_account': 'lj', 'uid': 1003,
    'launcher_pid': MAIN, 'worker_pids': WORKERS, 'not_a_training_resume_request': True}
identities = [process_identity(pid) for pid in [MAIN] + WORKERS]
before = checkpoint_snapshot()
time.sleep(2)
assert checkpoint_snapshot() == before, 'Checkpoint is changing; do not interrupt a save'
write('stop-before.json', {'at': stamp(), 'authorization': authorization, 'processes': identities, 'checkpoint': before})
owner_code = '''
import hashlib,json,os,signal
from pathlib import Path
expected=json.loads(os.environ['AUTHORIZED_TRAINING_IDENTITIES'])
base='/data/lj/Coding/G-LLaVA'
for row in expected:
 p=Path('/proc')/str(row['pid'])
 assert hashlib.sha256((p/'cmdline').read_bytes()).hexdigest()==row['cmdline_sha256']
 assert (p/'stat').read_text().rsplit(')',1)[1].split()[19]==row['start_ticks']
 assert os.readlink(p/'cwd')==base
 assert os.getpgid(row['pid'])==row['process_group']
 for fd in (p/'fd').iterdir():
  try:
   target=os.readlink(fd)
   flags=next(s.split()[1] for s in (p/'fdinfo'/fd.name).read_text().splitlines() if s.startswith('flags:'))
   assert not (target.startswith(base+'/checkpoints/llava1.5_7b_qa_full/') and int(flags,8)&3), 'Checkpoint writer still open'
  except FileNotFoundError: pass
os.kill(expected[0]['pid'],signal.SIGTERM)
print(json.dumps({'signal':'SIGTERM','target_pid':expected[0]['pid'],'sender_uid':os.getuid(),'direct_worker_signals_sent':False}),flush=True)
'''
env = dict(os.environ, AUTHORIZED_TRAINING_IDENTITIES=json.dumps(identities))
result = subprocess.run(['runuser', '-u', 'lj', '--', 'python', '-c', owner_code], env=env,
                        text=True, capture_output=True)
write('stop-signal.json', {'at': stamp(), 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
assert result.returncode == 0, 'Stop was not sent successfully; inspect receipt'
print(result.stdout, flush=True)
deadline = time.monotonic() + 150
while time.monotonic() < deadline:
    alive = []
    for row in identities:
        proc = Path('/proc') / str(row['pid'])
        try:
            fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[19] == row['start_ticks'] and fields[0] != 'Z':
                alive.append(row['pid'])
        except FileNotFoundError:
            pass
    if not alive:
        break
    time.sleep(1)
after = checkpoint_snapshot()
unchanged = after == before
write('stop-after.json', {'at': stamp(), 'remaining_owned_processes': alive, 'checkpoint': after,
    'latest_checkpoint_unchanged': unchanged, 'direct_SIGKILL_sent': False,
    'launcher_behavior': 'Its existing handler terminates only its worker trees, with built-in 30-second fallback per tree.',
    'unsaved_training_progress': 'Not saved at signal; resume from the verified latest checkpoint.'})
assert not alive, 'Launcher/worker remains; no extra forced signal sent'
assert unchanged, 'Checkpoint snapshot changed; inspect before/after evidence'
print(json.dumps({'status': 'training_stopped', 'checkpoint': before['checkpoint'],
                  'checkpoint_unchanged': unchanged}), flush=True)
