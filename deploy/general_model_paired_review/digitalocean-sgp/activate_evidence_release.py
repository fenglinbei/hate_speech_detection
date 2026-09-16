#!/usr/bin/env python3
"""Activate a preverified hsd-only release; preserve both authoritative sessions.

Run on digitalocean-sgp as the deployment user after uploading the archive and
reviewed unit. Does not touch Nginx, PDF files or PDF processes.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import pwd
import re
import subprocess
import sys
import tarfile
import time
from urllib.request import urlopen

ROOT = Path('/opt/hsd-general-model-paired-review')
STATE = Path('/var/lib/hsd-general-model-paired-review')
UNIT = Path('/etc/systemd/system/hsd-general-model-paired-review.service')
SERVICE = 'hsd-general-model-paired-review.service'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def run(*args):
    subprocess.run(args, check=True, capture_output=True, timeout=45)

def readonly_api(path):
    with urlopen('http://127.0.0.1:8772' + path, timeout=5) as response:
        return json.loads(response.read())

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive', type=Path, required=True)
parser.add_argument('--sha256', required=True)
parser.add_argument('--unit', type=Path, required=True)
parser.add_argument('--unit-sha256', required=True)
args = parser.parse_args()
if sys.flags.optimize:
    raise RuntimeError('Run without -O/PYTHONOPTIMIZE; release verification is required')
assert re.fullmatch('[a-f0-9]{64}', args.sha256)
archive_raw, unit_raw = args.archive.read_bytes(), args.unit.read_bytes()
assert sha(archive_raw) == args.sha256 and sha(unit_raw) == args.unit_sha256
release = ROOT / 'releases' / args.sha256
with tarfile.open(fileobj=io.BytesIO(archive_raw), mode='r:gz') as archive:
    members = archive.getmembers()
    assert len({m.name for m in members}) == len(members)
    for member in members:
        path = Path(member.name)
        assert path.parts[0] == 'release' and not path.is_absolute() and '..' not in path.parts
        assert member.isdir() or member.isfile()
    payload = {str(Path(m.name).relative_to('release')): archive.extractfile(m).read() for m in members if m.isfile()}
manifest = json.loads(payload['release_manifest.json'])
assert set(payload) == set(manifest['files']) | {'release_manifest.json'}
for name, expected in manifest['files'].items():
    assert sha(payload[name]) == expected['sha256'] and len(payload[name]) == expected['bytes']
assert sha(payload['data/manifest.json']) == manifest['source_manifest_sha256']
bundle = json.loads(payload['evidence/evidence_bundle.json'])
assert bundle['preparation']['human_confirmed'] == 0 and len(bundle['cases']) == 32
if release.exists():
    assert {p.relative_to(release).as_posix() for p in release.rglob('*') if p.is_file()} == set(payload)
    assert all((release / name).read_bytes() == content for name, content in payload.items())
else:
    release.mkdir(mode=0o755)
    for name, content in payload.items():
        path = release / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        path.write_bytes(content)
        path.chmod(0o644)
old_release = (ROOT / 'current').resolve()
old_unit = UNIT.read_bytes()
old_path = STATE / 'session.json'
assert old_path.is_file() and not old_path.is_symlink()
assert sha(old_path.read_bytes()) == bundle['source_identity']['parent_review']['sha256'], 'Parent human records changed; refresh read-only reference before activation'
backup = STATE / ('evidence-deploy-backup-' + args.sha256)
backup.mkdir(mode=0o700, exist_ok=False)
(backup / 'prior.service').write_bytes(old_unit)
run('systemctl', 'stop', SERVICE)
try:
    old_bytes = old_path.read_bytes()
    assert sha(old_bytes) == bundle['source_identity']['parent_review']['sha256']
    (backup / 'paired-session.json').write_bytes(old_bytes)
    (backup / 'paired-session.json').chmod(0o600)
    evidence_dir = STATE / 'evidence-applicability-v1'
    evidence_dir.mkdir(mode=0o700, exist_ok=True)
    owner = pwd.getpwnam('hsd-review')
    os.chown(evidence_dir, owner.pw_uid, owner.pw_gid)
    initialize = "import sys;from pathlib import Path;r=Path(sys.argv[1]);sys.path[:0]=[str(r),str(r/'src')];from tools.general_model_paired_review_ui.evidence_store import EvidenceReviewStore;s=EvidenceReviewStore(bundle_path=r/'evidence/evidence_bundle.json',session_path=Path(sys.argv[2]),reviewer_id='liaozijie');assert s.bootstrap()['status']['confirmed_count']==0"
    evidence_path = evidence_dir / 'session.json'
    if evidence_path.exists():
        raise RuntimeError('First activation requires a fresh evidence path; preserve existing records and inspect explicitly')
    run('runuser', '-u', 'hsd-review', '--', '/usr/bin/python3', '-B', '-S', '-c', initialize, str(release), str(evidence_path))
    evidence_bytes = evidence_path.read_bytes()
    def point(target):
        temporary = ROOT / 'current.evidence-next'
        assert not temporary.exists() and not temporary.is_symlink()
        temporary.symlink_to(target)
        temporary.replace(ROOT / 'current')
    point(release)
    UNIT.write_bytes(unit_raw)
    run('systemctl', 'daemon-reload')
    run('systemctl', 'start', SERVICE)
    for attempt in range(20):
        try:
            health = readonly_api('/api/health')
            evidence_boot = readonly_api('/api/evidence/bootstrap')
            paired_boot = readonly_api('/api/bootstrap')
            break
        except Exception:
            if attempt == 19:
                raise
            time.sleep(0.5)
    assert health['status'] == 'ok'
    assert evidence_boot['status']['confirmed_count'] == evidence_boot['status']['confirmed_object_count'] == 0
    assert paired_boot['status']['confirmed_count'] == sum(r['status'] == 'confirmed' for r in json.loads(old_bytes)['records'].values())
    assert old_path.read_bytes() == old_bytes and evidence_path.read_bytes() == evidence_bytes
    run('systemctl', 'restart', SERVICE)
    for attempt in range(20):
        try:
            reopened = readonly_api('/api/evidence/bootstrap')
            break
        except Exception:
            if attempt == 19:
                raise
            time.sleep(0.5)
    assert reopened['revision'] == evidence_boot['revision']
    assert old_path.read_bytes() == old_bytes and evidence_path.read_bytes() == evidence_bytes
    print(json.dumps({'status': 'active', 'release': str(release), 'archive_sha256': args.sha256, 'old_session_sha256': sha(old_bytes), 'old_confirmed': paired_boot['status']['confirmed_count'], 'evidence_session_sha256': sha(evidence_bytes), 'evidence_status': reopened['status'], 'restart_preserved_both_sessions': True, 'nginx_changed': False, 'pdf_changed': False}))
except Exception:
    run('systemctl', 'stop', SERVICE)
    temporary = ROOT / 'current.evidence-rollback'
    temporary.symlink_to(old_release)
    temporary.replace(ROOT / 'current')
    UNIT.write_bytes(old_unit)
    run('systemctl', 'daemon-reload')
    run('systemctl', 'start', SERVICE)
    raise
