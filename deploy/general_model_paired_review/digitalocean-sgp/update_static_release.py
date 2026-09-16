#!/usr/bin/env python3
"""Update only the three evidence UI assets, preserving both live sessions.

Run on digitalocean-sgp with --archive and its independently verified --sha256.
--check-only performs read-only validation. No session initialization, record
restoration, unit change, Nginx operation or PDF service operation is performed.
"""

import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.request import urlopen


ROOT = Path('/opt/hsd-general-model-paired-review')
STATE = Path('/var/lib/hsd-general-model-paired-review')
UNIT = Path('/etc/systemd/system/hsd-general-model-paired-review.service')
SERVICE = 'hsd-general-model-paired-review.service'
STATIC = {'tools/general_model_paired_review_ui/evidence.' + suffix
          for suffix in ('css', 'html', 'js')}
SESSIONS = {'paired': 'session.json', 'evidence': 'evidence-applicability-v1/session.json'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False) + '\n').encode()


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, timeout=45)


def validate_payload(payload):
    manifest = json.loads(payload['release_manifest.json'])
    require(set(payload) == set(manifest['files']) | {'release_manifest.json'},
            'Release inventory differs from its manifest')
    for name, expected in manifest['files'].items():
        require(sha(payload[name]) == expected['sha256'] and
                len(payload[name]) == expected['bytes'], 'File hash/size differs: ' + name)
    require(sha(payload['data/manifest.json']) == manifest['source_manifest_sha256'],
            'Frozen source manifest hash differs')
    require(sha(payload['evidence/evidence_bundle.json']) == manifest['evidence']['sha256'],
            'Evidence bundle hash differs')


def archive_payload(path, expected_sha):
    require(re.fullmatch('[a-f0-9]{64}', expected_sha) is not None, 'Invalid archive SHA-256')
    raw = path.read_bytes()
    require(sha(raw) == expected_sha, 'Archive SHA-256 differs')
    payload, names = {}, set()
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        for member in archive:
            name = member.name.rstrip('/') if member.isdir() else member.name
            parts = name.split('/')
            require(parts[0] == 'release' and all(p not in {'', '.', '..'} for p in parts)
                    and '\\' not in name and name not in names, 'Unsafe or duplicate archive path')
            require(member.isdir() or member.isfile(), 'Archive links and special files are forbidden')
            require(not member.isfile() or len(parts) > 1, 'Archive root must be a directory')
            names.add(name)
            if member.isfile():
                payload['/'.join(parts[1:])] = archive.extractfile(member).read()
    validate_payload(payload)
    return payload


def disk_payload(release):
    require(release.is_dir() and not release.is_symlink(), 'Release is not a real directory')
    payload = {}
    for path in release.rglob('*'):
        require(not path.is_symlink() and (path.is_dir() or path.is_file()),
                'Release contains a link or special file')
        if path.is_file():
            payload[path.relative_to(release).as_posix()] = path.read_bytes()
    validate_payload(payload)
    return payload


def current_release():
    link = ROOT / 'current'
    require(link.is_symlink(), 'Current release must be a symlink')
    target = link.resolve(strict=True)
    require(target.parent == ROOT / 'releases' and re.fullmatch('[a-f0-9]{64}', target.name),
            'Current release is outside the immutable release tree')
    return target


def validate_static_update(old, new):
    require(set(old) == set(new), 'Static update cannot add or remove release files')
    changed = {name for name in old if old[name] != new[name]}
    require(changed <= STATIC | {'release_manifest.json'},
            'Non-static files changed: ' + ', '.join(sorted(changed - STATIC - {'release_manifest.json'})))
    require(bool(changed & STATIC), 'No evidence UI assets changed')
    return sorted(changed & STATIC)


def session_snapshot():
    result = {}
    for name, relative in SESSIONS.items():
        path = STATE / relative
        require(path.is_file() and not path.is_symlink(), 'Authoritative session is missing or is a symlink')
        raw = path.read_bytes()
        value = json.loads(raw)
        require(value.get('revision') == sha(canonical({k: v for k, v in value.items() if k != 'revision'})),
                'Authoritative session revision is invalid')
        require(isinstance(value.get('records'), dict) and value['records'], 'Authoritative records are empty')
        result[name] = {'raw': raw, 'value': value}
    return result


def summary(snapshot):
    return {name: {'sha256': sha(row['raw']), 'revision': row['value']['revision'],
                   'record_count': len(row['value']['records']),
                   'confirmed_count': sum(r['status'] == 'confirmed' for r in row['value']['records'].values()),
                   'confirmed_object_count': sum(r['status'] == 'confirmed' for r in row['value'].get('objects', {}).values())}
            for name, row in snapshot.items()}


def retained_history(before, after):
    """Accept normal new saves/reopens, while requiring earlier history to survive."""
    for name in SESSIONS:
        old, new = before[name]['value'], after[name]['value']
        for field in ('schema_version', 'reviewer_id', 'source_identity', 'source_manifest_sha256',
                      'bundle_sha256', 'policy', 'created_at'):
            require(old.get(field) == new.get(field), 'Session identity changed: ' + name)
        require(set(old['records']) == set(new['records']) and
                new.get('events', [])[:len(old.get('events', []))] == old.get('events', []),
                'Earlier session records/history were removed: ' + name)
        require(set(old.get('objects', {})) == set(new.get('objects', {})), 'Evidence objects were removed')
        for oid, previous in old.get('objects', {}).items():
            current = new['objects'][oid]
            require(current['version'] >= previous['version'], 'Human object version moved backwards')
            require(current['version'] != previous['version'] or current == previous,
                    'Human object changed without a new version')
        for key, previous in old['records'].items():
            current = new['records'][key]
            if name == 'evidence':
                snapshots = previous['material_snapshots']
                require(current['material_snapshots'][:len(snapshots)] == snapshots,
                        'Earlier evidence material snapshot was removed')
            elif previous.get('resources_locked_at'):
                require(all(current.get(k) == previous.get(k) for k in
                            ('resources', 'resources_locked_at', 'resources_sha256')),
                        'Earlier paired material snapshot changed')


def readonly_api(path):
    with urlopen('http://127.0.0.1:8772' + path, timeout=5) as response:
        return json.loads(response.read())


def wait_ready(before):
    last_error = None
    for _ in range(20):
        try:
            require(readonly_api('/api/health')['status'] == 'ok', 'Health check failed')
            observed = session_snapshot()
            boots = {'paired': readonly_api('/api/bootstrap'),
                     'evidence': readonly_api('/api/evidence/bootstrap')}
            latest = session_snapshot()
            retained_history(before, latest)
            # A user may save during these read-only probes. Retry for one stable
            # snapshot; never replace their session with the deployment backup.
            require(all(observed[name]['raw'] == latest[name]['raw'] and
                        boots[name]['revision'] == latest[name]['value']['revision'] and
                        boots[name]['reviewer_id'] == latest[name]['value']['reviewer_id'] and
                        boots[name]['status']['item_count'] == len(latest[name]['value']['records'])
                        for name in SESSIONS), 'Concurrent save during bootstrap verification')
            return latest
        except Exception as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError('Read-only startup verification failed') from last_error


def private_write(path, content):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def install_release(release, payload):
    if release.exists() or release.is_symlink():
        require(disk_payload(release) == payload, 'Existing release differs from the uploaded archive')
        return
    stage = Path(tempfile.mkdtemp(prefix='.static-release-', dir=ROOT / 'releases'))
    try:
        stage.chmod(0o755)
        for name, content in payload.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            path.write_bytes(content)
            path.chmod(0o644)
        require(disk_payload(stage) == payload, 'Staged release verification failed')
        stage.rename(release)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def point_release(target):
    temporary = ROOT / ('.current-static-' + str(os.getpid()))
    temporary.symlink_to(target)
    try:
        temporary.replace(ROOT / 'current')
        descriptor = os.open(ROOT, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.is_symlink():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    payload = archive_payload(args.archive, args.sha256)
    old_release = current_release()
    old_payload = disk_payload(old_release)
    changed = validate_static_update(old_payload, payload)
    release = ROOT / 'releases' / args.sha256
    if args.check_only:
        print(json.dumps({'status': 'verified_read_only', 'current': str(old_release),
                          'candidate': str(release), 'changed_assets': changed,
                          'sessions': summary(session_snapshot())}))
        return 0
    unit_raw = UNIT.read_bytes()
    # A deployment lock serializes this updater; it is separate from review locks.
    with (STATE / '.static-release.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(current_release() == old_release and disk_payload(old_release) == old_payload,
                'Current release changed during validation')
        install_release(release, payload)
        stopped, switched, before = False, False, None
        try:
            stopped = True
            run('systemctl', 'stop', SERVICE)
            before = session_snapshot()
            backup = Path(tempfile.mkdtemp(prefix='static-backup-' + args.sha256[:12] + '-', dir=STATE))
            backup.chmod(0o700)
            for name, row in before.items():
                private_write(backup / (name + '-session.json'), row['raw'])
            private_write(backup / 'metadata.json', canonical({'previous_release': str(old_release),
                          'next_release': str(release), 'sessions': summary(before), 'unit_sha256': sha(unit_raw)}))
            require(current_release() == old_release and UNIT.read_bytes() == unit_raw,
                    'Release or unit changed before activation')
            switched = True
            point_release(release)
            run('systemctl', 'start', SERVICE)
            after = wait_ready(before)
            require(UNIT.read_bytes() == unit_raw, 'Service unit changed during update')
            print(json.dumps({'status': 'active', 'release': str(release), 'backup': str(backup),
                              'changed_assets': changed, 'before': summary(before), 'after': summary(after),
                              'subsequent_writes_preserved': {name: before[name]['raw'] != after[name]['raw'] for name in SESSIONS},
                              'unit_changed': False, 'nginx_changed': False, 'pdf_changed': False}))
        except BaseException:
            if stopped:
                run('systemctl', 'stop', SERVICE)
                if switched:
                    point_release(old_release)
                run('systemctl', 'start', SERVICE)
                if before is not None:
                    wait_ready(before)
                print(json.dumps({'status': 'code_rolled_back', 'release': str(old_release),
                                  'records_restored_from_backup': False}), file=sys.stderr)
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
