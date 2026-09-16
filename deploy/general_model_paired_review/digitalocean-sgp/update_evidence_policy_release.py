#!/usr/bin/env python3
"""Activate a policy-aware HSD release and migrate only its evidence session.

Upload this script beside update_static_release.py. --check-only validates the
archive/unit and performs the migration on private temporary copies. Activation
stops only HSD, verifies a stopped writer, backs up the latest bytes, then migrates
the evidence session. No Nginx or PDF commands exist in this updater.

Before migration, failures restore only code/unit. Once evidence bytes change,
failures leave HSD stopped with the candidate code and latest records intact:
an old binary must never reopen a new-policy session, and backups are not a
rollback mechanism for human decisions.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import update_static_release as common


POLICY = 'evidence/evidence_policy.json'
POLICY_OPTION = ' --evidence-policy /opt/hsd-general-model-paired-review/current/' + POLICY
ALLOWED_CHANGES = {
    'scripts/stage1/general_model_paired_review.py',
    'tools/general_model_paired_review_ui/server.py',
    'tools/general_model_paired_review_ui/evidence_store.py',
    'tools/general_model_paired_review_ui/evidence_schema.py',
    'tools/general_model_paired_review_ui/evidence_policy.py',
    'tools/general_model_paired_review_ui/evidence.html',
    'tools/general_model_paired_review_ui/evidence.js',
    'tools/general_model_paired_review_ui/evidence.css',
    POLICY, 'release_manifest.json',
}
require, sha, canonical = common.require, common.sha, common.canonical


def validate_update(old, new):
    require(set(old) <= set(new), 'Policy release cannot remove existing files')
    changed = {name for name in new if old.get(name) != new[name]}
    require(changed <= ALLOWED_CHANGES,
            'Out-of-scope release changes: ' + ', '.join(sorted(changed - ALLOWED_CHANGES)))
    require(POLICY in new, 'Release must include a separate active evidence policy')
    metadata = json.loads(new['release_manifest.json'])['evidence']['active_policy']
    policy = json.loads(new[POLICY])
    require(metadata == {'path': POLICY, 'sha256': sha(new[POLICY]), 'version': policy['version']},
            'Active policy metadata differs from packaged bytes')
    require(old['data/manifest.json'] == new['data/manifest.json'] and
            old['evidence/evidence_bundle.json'] == new['evidence/evidence_bundle.json'],
            'Frozen source manifest and AI bundle must retain exact bytes')
    return sorted(changed - {'release_manifest.json'})


def validate_unit(old, new):
    """The only allowed unit edit appends this exact optional policy argument."""
    lines = old.decode('utf-8').splitlines(keepends=True)
    matches = [index for index, line in enumerate(lines) if line.startswith('ExecStart=')]
    require(len(matches) == 1, 'Expected exactly one existing ExecStart')
    index = matches[0]
    line = lines[index].rstrip('\n')
    require(line.startswith('ExecStart=/usr/bin/python3 -B -S scripts/stage1/general_model_paired_review.py ')
            and ' --host 127.0.0.1 --port 8772 ' in line and
            ' --evidence-bundle ' in line and ' --evidence-session ' in line,
            'Existing HSD unit does not match the reviewed service')
    if ' --evidence-policy ' not in line:
        lines[index] = line + POLICY_OPTION + ('\n' if lines[index].endswith('\n') else '')
    else:
        require(line.endswith(POLICY_OPTION), 'Existing unit has an unexpected evidence policy path')
    require(new == ''.join(lines).encode('utf-8'),
            'Unit changes must be limited to the exact evidence-policy argument')


def write_unit(content):
    descriptor, temporary = tempfile.mkstemp(prefix='.hsd-policy-unit-', dir=common.UNIT.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o644)
        os.replace(temporary, common.UNIT)
        descriptor = os.open(common.UNIT.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


_MIGRATE_PROGRAM = r'''
import json
from pathlib import Path
import sys
release, session = Path(sys.argv[1]), Path(sys.argv[2])
reviewer, revision, session_sha256 = sys.argv[3], sys.argv[4], sys.argv[5]
sys.path[:0] = [str(release), str(release / 'src')]
from tools.general_model_paired_review_ui.evidence_policy import migrate_policy_session
from tools.general_model_paired_review_ui.evidence_store import EvidenceReviewStore
result = migrate_policy_session(
    bundle_path=release / 'evidence/evidence_bundle.json', policy_path=release / 'evidence/evidence_policy.json',
    session_path=session, reviewer_id=reviewer, expected_revision=revision,
    expected_session_sha256=session_sha256,
    actor='deployment:hsd-evidence-policy-release',
)
store = EvidenceReviewStore(bundle_path=release / 'evidence/evidence_bundle.json',
                           policy_path=release / 'evidence/evidence_policy.json',
                           session_path=session, reviewer_id=reviewer)
bootstrap = store.bootstrap()
reopened = EvidenceReviewStore(bundle_path=release / 'evidence/evidence_bundle.json',
                              policy_path=release / 'evidence/evidence_policy.json',
                              session_path=session, reviewer_id=reviewer)
if reopened.bootstrap() != bootstrap:
    raise RuntimeError('Migrated session did not resume')
for name, module in list(sys.modules.items()):
    if name.startswith(('tools.', 'build_lex.', 'rag.')) and getattr(module, '__file__', None):
        if not Path(module.__file__).resolve().is_relative_to(release):
            raise RuntimeError('Module imported outside the release: ' + name)
print(json.dumps({'policy_version': bootstrap['policy']['version'], 'revision': bootstrap['revision'], 'status': bootstrap['status']}))
'''


def migrate(release, path, snapshot, *, production=False):
    command = ['/usr/bin/python3', '-I', '-B', '-S', '-c', _MIGRATE_PROGRAM,
               str(release), str(path), snapshot['value']['reviewer_id'], snapshot['value']['revision'],
               sha(snapshot['raw'])]
    if production:
        command[:0] = ['runuser', '-u', 'hsd-review', '--']
    result = subprocess.run(command, capture_output=True, timeout=60, check=False)
    # Do not echo subprocess output: exception messages could include human text.
    require(result.returncode == 0, 'Isolated policy migration/resume failed; inspect private state and code')
    return json.loads(result.stdout)


def retained_migration(before, after):
    """Require the original human layer to survive the explicit version change."""
    require(before['paired']['raw'] == after['paired']['raw'], 'Paired session changed while stopped')
    old, new = before['evidence']['value'], after['evidence']['value']
    for field in ('schema_version', 'reviewer_id', 'source_identity', 'created_at',
                  'bundle_sha256', 'review_mode', 'blind_review_claimed'):
        require(old.get(field) == new.get(field), 'Evidence identity changed: ' + field)
    require(new.get('events', [])[:len(old.get('events', []))] == old.get('events', []),
            'Earlier evidence events were removed')
    require(set(old['records']) == set(new['records']) and set(old['objects']) == set(new['objects']),
            'Evidence record/object identities changed')
    for key, previous in old['records'].items():
        current = new['records'][key]
        for field, value in previous.items():
            require(current.get(field) == value, 'Prior case decision changed: ' + field)
    for oid, previous in old['objects'].items():
        current = new['objects'][oid]
        for field, value in previous.items():
            require(current.get(field) == value, 'Prior human object changed: ' + field)


def preflight(payload, before):
    """Run actual migration on private throwaway copies, including saved records."""
    with tempfile.TemporaryDirectory(prefix='hsd-policy-preflight-', dir='/tmp') as directory:
        temporary = Path(directory)
        release = temporary / 'release'
        release.mkdir()
        for name, content in payload.items():
            path = release / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        session = temporary / 'reviews/evidence/session.json'
        session.parent.mkdir(parents=True)
        common.private_write(session, before['evidence']['raw'])
        receipt = migrate(release, session, before['evidence'])
        raw = session.read_bytes()
        retained_migration(before, {**before, 'evidence': {'raw': raw, 'value': json.loads(raw)}})
        return receipt


def require_stopped():
    result = common.run('systemctl', 'show', common.SERVICE, '--property=ActiveState', '--property=MainPID')
    fields = dict(line.split('=', 1) for line in result.stdout.decode().splitlines())
    require(fields.get('ActiveState') in {'inactive', 'failed'} and fields.get('MainPID') == '0',
            'HSD writer is not fully stopped')


def recover(before, old_release, old_unit, release, new_unit, *, stopped, migration_started):
    if not stopped:
        return 'not_started'
    common.run('systemctl', 'stop', common.SERVICE)
    require_stopped()
    evidence_path = common.STATE / common.SESSIONS['evidence']
    # A migration subprocess may fail after its atomic write. Inspect disk rather
    # than trusting a Python flag or a successful subprocess return.
    changed = migration_started and (before is None or not evidence_path.is_file()
                                    or evidence_path.read_bytes() != before['evidence']['raw'])
    if changed:
        common.point_release(release)
        write_unit(new_unit)
        common.run('systemctl', 'daemon-reload')
        return 'policy_migrated_service_stopped_forward_repair_required'
    common.point_release(old_release)
    write_unit(old_unit)
    common.run('systemctl', 'daemon-reload')
    common.run('systemctl', 'start', common.SERVICE)
    if before is not None:
        common.wait_ready(before)
    return 'code_and_unit_rolled_back_records_preserved'


def deploy(args):
    require(not sys.flags.optimize, 'Run without Python optimization; verification is required')
    payload = common.archive_payload(args.archive, args.sha256)
    require(re.fullmatch('[a-f0-9]{64}', args.unit_sha256) is not None, 'Invalid unit SHA-256')
    new_unit = args.unit.read_bytes()
    require(sha(new_unit) == args.unit_sha256, 'Candidate unit SHA-256 differs')
    old_release = common.current_release()
    old_payload, old_unit = common.disk_payload(old_release), common.UNIT.read_bytes()
    changed = validate_update(old_payload, payload)
    validate_unit(old_unit, new_unit)
    observed = common.session_snapshot()
    check = preflight(payload, observed)
    release = common.ROOT / 'releases' / args.sha256
    if args.check_only:
        return {'status': 'verified_read_only', 'current': str(old_release), 'candidate': str(release),
                'changed_files': changed, 'sessions': common.summary(observed), 'isolated_migration': check}
    # Shares the existing updater's lock, preventing simultaneous static/policy deployments.
    with (common.STATE / '.static-release.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(common.current_release() == old_release and common.disk_payload(old_release) == old_payload
                and common.UNIT.read_bytes() == old_unit, 'Release or unit changed during validation')
        common.install_release(release, payload)
        stopped, migration_started, before, backup = False, False, None, None
        try:
            stopped = True
            common.run('systemctl', 'stop', common.SERVICE)
            require_stopped()
            before = common.session_snapshot()
            backup = Path(tempfile.mkdtemp(prefix='policy-backup-' + args.sha256[:12] + '-', dir=common.STATE))
            backup.chmod(0o700)
            for name, row in before.items():
                common.private_write(backup / (name + '-session.json'), row['raw'])
                require((backup / (name + '-session.json')).read_bytes() == row['raw'], 'Backup bytes differ')
            common.private_write(backup / 'prior.service', old_unit)
            common.private_write(backup / 'metadata.json', canonical({
                'previous_release': str(old_release), 'next_release': str(release),
                'sessions': common.summary(before), 'unit_sha256': sha(old_unit),
                'candidate_unit_sha256': args.unit_sha256,
            }))
            sync_directory(backup)
            sync_directory(common.STATE)
            # A reviewer may save after read-only preflight. Validate that exact
            # latest stopped snapshot again before committing its migration.
            preflight(payload, before)
            require_stopped()
            require(common.current_release() == old_release and common.UNIT.read_bytes() == old_unit,
                    'Release or unit changed while preparing migration')
            require(common.session_snapshot() == before, 'Sessions changed despite stopped writer')
            migration_started = True
            migration = migrate(release, common.STATE / common.SESSIONS['evidence'],
                                before['evidence'], production=True)
            migrated = common.session_snapshot()
            retained_migration(before, migrated)
            common.private_write(backup / 'migration-receipt.json', canonical({
                'before': common.summary(before), 'migrated': common.summary(migrated),
                'migration': migration,
            }))
            common.point_release(release)
            write_unit(new_unit)
            common.run('systemctl', 'daemon-reload')
            common.run('systemctl', 'start', common.SERVICE)
            after = common.wait_ready(migrated)
            require(common.UNIT.read_bytes() == new_unit, 'Service unit changed during activation')
            return {'status': 'active', 'release': str(release), 'backup': str(backup),
                    'changed_files': changed, 'before': common.summary(before),
                    'migrated': common.summary(migrated), 'after': common.summary(after),
                    'records_restored_from_backup': False, 'nginx_changed': False, 'pdf_changed': False}
        except BaseException:
            recovery = recover(before, old_release, old_unit, release, new_unit,
                               stopped=stopped, migration_started=migration_started)
            print(json.dumps({'status': recovery, 'records_restored_from_backup': False,
                              'backup': str(backup) if backup else None}), file=sys.stderr)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--unit', type=Path, required=True)
    parser.add_argument('--unit-sha256', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(deploy(args), ensure_ascii=False))
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, 'Policy release failed: ' + str(exc) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
