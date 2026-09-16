"""Deployment safety tests on temporary synthetic sessions; no network/services."""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPOSITORY), str(REPOSITORY / 'src')]
import update_evidence_policy_release as updater

common = updater.common


def revised(value):
    value = copy.deepcopy(value)
    value.pop('revision', None)
    value['revision'] = common.sha(common.canonical(value))
    return value


def packed(files, *, policy=False):
    files = dict(files)
    metadata = {'source_manifest_sha256': common.sha(files['data/manifest.json']),
                'evidence': {'sha256': common.sha(files['evidence/evidence_bundle.json'])},
                'files': {name: {'sha256': common.sha(raw), 'bytes': len(raw)}
                          for name, raw in files.items()}}
    if policy:
        metadata['evidence']['active_policy'] = {
            'path': updater.POLICY, 'sha256': common.sha(files[updater.POLICY]),
            'version': json.loads(files[updater.POLICY])['version'],
        }
    files['release_manifest.json'] = common.canonical(metadata)
    return files


class DeploymentSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='hsd-policy-deploy-test-')
        self.root = Path(self.temporary.name)
        self.release_root, self.state = self.root / 'opt', self.root / 'state'
        self.release_root.mkdir()
        (self.release_root / 'releases').mkdir()
        self.state.mkdir()
        self.unit = self.root / 'hsd.service'
        self.new_unit = (HERE / 'hsd-general-model-paired-review.service').read_bytes()
        self.old_unit = self.new_unit.replace(updater.POLICY_OPTION.encode(), b'')
        self.unit.write_bytes(self.old_unit)
        self.candidate_unit = self.root / 'candidate.service'
        self.candidate_unit.write_bytes(self.new_unit)
        self.old_files = packed({'data/manifest.json': b'{"frozen":true}',
                                 'evidence/evidence_bundle.json': b'{"frozen_ai":true}',
                                 'tools/general_model_paired_review_ui/evidence_store.py': b'old-code'})
        self.new_files = packed({**{k: v for k, v in self.old_files.items() if k != 'release_manifest.json'},
                                 'tools/general_model_paired_review_ui/evidence_store.py': b'new-code',
                                 updater.POLICY: b'{"version":"test/v2"}'}, policy=True)
        self.old_release = self.release_root / 'releases' / ('a' * 64)
        for name, raw in self.old_files.items():
            path = self.old_release / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        (self.release_root / 'current').symlink_to(self.old_release)
        session = revised({'schema_version': 'synthetic-test/v1', 'reviewer_id': 'automated-deployment-test',
                           'records': {'one': {'status': 'confirmed', 'assessment': {'group': []},
                                               'material_snapshots': [], 'prior_exposure': {'ai': 'seen'},
                                               'confirmed_at': 'test-before'}},
                           'objects': {'q-one': {'status': 'confirmed', 'version': 1,
                                                 'values': {'note': 'synthetic only'}}},
                           'events': [{'action': 'synthetic-confirm'}], 'policy': {'version': 'test/v1'}})
        for relative in common.SESSIONS.values():
            path = self.state / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(common.canonical(session))
        self.calls = []
        self.patches = [patch.object(common, 'ROOT', self.release_root),
                        patch.object(common, 'STATE', self.state), patch.object(common, 'UNIT', self.unit),
                        patch.object(common, 'archive_payload', return_value=self.new_files),
                        patch.object(common, 'run', side_effect=self.service),
                        patch.object(common, 'wait_ready', side_effect=lambda _: common.session_snapshot()),
                        patch.object(updater, 'preflight', return_value={'status': 'isolated-test'})]
        for mock in self.patches:
            mock.start()
        self.before = common.session_snapshot()
        self.args = argparse.Namespace(archive=self.root / 'archive.tar.gz', sha256='b' * 64,
                                       unit=self.candidate_unit, unit_sha256=common.sha(self.new_unit),
                                       check_only=False)

    def tearDown(self):
        for mock in reversed(self.patches):
            mock.stop()
        self.temporary.cleanup()

    def service(self, *args):
        self.calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=b'ActiveState=inactive\nMainPID=0\n')

    def migration_write(self, *args, **kwargs):
        path = self.state / common.SESSIONS['evidence']
        session = json.loads(path.read_bytes())
        session['policy'] = {'version': 'test/v2'}
        session['records']['one']['policy_group_pending'] = True
        session['events'].append({'action': 'policy_migrated'})
        path.write_bytes(common.canonical(revised(session)))
        return {'status': 'synthetic-migrated'}

    def test_readonly_does_not_touch_sessions_unit_link_or_services(self):
        self.args.check_only = True
        receipt = updater.deploy(self.args)
        self.assertEqual(receipt['status'], 'verified_read_only')
        self.assertEqual(common.session_snapshot(), self.before)
        self.assertEqual(self.unit.read_bytes(), self.old_unit)
        self.assertEqual(common.current_release(), self.old_release)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.state / '.static-release.lock').exists())

    def test_success_preserves_all_old_records_and_exact_backup(self):
        with patch.object(updater, 'migrate', side_effect=self.migration_write):
            receipt = updater.deploy(self.args)
        self.assertEqual(receipt['status'], 'active')
        updater.retained_migration(self.before, common.session_snapshot())
        for name, row in self.before.items():
            self.assertEqual((Path(receipt['backup']) / (name + '-session.json')).read_bytes(), row['raw'])
        self.assertTrue(all(call[0] == 'systemctl' and (len(call) < 3 or call[2] == common.SERVICE)
                            for call in self.calls))

    def test_failure_before_write_rolls_back_only_code_unit(self):
        with patch.object(updater, 'migrate', side_effect=RuntimeError('before atomic write')):
            with self.assertRaises(RuntimeError):
                updater.deploy(self.args)
        self.assertEqual(common.current_release(), self.old_release)
        self.assertEqual(self.unit.read_bytes(), self.old_unit)
        self.assertEqual(common.session_snapshot(), self.before)
        self.assertIn(('systemctl', 'start', common.SERVICE), self.calls)

    def test_failure_after_atomic_write_never_starts_old_writer(self):
        def partial(*args, **kwargs):
            self.migration_write()
            raise RuntimeError('after atomic write')
        with patch.object(updater, 'migrate', side_effect=partial):
            with self.assertRaises(RuntimeError):
                updater.deploy(self.args)
        self.assertEqual(common.current_release().name, 'b' * 64)
        self.assertEqual(self.unit.read_bytes(), self.new_unit)
        updater.retained_migration(self.before, common.session_snapshot())
        self.assertNotIn(('systemctl', 'start', common.SERVICE), self.calls)

    def test_startup_failure_keeps_new_user_save(self):
        def failed_probe(_):
            path = self.state / common.SESSIONS['evidence']
            value = json.loads(path.read_bytes())
            value['events'].append({'action': 'subsequent-user-save'})
            path.write_bytes(common.canonical(revised(value)))
            raise RuntimeError('read-only startup probe failed')
        with patch.object(updater, 'migrate', side_effect=self.migration_write), \
             patch.object(common, 'wait_ready', side_effect=failed_probe):
            with self.assertRaises(RuntimeError):
                updater.deploy(self.args)
        latest = common.session_snapshot()['evidence']['value']
        self.assertEqual(latest['events'][-1]['action'], 'subsequent-user-save')
        self.assertEqual(common.current_release().name, 'b' * 64)
        self.assertEqual(self.calls.count(('systemctl', 'start', common.SERVICE)), 1)

    def test_scope_guards_reject_inputs_reserve_and_unit_changes(self):
        for name in ('data/manifest.json', 'evidence/evidence_bundle.json', 'data/reserve.json'):
            mutated = dict(self.new_files)
            mutated[name] = b'changed'
            with self.assertRaises(RuntimeError):
                updater.validate_update(self.old_files, mutated)
        with self.assertRaises(RuntimeError):
            updater.validate_unit(self.old_unit, self.new_unit.replace(b'RestartSec=3s', b'RestartSec=1s'))

    def test_identity_and_history_loss_are_rejected(self):
        for layer, field in (('paired', 'records'), ('evidence', 'objects'), ('evidence', 'events')):
            broken = copy.deepcopy(self.before)
            broken[layer]['value'][field] = {} if field != 'events' else []
            broken[layer]['raw'] = common.canonical(broken[layer]['value'])
            with self.assertRaises(RuntimeError):
                updater.retained_migration(self.before, broken)


class ActualMigrationPreflightTests(unittest.TestCase):
    def test_actual_stdlib_migration_with_saved_confirmed_fixture(self):
        from tools.general_model_paired_review_ui import test_evidence_store as fixtures
        fixture = fixtures.EvidenceStoreTests()
        fixture.setUp()
        try:
            fixture.materials('one')
            fixture.action('confirm', assessment=fixture.assessment())
            obj = fixture.obj('q-two', 'two')
            values = obj['effective_values']
            values['note'] = 'isolated draft retained across policy migration'
            fixture.action('save_object', key='two', object_id='q-two', object_version=0, values=values)
            raw = fixture.session.read_bytes()
            bundle_raw = fixture.bundle.read_bytes()
            bundle = json.loads(bundle_raw)
            document = '# Synthetic isolated group scope v2\n'
            policy = {'schema_version': 'general-model-evidence-policy-amendment/v1',
                      'version': 'test-policy/v2', 'sha256': hashlib.sha256(document.encode()).hexdigest(),
                      'document_text': document, 'parent_policy': bundle['policy'],
                      'approval': {'reviewer_id': 'automated-deployment-test', 'confirmed_at': '2026-09-09T00:00:00Z'},
                      'bundle_sha256': hashlib.sha256(bundle_raw).hexdigest(),
                      'impact': {'tasks': ['group'], 'object_ids': [oid for oid, obj in bundle['objects'].items()
                                                if obj['kind'] in {'query', 'demo', 'relation'}],
                                 'case_ids': bundle['order']}}
            builder_spec = importlib.util.spec_from_file_location('policy_test_builder', HERE.parent / 'build_release.py')
            builder = importlib.util.module_from_spec(builder_spec)
            builder_spec.loader.exec_module(builder)
            payload = {name: (REPOSITORY / name).read_bytes() for name in builder.CODE_FILES}
            payload['evidence/evidence_bundle.json'] = bundle_raw
            payload[updater.POLICY] = common.canonical(policy)
            before = {'paired': {'raw': b'synthetic-paired-layer', 'value': {}},
                      'evidence': {'raw': raw, 'value': json.loads(raw)}}
            receipt = updater.preflight(payload, before)
            self.assertEqual(receipt['policy_version'], 'test-policy/v2')
            self.assertEqual(fixture.session.read_bytes(), raw)
            self.assertEqual(fixture.bundle.read_bytes(), bundle_raw)
        finally:
            fixture.tearDown()


if __name__ == '__main__':
    unittest.main()
