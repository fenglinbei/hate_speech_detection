"""Checks for the private model copy and scientific-input-preserving path amendment."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.review import freeze_evidence_lexicon_execution_v2 as freeze
from scripts.review.prepare_evidence_model_copy import copy_inventory
from diagnostics.general_model_evidence_evaluation import read_json


class ModelPathTests(unittest.TestCase):
    def test_copy_follows_source_link_but_creates_independent_regular_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); src = root / 'source'; src.mkdir(); shared = root / 'shared'
            shared.write_bytes(b'unchanged model bytes'); (src / 'weights').symlink_to(shared)
            before = shared.stat(); expected = hashlib.sha256(shared.read_bytes()).hexdigest()
            snapshot = {'files': [{'path': 'weights', 'size': shared.stat().st_size, 'sha256': expected}]}
            target = root / 'copy'; rows = copy_inventory(snapshot, src, target)
            self.assertFalse((target / 'weights').is_symlink()); self.assertTrue((src / 'weights').is_symlink())
            self.assertEqual((target / 'weights').read_bytes(), shared.read_bytes())
            self.assertNotEqual((target / 'weights').stat().st_ino, shared.stat().st_ino)
            self.assertEqual(before.st_ctime_ns, shared.stat().st_ctime_ns)
            self.assertEqual(rows[0]['sha256'], expected)

    def test_copy_rejects_changed_bytes_and_refuses_existing_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); src = root / 'source'; src.mkdir(); (src / 'weights').write_bytes(b'changed')
            snapshot = {'files': [{'path': 'weights', 'size': 7, 'sha256': '0' * 64}]}
            with self.assertRaisesRegex(ValueError, 'digest differs'): copy_inventory(snapshot, src, root / 'copy')
            with self.assertRaisesRegex(ValueError, 'already exists'): copy_inventory(snapshot, src, root / 'copy')

    def test_amendment_changes_only_model_load_path_and_provenance(self):
        import json
        files, sources = freeze.build()
        parent = read_json(freeze.PARENT / 'plan.json'); changed = json.loads(files['plan.json'])
        for field in ('model_path_amendment', 'preparation_plan_id', 'preparation_manifest_sha256'): changed.pop(field)
        changed['runtime_parent_plan']['package_path'] = parent['runtime_parent_plan']['package_path']
        for field in ('plan_id', 'code_sha256', 'source_files'): changed[field] = deepcopy(parent[field])
        self.assertEqual(changed, parent)
        for name, raw in files.items():
            if name not in freeze.CHANGED | {'model-path-amendment.json'}:
                self.assertEqual(raw, (freeze.PARENT / name).read_bytes())


if __name__ == '__main__': unittest.main()
