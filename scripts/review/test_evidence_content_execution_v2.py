"""CPU acceptance of the allocation amendment and unchanged execution gates."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.review import freeze_evidence_content_execution_v2 as freeze
from scripts.review import run_evidence_content_execution_v2 as execution
from scripts.review import test_evidence_content_decomposition as original_tests
from diagnostics.evidence_label_calibration_execution import partition_groups
from diagnostics.general_model_evidence_evaluation import json_bytes, read_json, sha, write_output


class TwoGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files, cls.sources = freeze.build()
        cls.plan = json.loads(cls.files['plan.json'])
        cls.contexts = [json.loads(line) for line in cls.files['contexts.jsonl'].splitlines()]
        cls.historical = [json.loads(line) for line in cls.files['historical-selected.jsonl'].splitlines()]
        cls.comps = [json.loads(line) for line in cls.files['comparisons.jsonl'].splitlines()]
        cls.manifest = json_bytes({'schema_version': 'evidence-content-decomposition-freeze/v1', 'status': 'frozen',
            'source_files': cls.sources, 'artifacts': {n: sha(b) for n, b in cls.files.items()}})

    def temporary_freeze(self, root):
        target = root / 'frozen-02'; write_output(target, {**self.files, 'manifest.json': self.manifest}); return target

    def test_only_allocation_and_provenance_change(self):
        parent = read_json(freeze.PARENT / 'plan.json')
        observed = deepcopy(self.plan)
        for field in ('execution_amendment', 'preparation_plan_id', 'preparation_manifest_sha256'): observed.pop(field)
        observed['config']['execution']['device_indices'] = [0, 1, 2, 3]
        for field in ('plan_id', 'code_sha256', 'source_files'): observed[field] = parent[field]
        self.assertEqual(observed, parent)
        for name, raw in self.files.items():
            if name not in freeze.CHANGED_ARTIFACTS | {'execution-amendment.json'}:
                self.assertEqual(raw, (freeze.PARENT / name).read_bytes())

    def test_both_devices_cover_every_context_and_replica_pass_moves_each(self):
        def assignments(shift):
            rows = partition_groups(self.contexts, self.plan['catalog'], 1, [1, 2], shift)
            self.assertEqual([len(r['contexts']) for r in rows], [104, 104])
            return {c['record_id']: r['physical_gpu_index'] for r in rows for c in r['contexts']}
        first, second = assignments(0), assignments(1)
        self.assertEqual(len(first), 208); self.assertTrue(all(first[k] != second[k] for k in first))

    def test_preflight_refuses_busy_or_wrong_card_before_loading_model(self):
        uuids = self.plan['execution_amendment']['selected_gpu_uuids']
        idle = '\n'.join(f'{i}, {uuids[str(i)]}, 0, 46068, 0' for i in (1, 2))
        from types import SimpleNamespace
        with patch('subprocess.run', return_value=SimpleNamespace(stdout=idle)):
            self.assertEqual([r['index'] for r in execution.gpu_preflight(self.plan)['devices']], [1, 2])
        for bad in (idle.replace(', 0, 46068, 0', ', 20000, 46068, 0', 1), idle.replace(uuids['1'], 'GPU-wrong')):
            with patch('subprocess.run', return_value=SimpleNamespace(stdout=bad)), self.assertRaises(ValueError): execution.gpu_preflight(self.plan)

    def test_all_eight_gates_checkpoint_and_analysis_on_two_gpu_plan(self):
        # Reuse the already-reviewed full lifecycle test with this version's
        # plan/runner; no GPU calls or model loading in this isolated test.
        with patch.object(original_tests, 'execution', execution), patch.object(execution, 'gpu_preflight', return_value={'synthetic_cpu_only': True}):
            original_tests.ContentDecompositionTests.test_complete_eight_pass_cpu_synthetic_lifecycle_and_sealed_rerun(self)


if __name__ == '__main__': unittest.main()
