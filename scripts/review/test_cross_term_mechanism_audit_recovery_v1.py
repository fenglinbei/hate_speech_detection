#!/usr/bin/env python3
"""CPU-only regression checks using sealed telemetry and adversarial mutations."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import audit_cross_term_mechanism_results_v1 as old
import audit_cross_term_mechanism_results_v2 as new

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'reviews/cross-term-mechanism-v1'


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_dir = WORK / 'run-01'
        cls.records = {}
        for file in sorted((cls.run_dir / 'records').glob('*/*.json')):
            record = old.read(file)
            key = record['job']['job_id'] if record['job'] else record['request_id']
            cls.records[record['stage'], key] = record
        cls.requests = {r['request_id']: r for r in map(json.loads, (WORK / 'prepared-01/scoring-inputs.jsonl').read_text().splitlines())}
        cls.profile = old.read(WORK / 'prepared-01/model-profile.json')

    def check_bundle(self, records=None):
        details = {}
        count = new.audit_branch_sources(self.run_dir, records or self.records, self.requests, self.profile, details)
        return count, details

    def test_original_failure_and_exact_recovery(self):
        with self.assertRaises(AssertionError):
            old.audit_branch_sources(self.run_dir, self.records, self.requests, self.profile)
        count, details = self.check_bundle()
        self.assertEqual(count, 960)
        self.assertEqual(details['format_branch_proofs'], 96)
        self.assertEqual(details['exact_unappended_comparators'], 88)
        self.assertEqual(details['exact_right_padding_comparators'], 8)

    def test_nonformat_must_not_fall_back_to_padded_vector(self):
        file = self.run_dir / 'format/B02-D02-restore-L26-attention.json'
        form = old.read(file)
        records = deepcopy(self.records)
        records['production', form['key']]['patch_proof']['restoration'][0]['before_sha256'] = form['steps'][0]['patch_proof']['restoration'][0]['before_sha256']
        with self.assertRaises(AssertionError):
            self.check_bundle(records)

    def test_wrong_installed_native_vector_rejected(self):
        records = deepcopy(self.records)
        rp = records['production', 'B02-D02-restore-L26-attention']['patch_proof']['restoration'][0]
        rp['replacement_sha256'] = rp['before_sha256']
        with self.assertRaises(AssertionError):
            self.check_bundle(records)

    def corrupt_format(self, mutate):
        target = self.run_dir / 'format/B02-D02-restore-joint.json'
        real_read = old.read
        def reader(path):
            value = real_read(path)
            if Path(path) == target:
                mutate(value)
            return value
        with patch.object(old, 'read', reader), self.assertRaises(AssertionError):
            self.check_bundle()

    def test_wrong_joint_before_branch_rejected(self):
        upstream = self.records['cross-right', 'B02-D02-upstream']
        with np.load(upstream['trajectory']['path']) as z:
            wrong = new.digest(z['branches'][28, 1])
        self.corrupt_format(lambda f: f['steps'][0]['patch_proof']['restoration'][1].update(before_sha256=wrong))

    def test_missing_prefix_rejected(self):
        self.corrupt_format(lambda f: f['steps'][0].update(prefix_tokens=[]))

    def test_format_norm_corruption_rejected(self):
        def mutate(f):
            item = f['steps'][0]['patch_proof']['restoration'][0]
            item['difference_l2'] += .01
        self.corrupt_format(mutate)

    def test_unmatched_hash_and_large_padded_drift_rejected(self):
        base = np.zeros((36, 2, 3), np.float32)
        other = base.copy(); other[26, 0, 0] = 1
        arrays = {('cross-probe', 'x'): base, ('cross-right', 'x'): other}
        proof = {'layer': 26, 'branch': 'attention', 'before_sha256': 'missing'}
        with self.assertRaises(AssertionError):
            new.choose_before('format', ('cross-probe', 'x'), arrays, proof)
        proof['before_sha256'] = new.digest(other[26, 0])
        with self.assertRaises(AssertionError):
            new.choose_before('format', ('cross-probe', 'x'), arrays, proof)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    receipt = {'status': 'pass' if result.wasSuccessful() else 'fail', 'tests': result.testsRun,
               'CPU_only_saved_telemetry': True, 'GPU_forwards': 0, 'research_weights_loaded': False,
               'failures': [str(x) for _, x in result.failures + result.errors],
               'source': old.info(Path(__file__))}
    with args.output.open('x') as file:
        json.dump(receipt, file, ensure_ascii=False, indent=2)
    print(json.dumps(receipt, ensure_ascii=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
