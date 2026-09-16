#!/usr/bin/env python3
"""Regression of stored nonzero-error receipts and unchanged fail-closed gates."""
from pathlib import Path
from copy import deepcopy
from contextlib import ExitStack
import json
import sys
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics import q01_mechanism_execution as old
from diagnostics import q01_mechanism_execution_v2 as fixed
from diagnostics.q01_mechanism_resume_v2 import OLD_LINE, NEW_LINE


class ReceiptTests(unittest.TestCase):
    def run_receipt(self, module, mutate=None, require_complete=False):
        spec = {'pass_id': 'science-reference', 'phase': 'science', 'mode': 'reference',
                'options': {'replica_shift': 0}, 'request_ids': []}
        plan = {'plan_id': 'fixture', 'schedule': [spec]}
        state = {'plan_id': 'fixture', 'plan_manifest_sha256': 'hash',
                 'completed_passes': ['science-reference'], 'status': 'paused',
                 'query_reference_loaded_during_scoring': False, 'formal_test_or_reserve_access': False,
                 'runtime_identities': [{'identity': {}}, {'identity': {}}], 'device_indices': [0, 1]}
        rebuilt = {'status': 'passed', 'derived': {'largest': {
            'key': ('cell', 'original/answer_sum', 'R', 'I', 'effect'),
            'error': 0.00038909912109375, 'bound': 0.0013427734375}, 'passed': True},
            'gates': [{'passed': True, 'max_error': 0.00038909912109375}]}
        stored = json.loads(json.dumps(rebuilt))
        if mutate: mutate(stored)
        manifest = {'status': 'complete', 'plan_id': 'fixture', 'pass_spec_sha256': module.digest(spec), 'artifacts': {}}
        def reader(path):
            return stored if Path(path).name == 'acceptance.json' else manifest
        with ExitStack() as stack:
            for name, kwargs in [('file_sha', {'return_value': 'hash'}),
                                 ('read_json', {'side_effect': reader}),
                                 ('read_lines', {'return_value': []}),
                                 ('validate_numeric_identity', {}), ('verify_capture_audit', {}),
                                 ('accept_pass', {'return_value': rebuilt})]:
                stack.enter_context(patch.object(module, name, **kwargs))
            return module.verify_completed_passes(plan, [], [], Path('/tmp/plan'), Path('/tmp/run'), state,
                                                   require_complete=require_complete)

    def test_original_rejects_json_roundtrip_but_fix_accepts(self):
        with self.assertRaisesRegex(ValueError, 'gate reconstruction differs'):
            self.run_receipt(old)
        self.assertIn('science', self.run_receipt(fixed))

    def test_changed_error_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, 'gate reconstruction differs'):
            self.run_receipt(fixed, lambda r: r['derived']['largest'].update(error=0.0004))

    def test_changed_threshold_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, 'gate reconstruction differs'):
            self.run_receipt(fixed, lambda r: r['derived']['largest'].update(bound=0.1))

    def test_changed_key_order_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, 'gate reconstruction differs'):
            self.run_receipt(fixed, lambda r: r['derived']['largest']['key'].reverse())

    def test_incomplete_analysis_gate_is_unchanged(self):
        with self.assertRaisesRegex(ValueError, 'all 12 sealed passes'):
            self.run_receipt(fixed, require_complete=True)

    def test_exactly_one_receipt_predicate_changed(self):
        before = Path(old.__file__).read_text()
        after = Path(fixed.__file__).read_text()
        self.assertEqual(before.count(OLD_LINE), 1)
        self.assertEqual(before.replace(OLD_LINE, NEW_LINE), after)

    def test_overnight_window_and_resumed_command(self):
        from scripts.review.schedule_q01_gpu_window_v2 import decision, command, epoch
        cfg = {'first_check_at': '2026-09-15T21:00:00+08:00',
               'checkpoint_at': '2026-09-16T00:45:00+08:00',
               'poll_interval_seconds': 1800, 'device_indices': [0, 1, 2, 3],
               'plan': '/tmp/plan', 'run': '/tmp/run'}
        start = epoch(cfg['first_check_at'])
        self.assertEqual(decision(cfg, start-1)['action'], 'wait')
        self.assertEqual(decision(cfg, start+1)['next_check_epoch'], start+1800)
        self.assertEqual(decision(cfg, epoch(cfg['checkpoint_at']))['action'], 'window_closed')
        args = command(cfg)
        self.assertTrue(args[1].endswith('run_q01_local_mechanism_v2.py'))
        self.assertEqual(args[-1], cfg['checkpoint_at'])
        with self.assertRaises(ValueError): command({**cfg, 'device_indices': [1, 2]})


if __name__ == '__main__': unittest.main(verbosity=2)
