#!/usr/bin/env python3
"""CPU tests of the 15-minute scheduling amendment; no GPU/model execution."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.q01_module_package import FREEZE, WORK
from diagnostics.q01_module_poll_v2 import decision, command, epoch, validate_config, verify_amendment
from diagnostics.general_model_evidence_evaluation import file_sha


def fixture():
    return {'schema_version': 'q01-module-idle-config/v2', 'poll_interval_seconds': 900,
            'device_indices': [0, 1, 2, 3], 'plan': str(FREEZE), 'run': str(WORK / 'run-01'),
            'phase': 'full', 'first_check_at': '2026-09-16T09:00:00+08:00',
            'checkpoint_at': None, 'release_deadline': None, 'user_window_end': None,
            'run_until_complete': True, 'amendment_sha256': 'fixture'}


class PollTests(unittest.TestCase):
    def test_exact_fifteen_minute_grid_and_no_early_poll(self):
        c = fixture()
        first = epoch(c['first_check_at'])
        self.assertEqual(decision(c, first - 1), {'action': 'wait', 'next_check_epoch': first})
        for elapsed, next_seconds in ((0, 900), (899, 900), (900, 1800), (1822, 2700)):
            self.assertEqual(decision(c, first + elapsed), {'action': 'check', 'next_check_epoch': first + next_seconds})

    def test_deadline_clips_polling_and_stops_without_new_work(self):
        c = fixture() | {'run_until_complete': False, 'checkpoint_at': '2026-09-16T16:52:00+08:00',
                         'release_deadline': '2026-09-16T16:56:00+08:00', 'user_window_end': '2026-09-16T17:00:00+08:00'}
        stop = epoch(c['checkpoint_at'])
        self.assertEqual(decision(c, stop - 1)['next_check_epoch'], stop)
        self.assertEqual(decision(c, stop)['action'], 'window_closed')

    def test_policy_rejects_wrong_interval_devices_scope_and_ambiguous_deadline(self):
        for changes in ({'poll_interval_seconds': 1800}, {'device_indices': [1, 2]}, {'phase': 'engineering'},
                        {'run': '/tmp/other-run'}, {'run_until_complete': False},
                        {'checkpoint_at': '2026-09-16T16:52:00+08:00'},
                        {'first_check_at': '2026-09-16T09:00:00'}):
            with self.assertRaises(ValueError):
                validate_config(fixture() | changes)

    def test_command_binds_exact_config_and_has_no_old_deadline(self):
        c = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'config.json'
            p.write_text(json.dumps(c))
            args = command(p, c)
            self.assertIn('run_q01_module_polled_v2.py', args[1])
            self.assertEqual(args[-1], file_sha(p))
            self.assertNotIn('--stop-at', args)
            p.write_text(json.dumps(c | {'first_check_at': '2026-09-16T10:00:00+08:00'}))
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                command(p, c)

    def test_source_amendment_rejects_modified_code(self):
        import diagnostics.q01_module_poll_v2 as module
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            freeze = work / 'freeze'
            freeze.mkdir()
            (freeze / 'manifest.json').write_text('frozen')
            code = work / 'scheduler.py'
            code.write_text('pinned')
            manifest = work / 'manifest.json'
            manifest.write_text(json.dumps({'status': 'frozen', 'schema_version': 'q01-module-idle-resume/v2',
                'parent_manifest_sha256': file_sha(freeze / 'manifest.json'), 'source_files': {'scheduler.py': file_sha(code)}}))
            with patch.object(module, 'AMENDMENT', manifest), patch.object(module, 'ROOT', work), \
                 patch.object(module, 'FREEZE', freeze), patch.object(module, 'PARENT_MANIFEST', file_sha(freeze / 'manifest.json')):
                verify_amendment(file_sha(manifest))
                code.write_text('changed')
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    verify_amendment(file_sha(manifest))

    def test_wrapper_delegates_full_run_to_unchanged_executor(self):
        from scripts.review.run_q01_module_polled_v2 import main
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            run = p / 'run'
            run.mkdir()
            (run / 'run_manifest.json').write_text('{"status":"paused"}')
            c = fixture() | {'run': str(run), 'paused_run_manifest_sha256': file_sha(run / 'run_manifest.json')}
            path = p / 'config.json'
            path.write_text(json.dumps(c))
            with patch.object(sys, 'argv', ['fixture', '--config', str(path), '--config-sha256', file_sha(path)]), \
                 patch('scripts.review.run_q01_module_polled_v2.validate_config'), \
                 patch('scripts.review.run_q01_module_polled_v2.verify_amendment'), \
                 patch('scripts.review.run_q01_module_refinement.preparation_check') as check, \
                 patch('diagnostics.q01_module_execution.execute', return_value={'status': 'complete', 'completed_passes': []}) as execute:
                main()
                check.assert_called_once_with(c['plan'], c['run'])
                execute.assert_called_once_with(c['plan'], c['run'], [0, 1, 2, 3], stop_epoch=None, through_pass=None)

    def test_wrapper_rejects_config_changed_after_dispatch(self):
        from scripts.review.run_q01_module_polled_v2 import main
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text(json.dumps(fixture()))
            with patch.object(sys, 'argv', ['fixture', '--config', str(path), '--config-sha256', 'wrong']), \
                 patch('diagnostics.q01_module_execution.execute') as execute:
                with self.assertRaisesRegex(ValueError, 'after dispatch'):
                    main()
                execute.assert_not_called()

    def test_cancellation_does_not_signal_an_already_exited_process(self):
        from scripts.review.schedule_q01_module_poll_v2 import terminate_owned
        process = SimpleNamespace(poll=lambda: 0, send_signal=lambda _: self.fail('signaled an exited process'))
        terminate_owned(process)

    def test_busy_then_idle_launches_once_after_900_seconds_and_analyzes(self):
        import scripts.review.schedule_q01_module_poll_v2 as scheduler
        import diagnostics.q01_module_poll_v2 as policy
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            freeze, run, window = work / 'frozen-01', work / 'run-01', work / 'window-test'
            for p in (freeze, run, window): p.mkdir()
            (freeze / 'manifest.json').write_text('synthetic freeze')
            (run / 'run_manifest.json').write_text('{"status":"paused"}')
            receipt = work / 'receipt.json'
            receipt.write_text('{"status":"passed","plan_id":"fixture"}')
            allocation = [{'physical_gpu_index': i, 'uuid': 'synthetic-' + str(i)} for i in range(4)]
            c = fixture() | {'plan': str(freeze), 'run': str(run), 'results': str(work / 'results'),
                'allocation': allocation, 'paused_run_manifest_sha256': file_sha(run / 'run_manifest.json'),
                'plan_manifest_sha256': file_sha(freeze / 'manifest.json'), 'cpu_receipt': str(receipt),
                'cpu_receipt_sha256': file_sha(receipt)}
            config = window / 'config.json'
            config.write_text(json.dumps(c))
            clock = [epoch(c['first_check_at'])]
            polls = []
            def inventory(indices):
                polls.append(clock[0])
                if len(polls) == 1:
                    raise ValueError('synthetic busy cards')
                return {'devices': allocation}
            def sleep(seconds): clock[0] += seconds
            def launch(*args, **kwargs):
                (run / 'run_manifest.json').write_text(json.dumps({'status': 'complete',
                    'completed_passes': ['all-synthetic-passes'],
                    'invocations': [{'all_workers_exited': True, 'all_workers_normal_exit': True}]}))
                return SimpleNamespace(pid=12345, poll=lambda: 0, returncode=0)
            with patch.object(policy, 'FREEZE', freeze), patch.object(policy, 'WORK', work), \
                 patch.object(scheduler, 'WORK', work), patch.object(scheduler, 'verify_amendment'), \
                 patch.object(scheduler, 'load_frozen', return_value=({'plan_id': 'fixture'}, [], {}, [])), \
                 patch.object(scheduler, 'preflight', side_effect=inventory), \
                 patch.object(scheduler.time, 'time', side_effect=lambda: clock[0]), \
                 patch.object(scheduler.time, 'sleep', side_effect=sleep), \
                 patch.object(scheduler.subprocess, 'Popen', side_effect=launch) as launched, \
                 patch.object(scheduler.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as analyzed:
                result = scheduler.supervise(config)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(polls[1] - polls[0], 900)
                launched.assert_called_once()
                self.assertTrue(launched.call_args.kwargs['start_new_session'])
                self.assertIn('analyze', analyzed.call_args.args[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
