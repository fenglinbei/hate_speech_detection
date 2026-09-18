#!/usr/bin/env python3
"""CPU-only regression tests, including the real worker with synthetic logits."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics import cross_term_behavior_execution_v1 as c
from diagnostics import cross_term_behavior_gpu_v1 as worker_module


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/review' / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


audit = load_script('audit_cross_term_behavior_v1')
cli = load_script('run_cross_term_behavior_v1')


class SyntheticLifecycle:
    """Only /tmp artifacts. Real weights and GPU APIs are replaced, not numerical gates."""
    def __init__(self, root):
        import numpy as np
        self.np = np
        self.prepared, self.run = root / 'prepared', root / 'run'
        self.prepared.mkdir(); self.run.mkdir()
        self.new = c.jsonl(c.MATERIAL / 'model-inputs.jsonl')
        links = c.read(c.MATERIAL / 'historical-bridges.json')
        lookup = {r['request_id']: r for r in c.jsonl(c.OLD_FREEZE / 'model-inputs.jsonl')}
        self.bridge = [lookup[b['request_id']] for b in links['records']]
        for name, records in [('model-inputs.jsonl', self.new), ('bridge-inputs.jsonl', self.bridge)]:
            (self.prepared / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
        self.identity = dict(c.read(c.ROOT / c.read(c.OLD_RUN / 'run_manifest.json')['invocations'][0]['runtime_identity']['path']),
                             uuid='SYNTHETIC-CPU', synthetic_CPU_only=True)
        c.atomic_json(root / 'expected-identity.json', self.identity)
        self.plan = {'runtime': c.old.runtime_snapshot(), 'model_files': [],
            'allocation': None, 'GPU_launch_authorized_by_this_preparation': False,
            'required_allocation': {'index': -1, 'uuid': 'SYNTHETIC-CPU', 'name': 'CPU fixture', 'driver': 'none', 'total_mib': 0},
            'required_producer_identity': c.file_info(root / 'expected-identity.json'),
            'analysis_plan': c.file_info(c.MATERIAL / 'analysis-plan.json'),
            'design': c.file_info(c.MATERIAL / 'design.json')}
        c.atomic_json(self.prepared / 'plan.json', self.plan)
        c.atomic_json(self.prepared / 'manifest.json', {'synthetic_CPU_only': True})
        index = []
        self.vectors = {}
        for entry, request in zip(links['records'], self.bridge):
            receipt = c.read(c.ROOT / entry['score_ref']['path'])
            index.append(dict(condition_id=entry['condition_id'], request_id=entry['request_id'],
                prompt_sha256=entry['prompt_sha256'], score_ref=entry['score_ref'], raw_logits=receipt['raw_logits'],
                physical_score_id=receipt['physical_score_id'], qualification_ref=receipt['readout']['qualification_ref']))
            self.vectors[tuple(request['input_ids'])] = np.load(c.ROOT / receipt['raw_logits']['path'], allow_pickle=False)
        for i, request in enumerate(self.new):
            vector = np.zeros(151936, dtype=np.float32)
            vector[18830] = 1.0
            vector[42192] = 1.0 + ((i % 9) - 4) / 2
            self.vectors[tuple(request['input_ids'])] = vector
        c.atomic_json(self.prepared / 'historical-index.json', {'records': index,
            'original_run': str(c.OLD_RUN.relative_to(ROOT)), 'old_N_inputs_retained_externally': links['old_N_inputs'],
            'original_results': links['records'][0]['results_ref']})
        self.bound = root / 'bound.json'
        c.atomic_json(self.bound, {'synthetic_CPU_only': True,
            'preparation_manifest': c.file_info(self.prepared / 'manifest.json'), 'allocation': self.plan['required_allocation'],
            'authorization_note': 'Synthetic CPU test; no GPU authorization or execution.'})
        c.atomic_json(self.run / 'binding.json', {'synthetic_CPU_only': True, 'run_id': 'SYNTHETIC-CPU',
            'preparation_manifest': c.file_info(self.prepared / 'manifest.json'), 'allocation': self.plan['required_allocation'],
            'runtime': self.plan['runtime'], 'device_binding': c.file_info(self.bound)})
        self.state = {'synthetic_CPU_only': True, 'status': 'created', 'invocations': [], 'completed_passes': [],
            'preparation_manifest': c.file_info(self.prepared / 'manifest.json'), 'binding': c.file_info(self.run / 'binding.json'),
            'analysis_reference_join_performed': False}
        c.atomic_json(self.run / 'run_manifest.json', self.state)
        self.forwards = 0; self.pause_at = None

    def forward(self, model, ids, padding, device):
        vector = self.vectors[tuple(ids)].copy()
        if padding != 'none':
            vector[42192] += 2 ** -14
        self.forwards += 1
        if self.forwards == self.pause_at:
            (self.run / 'STOP').touch()
        return c.old.padded_input(ids, padding), vector

    def invocation(self, name):
        state = c.read(self.run / 'run_manifest.json')
        state['status'] = 'launching'
        state['invocations'].append({'invocation_id': name, 'worker_pid': None, 'controller_pid': os.getpid()})
        c.atomic_json(self.run / 'run_manifest.json', state)
        return worker_module.worker(self.prepared, self.bound, self.run, name)

    def release(self, complete):
        state = c.read(self.run / 'run_manifest.json')
        path = self.run / ('synthetic-release-' + state['invocations'][-1]['invocation_id'] + '.json')
        c.atomic_json(path, {'synthetic_CPU_only': True, 'owned_worker_absent': True,
                             'worker_exit_code': 0, 'worker_pid': os.getpid()}, replace=False)
        state.update(status='complete' if complete else 'paused', worker_exit_code=0,
                     owned_worker_absent=True, resource_release=c.file_info(path))
        c.atomic_json(self.run / 'run_manifest.json', state)

    def mocks(self):
        stack = ExitStack()
        stack.enter_context(patch.object(c, 'check_prepared', return_value=(self.plan, {'new': self.new, 'bridge': self.bridge})))
        stack.enter_context(patch.object(c.old, 'gpu_inventory', side_effect=AssertionError('GPU inventory forbidden in CPU tests')))
        stack.enter_context(patch.object(worker_module.kernel, 'load_model', return_value=(object(), None, self.identity)))
        stack.enter_context(patch.object(worker_module.kernel, 'forward_logits', side_effect=self.forward))
        stack.enter_context(patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'SYNTHETIC-CPU'}))
        return stack


class ExecutionTests(unittest.TestCase):
    def test_01_reviewed_inventory_and_source_isolation(self):
        new = c.jsonl(c.MATERIAL / 'model-inputs.jsonl')
        old_inputs = {r['request_id']: r for r in c.jsonl(c.OLD_FREEZE / 'model-inputs.jsonl')}
        bridge = [old_inputs[e['request_id']] for e in c.read(c.MATERIAL / 'historical-bridges.json')['records']]
        c.validate_input_sets(new, bridge)
        bad = copy.deepcopy(new); bad[0]['human_reference'] = '无'
        with self.assertRaisesRegex(ValueError, 'fields'):
            c.validate_input_sets(bad, bridge)
        bad = copy.deepcopy(new); bad[0]['input_ids'][-1] += 1
        with self.assertRaisesRegex(ValueError, 'token hash'):
            c.validate_input_sets(bad, bridge)

    def test_02_bridge_is_exact_and_complete(self):
        rows = {str(i): {'m': float(i), 'log_p_no': float(i), 'log_p_yes': 0.0} for i in range(84)}
        self.assertEqual(c.bridge_values(rows, rows)['transport_allowance'], 0)
        altered = copy.deepcopy(rows); altered['0']['m'] = 1e-12
        with self.assertRaisesRegex(ValueError, 'bridge numerical'):
            c.bridge_values(rows, altered)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            c.bridge_values(rows, {k: v for k, v in rows.items() if k != '0'})

    def test_03_new_bound_is_not_inherited(self):
        score = dict(m=0., z_no=0., z_yes=0., log_p_no=-1., log_p_yes=-1., legal_mass=.5,
                     log_legal_mass=-1., pair_support_no=.5)
        passes = {name: {str(i): dict(score) for i in range(36)} for name in c.ENGINEERING}
        self.assertEqual(c.qualification_values(passes)['margin_error_bound'], 1e-6)
        for name in c.ENGINEERING[2:4]:
            passes[name]['0'].update(m=2 ** -14, log_p_no=-1 + 2 ** -14)
        self.assertEqual(c.qualification_values(passes)['margin_error_bound'], 2 ** -13)
        for name, key, value, message in [('engineering-repeat', 'm', 1e-7, 'repeat/order'),
                                         ('engineering-left-padding', 'm', .002, 'padding')]:
            broken = copy.deepcopy(passes); broken[name]['0'].update({key: value, 'log_p_no': -1 + value})
            with self.assertRaisesRegex(ValueError, message):
                c.qualification_values(broken)
        broken = copy.deepcopy(passes); broken['engineering-repeat']['0']['log_p_no'] += .01
        with self.assertRaisesRegex(ValueError, 'identity'):
            c.qualification_values(broken)
        del passes['engineering-repeat']['0']
        with self.assertRaisesRegex(ValueError, 'coverage'):
            c.qualification_values(passes)

    def test_04_mixed_physical_bounds_cancel_aliases(self):
        maths = c.math_module()
        a = {'m': 3., 'physical_score_id': 'old', 'prompt_sha256': 'a', 'margin_error_bound': .125, 'qualification_ref': 'old-q'}
        b = {'m': 1., 'physical_score_id': 'new', 'prompt_sha256': 'b', 'margin_error_bound': .25, 'qualification_ref': 'new-q'}
        effect = maths.linear_effect([{'condition_id': k, 'coefficient': v} for k, v in [('a', 1), ('alias', -1), ('b', 2)]],
                                    {'a': a, 'alias': a, 'b': b})
        self.assertEqual((effect['value'], effect['bound']), (2., .5))
        with self.assertRaisesRegex(ValueError, 'conflicting aliases'):
            maths.linear_effect([{'condition_id': 'a', 'coefficient': 1}, {'condition_id': 'alias', 'coefficient': -1}],
                                {'a': a, 'alias': dict(a, qualification_ref='wrong')})

    def test_05_all_formulas_and_transitions_independent_decimal(self):
        for seed in range(7):
            result = audit.synthetic_registered_analysis(seed)
            report = audit.arithmetic(result['scores'], result['expressions'])
            self.assertEqual(report['expression_values_and_bounds'], 168)
            self.assertEqual(len(result['arm_readouts']), 10)
            self.assertEqual(len(result['family_equal_summaries']), 14)
            self.assertEqual(sum(r['exposure'] == 'contains_new_unscored_input' for r in result['expressions']), 138)

    def test_06_terminal_and_changed_bindings_reject_resume(self):
        for status in ['complete', 'failed', 'running', 'launching']:
            with self.assertRaisesRegex(ValueError, 'paused resume'):
                cli.validate_resume_state({'status': status}, True, {}, {})
        state = {'status': 'paused', 'preparation_manifest': 'a', 'binding': 'b'}
        cli.validate_resume_state(state, True, 'a', 'b')
        with self.assertRaisesRegex(ValueError, 'binding'):
            cli.validate_resume_state(state, True, 'other', 'b')

    def test_07_no_reference_join_before_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for status in ['paused', 'running', 'failed', 'scoring_complete_releasing']:
                c.atomic_json(run / 'run_manifest.json', {'status': status})
                with patch.object(c, 'check_completed', side_effect=AssertionError('must gate before reading results')):
                    with self.assertRaisesRegex(ValueError, 'released worker'):
                        c.analyze(run / 'missing', run, run / 'output')

    def test_08_cpu_tiny_model_reuses_actual_scorer(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.set_num_threads(2); torch.random.default_generator.manual_seed(13)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=64, _attn_implementation='eager')).float().eval()
        ids = [10, 20, 30, 40]
        _, reference = worker_module.kernel.forward_logits(model, ids, 'none', 'cpu')
        for padding in ['left_to_next_strict_multiple_of_16', 'right_to_next_strict_multiple_of_16']:
            prepared, actual = worker_module.kernel.forward_logits(model, ids, padding, 'cpu')
            torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(reference), atol=1e-6, rtol=1e-6)
            self.assertEqual(prepared['last_valid_index'], 15 if padding.startswith('left') else 3)
        self.assertFalse(torch.cuda.is_initialized())

    def test_09_full_worker_pause_resume_gates_analysis_corruption(self):
        with tempfile.TemporaryDirectory(prefix='cross-term-behavior-CPU-only-') as temporary:
            fixture = SyntheticLifecycle(Path(temporary))
            real_read = c.read

            def no_human_read(path):
                if Path(path).name in ['analysis-plan.json', 'design.json', 'relations.json']:
                    raise AssertionError('GPU worker attempted to parse human-reference/design metadata')
                return real_read(path)

            with fixture.mocks():
                fixture.pause_at = 88
                with patch.object(c, 'read', side_effect=no_human_read):
                    self.assertEqual(fixture.invocation('first'), 'paused')
                self.assertEqual(fixture.forwards, 88)
                fixture.release(complete=False)
                saved = {p: c.sha(p) for p in (fixture.run / 'scores').rglob('*') if p.is_file()}
                receipt = c.resume_check(fixture.prepared, fixture.run)
                self.assertEqual(receipt['saved_requests'], 88)
                self.assertEqual(receipt['sealed_passes'], ['historical-bridge'])
                (fixture.run / 'STOP').unlink()
                with patch.object(c, 'read', side_effect=no_human_read):
                    self.assertEqual(fixture.invocation('second'), 'scoring_complete_releasing')
                self.assertEqual(fixture.forwards, 300)
                self.assertTrue(all(c.sha(p) == digest for p, digest in saved.items()))
                self.assertEqual(c.read(fixture.run / 'run_manifest.json')['invocations'][-1]['reused_requests'], 88)
                # Raw/gate checks do not parse human reference files either.
                with patch.object(c, 'read', side_effect=no_human_read):
                    self.assertEqual(c.check_completed(fixture.prepared, fixture.run)['prompt_forwards'], 300)
                with self.assertRaisesRegex(ValueError, 'released worker'):
                    c.analyze(fixture.prepared, fixture.run, Path(temporary) / 'results')
                fixture.release(complete=True)
                result_dir = Path(temporary) / 'results'
                c.analyze(fixture.prepared, fixture.run, result_dir)
                before = {p: c.sha(p) for p in result_dir.iterdir()}
                report = audit.audit_results(fixture.prepared, fixture.run, result_dir)
                self.assertEqual(report['raw_vectors'], 384)
                self.assertTrue(all(c.sha(p) == digest for p, digest in before.items()))
                rows = c.read(result_dir / 'results.json')['scores']
                self.assertEqual(sum(r['physical_score_id'].startswith('SYNTHETIC-CPU:science') for r in rows), 36)
                self.assertEqual({r['margin_error_bound'] for r in rows if r['input_status'] == 'new_unscored'}, {2 ** -13})
                self.assertEqual({r['margin_error_bound'] for r in rows if r['input_status'] == 'historical_reuse'}, {.00045013427734375})
                target = next((fixture.run / 'scores/science').glob('*.npy'))
                original = target.read_bytes(); damaged = bytearray(original); damaged[-1] ^= 1; target.write_bytes(damaged)
                with self.assertRaisesRegex(ValueError, 'raw vector changed'):
                    c.check_completed(fixture.prepared, fixture.run, full=False)
                target.write_bytes(original)
                receipt_path = next((fixture.run / 'scores/science').glob('*.json'))
                original = receipt_path.read_bytes(); record = c.read(receipt_path)
                record['producer']['uuid'] = 'OTHER-GPU'; c.atomic_json(receipt_path, record)
                with self.assertRaisesRegex(ValueError, 'producer PID/UUID'):
                    c.check_completed(fixture.prepared, fixture.run, full=False)
                receipt_path.write_bytes(original)
                with self.assertRaisesRegex(ValueError, 'paused run'):
                    c.resume_check(fixture.prepared, fixture.run)

    def test_10_preparation_does_not_call_GPU_apis(self):
        # Static call inventory complements execution tests: the sole GPU inventory
        # is the separate bind action, never the CPU prepare/validate paths.
        import ast
        tree = ast.parse(Path(c.__file__).read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in ['create_preparation', 'check_prepared']:
                calls = [n.func.attr for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
                self.assertFalse({'gpu_inventory', 'idle_device', 'load_model', 'forward_logits'} & set(calls))


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('Run tests with CUDA_VISIBLE_DEVICES empty.')
    unittest.main(verbosity=2)
