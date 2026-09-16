#!/usr/bin/env python3
"""CPU tests of the new module frame, actual hooks, gates and phase isolation."""
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
from diagnostics.q01_module_inputs import build_requests, capture_specs, schedule, budget, select_contexts, UNITS
from diagnostics.q01_module_package import PARENT, read_json, read_lines
from diagnostics.q01_module_hooks import HookRuntime
from diagnostics.q01_module_scoring import ReadoutBuilder, nominate
from diagnostics.q01_module_execution import accept_pass, execute
from diagnostics.evidence_label_calibration import readouts
from diagnostics.general_model_numeric_analysis import candidate_scores
from scripts.review.test_q01_local_mechanism import (
    synthetic_scores, tiny_runner, tiny_context, tiny_positions, tiny_catalog, options)


class FrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = read_json(PARENT / 'plan.json')
        cls.contexts, cls.index = select_contexts(read_lines(PARENT / 'contexts.jsonl'))
        cls.positions = {r['record_id']: r for r in read_lines(PARENT / 'positions.jsonl')}
        cls.requests, cls.proofs = build_requests(cls.contexts, cls.index, cls.positions)
        cls.schedule = schedule(cls.requests, cls.contexts, cls.parent['catalog'])
        cls.science = [r for r in cls.requests if r['category'] != 'engineering']
        cls.rows = synthetic_scores(cls.contexts, cls.science, cls.parent['catalog'], options())
        cls.comparisons = ReadoutBuilder(cls.contexts, cls.requests, cls.rows, cls.parent['numeric_policy']['epsilon']).comparisons()
        by_id = {r['request_id']: r for r in cls.requests}
        cls.plan = {'catalog': cls.parent['catalog'], 'numeric_policy': cls.parent['numeric_policy'], 'eos_token_id': 151645,
                    'benchmark_readouts': {r['request_id']: readouts(r) for r in cls.rows
                        if by_id[r['request_id']]['category'] in ('block_benchmark', 'neutral_benchmark', 'output_diagnostic')}}

    def test_complete_six_module_frame_and_capture_coverage(self):
        b = budget(self.requests, self.schedule)
        self.assertEqual(b['primary_candidates_one_pass'], 1536)
        self.assertEqual(len(self.proofs), 96)
        self.assertEqual({(r['layer'], r['module'], r['role']) for r in self.requests if r['category'] == 'primary'},
                         {(layer, module, 'pre_answer') for layer, module in UNITS})
        for r in self.requests:
            if r['donor'] is None:
                continue
            specs = capture_specs(self.positions[r['donor']])
            matches = [s for s in specs if (s['module'], s['layer'], s['role']) == (r['module'], r['layer'], r['role'])]
            self.assertEqual(len(matches), 1)
            self.assertTrue(set(r['positions']) <= set(matches[0]['positions']))

    def test_final_layer_control_does_not_leak_to_earlier_modules(self):
        controls = [r for r in self.requests if r['kind'] == 'terminal_nonpropagation']
        self.assertTrue(controls)
        self.assertEqual({(r['layer'], r['module']) for r in controls}, {(35, 'attention'), (35, 'mlp')})
        self.assertTrue(all(r['expected_baseline'] == r['recipient'] for r in controls))
        self.assertTrue(all(r['expected_baseline'] is None for r in self.requests if r['category'] == 'site_control'))

    def test_both_modules_at_same_layer_remain_separate_nominees(self):
        result = nominate(self.comparisons)
        for metric in ('C', 'I'):
            self.assertEqual(len(result['rankings'][metric]['all_units']), 6)
        changed = deepcopy(self.comparisons)
        for r in changed:
            if r['category'] == 'primary' and r['view'] == 'original/answer_sum' and r['module'] == 'mlp':
                r['metrics']['C']['eligible_direction'] = False
        revised = nominate(changed)
        self.assertEqual(len(revised['rankings']['C']['eligible_ranking']), 3)
        self.assertTrue(all(r['module'] == 'attention' for r in revised['rankings']['C']['eligible_ranking']))

    def test_fixed_ncc_and_full_probe_coverage(self):
        index = {(r['cell_id'], r['view'], r['direction']): r for r in self.comparisons}
        for r in self.comparisons:
            if r['view'] == 'ncc_fixed_recipient':
                raw = index[r['cell_id'], 'original/answer_mean', r['direction']]
                for metric in ('C', 'I'):
                    for field in ('target', 'effect', 'residual'):
                        self.assertEqual(r['metrics'][metric][field], raw['metrics'][metric][field])
        for cell in {r['cell_id'] for r in self.comparisons}:
            views = {r['view'] for r in self.comparisons if r['cell_id'] == cell}
            self.assertEqual(sum(v.startswith('ncc_single/') for v in views), 5)
            self.assertEqual(sum(v.startswith('ncc_loo/') for v in views), 5)

    def test_all_twelve_passes_and_module_benchmark_gates(self):
        by_id = {r['request_id']: r for r in self.requests}
        history = [r for r in self.rows if by_id[r['request_id']]['kind'] == 'baseline']
        references, bridge = {}, None
        for spec in self.schedule:
            rows = synthetic_scores(self.contexts, [by_id[rid] for rid in spec['request_ids']], self.parent['catalog'], spec['options'])
            receipt = accept_pass(self.plan, self.contexts, self.requests, rows, spec, history,
                                  references.get(spec['phase']), bridge if spec['phase'] == 'science' else None)
            self.assertEqual(receipt['status'], 'passed')
            if spec['phase'] == 'science':
                self.assertTrue(next(g for g in receipt['gates'] if g['name'] == 'parent_block_benchmark')['passed'])
            if spec['mode'] == 'reference':
                references[spec['phase']] = rows
                if spec['phase'] == 'engineering':
                    bridge = {r['record_id']: r for r in rows if by_id[r['request_id']]['kind'] == 'baseline'}

    def test_changed_parent_benchmark_fails_independently(self):
        spec = next(p for p in self.schedule if p['pass_id'] == 'science-reference')
        plan = deepcopy(self.plan)
        first = next(iter(plan['benchmark_readouts'].values()))
        first[next(iter(first))] += 1.0
        history = [r for r in self.rows if next(q for q in self.requests if q['request_id'] == r['request_id'])['kind'] == 'baseline']
        with self.assertRaisesRegex(ValueError, 'historical benchmark changed'):
            accept_pass(plan, self.contexts, self.requests, self.rows, spec, history)

    def test_zero_targets_do_not_generate_nominees(self):
        rows = deepcopy(self.rows)
        for r in rows:
            for c in r['candidates']:
                c['token_logprobs'] = [-1.0] * c['answer_tokens']
                c.update(candidate_scores(c['token_logprobs'], c['eos_logprob']))
                c['scores'] = candidate_scores(c['token_logprobs'], c['eos_logprob'])
        comparisons = ReadoutBuilder(self.contexts, self.requests, rows, self.parent['numeric_policy']['epsilon']).comparisons()
        self.assertTrue(nominate(comparisons)['stop_this_scan_range'])

    def test_analysis_keeps_module_column_and_reconstructs_exactly(self):
        from diagnostics.q01_module_execution import analyze
        from diagnostics.general_model_evidence_evaluation import jsonl
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            freeze, run, output = work / 'freeze', work / 'run', work / 'results'
            freeze.mkdir()
            (run / 'science-reference').mkdir(parents=True)
            (freeze / 'analysis-reference.json').write_bytes((PARENT / 'analysis-reference.json').read_bytes())
            (run / 'science-reference/scores.jsonl').write_bytes(jsonl(self.rows))
            (run / 'run_manifest.json').write_text('{"fixture": "synthetic only"}')
            plan = {**self.plan, 'plan_id': 'synthetic-module-analysis'}
            with patch('diagnostics.q01_module_execution.load_frozen', return_value=(plan, self.contexts, self.positions, self.requests)), \
                 patch('diagnostics.q01_module_execution.check_run', return_value={'science': self.rows}):
                first = analyze(freeze, run, output)
                before = {p.name: p.read_bytes() for p in output.iterdir()}
                second = analyze(freeze, run, output)
                self.assertEqual(first, second)
                self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
                self.assertIn('module', (output / 'comparison-summary.csv').read_text(encoding='utf-8-sig').splitlines()[0])


class ModuleHookTests(unittest.TestCase):
    def setUp(self):
        self.contexts = [tiny_context('O', 'abcdefghijklmnopqr', 'O'), tiny_context('N', 'abcdjjjhijklmnopqr', 'N1')]
        self.positions = {c['record_id']: tiny_positions(c) for c in self.contexts}
        self.runner = tiny_runner()
        self.catalog = tiny_catalog()
        self.runtime = HookRuntime(self.runner, self.contexts, self.positions, 'module-cpu-test', expected_hidden=8)

    def request(self, **changes):
        return {'request_id': 'fixture', 'kind': 'patch', 'recipient': 'O', 'donor': 'N',
                'module': 'attention', 'layer': 34, 'role': 'pre_answer', 'positions': [17],
                'encoding': 'original', 'probe_id': None, **changes}

    def test_all_six_actual_module_hooks_self_patch_and_prefix_share_prompt_cache(self):
        baseline = self.runtime.score(self.request(kind='baseline'), self.catalog, options())
        for layer, module in UNITS:
            r = self.request(module=module, layer=layer, donor='O')
            row = self.runtime.score(r, self.catalog, options())
            self.assertAlmostEqual(readouts(row)['margin/answer_sum'], readouts(baseline)['margin/answer_sum'], places=5)
            r['donor'] = 'N'
            full = self.runtime.score(r, self.catalog, options())
            prefix = self.runtime.score(r, self.catalog, options(reference=False, prefix=True))
            self.assertEqual(full['hook_receipt']['donor_vector_sha256'], prefix['hook_receipt']['donor_vector_sha256'])
            self.assertAlmostEqual(readouts(full)['margin/answer_sum'], readouts(prefix)['margin/answer_sum'], places=5)
        self.assertEqual(self.runtime.forward_capture_count, 2)
        self.assertTrue(all(not m._forward_hooks for m in self.runner.model.modules()))

    def test_final_module_nonpropagation_and_earlier_module_propagation(self):
        baseline = self.runtime.score(self.request(kind='baseline'), self.catalog, options())
        for module in ('attention', 'mlp'):
            final = self.runtime.score(self.request(module=module, layer=35, role='lexicon_end', positions=[6]), self.catalog, options())
            self.assertEqual(readouts(final), readouts(baseline))
            earlier = self.runtime.score(self.request(module=module, layer=34, role='unrelated_demo_end', positions=[14]), self.catalog, options())
            self.assertNotEqual(readouts(earlier), readouts(baseline))

    def test_module_cache_provenance_rejects_corruption(self):
        import torch
        r = self.request()
        self.runtime.donor(r)
        with torch.inference_mode():
            self.runtime.cache['N']['attention', 34, 'pre_answer'].vectors[0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'cached donor was modified'):
            self.runtime.score(r, self.catalog, options())

    def test_tiny_random_qwen3_actual_module_outputs(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=36,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=8)
        config._attn_implementation = 'eager'
        self.runner.model = Qwen3ForCausalLM(config).float().eval()
        runtime = HookRuntime(self.runner, self.contexts, self.positions, 'random-qwen3-cpu', expected_hidden=16)
        with torch.inference_mode():
            baseline = runtime.score(self.request(kind='baseline'), self.catalog, options())
            for layer, module in UNITS:
                row = runtime.score(self.request(layer=layer, module=module, donor='O'), self.catalog, options())
                self.assertAlmostEqual(readouts(row)['margin/answer_sum'], readouts(baseline)['margin/answer_sum'], places=5)
            for module in ('attention', 'mlp'):
                row = runtime.score(self.request(layer=35, module=module, role='lexicon_end', positions=[6]), self.catalog, options())
                self.assertEqual(readouts(row), readouts(baseline))
        self.assertTrue(all(not m._forward_hooks for m in self.runner.model.modules()))

    def test_actual_module_worker_checkpoint_reuse_and_capture_audit(self):
        from diagnostics.q01_module_execution import _worker, verify_capture_audit
        from diagnostics.general_model_evidence_evaluation import file_sha
        import diagnostics.q01_module_hooks as hook_module
        requests = [self.request(request_id='r1', kind='baseline', donor=None), self.request(request_id='r2')]
        plan = {'code_sha256': {}, 'runtime_parent_plan': {}, 'runtime': {}, 'plan_id': 'module-cpu-test',
                'catalog': self.catalog, 'eos_token_id': 31}
        hardware = {'uuid': 'CPU-test-only', 'physical_gpu_index': 0}
        class Connection:
            def __init__(self, jobs): self.jobs = iter(jobs); self.sent = []
            def recv(self): return next(self.jobs)
            def send(self, value): self.sent.append(value)
            def close(self): pass
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / 'shards/0'
            previous = None
            for stop, wanted in ((0, 0), (None, 2), (None, 2)):
                connection = Connection([{'kind': 'pass', 'requests': requests, 'options': options(),
                    'directory': str(shard), 'stop_epoch': stop}, {'kind': 'stop'}])
                constructor = lambda runner, cs, ps, pid: HookRuntime(runner, cs, ps, pid, expected_hidden=8)
                with patch('diagnostics.general_model_numeric_kernel_v2.NumericRunner', return_value=self.runner), \
                     patch('diagnostics.evidence_label_calibration_execution.hardware_identity', return_value=hardware), \
                     patch('diagnostics.q01_module_execution.validate_numeric_identity'), \
                     patch.object(hook_module, 'HookRuntime', side_effect=constructor):
                    _worker(connection, plan, self.contexts, self.positions, 0, hardware, str(ROOT))
                self.assertEqual(connection.sent[-1]['completed'], wanted)
                if wanted:
                    current = (shard / 'scores.jsonl').read_bytes()
                    if previous is not None:
                        self.assertEqual(current, previous)
                    previous = current
            artifacts = {str(p.relative_to(tmp)): file_sha(p) for p in shard.iterdir()}
            verify_capture_audit(plan, self.contexts, requests, read_lines(shard / 'scores.jsonl'),
                                 Path(tmp), {'artifacts': artifacts}, {}, {}, hidden_size=8)


class ExecutionBoundaryTests(unittest.TestCase):
    def test_engineering_only_cannot_resume_into_science_when_already_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            run = work / 'run-01'
            run.mkdir()
            (run / 'run_manifest.json').write_text(json.dumps({'status': 'paused', 'plan_id': 'fixture',
                'device_indices': [0, 1, 2, 3], 'completed_passes': ['engineering-replica']}))
            fake_plan = {'plan_id': 'fixture', 'schedule': [{'pass_id': 'engineering-replica'}]}
            with patch('diagnostics.q01_module_execution.WORK', work), \
                 patch('diagnostics.q01_module_execution.load_frozen', return_value=(fake_plan, [], {}, [])), \
                 patch('diagnostics.q01_module_execution.verify_completed_passes'), \
                 patch('diagnostics.q01_module_execution.HookPool') as pool:
                with self.assertRaisesRegex(ValueError, 'already sealed'):
                    execute(work / 'freeze', run, [0, 1, 2, 3], through_pass='engineering-replica')
                pool.assert_not_called()

    def test_unsealed_analysis_does_not_parse_reference(self):
        from diagnostics.q01_module_execution import analyze
        with tempfile.TemporaryDirectory() as tmp, \
             patch('diagnostics.q01_module_execution.load_frozen', return_value=({}, [], {}, [])), \
             patch('diagnostics.q01_module_execution.check_run', side_effect=ValueError('unsealed')), \
             patch('diagnostics.q01_module_execution.read_json') as reader:
            with self.assertRaisesRegex(ValueError, 'unsealed'):
                analyze(tmp, tmp, Path(tmp) / 'results')
            reader.assert_not_called()

    def test_window_full_phase_requires_completion_budget(self):
        from scripts.review.schedule_q01_module_window import command, decision, epoch
        config = {'phase': 'engineering', 'device_indices': [0, 1, 2, 3], 'plan': '/tmp/plan', 'run': '/tmp/run',
                  'first_check_at': '2026-09-16T00:25:00+08:00', 'checkpoint_at': '2026-09-16T00:50:00+08:00',
                  'poll_interval_seconds': 1800}
        self.assertEqual(command(config)[-1], 'engineering')
        with self.assertRaisesRegex(ValueError, 'completion budget'):
            command({**config, 'phase': 'full'})
        self.assertEqual(command({**config, 'phase': 'full', 'full_completion_budget_accepted': True})[-1], 'full')
        self.assertEqual(decision(config, epoch(config['checkpoint_at']))['action'], 'window_closed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
