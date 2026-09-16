"""Meaningful CPU guards for aliases, reference isolation and both execution stages."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.review import freeze_evidence_functional_queries as freeze
from scripts.review import run_evidence_functional_queries as execution
from scripts.review import analyze_evidence_functional_queries as analysis
from scripts.review import audit_evidence_functional_queries as independent
from scripts.review.test_evidence_label_calibration import block
from diagnostics import evidence_label_calibration_execution as backend
from diagnostics import evidence_functional_queries as mathlib
from diagnostics.general_model_evidence_evaluation import json_bytes, read_json, read_lines, sha, write_output


class FunctionalQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.built = {}
        for stage in (1, 2):
            files, sources = freeze.build(stage)
            manifest = json_bytes({'schema_version': 'evidence-functional-query-freeze/v1', 'status': 'frozen',
                'source_files': sources, 'artifacts': {n: sha(b) for n, b in files.items()}})
            cls.built[stage] = {**files, 'manifest.json': manifest}
        cls.plans = {s: json.loads(f['plan.json']) for s, f in cls.built.items()}
        cls.contexts = {s: [json.loads(x) for x in f['contexts.jsonl'].splitlines()] for s, f in cls.built.items()}
        cls.comps = {s: [json.loads(x) for x in f['comparisons.jsonl'].splitlines()] for s, f in cls.built.items()}

    def synthetic(self, stage):
        return [block(c, .15 + (i % 19) * .031, self.plans[stage]['catalog'][c['task']])
                for i, c in enumerate(self.contexts[stage])]

    def test_bulk_acceptance_preserves_draft_and_absent_original_gold(self):
        approval = freeze.acceptance()
        self.assertEqual(len(approval['records']), 8)
        for stage, files in self.built.items():
            refs = json.loads(files['analysis_references.json'])
            self.assertEqual([r['adjudicated_label'] for r in refs if r['query_id'] != '3169'], ['non-hate', 'hate'] * 4)
            self.assertTrue(all(r['original_label'] is None for r in refs if r['query_id'] != '3169'))
        source = read_json(freeze.DRAFT / 'materials-source.json')
        self.assertTrue(all(q['human_applicability'] is q['human_label'] is None for q in source['queries']))

    def test_stage_dedup_coverage_and_exact_cross_stage_inputs(self):
        self.assertEqual([len(self.contexts[s]) for s in (1, 2)], [476, 312])
        a, b = [{c['record_id']: c for c in self.contexts[s]} for s in (1, 2)]
        shared = a.keys() & b.keys(); self.assertEqual(len(shared), 80)
        self.assertTrue(all(a[k] == b[k] for k in shared))
        for s in (1, 2):
            values, _, _, _ = mathlib.numeric_values(self.contexts[s], self.synthetic(s), self.plans[s]['catalog'])
            self.assertEqual(len(values), 132 if s == 1 else 84)
        self.assertEqual(sum(p['counts']['candidate_evaluations'] for p in self.plans.values()), 9712)

    def test_shared_background_cancellation_and_uncancelled_local_effect(self):
        raw = self.synthetic(1)
        values, _, _, aliases = mathlib.numeric_values(self.contexts[1], raw)
        cross = [c for c in self.comps[1] if c['cross_query']]
        self.assertTrue(cross)
        for c in cross:
            raw_mean, raw_factor, _ = mathlib.linear_value(c, 'original/answer_mean', values, aliases)
            for view in ('ncc', 'single_probe/space', 'leave_one_out/na'):
                value, factor, residual = mathlib.linear_value(c, view, values, aliases)
                self.assertEqual(value, raw_mean); self.assertEqual(factor, raw_factor)
                self.assertLess(abs(residual), 1e-10)
        c = next(c for c in self.comps[1] if c['contrast_id'].endswith('O-N1-I'))
        _, factor, residual = mathlib.linear_value(c, 'ncc', values, aliases)
        self.assertEqual(factor, 8); self.assertIsNone(residual)
        changed = deepcopy(raw); changed[0] = block(self.contexts[1][0], 2., self.plans[1]['catalog'][self.contexts[1][0]['task']])
        self.assertFalse(mathlib.compare_derived(self.contexts[1], raw, changed, self.comps[1], .001)['passed'])

    def test_duplicate_alias_and_missing_real_input_rejected(self):
        raw = self.synthetic(1); contexts = deepcopy(self.contexts[1])
        contexts[0]['bindings'].append(deepcopy(contexts[0]['bindings'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate'): mathlib.numeric_values(contexts, raw)
        with self.assertRaises(ValueError): mathlib.numeric_values(self.contexts[1], raw[:-1])

    def test_no_reference_parse_in_loader_and_unsealed_analysis_guard(self):
        with tempfile.TemporaryDirectory(prefix='functional-load-') as temp:
            path = Path(temp) / 'frozen'; write_output(path, self.built[1])
            reader = freeze.read_json
            def guard(p):
                self.assertNotIn(Path(p).name, ('analysis_references.json', 'feedback.json', 'feedback-01.json'))
                return reader(p)
            with patch.object(freeze, 'read_json', side_effect=guard): freeze.load_frozen(path)
            (path / 'contexts.jsonl').write_bytes(self.built[1]['contexts.jsonl'] + b'\n')
            with self.assertRaisesRegex(ValueError, 'artifact changed'): freeze.load_frozen(path)
        with patch.object(execution, 'check_run', side_effect=ValueError('unsealed')), \
             patch.object(analysis, 'read_json', side_effect=AssertionError('early reference read')):
            with self.assertRaisesRegex(ValueError, 'unsealed'): analysis.analyze('/tmp/p', '/tmp/r', '/tmp/o')

    def test_four_gpu_assignment_and_busy_device_guard(self):
        assignments = [backend.partition_groups(self.contexts[1], self.plans[1]['catalog'], 1, [0, 1, 2, 3], shift) for shift in (0, 1)]
        maps = [{c['record_id']: a['physical_gpu_index'] for a in aa for c in a['contexts']} for aa in assignments]
        self.assertTrue(all(maps[0][k] != maps[1][k] for k in maps[0]))
        self.assertEqual({len(a['contexts']) for a in assignments[0]}, {119})
        idle = '\n'.join(f'{i}, {freeze.DEVICES[str(i)]}, 0, 46068, 0' for i in range(4))
        with patch('subprocess.run', return_value=SimpleNamespace(stdout=idle)):
            self.assertEqual(len(execution.gpu_preflight(self.plans[1])['devices']), 4)
        with patch('subprocess.run', return_value=SimpleNamespace(stdout=idle.replace(', 0, 46068, 0', ', 15000, 46068, 0', 1))):
            with self.assertRaises(ValueError): execution.gpu_preflight(self.plans[1])

    def test_two_stage_full_synthetic_lifecycle_seal_bridge_and_decimal(self):
        saved_score = backend.score_pass; calls, closed = [], []
        historical = {r['record_id']: r for r in [json.loads(x) for x in self.built[1]['historical-selected.jsonl'].splitlines()]}
        def candidates(c, plan):
            # Same physical prompt receives the same synthetic score in either stage.
            val = .17 + (int(c['record_id'][-4:], 16) % 29) * .013
            r = deepcopy(historical.get(c['record_id']) or block(c, val, plan['catalog'][c['task']]))
            for candidate in r['candidates']:
                candidate['reference_scores'] = {'token_logprobs': candidate['token_logprobs'], 'eos_logprob': candidate['eos_logprob']}
            return r['candidates']
        pool = SimpleNamespace(identity={'test': 'synthetic_cpu_only'}, close=lambda terminate=False: closed.append(terminate))
        def score(pool, contexts, plan, output, **options):
            def scorer(runner, items, reference=False):
                calls.extend(i['context']['record_id'] for i in items)
                return [candidates(i['context'], plan)[i['candidate']['ordinal']] for i in items]
            def prefix(runner, c, catalog, reference=False):
                calls.extend([c['record_id']] * len(catalog)); return candidates(c, plan)
            return saved_score(pool, contexts, plan, output, scorer=scorer, prefix_scorer=prefix, **options)
        def geometry(rows, contexts, plan, **options):
            self.assertEqual([r['record_id'] for r in rows], [c['record_id'] for c in contexts])
            return {'synthetic_cpu_only': True, 'blocks': len(rows), 'options': options}
        with tempfile.TemporaryDirectory(prefix='functional-lifecycle-', dir=freeze.BASE / 'reviews') as temp:
            work = Path(temp)
            for s in (1, 2): write_output(work / f'frozen-stage-{s}-01', self.built[s])
            with patch.object(execution, 'WORK', work), patch.object(analysis, 'WORK', work), \
                 patch.object(execution, 'gpu_preflight', return_value={'synthetic_cpu_only': True}), \
                 patch.object(backend, 'PersistentNumericPool', return_value=pool), patch.object(backend, 'score_pass', side_effect=score), \
                 patch.object(backend, 'validate_sealed_pass', side_effect=lambda p, plan: read_json(p / 'manifest.json')), \
                 patch.object(backend, 'validate_geometry', side_effect=geometry), patch.object(backend, 'replica_proof', return_value={'synthetic_cpu_only': True}), \
                 patch('diagnostics.general_model_nolabel_execution.validate_runtime'), redirect_stdout(io.StringIO()):
                for s in (1, 2):
                    p, r, o = (work / f'{name}-stage-{s}-01' for name in ('frozen', 'run', 'results'))
                    execution.execute(p, r); execution.check_run(p, r)
                    analysis.analyze(p, r, o); analysis.analyze(p, r, o, check=True)
                    self.assertEqual(independent.results(p, r, o)['status'], 'passed')
                    with self.assertRaisesRegex(ValueError, 'terminal'): execution.execute(p, r)
                self.assertEqual(read_json(work / 'run-stage-2-01/stage-bridge-differences.json')['max_abs_error'], 0)
            self.assertEqual(len(calls), 9712); self.assertEqual(closed, [False, False])


if __name__ == '__main__': unittest.main()
