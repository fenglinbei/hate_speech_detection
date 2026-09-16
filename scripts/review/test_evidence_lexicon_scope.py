"""CPU checks of scientific contrasts, input isolation and the full run lifecycle."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.review import freeze_evidence_lexicon_scope as freeze
from scripts.review import run_evidence_lexicon_scope as execution
from scripts.review import analyze_evidence_lexicon_scope as analysis
from scripts.review import audit_evidence_lexicon_scope as independent
from scripts.review.test_evidence_label_calibration import block
from diagnostics import evidence_label_calibration_execution as backend
from diagnostics.evidence_label_calibration import compare_derived
from diagnostics.general_model_evidence_evaluation import json_bytes, read_json, read_lines, sha, write_output


class LexiconScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files, cls.sources, cls.audit = freeze.build()
        cls.plan = json.loads(cls.files['plan.json'])
        cls.contexts = [json.loads(s) for s in cls.files['contexts.jsonl'].splitlines()]
        cls.historical = [json.loads(s) for s in cls.files['historical-selected.jsonl'].splitlines()]
        cls.comps = [json.loads(s) for s in cls.files['comparisons.jsonl'].splitlines()]
        cls.source = json.loads(cls.files['materials.json'])
        cls.manifest = json_bytes({'schema_version': 'evidence-lexicon-scope-freeze/v1', 'status': 'frozen',
            'source_files': cls.sources, 'artifacts': {n: sha(b) for n, b in cls.files.items()}})

    def temporary_freeze(self, root):
        path = root / 'frozen-01'; write_output(path, {**self.files, 'manifest.json': self.manifest}); return path

    def test_dictionary_only_edit_and_no_inferred_human_adoption(self):
        self.assertIs(freeze.validate_materials(self.source), self.source)
        c = next(c for c in self.contexts if c['lexicon_arm'] == 'O')
        before = deepcopy(c)
        for arm in freeze.ARMS:
            messages = freeze.mutate_messages(c, arm, self.source)
            self.assertEqual(messages[0], c['messages'][0])
            self.assertEqual(messages[1]['content'].split('参考示例：\n')[1], c['messages'][1]['content'].split('参考示例：\n')[1])
        self.assertEqual(c, before)
        wrong = deepcopy(self.source); wrong['definition_notes'][0]['human_review'] = {'confirmed': True}
        with self.assertRaisesRegex(ValueError, 'invented'): freeze.validate_materials(wrong)
        wrong = deepcopy(self.source); wrong['original_entry_block'] += ' '
        with self.assertRaisesRegex(ValueError, 'hash'): freeze.validate_materials(wrong)

    def test_full_lengths_positions_and_history_with_independent_reconstruction(self):
        self.assertEqual((len(self.contexts), len(self.historical), len(self.comps)), (320, 32, 124))
        expected = {'O': 743, 'D': 710, 'N1': 743, 'E1': 743, 'N2': 743, 'E2': 743,
                    'P1': 757, 'X1': 757, 'P2': 756, 'X2': 756}
        for c in self.contexts:
            if c['encoding'] == 'original' and c['probe_id'] is None:
                self.assertEqual(c['prompt_tokens'], expected[c['lexicon_arm']])
        with tempfile.TemporaryDirectory(prefix='els-input-') as temp:
            path = self.temporary_freeze(Path(temp)); receipt = independent.audit_inputs(path)
            self.assertEqual(receipt['historical_payload_replays'], 32)
            self.assertEqual(receipt['geometry_proofs'], 352)

    def test_loader_does_not_parse_query_references_and_refuses_tampering(self):
        with tempfile.TemporaryDirectory(prefix='els-load-') as temp:
            path = self.temporary_freeze(Path(temp)); original = freeze.read_json
            def guard(p):
                self.assertNotEqual(Path(p).name, 'analysis_references.json'); return original(p)
            with patch.object(freeze, 'read_json', guard):
                plan, contexts, history = freeze.load_frozen(path)
                self.assertEqual(plan['plan_id'], self.plan['plan_id']); self.assertEqual(len(history), 32)
            with self.assertRaises(ValueError): write_output(path, self.files)
            (path / 'contexts.jsonl').write_bytes(self.files['contexts.jsonl'] + b'\n')
            with self.assertRaisesRegex(ValueError, 'artifact changed'): freeze.load_frozen(path)

    def test_pair_change_common_shift_reversal_zero_and_all_probe_views(self):
        settings = {'O': (-2., 0.), 'D': (-.5, 1.), 'N1': (0., 2.), 'N2': (-2., 1.),
                    'E1': (1., 2.), 'E2': (-2., 1.), 'P1': (-2., .3), 'X1': (-.1, 1.),
                    'P2': (-2., .3), 'X2': (2., 1.)}
        raw = []
        for c in self.contexts:
            delta, common = settings[c['lexicon_arm']]
            margin = .2 if c['probe_id'] is not None else common + delta * (.5 if c['form'] == 'H' else -.5)
            if c['encoding'] == 'ab_reverse': margin *= -1
            raw.append(block(c, margin, self.plan['catalog'][c['task']]))
        refs = [{'query_id': '3169', 'task': 'hate', 'original_label': 'non-hate', 'adjudicated_label': 'non-hate'}]
        rows, cal, effects, probes, summaries = analysis.analyze_rows(self.contexts, raw, refs, self.comps, self.plan['catalog'], self.plan['numeric_policy']['epsilon'])
        changes = analysis.pair_changes(effects, probes)
        index = {(r['template'], r['from_arm'], r['to_arm'], r['view']): r for r in changes}
        r = index[1, 'O', 'D', 'ncc']
        self.assertAlmostEqual(r['delta_before'], -2.); self.assertAlmostEqual(r['delta_after'], -.5)
        self.assertAlmostEqual(r['interaction'], 1.5); self.assertAlmostEqual(r['common_shift'], 1.)
        self.assertEqual(r['interaction_bound'], 8 * self.plan['numeric_policy']['epsilon'])
        self.assertEqual(r['common_shift_bound'], 4 * self.plan['numeric_policy']['epsilon'])
        self.assertEqual(index[1, 'O', 'N1', 'ncc']['descriptive_status'], 'pair_numerically_unresolved')
        self.assertEqual(index[1, 'O', 'N2', 'ncc']['descriptive_status'], 'common_shift_without_resolved_pair_change')
        self.assertTrue(index[2, 'P2', 'X2', 'ncc']['pair_direction_reversed'])
        self.assertIsNone(index[1, 'N1', 'E1', 'ncc']['signed_remaining_ratio'])
        self.assertAlmostEqual(index[1, 'O', 'D', 'ab_forward/answer_sum']['interaction'],
                               -index[1, 'O', 'D', 'ab_reverse/answer_sum']['interaction'])
        self.assertEqual(len(changes), 676)
        proof = compare_derived(self.contexts, raw, raw, self.comps, self.plan['numeric_policy']['epsilon'])
        self.assertEqual(proof['readouts'], 1968); self.assertTrue(proof['passed'])

    def test_unsealed_run_blocks_analysis_reference_read(self):
        with patch.object(execution, 'check_run', side_effect=ValueError('unsealed')), \
             patch.object(analysis, 'read_json', side_effect=AssertionError('premature reference read')):
            with self.assertRaisesRegex(ValueError, 'unsealed'):
                analysis.analyze(Path('/tmp/no-plan'), Path('/tmp/no-run'), Path('/tmp/no-output'))

    def test_allocation_and_busy_gpu_guard(self):
        def assignments(shift):
            rows = backend.partition_groups(self.contexts, self.plan['catalog'], 1, [1, 2], shift)
            self.assertEqual([len(r['contexts']) for r in rows], [160, 160])
            return {c['record_id']: r['physical_gpu_index'] for r in rows for c in r['contexts']}
        a, b = assignments(0), assignments(1)
        self.assertTrue(all(a[k] != b[k] for k in a))
        uuids = self.plan['execution_amendment']['selected_gpu_uuids']
        idle = '\n'.join(f'{i}, {uuids[str(i)]}, 0, 46068, 0' for i in (1, 2))
        with patch('subprocess.run', return_value=SimpleNamespace(stdout=idle)):
            self.assertEqual(len(execution.gpu_preflight(self.plan)['devices']), 2)
        for bad in (idle.replace(', 0, 46068, 0', ', 20000, 46068, 0', 1), idle.replace(uuids['1'], 'wrong')):
            with patch('subprocess.run', return_value=SimpleNamespace(stdout=bad)), self.assertRaises(ValueError): execution.gpu_preflight(self.plan)

    def test_eight_pass_lifecycle_checkpoint_decimal_audit_and_sealed_rerun(self):
        saved_score = backend.score_pass; historical = {r['record_id']: r for r in self.historical}
        calls, closed = [], []
        def candidates(c):
            row = deepcopy(historical.get(c['record_id']) or block(c, .2, self.plan['catalog'][c['task']]))
            for candidate in row['candidates']:
                candidate['reference_scores'] = {'token_logprobs': candidate['token_logprobs'], 'eos_logprob': candidate['eos_logprob']}
            return row['candidates']
        def scorer(runner, items, reference=False):
            calls.extend(i['context']['record_id'] for i in items)
            return [candidates(i['context'])[i['candidate']['ordinal']] for i in items]
        def prefix_scorer(runner, c, catalog, reference=False):
            calls.extend([c['record_id']] * len(catalog)); return candidates(c)
        pool = SimpleNamespace(identity={'test': 'synthetic_cpu_only'}, close=lambda terminate=False: closed.append(terminate))
        def score(pool, contexts, plan, output, **options):
            return saved_score(pool, contexts, plan, output, scorer=scorer, prefix_scorer=prefix_scorer, **options)
        def geometry(rows, contexts, plan, **options):
            self.assertEqual([r['record_id'] for r in rows], [c['record_id'] for c in contexts])
            return {'synthetic_cpu_test_only': True, 'blocks': len(rows), 'options': options}
        with tempfile.TemporaryDirectory(prefix='els-lifecycle-', dir=freeze.BASE / 'reviews') as temp:
            work = Path(temp); path = self.temporary_freeze(work); run = work / 'run-01'; result = work / 'results-01'
            with patch.object(execution, 'WORK', work), patch.object(analysis, 'WORK', work), \
                 patch.object(execution, 'gpu_preflight', return_value={'synthetic_cpu_only': True}), \
                 patch.object(backend, 'PersistentNumericPool', return_value=pool), patch.object(backend, 'score_pass', side_effect=score), \
                 patch.object(backend, 'validate_sealed_pass', side_effect=lambda p, plan: read_json(p / 'manifest.json')), \
                 patch.object(backend, 'validate_geometry', side_effect=geometry), patch.object(backend, 'replica_proof', return_value={'synthetic_cpu_only': True}), \
                 patch('diagnostics.general_model_nolabel_execution.validate_runtime'), redirect_stdout(io.StringIO()):
                execution.execute(path, run)
                self.assertEqual(len(calls), 3968); self.assertEqual(closed, [False])
                execution.check_run(path, run)
                analysis.analyze(path, run, result); analysis.analyze(path, run, result, check=True)
                receipt = independent.audit_results(path, run, result)
                self.assertGreater(receipt['scalar_checks'], 10000)
                self.assertEqual(read_json(result / 'summary.json')['paired_change_rows'], 676)
                score(pool, self.contexts, self.plan, run / self.plan['raw_pass'], batch_size=1, reference=True)
                self.assertEqual(len(calls), 3968)
                with self.assertRaisesRegex(ValueError, 'terminal run is sealed'): execution.execute(path, run)
                state_path = run / 'run_manifest.json'; state = read_json(state_path)
                state['checks'][0]['limit'] *= 2; state_path.write_bytes(json_bytes(state))
                with self.assertRaisesRegex(ValueError, 'raw gate receipt differs'): execution.check_run(path, run)


if __name__ == '__main__': unittest.main()
