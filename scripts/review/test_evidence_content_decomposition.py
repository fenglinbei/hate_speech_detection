"""CPU-only content protocol, input integrity, and synthetic executor acceptance.

Synthetic scorer tests exercise orchestration/checkpoint/math, not GPU numerics.
All synthetic run records live in temporary directories and are removed.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.review import freeze_evidence_content_decomposition as freeze
from scripts.review import run_evidence_content_decomposition as execution
from scripts.review import analyze_evidence_content_decomposition as analysis
from scripts.review.test_evidence_label_calibration import block
from diagnostics import evidence_label_calibration_execution as backend
from diagnostics.evidence_label_calibration import compare_derived
from diagnostics.general_model_evidence_evaluation import json_bytes, read_json, read_lines, sha, write_output


class ContentDecompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files, cls.sources, cls.audit = freeze.build()
        cls.plan = json.loads(cls.files['plan.json'])
        cls.contexts = [json.loads(line) for line in cls.files['contexts.jsonl'].splitlines()]
        cls.historical = [json.loads(line) for line in cls.files['historical-selected.jsonl'].splitlines()]
        cls.comps = [json.loads(line) for line in cls.files['comparisons.jsonl'].splitlines()]
        cls.manifest = json_bytes({'schema_version': 'evidence-content-decomposition-freeze/v1', 'status': 'frozen',
            'source_files': cls.sources, 'artifacts': {name: sha(raw) for name, raw in cls.files.items()}})

    def temporary_freeze(self, root):
        target = root / 'frozen-01'
        write_output(target, {**self.files, 'manifest.json': self.manifest})
        return target

    def test_semantic_pair_invariants_and_no_inferred_human_adoption(self):
        source = json.loads(self.files['materials.json']); self.assertEqual(len(freeze.review_materials(source)), 16)
        wrong = deepcopy(source); wrong['materials'][0]['presented_answer'] = 'hate'
        with self.assertRaisesRegex(ValueError, 'mismatch'): freeze.review_materials(wrong)
        wrong = deepcopy(source); wrong['materials'][0]['human_review']['naturalness'] = True
        with self.assertRaisesRegex(ValueError, 'invented'): freeze.review_materials(wrong)
        wrong = deepcopy(source); row = wrong['materials'][-1]; row['text'] += '哈'; row['text_sha256'] = sha(row['text'].encode())
        with self.assertRaisesRegex(ValueError, 'unrelated wording'): freeze.review_materials(wrong)

    def test_full_prompt_external_anchors_and_encoding_probe_frame(self):
        self.assertEqual((len(self.contexts), len(self.historical), len(self.comps)), (208, 80, 30))
        primary = [c for c in self.comps if c['role'] == 'primary']
        self.assertEqual([sum(c['query_id'] == q for c in primary) for q in ('541', '3169')], [10, 4])
        self.assertFalse(any('interaction' in c['kind'] for c in primary if c['query_id'] == '3169'))
        new = [c for c in self.contexts if not c['baseline_replay'] and c['encoding'] == 'original' and c['probe_id'] is None]
        self.assertEqual({c['prompt_tokens'] for c in new if c['query_id'] == '541'}, {918})
        self.assertEqual({c['prompt_tokens'] for c in new if c['query_id'] == '3169'}, {743})
        self.assertEqual(self.audit['matched_contrast_variant_proofs'], 112)
        self.assertEqual(self.audit['full_label_boundaries'], 416)
        self.assertFalse(self.audit['model_forward_executed'])

    def test_loader_never_parses_query_references_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory(prefix='ecd-test-') as temp:
            path = self.temporary_freeze(Path(temp)); original_read = freeze.read_json
            def guard(p):
                self.assertNotEqual(Path(p).name, 'analysis_references.json')
                return original_read(p)
            with patch.object(freeze, 'read_json', guard):
                plan, contexts, history = freeze.load_frozen(path)
                self.assertEqual(plan['plan_id'], self.plan['plan_id']); self.assertEqual(len(history), 80)
            with self.assertRaises(ValueError): write_output(path, self.files)
            p = path / 'contexts.jsonl'; p.write_bytes(p.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'artifact changed'): freeze.load_frozen(path)

    def test_synthetic_analysis_keeps_conditional_interaction_reverse_and_null(self):
        raw = []
        for c in self.contexts:
            condition = c['root_condition']; cell = condition.split('-', 1)[-1]
            if c['query_id'] == '541':
                real = {'T0R0': 0., 'T1R0': 1., 'T0R1': 2., 'T1R1': 4.}.get(cell, 0.)
                prior = 1. if cell in ('T0R1', 'T1R1') else 0.
            else:
                real = {'G_H': -3., 'G_E': 0., 'L_H': 1., 'L_O': 0.}.get(cell, 0.)
                if condition == 'F2-L_H': real = 0.
                prior = 0.
            margin = prior if c['probe_id'] is not None else real
            if c['encoding'] == 'ab_reverse': margin = -margin
            raw.append(block(c, margin, self.plan['catalog'][c['task']]))
        refs = [{'query_id': q, 'task': 'hate', 'original_label': 'non-hate', 'adjudicated_label': 'non-hate'} for q in ('541', '3169')]
        rows, cal, effects, probes, summaries = analysis.analyze_rows(self.contexts, raw, refs, self.comps, self.plan['catalog'], self.plan['numeric_policy']['epsilon'])
        self.assertEqual((len(rows), len(cal), len(effects), len(probes), len(summaries)), (312, 26, 480, 300, 30))
        lookup = {(r['contrast_id'], r['view']): r for r in effects}
        interaction = lookup['ECD-541-F1-topic_rule_interaction', 'ncc']
        self.assertAlmostEqual(interaction['effect'], 1.)
        self.assertEqual(interaction['numeric_bound'], 8 * self.plan['numeric_policy']['epsilon'])
        self.assertEqual(lookup['ECD-3169-F1-group_designation_substitution', 'ncc']['direction'], 'negative')
        self.assertEqual(lookup['ECD-3169-F1-ordinary_laughter_form_substitution', 'ncc']['direction'], 'positive')
        self.assertEqual(lookup['ECD-3169-F2-ordinary_laughter_form_substitution', 'ncc']['direction'], 'unresolved')
        self.assertFalse(next(r for r in summaries if r['contrast_id'] == 'ECD-3169-F1-group_designation_substitution')['ab_directions_agree_resolved'])
        for c in self.comps:
            values = {v: lookup[c['contrast_id'], v]['effect'] for v in ('original/answer_mean', 'background', 'ncc')}
            self.assertAlmostEqual(values['original/answer_mean'] - values['background'], values['ncc'])
        derived = compare_derived(self.contexts, raw, raw, self.comps, self.plan['numeric_policy']['epsilon'])
        self.assertEqual(derived['readouts'], 672); self.assertTrue(derived['passed'])

    def test_unsealed_or_failed_run_blocks_reference_loading(self):
        with patch.object(execution, 'check_run', side_effect=ValueError('unsealed')), patch.object(analysis, 'read_json', side_effect=AssertionError('reference read before gate')):
            with self.assertRaisesRegex(ValueError, 'unsealed'): analysis.analyze(Path('/tmp/unsealed'), Path('/tmp/no-run'), Path('/tmp/no-results'))

    def test_candidate_partition_covers_each_context_once_and_changes_replica(self):
        assignments = backend.partition_groups(self.contexts, self.plan['catalog'], 1, [0, 1, 2, 3])
        shifted = backend.partition_groups(self.contexts, self.plan['catalog'], 1, [0, 1, 2, 3], 1)
        def mapping(rows): return {c['record_id']: s['physical_gpu_index'] for s in rows for c in s['contexts']}
        a, b = mapping(assignments), mapping(shifted)
        self.assertEqual(len(a), 208); self.assertTrue(all(a[k] != b[k] for k in a))
        self.assertEqual([len(s['contexts']) for s in assignments], [52, 52, 52, 52])

    def test_complete_eight_pass_cpu_synthetic_lifecycle_and_sealed_rerun(self):
        # Actual checkpoint/scoring orchestration and all calibration gates, with
        # explicitly synthetic logits. Hardware/runtime/geometry validators are
        # replaced only here; this test makes no physical GPU validation claim.
        saved_score = backend.score_pass
        by_id = {r['record_id']: r for r in self.historical}
        calls, closed = [], []
        def candidates(c):
            row = deepcopy(by_id.get(c['record_id']) or block(c, .2, self.plan['catalog'][c['task']]))
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
        with tempfile.TemporaryDirectory(prefix='ecd-lifecycle-', dir=freeze.BASE / 'reviews') as temp:
            work = Path(temp); path = self.temporary_freeze(work); run = work / 'run-01'
            with patch.object(execution, 'WORK', work), patch.object(analysis, 'WORK', work), \
                 patch.object(backend, 'PersistentNumericPool', return_value=pool), \
                 patch.object(backend, 'score_pass', side_effect=score), \
                 patch.object(backend, 'validate_sealed_pass', side_effect=lambda output, plan: read_json(output / 'manifest.json')), \
                 patch.object(backend, 'validate_geometry', side_effect=geometry), \
                 patch.object(backend, 'replica_proof', return_value={'synthetic_cpu_test_only': True}), \
                 patch('diagnostics.general_model_nolabel_execution.validate_runtime'), redirect_stdout(io.StringIO()):
                execution.execute(path, run)
                self.assertEqual(len(calls), 2816); self.assertEqual(closed, [False])
                execution.check_run(path, run)
                analysis.analyze(path, run, work / 'results-01')
                analysis.analyze(path, run, work / 'results-01', check=True)
                summary = read_json(work / 'results-01/summary.json')
                self.assertEqual(summary['comparison_counts_by_role'], {'primary': 14, 'bridge': 12, 'historical_anchor': 4})
                self.assertEqual(len(calls), 2816)
                # The sealed pass itself can be replayed from checkpoints with
                # no new synthetic calls; the terminal whole run rejects run.
                score(pool, self.contexts, self.plan, run / self.plan['raw_pass'], batch_size=1, reference=True)
                self.assertEqual(len(calls), 2816)
                with self.assertRaisesRegex(ValueError, 'terminal run is sealed'): execution.execute(path, run)
                state_path = run / 'run_manifest.json'; state = read_json(state_path)
                state['checks'][0]['limit'] *= 2; state_path.write_bytes(json_bytes(state))
                with self.assertRaisesRegex(ValueError, 'raw gate receipt differs'): execution.check_run(path, run)


if __name__ == '__main__': unittest.main()
