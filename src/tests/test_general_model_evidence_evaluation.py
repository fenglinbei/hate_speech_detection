"""Scientific-contract and failure-path tests; all review records are synthetic."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from diagnostics import general_model_evidence_evaluation as ev
from diagnostics.general_model_numeric_analysis import candidate_scores, _validated_candidates


def fixture():
    cards, overlays, eligibility, raw = {}, {}, {}, {}
    q = 'synthetic-query'
    cards[q] = {'profile': {'conditions': {}}, 'contexts': []}
    for task in ev.TASKS:
        old = 'hate' if task == 'hate' else ['Racism']
        reviewed = 'non-hate' if task == 'hate' else []
        overlays[q, task] = {
            'original_label': old, 'adjudicated_label': reviewed, 'query_material_label': reviewed,
            'text_sha256': 'synthetic-text', 'native_case_policy': {'version': 'synthetic-case'},
            'query_material_policy': {'version': 'synthetic-material'}, 'original_status': 'suspected_error',
        }
        eligibility[q, task] = {
            'reference_analysis_eligible': True, 'unavailable_reasons': [], 'original_bucket': 'synthetic',
            'native_eligibility': {'status': 'confirmed', 'availability': 'resolved'},
        }
        for condition in ev.CONDITIONS:
            winner = 1 if task == 'hate' else 0
            labels = [reviewed] if task == 'hate' else reviewed
            ctx = {'task': task, 'condition': condition, 'context_sha256': 'context:' + condition,
                   'prompt_sha256': 'prompt:' + task + condition, 'prompt_token_ids_sha256': 'tokens',
                   'prompt_tokens': 20}
            candidates = ev.candidate_catalog()[task]
            for candidate in candidates:
                value = -1.0 if candidate['ordinal'] == winner else -3.0
                candidate.update(answer_token_ids=[10], token_logprobs=[value],
                                 scores=candidate_scores([value], -0.5),
                                 prompt_token_ids_sha256='tokens', prompt_tokens=20)
            raw[q, task, condition] = {
                'query_id': q, 'task': task, 'condition': condition, 'record_id': f'{q}:{task}:{condition}',
                'plan_id': 'synthetic-plan', 'pass_name': 'dev-b1', 'repetition': 0,
                'context_sha256': ctx['context_sha256'], 'prompt_sha256': ctx['prompt_sha256'],
                'candidates': candidates,
            }
            cards[q]['contexts'].append(ctx)
            cards[q]['profile']['conditions'].setdefault(condition, {})[task] = {
                'prediction': {'labels': labels, 'ordinal': winner, 'top_score_gap': 2.0,
                               'tied_top_count': 1, 'within_two_epsilon': False},
                'correct': False, 'readouts': {'answer_sum/gold/best_nongold_margin': -2.0},
                'context': ctx, 'score_mode_predictions': {mode: labels for mode in ev.SCORE_MODES},
                'score_mode_sensitive': False,
            }
    return cards, overlays, eligibility, raw


class ScientificContractTests(unittest.TestCase):
    def test_changed_group_margin_requires_full_competitor_set(self):
        candidates = ev.candidate_catalog()['group']
        for c in candidates:
            c['scores'] = {'answer_sum': {0: -3.0, 1: -1.0, 2: -2.0}.get(c['ordinal'], -10.0)}
        self.assertEqual(ev.margin('group', candidates, ['Racism']), 1.0)
        self.assertEqual(ev.margin('group', candidates, []), -2.0)
        self.assertIsNone(ev.margin('group', candidates, None))

    def test_exact_ties_and_near_ties_keep_historical_argmax(self):
        candidates = [{'ordinal': i, 'labels': [str(i)], 'scores': {'answer_sum': -1.0}} for i in (5, 2, 1)]
        tied = ev.ranking(candidates, 0.001)
        self.assertEqual(tied['ordinal'], 1)
        self.assertEqual(tied['tied_top_count'], 3)
        candidates[0]['scores']['answer_sum'] += 0.0001
        near = ev.ranking(candidates, 0.001)
        self.assertEqual(near['ordinal'], 5)
        self.assertEqual(near['tied_top_count'], 1)
        self.assertTrue(near['within_two_epsilon'])

    def test_metrics_against_hand_counted_confusions(self):
        hate = ev.metrics('hate', ['hate', 'hate', 'non-hate'], ['hate', 'non-hate', 'non-hate'])
        self.assertAlmostEqual(hate['exact_accuracy'], 2 / 3)
        self.assertAlmostEqual(hate['macro_f1'], 2 / 3)
        group = ev.metrics('group', [[], ['Racism', 'Sexism'], ['others']], [[], ['Sexism'], []])
        self.assertEqual(group['exact_accuracy'], 1 / 3)
        self.assertEqual(group['macro_f1'], 0.2)
        self.assertEqual(group['micro_f1'], 0.5)
        self.assertEqual(group['per_label'][0]['fp'], 1)
        self.assertIsNone(ev.metrics('group', [], [])['micro_f1'])

    def test_empty_group_valid_unknown_hate_excluded_from_both_references(self):
        cards, overlays, eligible, raw = fixture()
        q = next(iter(cards))
        overlays[q, 'hate']['adjudicated_label'] = None
        eligible[q, 'hate'].update(reference_analysis_eligible=False, unavailable_reasons=['unresolved'])
        eligible[q, 'hate']['native_eligibility']['availability'] = 'unresolved'
        blocks = ev.evaluate_blocks(cards, overlays, eligible, raw, 0.001)
        tables = ev.aggregate(blocks, [q])
        hate = tables['coverage'][0]
        self.assertEqual((hate['queue_n'], hate['paired_reference_n'], hate['confirmed_n'], hate['unresolved_n']), (1, 0, 1, 1))
        for row in tables['metrics']:
            self.assertEqual(row['n'], 0 if row['task'] == 'hate' else 1)
        groups = [b for b in blocks if b['task'] == 'group']
        self.assertTrue(all(b['reviewed_correct'] and b['revised_margin'] == 2.0 for b in groups))
        self.assertTrue(all(b['reviewed_correct'] is None and b['revised_margin'] is None for b in blocks if b['task'] == 'hate'))
        self.assertEqual(len(tables['core_mask_counts']), 64)
        self.assertEqual(len(tables['core_mask_transitions']), 512)
        changed = [r for r in tables['core_mask_transitions'] if r['count']]
        self.assertEqual(changed, [{'task': 'group', 'original_mask': '0000', 'reviewed_mask': '1111', 'count': 1}])

    def test_missing_candidates_preserve_discrete_results_with_null_margins(self):
        cards, overlays, eligible, raw = fixture()
        scored = ev.evaluate_blocks(cards, overlays, eligible, raw, 0.001)
        missing = ev.evaluate_blocks(cards, overlays, eligible, {}, 0.001)
        for a, b in zip(scored, missing):
            for k in ('prediction', 'original_correct', 'reviewed_correct', 'correctness_relabel_transition'):
                self.assertEqual(a[k], b[k])
            self.assertIsNone(b['revised_margin'])
            self.assertIsNone(b['original_margin'])
            self.assertIsNone(b['hate_score_direction'])
            self.assertEqual(b['continuous_status'], 'candidate_scores_missing')
        tables = ev.aggregate(missing, list(cards))
        self.assertTrue(all(c['SD_minus_S_minus_D_plus_0'] is None for c in tables['margin_interactions']))

    def test_context_or_saved_prediction_drift_rejected(self):
        for mutation in ('context', 'prediction', 'sensitivity'):
            with self.subTest(mutation=mutation):
                cards, overlays, eligible, raw = fixture()
                q = next(iter(cards))
                if mutation == 'context':
                    raw[q, 'group', 'C0']['candidates'][2]['prompt_token_ids_sha256'] = 'different'
                elif mutation == 'prediction':
                    cards[q]['profile']['conditions']['C0']['group']['prediction']['ordinal'] = 1
                else:
                    cards[q]['profile']['conditions']['C0']['group']['score_mode_sensitive'] = True
                with self.assertRaises(ValueError):
                    ev.evaluate_blocks(cards, overlays, eligible, raw, 0.001)

    def test_selected_raw_scope_and_complete_matrix(self):
        cards, _, _, raw = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'raw.jsonl'
            # An unselected payload is deliberately undecodable; only its ID may be read.
            path.write_bytes(b'{"query_id":"unselected","candidates":not-decoded}\n' + ev.jsonl(raw.values()))
            selected = ev.selected_raw(path, set(cards), 'synthetic-plan')
            self.assertEqual(len(selected), 12)
            row = next(iter(raw.values()))
            with path.open('ab') as stream:
                stream.write(ev.jsonl([row]))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                ev.selected_raw(path, set(cards), 'synthetic-plan')
            path.write_bytes(ev.jsonl(list(raw.values())[:-1]))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                ev.selected_raw(path, set(cards), 'synthetic-plan')

    def test_group_candidates_cannot_be_partial_reordered_or_corrupt(self):
        candidates = fixture()[3]['synthetic-query', 'group', 'C0']['candidates']
        for mutation in ('missing', 'order', 'score'):
            changed = deepcopy(candidates)
            if mutation == 'missing':
                changed.pop()
            elif mutation == 'order':
                changed.reverse()
            else:
                changed[3]['scores']['answer_sum'] = -99
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                _validated_candidates('group', changed)


class IntegrityAndCpuTests(unittest.TestCase):
    def test_cpu_only_config_and_import(self):
        config = ev.read_json(ROOT / 'config/stage1/general_model_evidence_dual_reference_v1.json')
        ev.validate_config(config)
        for flag in ('allow_gpu', 'allow_model_forward', 'allow_test_or_reserve_analysis'):
            changed = {**config, flag: True}
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'CPU-only'):
                ev.validate_config(changed)
        code = ("import sys; sys.path.insert(0, 'src'); "
                "import diagnostics.general_model_evidence_evaluation; "
                "assert not {'torch', 'transformers', 'numpy'} & set(sys.modules)")
        subprocess.run([sys.executable, '-S', '-c', code], cwd=ROOT, check=True)

    def test_hash_mismatch_does_not_fall_back_to_missing_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'scores.jsonl'
            path.write_text('corrupt')
            sources = ev.Sources(root)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                sources.verify(path, '0' * 64)

    def test_strict_json_and_duplicate_rows(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                ev.loads(raw)
        with self.assertRaisesRegex(ValueError, 'duplicate row'):
            ev.unique([{'id': 1}, {'id': 1}], lambda v: v['id'])
        with self.assertRaises(ValueError):
            ev.label('group', ['others', 'Racism'])
        self.assertEqual(ev.label('group', []), [])
        self.assertIsNone(ev.label('group', None, nullable=True))

    def test_atomic_export_no_overwrite_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'result'
            payload = b'original\n'
            manifest = {'schema_version': 'evidence-dual-reference-run/v1', 'status': 'complete',
                        'artifacts': {'data.jsonl': ev.sha(payload)}}
            ev.write_output(target, {'data.jsonl': payload, 'manifest.json': ev.json_bytes(manifest)})
            ev.verify_output(target)
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                ev.write_output(target, {'data.jsonl': b'changed'})
            self.assertEqual((target / 'data.jsonl').read_bytes(), payload)
            (target / 'data.jsonl').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                ev.verify_output(target)


if __name__ == '__main__':
    unittest.main()
