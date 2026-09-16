"""Fail-closed checks for the authorized intervention frame and numerical gates."""
from copy import deepcopy
import unittest

from scripts.review.run_evidence_interventions import PRIORITY, gate, scoring_contexts
from data.stage1_data import canonical_json_sha256


def prepared_frame():
    rows = []
    for query in PRIORITY:
        for task in ('hate', 'group'):
            count = 4 if (query, task) == ('5086', 'hate') else 2
            for index in range(count):
                rows.append({'record_id': f'{query}:{task}:{index}', 'query_id': query,
                             'task': task, 'condition': str(index), 'protocol_id': f'{query}:{task}',
                             'baseline_replay': index == 0, 'tokenization_status': 'verified',
                             'prompt_token_ids': [1, 2], 'prompt_tokens': 2,
                             'prompt_token_ids_sha256': canonical_json_sha256([1, 2])})
    return rows


class ExecutionBoundaryTests(unittest.TestCase):
    def test_prepared_frame_retains_all_arms_and_original_objects(self):
        original = prepared_frame()
        saved = deepcopy(original)
        actual = scoring_contexts(original)
        self.assertEqual(original, saved)
        self.assertEqual(len(actual), 26)
        self.assertEqual({r['record_id'] for r in actual}, {r['record_id'] for r in original})
        self.assertTrue(all(len(r['context_sha256']) == 64 for r in actual))

    def test_incomplete_or_duplicate_conditions_rejected(self):
        rows = prepared_frame()
        with self.assertRaises(ValueError):
            scoring_contexts(rows[:-1])
        rows[-1] = rows[-2]
        with self.assertRaises(ValueError):
            scoring_contexts(rows)

    def test_baseline_per_protocol_is_required(self):
        rows = prepared_frame()
        rows[0]['baseline_replay'] = False
        rows[3]['baseline_replay'] = True
        with self.assertRaises(ValueError):
            scoring_contexts(rows)

    def test_query_reference_labels_cannot_enter_worker_contexts(self):
        rows = prepared_frame()
        rows[0]['reviewed_label'] = 'non-hate'
        with self.assertRaises(ValueError):
            scoring_contexts(rows)

    def test_token_drift_and_out_of_scope_queries_rejected(self):
        for field, value in (('prompt_token_ids', [1, 3]), ('query_id', 'reserved-query')):
            rows = prepared_frame()
            rows[0][field] = value
            with self.assertRaises(ValueError):
                scoring_contexts(rows)

    def test_failed_and_nonfinite_numerical_gates_stop(self):
        gate({'max_abs_error': 1e-4}, 1e-4, 'repeat')
        for error in (1.0001e-4, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                gate({'max_abs_error': error}, 1e-4, 'repeat')


if __name__ == '__main__':
    unittest.main()
