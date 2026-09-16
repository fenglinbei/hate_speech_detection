"""Analysis invariants using synthetic candidate scores, never human records."""
import unittest

from scripts.review.analyze_evidence_interventions import condition_metrics, contrast, direction
from diagnostics.general_model_numeric_analysis import candidate_catalog, candidate_scores


def block(task, values):
    candidates = []
    for canonical, logprobs in zip(candidate_catalog()[task], values, strict=True):
        candidates.append({**canonical, 'answer_token_ids': list(range(1, len(logprobs) + 1)),
                           'token_logprobs': logprobs, 'scores': candidate_scores(logprobs, -0.1)})
    return {'task': task, 'candidates': candidates}


def context(task):
    return {'record_id': 'test', 'protocol_id': 'test', 'query_id': 'synthetic', 'task': task,
            'condition': 'test', 'prompt_tokens': 8, 'prompt_sha256': 'a' * 64, 'baseline_replay': True}


class AnalysisTests(unittest.TestCase):
    def test_group_fixed_foil_does_not_follow_best_other_candidate(self):
        protocol = {'original_reference_label': [], 'reference_label': [],
                    'group_foil': {'labels': ['Racism'], 'ordinal': 1}}
        scores = [[-20.0] for _ in range(32)]
        scores[:3] = [[-4.0], [-2.0], [-1.0]]
        result = condition_metrics(block('group', scores), protocol, context('group'), candidate_catalog()['group'], 'answer_sum')
        self.assertEqual(result['reviewed_margin'], -3.0)
        self.assertEqual(result['fixed_foil_margin'], -2.0)
        self.assertEqual(result['prediction'], ['Region'])
        self.assertEqual(result['fixed_foil'], ['Racism'])

    def test_empty_reference_and_canonical_tie(self):
        protocol = {'original_reference_label': [], 'reference_label': [],
                    'group_foil': {'labels': ['Racism'], 'ordinal': 1}}
        scores = [[-20.0] for _ in range(32)]
        scores[:2] = [[-1.0], [-1.0]]
        result = condition_metrics(block('group', scores), protocol, context('group'), candidate_catalog()['group'], 'answer_sum')
        self.assertEqual(result['prediction'], [])
        self.assertEqual(result['tied_top_count'], 2)
        self.assertTrue(result['reviewed_correct'])

    def test_auxiliary_length_normalization_can_change_prediction(self):
        protocol = {'original_reference_label': 'hate', 'reference_label': 'non-hate', 'group_foil': None}
        row = block('hate', [[-1.0, -1.0], [-1.5]])
        result = [condition_metrics(row, protocol, context('hate'), candidate_catalog()['hate'], mode)
                  for mode in ('answer_sum', 'answer_mean')]
        self.assertEqual([r['prediction'] for r in result], ['non-hate', 'hate'])
        self.assertEqual([r['reviewed_correct'] for r in result], [True, False])

    def test_four_cell_interaction_and_group_conservative_bound(self):
        protocol = {'protocol_id': 'test', 'query_id': 'synthetic', 'task': 'group',
                    'operation': 'synthetic', 'strict_U_branch': False}
        rows = {condition: {'hate_direction': None, 'original_margin': value, 'reviewed_margin': value,
                            'fixed_foil_margin': value, 'label_margins': {'Racism': value}, 'score_mode': 'answer_sum'}
                for condition, value in (('T0_L0', 1), ('T1_L0', 2), ('T0_L1', 3), ('T1_L1', 8))}
        effect = contrast(protocol, rows, {'T1_L1': 1, 'T1_L0': -1, 'T0_L1': -1, 'T0_L0': 1}, 'interaction', .01)
        self.assertEqual(effect['effects']['reviewed_margin']['value'], 4)
        self.assertEqual(effect['effects']['fixed_foil_margin']['numeric_bound'], .08)
        self.assertEqual(effect['effects']['label_margin/Racism']['numeric_bound'], .04)

    def test_zero_and_reverse_effects_are_retained(self):
        self.assertEqual(direction(0.0, .01)['direction'], 'numerically_unresolved')
        self.assertEqual(direction(-2, .01)['value'], -2)
        self.assertEqual(direction(-2, .01)['direction'], 'negative')


if __name__ == '__main__':
    unittest.main()
