"""Reject confounded swaps and unexplained downstream token displacement."""
from copy import deepcopy
import unittest

from scripts.review.prepare_evidence_position_length_materials import length_proof, swap_proof


def swap_fixture():
    before = {
        'demo_ids': ['target', 'partner'],
        'texts_by_demo': {'target': 'one', 'partner': 'two'},
        'answers_by_demo': {'target': 'hate', 'partner': 'hate'},
        'prompt_token_ids': [1, 2, 101, 102, 4, 5, 6, 201, 202, 4, 8, 9],
        'layout': {'demos': {
            'target': {'slot': 1, 'block': [0, 5], 'text': [2, 4], 'answer': [4, 5]},
            'partner': {'slot': 2, 'block': [5, 10], 'text': [7, 9], 'answer': [9, 10]},
        }, 'query_block': [10, 11], 'generation_start': 12},
    }
    after = deepcopy(before)
    after['demo_ids'] = ['partner', 'target']
    after['prompt_token_ids'][2:4], after['prompt_token_ids'][7:9] = [201, 202], [101, 102]
    after['layout']['demos']['target'], after['layout']['demos']['partner'] = (
        deepcopy(before['layout']['demos']['partner']), deepcopy(before['layout']['demos']['target']))
    return before, after


class PositionLengthTests(unittest.TestCase):
    def test_swap_preserves_labels_at_slots_and_marks_joint_scope(self):
        before, after = swap_fixture()
        proof = swap_proof(before, after, 'target', 'partner')
        self.assertTrue(proof['passed'])
        self.assertFalse(proof['absolute_single_example_position_isolated'])
        before['answers_by_demo']['partner'] = after['answers_by_demo']['partner'] = 'non-hate'
        with self.assertRaisesRegex(ValueError, 'labels by position'):
            swap_proof(before, after, 'target', 'partner')

    def test_equal_total_length_does_not_hide_a_moved_slot_or_query(self):
        for field in ('slot', 'query'):
            with self.subTest(field=field):
                before, after = swap_fixture()
                if field == 'slot':
                    after['layout']['demos']['target']['answer'] = [8, 9]
                else:
                    after['layout']['query_block'] = [9, 10]
                with self.assertRaises(ValueError):
                    swap_proof(before, after, 'target', 'partner')

    def test_swap_cannot_edit_payloads_or_other_tokens(self):
        for kind in ('text', 'query_token', 'moved_token'):
            with self.subTest(kind=kind):
                before, after = swap_fixture()
                if kind == 'text':
                    after['texts_by_demo']['target'] = 'different'
                elif kind == 'query_token':
                    after['prompt_token_ids'][10] = 99
                else:
                    after['prompt_token_ids'][7] = 99
                with self.assertRaises(ValueError):
                    swap_proof(before, after, 'target', 'partner')

    def test_length_shift_checks_every_downstream_answer_and_query(self):
        before, _ = swap_fixture()
        after = deepcopy(before)
        after['texts_by_demo']['target'] = 'one expanded'
        after['prompt_token_ids'][4:4] = [103, 104, 105, 106]
        after['layout']['demos']['target']['text'][1] += 4
        after['layout']['demos']['target']['block'][1] += 4
        after['layout']['demos']['target']['answer'] = [8, 9]
        for key in ('block', 'text', 'answer'):
            after['layout']['demos']['partner'][key] = [v + 4 for v in before['layout']['demos']['partner'][key]]
        after['layout']['query_block'] = [14, 15]
        after['layout']['generation_start'] = 16
        self.assertEqual(length_proof(before, after, 'target')['token_delta'], 4)
        after['layout']['query_block'] = [13, 14]
        with self.assertRaisesRegex(ValueError, 'query shift'):
            length_proof(before, after, 'target')

    def test_length_proof_cannot_hide_a_second_material_change(self):
        before, after = swap_fixture()
        after['demo_ids'] = before['demo_ids'][:]
        after['texts_by_demo'] = {'target': 'changed one', 'partner': 'changed two'}
        with self.assertRaisesRegex(ValueError, 'more than its selected text'):
            length_proof(before, after, 'target')


if __name__ == '__main__':
    unittest.main()
