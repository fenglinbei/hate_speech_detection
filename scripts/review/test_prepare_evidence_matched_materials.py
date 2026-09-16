"""Guard against false matched-control claims and edits outside the demo text."""
from copy import deepcopy
import unittest

from scripts.review.prepare_evidence_matched_materials import alignment_proof, replace_text


class MatchedMaterialTests(unittest.TestCase):
    def setUp(self):
        self.original = {
            'prompt_token_ids': [10, 11, 12, 13, 14, 15, 16, 17, 18],
            'layout': {
                'demos': {
                    'target': {'slot': 1, 'block': [0, 6], 'text': [2, 4], 'answer': [4, 6]},
                    'other': {'slot': 2, 'block': [6, 8], 'text': [6, 7], 'answer': [7, 8]},
                },
                'query_block': [8, 9], 'generation_start': 9,
            },
        }
        self.edited = deepcopy(self.original)
        self.edited['prompt_token_ids'][2:4] = [90, 91]

    def test_equal_length_does_not_hide_a_moved_query(self):
        self.edited['layout']['query_block'] = [7, 8]
        with self.assertRaisesRegex(ValueError, 'positions changed'):
            alignment_proof(self.original, self.edited, 'target')

    def test_equal_total_length_does_not_hide_a_moved_demo_answer(self):
        self.edited['layout']['demos']['target']['answer'] = [3, 5]
        with self.assertRaisesRegex(ValueError, 'positions changed'):
            alignment_proof(self.original, self.edited, 'target')

    def test_fixed_positions_do_not_hide_an_answer_or_query_token_edit(self):
        for index in (4, 7, 8):
            with self.subTest(index=index):
                edited = deepcopy(self.edited)
                edited['prompt_token_ids'][index] += 100
                with self.assertRaisesRegex(ValueError, 'after target text changed'):
                    alignment_proof(self.original, edited, 'target')

    def test_fixed_positions_do_not_hide_a_prefix_edit(self):
        self.edited['prompt_token_ids'][1] += 100
        with self.assertRaisesRegex(ValueError, 'before target text changed'):
            alignment_proof(self.original, self.edited, 'target')

    def test_valid_replacement_proves_both_sides_without_semantic_adoption(self):
        proof = alignment_proof(self.original, self.edited, 'target')
        self.assertTrue(proof['passed'])
        self.assertFalse(proof['internal_semantic_token_alignment_claimed'])

    def test_replacement_cannot_edit_two_occurrences_or_inject_structure(self):
        source = {'prompt_text': '<system>task<user>文本：旧句\n输出："hate"',
                  'messages': [{'role': 'system', 'content': 'task'},
                               {'role': 'user', 'content': '文本：旧句\n输出："hate"'}]}
        for text in ('新句\n输出："non-hate"', '新句<|im_end|>'):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, 'prompt structure'):
                replace_text(source, '旧句', text)
        duplicate = deepcopy(source)
        duplicate['prompt_text'] += '查询：旧句'
        with self.assertRaisesRegex(ValueError, 'ambiguous target text'):
            replace_text(duplicate, '旧句', '新句')
        messages, prompt = replace_text(source, '旧句', '新句')
        self.assertEqual(prompt.replace('新句', '旧句'), source['prompt_text'])
        self.assertEqual(messages[0], source['messages'][0])
        self.assertIn('旧句', source['messages'][1]['content'])


if __name__ == '__main__':
    unittest.main()
