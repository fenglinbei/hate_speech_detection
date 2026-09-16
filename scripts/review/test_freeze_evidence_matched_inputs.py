"""Freeze/acceptance boundaries and score-definition-sensitive contrast math."""
from copy import deepcopy
import unittest

from scripts.review.freeze_evidence_matched_inputs import (
    CONDITIONS, canonical, registered_contrasts, scoring_contexts, sha, verify_feedback,
)
from scripts.review.run_evidence_matched_inputs import analyze_rows
from diagnostics.general_model_numeric_analysis import candidate_catalog, candidate_scores


def frame():
    rows = []
    for q in ('541', '3169'):
        for i, c in enumerate(CONDITIONS):
            text = q + ':' + c
            tokens = [int(q), i]
            rows.append({'record_id': text, 'query_id': q, 'task': 'hate', 'condition': c, 'family': 'test',
                         'messages': [{'role': 'system', 'content': 'task'}, {'role': 'user', 'content': text}],
                         'prompt_text': text, 'prompt_sha256': sha(text.encode()), 'prompt_token_ids': tokens,
                         'prompt_tokens': 2, 'prompt_token_ids_sha256': sha(canonical(tokens).encode())})
    return rows


class MatchedFreezeTests(unittest.TestCase):
    def test_batch_feedback_must_bind_every_exact_text_and_preserve_scope(self):
        rows = [{'material_id': 'x', 'text': '原文', 'text_sha256': sha('原文'.encode())}]
        feedback = {'source_manifest_sha256': 'm', 'batch_overall_no_objection': True,
                    'individual_field_decisions': [], 'new_mechanism_judgments': 0,
                    'materials': [{'material_id': 'x', 'text_sha256': rows[0]['text_sha256']}]}
        verify_feedback(feedback, rows, 'm')
        for change in ({'materials': []}, {'source_manifest_sha256': 'other'}, {'individual_field_decisions': [{'hate': 'hate'}]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify_feedback(feedback | change, rows, 'm')
        rows[0]['text'] = '改文'
        with self.assertRaises(ValueError): verify_feedback(feedback, rows, 'm')

    def test_exact_context_frame_and_token_identity_required(self):
        source = frame(); saved = deepcopy(source)
        result = scoring_contexts(source)
        self.assertEqual(source, saved)
        self.assertEqual(sum(r['baseline_replay'] for r in result), 4)
        for mutation in ('missing', 'reserve', 'tokens'):
            changed = deepcopy(source)
            if mutation == 'missing': changed.pop()
            elif mutation == 'reserve': changed[0]['query_id'] = 'reserved'
            else: changed[0]['prompt_token_ids'][0] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): scoring_contexts(changed)

    def test_nested_query_reference_rejected(self):
        source = frame(); source[0]['notes'] = {'adjudicated_label': 'non-hate'}
        with self.assertRaisesRegex(ValueError, 'query reference'): scoring_contexts(source)

    def test_comparison_bounds_and_no_extra_conditions(self):
        source = [{'contrast_id': f'x{i}', 'query_id': '541', 'task': 'hate', 'kind': 'test',
                   'terms': [{'condition': 'P1-D1', 'coefficient': 1}, {'condition': 'P0-D1', 'coefficient': -1}]} for i in range(38)]
        source[1]['terms'] += [{'condition': 'P1-C1', 'coefficient': -1}, {'condition': 'P0-C1', 'coefficient': 1}]
        rows = registered_contrasts(source, 0.0013427734375)
        self.assertEqual(len(rows), 64)
        self.assertEqual(rows[1]['numeric_bound'], 0.00537109375)
        self.assertTrue(all(r['numeric_bound'] == 0.002685546875 for i, r in enumerate(rows) if i != 1))
        source[0]['terms'][0]['condition'] = 'unregistered'
        with self.assertRaises(ValueError): registered_contrasts(source, 0.0013427734375)

    def test_mean_and_sum_can_reverse_contrast_and_are_both_retained(self):
        contexts = scoring_contexts(frame())[:2]
        # Keep a single query with two input conditions for the isolated analysis.
        contexts[0].update(query_id='541', condition='X')
        contexts[1].update(query_id='541', condition='Y')
        raw = []
        for index, c in enumerate(contexts):
            candidates = candidate_catalog()['hate']
            token_scores = ([-1.0, -1.0, -1.0], [-1.0] * 5) if index == 0 else ([-2.0] * 3, [-1.8] * 5)
            for candidate, values in zip(candidates, token_scores):
                candidate.update(answer_token_ids=list(range(len(values))), token_logprobs=values, scores=candidate_scores(values, 0.0))
            raw.append({'record_id': c['record_id'], 'query_id': '541', 'task': 'hate', 'prompt_sha256': c['prompt_sha256'],
                        'context_sha256': c['context_sha256'], 'candidates': candidates})
        comparison = [{'contrast_id': 'Y-X', 'query_id': '541', 'kind': 'test',
                       'terms': [{'condition': 'Y', 'coefficient': 1}, {'condition': 'X', 'coefficient': -1}], 'numeric_bound': 0.01}]
        rows, effects = analyze_rows(contexts, raw, [{'query_id': '541', 'task': 'hate', 'original_label': 'non-hate', 'adjudicated_label': 'non-hate'}], comparison, candidate_catalog())
        by_mode = {r['score_mode']: r['non_hate_margin_effect'] for r in effects}
        self.assertEqual(len(rows), 8)
        self.assertEqual(by_mode['answer_sum']['direction'], 'negative')
        self.assertEqual(by_mode['answer_mean']['direction'], 'positive')


if __name__ == '__main__': unittest.main()
