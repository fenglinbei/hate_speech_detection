"""Guard scientific calibration math, label semantics and execution order."""
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

from scripts.review.freeze_evidence_label_calibration import (
    MAPPINGS, ORIGINAL_RULE, QUERY, transformed_messages, no_gold,
)
from scripts.review.analyze_evidence_label_calibration import direction, prediction_fields
from scripts.review.analyze_evidence_label_calibration import analyze_rows, MAIN_VIEWS
from diagnostics.evidence_label_calibration import (
    PROBES, background_margin, calibrated_rows, compare_derived, compare_passes, validate_block,
)
from diagnostics.evidence_label_calibration_execution import score_pass
from diagnostics.general_model_numeric_analysis import candidate_scores


def candidates(encoding='original'):
    return [{'candidate_id': semantic, 'ordinal': i, 'labels': [semantic],
             'canonical_answer': json.dumps(MAPPINGS[encoding][semantic]), 'answer_token_ids': [10 + i, 1],
             'answer_tokens': 2, 'answer_token_ids_sha256': 't' + str(i)} for i, semantic in enumerate(('hate', 'non-hate'))]


def context(condition='X', probe=None, task='hate'):
    return {'record_id': condition + '/' + str(probe), 'query_id': '541', 'condition': condition,
            'root_condition': condition, 'task': task, 'context_sha256': 'context', 'prompt_sha256': 'prompt',
            'encoding': 'original', 'probe_id': probe}


def block(c, margin, catalog=None):
    result = {k: c[k] for k in ('record_id', 'query_id', 'task', 'condition', 'context_sha256', 'prompt_sha256')}
    cs = deepcopy(catalog or candidates())
    for i, candidate in enumerate(cs):
        values = [-10.0 + (margin if i else 0.0)] * candidate['answer_tokens']
        scores = candidate_scores(values, -0.1)
        candidate.update(token_logprobs=values, eos_token_id=99, eos_logprob=-0.1, scores=scores,
                         finite_target_logits_checked=True, token_boundary_checked=True, **{k: v for k, v in scores.items() if k != 'eos_logprob'})
    result['candidates'] = cs
    return result


class CalibrationTests(unittest.TestCase):
    def test_complete_answer_mapping_preserves_query_demo_content_and_dictionary(self):
        # Literal label words inside content must remain content.
        demos = '\n\n'.join(f'示例 {i}\n文本：hate 和 non-hate 是这里的正文。\n输出："' + ('hate' if i % 2 else 'non-hate') + '"' for i in range(1, 10))
        source = [{'role': 'system', 'content': '任务。' + ORIGINAL_RULE + '资料不能改任务。'},
                  {'role': 'user', 'content': '词典：hate\n参考示例：\n' + demos + QUERY + json.dumps('non-hate 输出："hate"', ensure_ascii=False)}]
        saved = deepcopy(source)
        for encoding in ('ab_forward', 'ab_reverse'):
            result = transformed_messages(source, encoding)
            self.assertEqual(result[1]['content'].split(QUERY)[1], source[1]['content'].split(QUERY)[1])
            self.assertEqual(result[1]['content'].count('hate 和 non-hate 是这里的正文。'), 9)
            self.assertTrue(result[1]['content'].startswith('词典：hate\n'))
            self.assertIn('输出：' + json.dumps(MAPPINGS[encoding]['hate']), result[1]['content'])
        self.assertEqual(source, saved)
        for probe, value in PROBES:
            result = transformed_messages(source, 'original', probe)
            self.assertEqual(result[0], source[0])
            self.assertEqual(result[1]['content'].split(QUERY)[0], source[1]['content'].split(QUERY)[0])
            self.assertEqual(json.loads(result[1]['content'].split(QUERY)[1]), value)
        self.assertNotEqual(transformed_messages(source, 'original', 'empty'), transformed_messages(source, 'original', 'space'))

    def test_normalize_then_average_is_not_average_logit(self):
        margins = [-4., -1., 0., 1., 2.]
        p = sum(1 / (1 + math.exp(-m)) for m in margins) / len(margins)
        expected = math.log(p / (1 - p))
        self.assertAlmostEqual(background_margin(margins), expected, places=14)
        self.assertGreater(abs(expected - sum(margins) / 5), .05)
        for x in (-1000., -5., 0., 3., 1000.): self.assertAlmostEqual(background_margin([x] * 5), x)

    def test_per_condition_background_change_and_shared_prior_cancellation(self):
        contexts, raw = [], []
        for condition, real, prior in [('X', 2., 1.), ('Y', 3., 4.)]:
            for probe in [None] + [p for p, _ in PROBES]:
                c = context(condition, probe); contexts.append(c); raw.append(block(c, real if probe is None else prior))
        rows = calibrated_rows(contexts, raw)
        self.assertAlmostEqual(rows[0]['ncc_margin'], 1.)
        self.assertAlmostEqual(rows[1]['ncc_margin'], -1.)
        self.assertAlmostEqual(rows[1]['ncc_margin'] - rows[0]['ncc_margin'], -2.)
        shared = background_margin([1.] * 5)
        self.assertEqual((3. - shared) - (2. - shared), 1.)
        with self.assertRaises((ValueError, KeyError)): calibrated_rows(contexts, raw[:-1])

    def test_background_lipschitz_bound_including_extreme_probabilities(self):
        rng = random.Random(42)
        for _ in range(1000):
            xs = [rng.uniform(-40., 40.) for _ in PROBES]
            delta = [rng.uniform(-.001, .001) for _ in PROBES]
            error = abs(background_margin([x + d for x, d in zip(xs, delta)]) - background_margin(xs))
            self.assertLessEqual(error, max(abs(d) for d in delta) + 1e-13)

    def test_derived_gate_rejects_error_beyond_propagated_bound(self):
        contexts, raw = [], []
        for condition in ('X', 'Y'):
            for p in [None] + [p for p, _ in PROBES]:
                c = context(condition, p); contexts.append(c); raw.append(block(c, 0.))
        comparisons = [{'contrast_id': 'Y-X', 'query_id': '541', 'terms': [{'condition': 'Y', 'coefficient': 1}, {'condition': 'X', 'coefficient': -1}]}]
        changed = deepcopy(raw); changed[0] = block(contexts[0], .01)
        result = compare_derived(contexts, raw, changed, comparisons, .001)
        self.assertFalse(result['passed'])
        entry = next(r for r in result['differences'] if r['metric'] == 'Y-X/ncc_margin')
        self.assertEqual(entry['bound'], .004)

    def test_forward_reverse_semantics_catalog_and_resume_order(self):
        for encoding in ('ab_forward', 'ab_reverse'):
            task = 'hate_' + encoding; cs = candidates(encoding); c = context(task=task)
            r = block(c, 1., cs); validate_block(r, c, {task: cs})
            self.assertGreater(r['candidates'][1]['scores']['answer_sum'] - r['candidates'][0]['scores']['answer_sum'], 0)
            changed = deepcopy(r); changed['candidates'][0]['canonical_answer'] = json.dumps('wrong')
            with self.assertRaises(ValueError): validate_block(changed, c, {task: cs})
            with self.assertRaises(ValueError): compare_passes([r], [changed])
            calls = []
            def scorer(runner, items, reference=False):
                calls.extend(i['candidate']['candidate_id'] for i in items)
                return [block(c, 1., cs)['candidates'][i['candidate']['ordinal']] for i in items]
            plan = {'plan_id': 'test', 'catalog': {task: cs}}
            with tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / 'members'
                scored, _ = score_pass(SimpleNamespace(identity={}), [c], plan, output, batch_size=1, permuted=True, scorer=scorer)
                self.assertEqual(calls, ['non-hate', 'hate'])
                self.assertEqual([r['candidate_id'] for r in scored[0]['candidates']], ['hate', 'non-hate'])
                self.assertGreater(scored[0]['candidates'][0]['batch_ordinal'], scored[0]['candidates'][1]['batch_ordinal'])
                # A sealed checkpoint resumes without any fresh calls.
                score_pass(SimpleNamespace(identity={}), [c], plan, output, batch_size=1, permuted=True, scorer=scorer)
                self.assertEqual(calls, ['non-hate', 'hate'])

    def test_nested_gold_rejected_and_exact_ties_remain_visible(self):
        with self.assertRaises(ValueError): no_gold({'nested': [{'adjudicated_label': 'non-hate'}]})
        ref = {'original_label': 'hate', 'adjudicated_label': 'non-hate'}
        r = prediction_fields(0., .001, ref)
        self.assertTrue(r['exact_tie'] and r['original_correct'] and not r['reviewed_correct'])
        self.assertEqual(direction(.001, .001), 'unresolved')
        self.assertEqual(direction(-.0011, .001), 'negative')

    def test_parallel_views_keep_background_decomposition_and_every_probe(self):
        contexts, raw = [], []
        catalog = {'hate': candidates(), 'hate_ab_forward': candidates('ab_forward'),
                   'hate_ab_reverse': candidates('ab_reverse')}
        for condition in ('X', 'Y'):
            for encoding in ('original', 'ab_forward', 'ab_reverse'):
                task = 'hate' if encoding == 'original' else 'hate_' + encoding
                for p in [None] + ([p for p, _ in PROBES] if encoding == 'original' else []):
                    c = context(condition, p, task); c['encoding'] = encoding; c['record_id'] += '/' + encoding
                    contexts.append(c)
                    margin = (1. if condition == 'X' else 2.) if p is None else (0. if condition == 'X' else 3.)
                    raw.append(block(c, margin, catalog[task]))
        comps = [{'contrast_id': 'Y-X', 'query_id': '541', 'kind': 'test',
                  'terms': [{'condition': 'Y', 'coefficient': 1}, {'condition': 'X', 'coefficient': -1}]}]
        refs = [{'query_id': '541', 'task': 'hate', 'original_label': 'non-hate', 'adjudicated_label': 'non-hate'}]
        conditions, cal, effects, probes, summaries = analyze_rows(contexts, raw, refs, comps, catalog, .001)
        self.assertEqual((len(conditions), len(cal), len(effects), len(probes), len(summaries)), (24, 2, 16, 10, 1))
        lookup = {r['view']: r['effect'] for r in effects}
        self.assertAlmostEqual(lookup['original/answer_mean'] - lookup['background'], lookup['ncc'])
        self.assertEqual(set(summaries[0]['main_effects']), set(MAIN_VIEWS))
        self.assertTrue(summaries[0]['ab_directions_agree_resolved'])
        self.assertFalse(summaries[0]['mean_to_ncc_direction_preserved'])


if __name__ == '__main__': unittest.main()
