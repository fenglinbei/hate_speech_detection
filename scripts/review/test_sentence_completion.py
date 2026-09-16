"""Read-only provenance checks and isolated amendment-history regression tests."""
import copy
import json
import unittest

from export_adjudication_preferences import resolve_amendments
from export_sentence_completion import DEFAULT_RUN, build


def event(event_id, value, field='group', oid='demo:1'):
    return {'event_id': event_id, 'object_id': oid, 'field': field,
            'value': value, 'recorded_at_utc': '2026-09-11T00:00:00+00:00'}


class AmendmentTests(unittest.TestCase):
    def test_explicit_amendment_preserves_previous_value(self):
        old = event('old', ['Racism'])
        new = {**event('new', ['Racism', 'others']),
               'supersedes_event_ids': ['old'], 'amendment_reason': 'Same-target rule clarified.'}
        amendments = resolve_amendments([old, new])
        self.assertEqual(old['value'], ['Racism'])
        self.assertEqual(old['superseded_by'], ['new'])
        self.assertEqual([e['value'] for e in [old, new] if not e.get('superseded_by')], [['Racism', 'others']])
        self.assertEqual(len(amendments), 1)

    def test_differing_values_do_not_silently_select_latest(self):
        records = [event('old', ['Racism']), event('new', ['Racism', 'others'])]
        original = copy.deepcopy(records)
        self.assertEqual(resolve_amendments(records), [])
        self.assertEqual(records, original)

    def test_other_field_or_case_cannot_be_superseded(self):
        for previous in [event('old', 2, field='attack_severity'), event('old', [], oid='demo:2')]:
            replacement = {**event('new', ['Racism']), 'supersedes_event_ids': ['old'], 'amendment_reason': 'Review'}
            with self.assertRaises(AssertionError):
                resolve_amendments([previous, replacement])

    def test_cycles_and_missing_reasons_are_rejected(self):
        a = {**event('a', []), 'supersedes_event_ids': ['b'], 'amendment_reason': 'Review'}
        b = {**event('b', []), 'supersedes_event_ids': ['a'], 'amendment_reason': 'Review'}
        with self.assertRaises(AssertionError):
            resolve_amendments([a, b])
        with self.assertRaises(AssertionError):
            resolve_amendments([event('old', []), {**event('new', []), 'supersedes_event_ids': ['old']}])


class CloseoutReadOnlyTests(unittest.TestCase):
    def test_complete_coverage_and_no_inferred_human_fields(self):
        _, artifacts, summary = build(DEFAULT_RUN)
        rows = {r['object_id']: r for r in json.loads(artifacts['current-sentences.json'])['records']}
        self.assertEqual(len(rows), 270)
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['severity_numeric_count'], 270)
        self.assertEqual(summary['new_human_field_event_counts'], {'attack_severity': 12, 'group': 8})
        self.assertEqual(summary['online_material_confirmations_added'], 0)
        for row in rows.values():
            self.assertEqual(row['values']['hate'], 'hate' if row['values']['attack_severity'] else 'non-hate')
            self.assertFalse(row['online_material_confirmed'])
        # A score-only reply must not turn a proposed group into a human decision.
        self.assertEqual(rows['demo:1660']['field_provenance']['group'], 'ai_note')
        self.assertEqual(rows['demo:1660']['field_provenance']['attack_severity'], 'user_discussion')
        # A group-only reply must not confirm the AI score.
        self.assertEqual(rows['demo:7927']['field_provenance']['group'], 'user_discussion')
        self.assertEqual(rows['demo:7927']['field_provenance']['attack_severity'], 'ai_note')
        self.assertEqual(rows['query:5423']['values']['group'], ['Racism', 'others'])
        self.assertEqual(rows['query:5423']['field_history']['group']['value'], ['Racism'])
        self.assertEqual(rows['query:1746']['values']['group'], ['Racism', 'Sexism', 'others'])
        self.assertEqual(rows['query:1746']['field_provenance']['group'], 'ai_note')
        # A later lexical clarification has its own explicit group amendment;
        # it retains the prior group and does not replace the scoring source.
        ana = rows['demo:5998']
        self.assertEqual(ana['values'], {'hate': 'hate', 'group': ['Sexism', 'others'], 'attack_severity': 1})
        self.assertEqual(ana['field_history']['group']['value'], ['others'])
        self.assertEqual(ana['field_status']['attack_severity'], 'retained')
        self.assertEqual(ana['current_field_sources']['attack_severity']['source']['path'],
                         'severity/batch-06/ai_annotations.json')
        self.assertFalse(ana['retained_task_acknowledgment']['new_numeric_or_hate_fields_counted'])
        circle = rows['query:4137']
        self.assertEqual(circle['values'], {'hate': 'non-hate', 'group': ['LGBTQ', 'Sexism'], 'attack_severity': 0})
        self.assertEqual(circle['field_provenance']['attack_severity'], 'user_discussion')
        self.assertEqual(circle['field_provenance']['group'], 'ai_note')
        self.assertEqual(circle['field_history']['attack_severity']['field_provenance'], 'ai_note')
        self.assertEqual(circle['historical_severity_assessment']['values']['attack_severity'], 0)


if __name__ == '__main__':
    unittest.main()
