"""Check that local resource exports preserve explicit human field scope."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from export_definition_reviews import apply_event, bound_question, build, digest, verify, DEFAULT_RUN


class HistoricalContextTests(unittest.TestCase):
    def test_only_exact_archived_sentence_pointer_is_accepted(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp)
            old = b'{"snapshot":"old"}\n'
            sha = digest(old)
            pointer = run / 'sentence_context.json'
            pointer.write_bytes(b'{"snapshot":"new"}\n')
            source = {'path': pointer.name, 'sha256': sha}
            with self.assertRaises(AssertionError):
                verify(source, run)
            archive = run / 'sentence-completion-v1/context-history' / (sha + '.json')
            archive.parent.mkdir(parents=True)
            archive.write_bytes(old)
            self.assertEqual(verify(source, run), archive)
            archive.write_bytes(b'changed')
            with self.assertRaises(AssertionError):
                verify(source, run)
            archive.write_bytes(old)
            other = run / 'frozen-input.json'
            other.write_bytes(b'changed')
            with self.assertRaises(AssertionError):
                verify({'path': other.name, 'sha256': sha}, run)


class ExplicitDefinitionFields(unittest.TestCase):
    def event(self, **extra):
        return {'event_id': 'test-only-1', 'scope': ['definition_verdict'],
                'values': {'definition_verdict': 'other_problem'},
                'review_kind': 'human_with_ai', 'online_material_confirmation_added': False,
                'rationale': '有其他问题', **extra}

    def test_verdict_does_not_confirm_rewrite_or_ai_explanation(self):
        active, history = {}, {}
        apply_event(active, history, self.event(), {'path': 'isolated-test'})
        self.assertEqual(set(active), {'definition_verdict'})
        self.assertEqual(active['definition_verdict']['rationale_verbatim'], '有其他问题')

    def test_ai_event_cannot_be_human(self):
        with self.assertRaises(AssertionError):
            apply_event({}, {}, self.event(review_kind='ai_note'), {})

    def test_scope_cannot_silently_expand(self):
        event = self.event(values={'definition_verdict': 'other_problem', 'adopted_definition': '未同意的改写'})
        with self.assertRaises(AssertionError):
            apply_event({}, {}, event, {})

    def test_revision_requires_reason_and_preserves_previous_field(self):
        active, history = {}, {}
        apply_event(active, history, self.event(), {})
        amendment = self.event(event_id='test-only-2', values={'definition_verdict': 'reasonable'})
        with self.assertRaises(AssertionError):
            apply_event(active, history, amendment, {})
        amendment['amends'] = {'definition_verdict': 'test-only-1'}
        with self.assertRaises(AssertionError):
            apply_event(active, history, amendment, {})
        amendment['amendment_reason'] = '明确保留单一义项'
        apply_event(active, history, amendment, {})
        self.assertEqual(active['definition_verdict']['value'], 'reasonable')
        self.assertEqual([r['value'] for r in history['definition_verdict']], ['other_problem', 'reasonable'])

    def test_uncertain_is_not_replaced_by_ai_reasonable(self):
        active, history = {}, {}
        apply_event(active, history, self.event(values={'definition_verdict': 'uncertain'}, rationale='无法判断'), {})
        self.assertEqual(active['definition_verdict']['value'], 'uncertain')

    def test_rewrite_only_does_not_confirm_original_quality(self):
        active, history = {}, {}
        event = self.event(scope=['adopted_definition'], values={'adopted_definition': '用户直接补写的定义'})
        apply_event(active, history, event, {})
        self.assertEqual(set(active), {'adopted_definition'})

    def test_followup_index_zero_is_bound_to_its_own_request(self):
        bindings = {('first-request', 0): {'object_id': 'first-case'},
                    ('followup-request', 0): {'object_id': 'followup-case'}}
        question = bound_question(bindings, ['request_user_input_async', 'followup-request', 0])
        self.assertEqual(question['object_id'], 'followup-case')
        with self.assertRaises(AssertionError):
            bound_question(bindings, ['request_user_input_async', 'unknown-request', 0])

    def test_current_export_is_read_only_and_keeps_field_provenance(self):
        doc = build(DEFAULT_RUN)
        self.assertEqual(doc['status']['online_material_confirmations_added'], 0)
        for record in doc['records']:
            human = record['human_fields']
            self.assertEqual(record['verdict_provenance'] == 'human_with_ai', 'definition_verdict' in human)
            self.assertEqual(record['adopted_definition'] is not None, 'adopted_definition' in human)
            self.assertEqual(record['assessment_status'] == 'human_answered', bool(human) and bool(record['definition_verdict']))
            self.assertFalse(record['online_material_confirmed'])
            self.assertEqual(record['ai_draft']['new_human_fields'], [])


if __name__ == '__main__':
    unittest.main()
