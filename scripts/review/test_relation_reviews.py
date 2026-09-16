import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from export_relation_reviews import (
    DEFAULT_RUN, FIELDS, apply_event, artifacts, build, check_reply,
    digest, encoded, ref, resolved_questions,
)


class ExplicitRelationFieldTests(unittest.TestCase):
    def event(self, **changes):
        return {'event_id': 'e1', 'scope': ['topic_hate'],
                'values': {'topic_hate': 'direct'}, 'review_kind': 'human_with_ai',
                'online_material_confirmation_added': False,
                'rationale': '话题直接相关', **changes}

    def test_one_field_does_not_confirm_other_fields(self):
        active, history = {}, {}
        apply_event(active, history, self.event(), {'path': 'isolated-test'})
        self.assertEqual(set(active), {'topic_hate'})
        self.assertEqual(set(history), {'topic_hate'})

    def test_amendment_needs_reference_and_reason_and_retains_history(self):
        active, history = {}, {}
        apply_event(active, history, self.event(), {})
        revised = self.event(event_id='e2', values={'topic_hate': 'partial'})
        with self.assertRaises(AssertionError):
            apply_event(copy.deepcopy(active), copy.deepcopy(history), revised, {})
        revised['amends'] = {'topic_hate': 'e1'}
        with self.assertRaises(AssertionError):
            apply_event(copy.deepcopy(active), copy.deepcopy(history), revised, {})
        revised['amendment_reason'] = '确认了不同语义'
        apply_event(active, history, revised, {})
        self.assertEqual(active['topic_hate']['value'], 'partial')
        self.assertEqual([h['value'] for h in history['topic_hate']], ['direct', 'partial'])

    def test_invalid_later_field_cannot_partially_apply_event(self):
        active, history = {}, {}
        event = self.event(scope=['topic_hate', 'lexicon_risk'],
                           values={'topic_hate': 'direct', 'lexicon_risk': 'direct'})
        with self.assertRaises(AssertionError):
            apply_event(active, history, event, {})
        self.assertEqual(active, {})
        self.assertEqual(history, {})

    def test_cannot_write_sentence_labels_or_online_confirmations(self):
        for event in [self.event(scope=['hate'], values={'hate': 'hate'}),
                      self.event(online_material_confirmation_added=True)]:
            with self.assertRaises(AssertionError):
                apply_event({}, {}, event, {})

    def test_multi_pair_question_stays_pending_after_partial_reply(self):
        questions = {0: {'object_ids': ['a', 'b'], 'allowed_user_fields': ['topic_hate']},
                     1: {'object_ids': ['c'], 'allowed_user_fields': ['rule_hate', 'rule_group']}}
        active = {'a': {'topic_hate': {}}, 'c': {'rule_hate': {}}}
        self.assertEqual(resolved_questions(questions, active), set())
        active['b'] = {'topic_hate': {}}
        self.assertEqual(resolved_questions(questions, active), {0})

    def test_reply_is_bound_to_question_scope_and_call(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp)
            folder = run / 'batch-01'
            folder.mkdir()
            question = {'question_index': 0, 'object_ids': ['relation:1:2'],
                        'allowed_user_fields': ['topic_hate'], 'question_text_verbatim': '只问话题？'}
            question_path = folder / 'questions.json'
            question_path.write_bytes(encoded({'records': [question]}))
            (folder / 'request-receipt.json').write_bytes(encoded({
                'request_call_id': 'test-call', 'questions_sha256': digest(question_path.read_bytes())}))
            qid = ['request_user_input_async', 'test-call', 0]
            reply = {'questionItemId': json.dumps(qid), 'question': '只问话题？', 'answer': '直接'}
            reply_path = folder / 'reply.json'
            reply_path.write_bytes(encoded({'replies': [reply]}))
            event = self.event(object_id='relation:1:2', rationale='直接', question_item_id=qid,
                               source_reply_reference=ref(reply_path, run), source_reply_index=0)
            self.assertEqual(check_reply(event, {0: question}, folder, run)[0], 0)
            wrong_scope = {**event, 'scope': ['rule_hate']}
            with self.assertRaises(AssertionError):
                check_reply(wrong_scope, {0: question}, folder, run)
            reply['questionItemId'] = json.dumps(['request_user_input_async', 'another-call', 0])
            reply_path.write_bytes(encoded({'replies': [reply]}))
            event.update(question_item_id=json.loads(reply['questionItemId']),
                         source_reply_reference=ref(reply_path, run))
            with self.assertRaises(AssertionError):
                check_reply(event, {0: question}, folder, run)


class ReadOnlyRelationIntegrationTests(unittest.TestCase):
    def test_authorized_coverage_exact_text_and_field_provenance(self):
        doc = build(DEFAULT_RUN)
        self.assertEqual(doc['status']['relations_total'], 283)
        self.assertEqual(doc['status']['prior_relationship_confirmations_referenced'], 37)
        self.assertEqual(len(doc['records']), 283)
        self.assertEqual(len({r['object_id'] for r in doc['records']}), 283)
        for row in doc['records']:
            self.assertFalse(row['online_material_confirmed'])
            for f in FIELDS:
                self.assertEqual(row['field_provenance'][f] == 'user_discussion', f in row['human_fields'])
            for span in row['values']['evidence']:
                self.assertEqual(row[span['source'] + '_text'][span['start']:span['end']], span['text'])

    def test_export_is_deterministic_and_read_only(self):
        doc = build(DEFAULT_RUN)
        original = copy.deepcopy(doc)
        first = artifacts(doc, DEFAULT_RUN)
        self.assertEqual(first, artifacts(build(DEFAULT_RUN), DEFAULT_RUN))
        self.assertEqual(doc, original)


if __name__ == '__main__':
    unittest.main()
