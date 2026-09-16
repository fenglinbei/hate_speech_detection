import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from export_hit_reviews import apply_event, build, check_reply, digest, effective_issues, encoded, ref, resolved_questions, DEFAULT_RUN


class ExplicitHitFieldTests(unittest.TestCase):
    def event(self, **changes):
        return {'event_id': 'e1', 'scope': ['source_fit'],
                'values': {'source_fit': 'valid_sense'}, 'review_kind': 'human_with_ai',
                'online_material_confirmation_added': False, 'rationale': '词义适配', **changes}

    def test_source_reply_does_not_confirm_query_fit(self):
        active, history = {}, {}
        apply_event(active, history, self.event(), {'path': 'isolated-test'})
        self.assertEqual(set(active), {'source_fit'})
        self.assertEqual(len(history['source_fit']), 1)

    def test_amendment_retains_old_decision_and_requires_reason(self):
        active, history = {}, {}
        apply_event(active, history, self.event(), {})
        revised = self.event(event_id='e2', values={'source_fit': 'wrong_sense'})
        with self.assertRaises(AssertionError):
            apply_event(copy.deepcopy(active), copy.deepcopy(history), revised, {})
        revised['amends'] = {'source_fit': 'e1'}
        with self.assertRaises(AssertionError):
            apply_event(copy.deepcopy(active), copy.deepcopy(history), revised, {})
        revised['amendment_reason'] = '根据补充上下文修订'
        apply_event(active, history, revised, {})
        self.assertEqual(active['source_fit']['value'], 'wrong_sense')
        self.assertEqual([r['value'] for r in history['source_fit']], ['valid_sense', 'wrong_sense'])

    def test_automatic_flags_do_not_retain_rejected_sense(self):
        flags = effective_issues({'source_fit': 'valid_sense', 'query_fit': 'uncertain'},
                                 ['wrong_sense', 'overly_narrow_definition'])
        self.assertEqual(flags, ['valid_sense', 'uncertain', 'overly_narrow_definition'])

    def test_cannot_write_sentence_labels_or_online_confirmations(self):
        for event in [self.event(scope=['hate'], values={'hate': 'hate'}),
                      self.event(online_material_confirmation_added=True)]:
            with self.assertRaises(AssertionError):
                apply_event({}, {}, event, {})

    def test_followup_is_bound_to_its_request_and_original_scope(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp)
            folder = run / 'batch-01'
            questions = {7: {'object_ids': ['hit:1'], 'allowed_user_fields': ['query_fit']}}
            question = {'question_index': 0, 'resolves_primary_question_index': 7,
                        'object_ids': ['hit:1'], 'allowed_user_fields': ['query_fit'],
                        'question_text_verbatim': '同词变体是否适配？'}
            path = folder / 'consistency-rechecks/variant-question.json'
            path.parent.mkdir(parents=True)
            path.write_bytes(encoded(question))
            receipt = {'request_call_id': 'followup-call', 'questions_sha256': digest(path.read_bytes())}
            path.with_name('variant-question-request-receipt.json').write_bytes(encoded(receipt))
            qid = ['request_user_input_async', 'followup-call', 0]
            reply_path = folder / 'reply.json'
            reply = {'questionItemId': json.dumps(qid), 'question': question['question_text_verbatim'],
                     'answer': '本例适配'}
            reply_path.write_bytes(encoded({'replies': [reply]}))
            event = self.event(object_id='hit:1', scope=['query_fit'], values={'query_fit': 'applicable'},
                               rationale=reply['answer'], question_item_id=qid,
                               source_question_reference=ref(path, run),
                               source_reply_reference=ref(reply_path, run), source_reply_index=0)
            self.assertEqual(check_reply(event, questions, folder, run)[0], 7)
            for qid_value, scope in [(['request_user_input_async', 'primary-call', 0], ['query_fit']),
                                     (qid, ['source_fit'])]:
                bad = copy.deepcopy(event)
                bad['question_item_id'] = qid_value
                bad['scope'] = scope
                reply['questionItemId'] = json.dumps(qid_value)
                reply_path.write_bytes(encoded({'replies': [reply]}))
                bad['source_reply_reference'] = ref(reply_path, run)
                with self.assertRaises(AssertionError):
                    check_reply(bad, questions, folder, run)

    def test_related_instruction_resolves_discussion_without_fit_confirmation(self):
        questions = {0: {'object_ids': ['a', 'b'], 'allowed_user_fields': ['source_fit']},
                     7: {'object_ids': ['c'], 'allowed_user_fields': ['query_fit']}}
        active = {'a': {'source_fit': {'value': 'valid_sense'}}}
        original = copy.deepcopy(active)
        related = {'c': [{'primary_question_index': 7}]}
        self.assertEqual(resolved_questions(questions, active, related), (set(), {7}))
        self.assertEqual(active, original)


class ReadOnlyHitIntegrationTests(unittest.TestCase):
    def test_authorized_sources_and_field_provenance(self):
        doc = build(DEFAULT_RUN)
        self.assertEqual(doc['status']['hits_total'], 292)
        self.assertEqual(len(doc['records']), len({r['object_id'] for r in doc['records']}))
        for row in doc['records']:
            hit = row['original_hit']['source']
            a, z = hit['raw_span']
            self.assertEqual(row['source_text'][a:z], hit['raw_surface'])
            self.assertFalse(row['online_material_confirmed'])
            for field in ('source_fit', 'query_fit'):
                self.assertEqual(row['field_provenance'][field] == 'user_discussion',
                                 field in row['human_fields'])
        missing = [r for r in doc['records'] if not r['original_definition']['text']]
        self.assertEqual(len(missing), 2)
        self.assertTrue(all(r['original_hit']['source']['source_id'] == '3898' for r in missing))


if __name__ == '__main__':
    unittest.main()
