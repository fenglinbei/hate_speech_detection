"""HTTP contract checks against synthetic sessions only."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tools.general_model_paired_review_ui.test_evidence_store import fixture
from tools.general_model_paired_review_ui.test_store import make_fixture
from tools.general_model_paired_review_ui.server import create_server


class EvidenceHTTPTests(unittest.TestCase):
    def test_explicit_policy_route_and_group_only_draft(self):
        from tools.general_model_paired_review_ui.test_evidence_policy import EvidencePolicyTests
        harness = EvidencePolicyTests(methodName="test_group_draft_confirm_and_export_preserve_hate_and_hits")
        harness.setUp()
        try:
            harness.confirmed_case()
            harness.migrate()
            root = harness.root
            old_session = root / 'http-state/old-paired.json'
            server = create_server(data_dir=make_fixture(root), session_path=old_session,
                                   reviewer_id='automated-deployment-test', port=0,
                                   evidence_bundle=harness.bundle, evidence_session=harness.session,
                                   evidence_policy=harness.policy_path)
            old_bytes = old_session.read_bytes()
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            base = 'http://127.0.0.1:' + str(server.server_address[1])
            def get(path):
                with urlopen(base + path) as response:
                    return json.loads(response.read())
            def post(path, data):
                request = Request(base + path, data=json.dumps(data).encode(), headers={'Content-Type': 'application/json', 'Origin': base})
                with urlopen(request) as response:
                    return json.loads(response.read())
            try:
                boot = get('/api/evidence/bootstrap')
                self.assertEqual(boot['policy']['version'], 'test-policy/v2')
                self.assertEqual(boot['bundle_policy']['version'], 'test-policy/v1')
                self.assertIn('policy_changed', boot['choices']['original_status'])
                item = get('/api/evidence/items/one')
                self.assertIsNotNone(item['comparison'])
                query = next(obj for obj in item['objects'] if obj['kind'] == 'query')
                self.assertTrue(query['group_only_recheck'])
                payload = {'session_token': boot['session_token'], 'expected_revision': boot['revision'],
                           'item_id': 'one', 'object_id': query['id'], 'object_version': query['review']['version'],
                           'values': {**query['effective_values'], 'note': 'forbidden shared change'}}
                before = harness.session.read_bytes()
                with self.assertRaises(HTTPError) as caught:
                    post('/api/evidence/save-object', payload)
                self.assertEqual(caught.exception.code, 422)
                self.assertEqual(harness.session.read_bytes(), before)
                payload['values'] = {**query['effective_values'], 'group': ['others'], 'group_reason': 'identity_target'}
                result = post('/api/evidence/save-object', payload)
                saved = next(obj for obj in result['current']['objects'] if obj['kind'] == 'query')
                self.assertTrue(saved['needs_group_recheck'])
                self.assertEqual(saved['task_reviews']['group']['status'], 'draft')
                self.assertEqual(saved['task_reviews']['hate']['status'], 'confirmed')
                self.assertEqual(old_session.read_bytes(), old_bytes)
            finally:
                server.shutdown()
                server.server_close()
                worker.join()
        finally:
            harness.tearDown()

    def test_routes_security_gating_and_separate_records(self):
        with tempfile.TemporaryDirectory(prefix='evidence-http-isolated-') as directory:
            root = Path(directory)
            server = create_server(data_dir=make_fixture(root), session_path=root / 'state/old.json',
                                   reviewer_id='automated-deployment-test', port=0,
                                   evidence_bundle=fixture(root), evidence_session=root / 'state/evidence.json')
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            base = 'http://127.0.0.1:' + str(server.server_address[1])
            def get(path):
                with urlopen(base + path) as response:
                    return response.read()
            def post(path, data, origin=base):
                req = Request(base + path, data=json.dumps(data).encode(), headers={'Content-Type': 'application/json', 'Origin': origin})
                with urlopen(req) as response:
                    return json.loads(response.read())
            try:
                old = (root / 'state/old.json').read_bytes()
                initial = (root / 'state/evidence.json').read_bytes()
                for path in ('/evidence/', '/evidence/evidence.js', '/evidence/evidence.css', '/', '/app.js'):
                    self.assertTrue(get(path))
                boot = json.loads(get('/api/evidence/bootstrap'))
                item = json.loads(get('/api/evidence/items/one'))
                self.assertNotIn('SECRET_QUERY_GOLD', json.dumps(item))
                self.assertIsNone(item['comparison'])
                self.assertEqual((root / 'state/evidence.json').read_bytes(), initial)
                common = {'session_token': boot['session_token'], 'expected_revision': boot['revision'], 'item_id': 'one'}
                for data, origin, code in (({**common, 'session_token': 'wrong'}, base, 403), (common, 'https://foreign.invalid', 403), (common, base, 422)):
                    with self.assertRaises(HTTPError) as caught:
                        post('/api/evidence/reveal', data, origin)
                    self.assertEqual(caught.exception.code, code)
                obj = item['objects'][0]
                data = {**common, 'object_id': obj['id'], 'object_version': 0, 'values': obj['effective_values']}
                saved = post('/api/evidence/save-object', data)
                self.assertEqual(saved['current']['objects'][0]['review']['status'], 'draft')
                self.assertEqual(saved['bootstrap']['status']['confirmed_object_count'], 0)
                with self.assertRaises(HTTPError) as caught:
                    post('/api/evidence/save-object', data)
                self.assertEqual(caught.exception.code, 409)
                self.assertEqual((root / 'state/old.json').read_bytes(), old)
                self.assertEqual(json.loads(get('/api/bootstrap'))['status']['confirmed_count'], 0)
            finally:
                server.shutdown()
                server.server_close()
                worker.join()


if __name__ == '__main__':
    unittest.main()
