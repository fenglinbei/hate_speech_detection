import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from build_lex.llm_lexicon_builder import CandidateStats
from scripts.stage1 import smoke_lexicon


class _Response:
    status_code = 200


class _ValidClient:
    def complete_json(self, stage, payload):
        del payload
        if stage == "context_judge":
            return {
                "supported": False,
                "category": "others",
                "categories": ["others"],
                "confidence": 0.1,
                "reason": "valid",
            }
        if stage == "web_evidence_judge":
            return {
                "supported": False,
                "confidence": 0.1,
                "reason": "valid",
                "evidence_ids": [],
            }
        return {
            "include": False,
            "category": "others",
            "categories": ["others"],
            "definition": "",
            "nonhateful_meaning": "",
            "variants": [],
            "confidence": 0.1,
            "reason": "valid",
            "evidence_ids": [],
        }


def _regression_candidates():
    candidates = [CandidateStats(term=f"term-{rank}", dataset="full", language="zh") for rank in range(1, 301)]
    for rank, term in smoke_lexicon.SMOKE_SELECTION:
        candidates[rank - 1].term = term
    for candidate in candidates:
        if candidate.term in smoke_lexicon.EXPECTED_OFFLINE_FILTERED_TERMS:
            candidate.reject_hint = "substring_fragment"
    selected = [
        candidate
        for candidate in candidates
        if candidate.term not in smoke_lexicon.EXPECTED_OFFLINE_FILTERED_TERMS
    ]
    return candidates, selected


class Stage1LexiconSmokeTests(unittest.TestCase):
    def test_regression_binding_preserves_terms_and_offline_status(self):
        corpus, selected = _regression_candidates()
        cases = smoke_lexicon._select_regression_cases(corpus, selected)
        self.assertEqual(
            [(case.baseline_rank, case.candidate.term) for case in cases],
            list(smoke_lexicon.SMOKE_SELECTION),
        )
        self.assertEqual(
            {case.candidate.term for case in cases if case.offline_filtered},
            smoke_lexicon.EXPECTED_OFFLINE_FILTERED_TERMS,
        )
        corpus[299].term = "drifted"
        with self.assertRaises(smoke_lexicon.SmokeLexiconError):
            smoke_lexicon._select_regression_cases(corpus, selected)

    def test_offline_filtered_probe_cannot_become_effectively_included(self):
        candidate = CandidateStats(
            term="德绑架",
            dataset="full",
            language="zh",
            total_count=3,
            hate_count=3,
            reject_hint="substring_fragment",
        )
        case = smoke_lexicon.PreparedCase(160, candidate, None, True)
        policy = smoke_lexicon._effective_smoke_policy(
            case,
            {"include": True, "confidence": 1.0},
            {"neutral_identity_terms": [], "min_count": 2, "confidence_threshold": 0.65},
        )
        self.assertFalse(policy["include"])
        self.assertTrue(policy["post_judgement_include"])

    def test_http_budget_blocks_request_25_before_dispatch(self):
        dispatch_count = 0
        redirect_values = []

        def fake_post(_url, *args, **kwargs):
            nonlocal dispatch_count
            del args
            dispatch_count += 1
            redirect_values.append(kwargs.get("allow_redirects"))
            return _Response()

        with patch.object(requests, "post", side_effect=fake_post):
            with smoke_lexicon.HTTPRequestBudget(
                "https://api.tavily.com/search",
                "https://api.deepseek.com/chat/completions",
            ):
                for _ in range(smoke_lexicon.WEB_REQUEST_LIMIT):
                    requests.post("https://api.tavily.com/search")
                with self.assertRaises(smoke_lexicon.SmokeLexiconError):
                    requests.post("https://api.tavily.com/search")
        self.assertEqual(dispatch_count, smoke_lexicon.WEB_REQUEST_LIMIT)
        self.assertEqual(redirect_values, [False] * smoke_lexicon.WEB_REQUEST_LIMIT)

    def test_http_budget_blocks_request_73_before_dispatch(self):
        dispatch_count = 0

        def fake_post(_url, *args, **kwargs):
            nonlocal dispatch_count
            del args, kwargs
            dispatch_count += 1
            return _Response()

        with patch.object(requests, "post", side_effect=fake_post):
            with smoke_lexicon.HTTPRequestBudget(
                "https://api.tavily.com/search",
                "https://api.deepseek.com/chat/completions",
            ):
                for _ in range(smoke_lexicon.LLM_REQUEST_LIMIT):
                    requests.post("https://api.deepseek.com/chat/completions")
                with self.assertRaises(smoke_lexicon.SmokeLexiconError):
                    requests.post("https://api.deepseek.com/chat/completions")
        self.assertEqual(dispatch_count, smoke_lexicon.LLM_REQUEST_LIMIT)

    def test_strict_stage_client_validates_raw_schema_and_caps_calls(self):
        client = smoke_lexicon.StrictStageClient(_ValidClient(), limit=1)
        result = client.complete_json("context_judge", {"candidate": {"term": "x"}})
        self.assertFalse(result["supported"])
        with self.assertRaises(smoke_lexicon.SmokeLexiconError):
            client.complete_json("context_judge", {"candidate": {"term": "x"}})

    def test_secret_scan_and_output_location_gates(self):
        with patch.dict("os.environ", {"TAVILY_API_KEY": "secret-value-123"}, clear=False):
            with self.assertRaises(smoke_lexicon.SmokeLexiconError):
                smoke_lexicon._secret_scan(("secret-value-123",))
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(smoke_lexicon.SmokeLexiconError):
                smoke_lexicon._diagnostics_output_root(Path(temp_dir))

    def test_live_ledger_persists_before_dispatch_and_blocks_rerun(self):
        corpus, selected = _regression_candidates()
        prepared = smoke_lexicon.PreparedSmoke(
            config={},
            preflight_public={"train_record_count": 5165},
            cases=tuple(smoke_lexicon._select_regression_cases(corpus, selected)),
            corpus_candidate_count=120566,
            eligible_candidate_count=1000,
            selection_sha256="b" * 64,
            request_budget={"web_request_limit": 24, "llm_request_limit": 72},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            diagnostics = Path(temp_dir) / "diagnostics"
            with patch.object(smoke_lexicon, "DIAGNOSTICS_ROOT", diagnostics):
                ledger = smoke_lexicon._start_run_ledger(diagnostics, prepared)

                def fake_post(_url, *args, **kwargs):
                    del args, kwargs
                    payload = json.loads(ledger.path.read_text(encoding="utf-8"))
                    self.assertEqual(payload["request_counts"]["tavily"], 1)
                    return _Response()

                with patch.object(requests, "post", side_effect=fake_post):
                    with smoke_lexicon.HTTPRequestBudget(
                        "https://api.tavily.com/search",
                        "https://api.deepseek.com/chat/completions",
                        before_request=ledger.before_request,
                    ):
                        requests.post("https://api.tavily.com/search")
                persisted = json.loads(ledger.path.read_text(encoding="utf-8"))
                self.assertEqual(persisted["request_counts"], {"tavily": 1, "deepseek": 0})
                with self.assertRaises(smoke_lexicon.SmokeLexiconError):
                    smoke_lexicon._start_run_ledger(diagnostics, prepared)

    def test_sensitive_retry_logger_is_disabled_only_inside_scope(self):
        logger = logging.getLogger("build_lex.llm_lexicon_builder")
        original = logger.disabled
        with smoke_lexicon.SensitiveLogSilencer():
            self.assertTrue(logger.disabled)
        self.assertEqual(logger.disabled, original)

    def test_dry_run_public_marks_zero_http_and_nonpublication(self):
        corpus, selected = _regression_candidates()
        prepared = smoke_lexicon.PreparedSmoke(
            config={},
            preflight_public={"train_record_count": 5165},
            cases=tuple(smoke_lexicon._select_regression_cases(corpus, selected)),
            corpus_candidate_count=120566,
            eligible_candidate_count=1000,
            selection_sha256="a" * 64,
            request_budget={"web_request_limit": 24, "llm_request_limit": 72},
        )
        result = smoke_lexicon._dry_run_public(prepared)
        self.assertEqual(result["actual_requests"], {"tavily": 0, "deepseek": 0})
        self.assertFalse(result["scientific_eligible"])
        self.assertFalse(result["formal_publication_eligible"])
        self.assertIsNone(result["published_ref"])
        self.assertFalse(result["writes_performed"])
        self.assertNotIn("target", result)
        self.assertNotIn("lexicon_path", result)


if __name__ == "__main__":
    unittest.main()
