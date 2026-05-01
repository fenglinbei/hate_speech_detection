import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import requests

from build_lex.llm_lexicon_builder import (
    DebugRecorder,
    LLMAPIError,
    OpenAICompatibleJudgementClient,
    build_candidates,
    build_lexicon,
    create_judgement_client,
    select_candidates,
    write_jsonl,
)
from build_lex.web_search import WebSearcher


class _NoopLogger:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def install_rag_import_stubs():
    sys.modules.setdefault("loguru", types.SimpleNamespace(logger=_NoopLogger()))
    sys.modules.setdefault("sentence_transformers", types.SimpleNamespace(SentenceTransformer=object))
    if "sklearn.metrics.pairwise" not in sys.modules:
        sklearn_mod = types.ModuleType("sklearn")
        metrics_mod = types.ModuleType("sklearn.metrics")
        pairwise_mod = types.ModuleType("sklearn.metrics.pairwise")
        cluster_mod = types.ModuleType("sklearn.cluster")

        def _cosine_similarity(a, b):
            a = np.asarray(a, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)
            a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
            b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
            return a @ b.T

        pairwise_mod.cosine_similarity = _cosine_similarity
        cluster_mod.KMeans = object
        metrics_mod.pairwise = pairwise_mod
        sklearn_mod.cluster = cluster_mod
        sklearn_mod.metrics = metrics_mod
        sys.modules.setdefault("sklearn", sklearn_mod)
        sys.modules.setdefault("sklearn.cluster", cluster_mod)
        sys.modules.setdefault("sklearn.metrics", metrics_mod)
        sys.modules.setdefault("sklearn.metrics.pairwise", pairwise_mod)
    reranker_mod = types.ModuleType("rag.reranker")
    reranker_mod.Reranker = object
    sys.modules.setdefault("rag.reranker", reranker_mod)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class FakeJudgeClient:
    def __init__(self, included_terms):
        self.included_terms = set(included_terms)
        self.calls = []

    def complete_json(self, stage, payload):
        term = payload["candidate"]["term"]
        self.calls.append((stage, term))
        include = term in self.included_terms
        if stage == "context_judge":
            return {
                "supported": include,
                "category": payload["candidate"].get("primary_category", "Racism"),
                "categories": [payload["candidate"].get("primary_category", "Racism")],
                "confidence": 0.9 if include else 0.1,
                "reason": "mock context judgement",
            }
        if stage == "web_evidence_judge":
            return {
                "supported": include,
                "confidence": 0.9 if include else 0.1,
                "reason": "mock web judgement",
                "evidence_ids": ["ev1"] if include else [],
            }
        if stage == "final_lexicon_judge":
            return {
                "include": include,
                "category": payload["candidate"].get("primary_category", "Racism"),
                "categories": [payload["candidate"].get("primary_category", "Racism")],
                "definition": "A mock discriminatory term used against a protected group." if include else "",
                "nonhateful_meaning": "",
                "variants": [],
                "confidence": 0.9 if include else 0.1,
                "reason": "mock final judgement",
                "evidence_ids": ["ev1"] if include else [],
            }
        raise AssertionError(stage)


class FailingWebJudgeClient(FakeJudgeClient):
    def complete_json(self, stage, payload):
        if stage == "web_evidence_judge":
            raise RuntimeError("mock API content filter")
        return super().complete_json(stage, payload)


class FakeWebSearcher:
    def __init__(self):
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return [
            {
                "id": "ev1",
                "query": query,
                "title": "Mock evidence",
                "snippet": "Mock evidence says the term is used as a slur.",
                "url": "https://example.test/evidence",
                "source": "mock",
            }
        ]

    def close(self):
        return None


class FakeTensor:
    is_cuda = False

    def __init__(self, values):
        self.values = values

    def numpy(self):
        return self.values


class FakeSentenceTransformer:
    def __init__(self, *_args, **_kwargs):
        pass

    def to(self, _device):
        return self

    def encode(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return FakeTensor(np.ones((len(texts), 3), dtype=np.float32))


class LLMLexiconBuilderTest(unittest.TestCase):
    def test_candidate_mining_suppresses_fragments_and_generic_words(self):
        records = [
            make_full_record("hate_1", "同性恋应该滚出这里，我不喜欢这种人", "同性恋", "LGBTQ", "hate"),
            make_full_record("hate_2", "真就高贵的同性恋，滚出我的视野", "同性恋", "LGBTQ", "hate"),
            make_full_record("normal_1", "同性恋群体应该得到平等尊重，我喜欢这种讨论", "同性恋", "non-hate", "non-hate"),
        ]
        settings = {
            "max_candidates": 20,
            "max_samples_per_candidate": 3,
            "zh_min_ngram": 2,
            "zh_max_ngram": 4,
            "zh_token_max_ngram": 4,
            "min_count_for_llm": 1,
            "min_hate_count_for_llm": 1,
            "suppressed_reject_hints": ["broken_fragment", "generic_word", "generic_phrase", "singleton_ngram", "substring_fragment"],
        }

        corpus = build_candidates("full", records, settings=settings, show_progress=False)
        all_terms = {candidate.term: candidate for candidate in corpus.candidates}
        selected_terms = {candidate.term: candidate for candidate in select_candidates(corpus, settings)}

        self.assertIn("同性恋", all_terms)
        self.assertIn("滚出", selected_terms)
        self.assertNotIn("喜欢", selected_terms)
        self.assertNotIn("性恋", selected_terms)
        self.assertEqual(all_terms["性恋"].substring_of, "同性恋")
        self.assertEqual(all_terms["性恋"].reject_hint, "substring_fragment")

    def test_cold_builder_uses_all_splits_and_writes_lexicon_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = root / "cold"
            output_dir = root / "generated"
            write_json(data_dir / "train.json", [make_cold_record("train_1", "黑火星人就是垃圾", "Racism")])
            write_json(data_dir / "val.json", [make_cold_record("val_1", "普通火星人电影很好", "non-hate")])
            write_json(data_dir / "test.json", [make_cold_record("test_1", "应该远离黑火星人", "Racism")])

            config = {
                "data_paths": {
                    "input_paths": [
                        str(data_dir / "train.json"),
                        str(data_dir / "val.json"),
                        str(data_dir / "test.json"),
                    ],
                    "output_dir": str(output_dir),
                },
                "candidate_settings": {
                    "max_candidates": 30,
                    "max_samples_per_candidate": 3,
                    "zh_min_ngram": 2,
                    "zh_max_ngram": 4,
                    "min_count_for_llm": 1,
                    "min_hate_count_for_llm": 1,
                },
                "web_settings": {"backend": "disabled"},
                "llm_settings": {"backend": "disabled"},
                "inclusion": {
                    "confidence_threshold": 0.65,
                    "single_mention_confidence": 0.85,
                    "min_count": 2,
                    "ambiguous_requires_nonhateful_meaning": True,
                },
                "runtime_settings": {"show_progress": False},
            }

            result = build_lexicon(
                "cold",
                config,
                judge_client=FakeJudgeClient({"黑火星人"}),
                web_searcher=FakeWebSearcher(),
            )

            self.assertEqual(result["total_records"], 3)
            lexicon = json.loads((output_dir / "lexicon.json").read_text(encoding="utf-8"))
            terms = {term["term"]: term for term in lexicon["terms"]}
            self.assertIn("黑火星人", terms)
            self.assertEqual(terms["黑火星人"]["category"], "Racism")
            self.assertEqual(terms["黑火星人"]["metadata"]["support"]["hate_count"], 2)
            self.assertTrue((output_dir / "candidates.jsonl").exists())
            self.assertTrue((output_dir / "web_evidence.jsonl").exists())
            self.assertTrue((output_dir / "llm_judgements.jsonl").exists())
            self.assertTrue((output_dir / "rejected.jsonl").exists())
            self.assertTrue((output_dir / "report.md").exists())

            install_rag_import_stubs()
            from rag.core import LexiconRetriever

            with patch("rag.core.SentenceTransformer", FakeSentenceTransformer):
                retriever = LexiconRetriever(
                    model_path="fake",
                    model_name="fake",
                    data_path=str(output_dir / "lexicon.json"),
                    enable_cache=False,
                    device="cpu",
                )
                exact = retriever.including_retrieve("这里出现了黑火星人这个词", top_k=-1, use_cache=False)
            self.assertEqual(len(exact), 1)

    def test_hatexplain_builder_outputs_hatebase_like_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = root / "hatexplain"
            output_dir = root / "generated"
            write_json(data_dir / "train.json", [make_hatexplain_record("train_1", "paki migrants")])
            write_json(data_dir / "val.json", [])
            write_json(data_dir / "test.json", [make_hatexplain_record("test_1", "paki migrants")])

            config = {
                "data_paths": {
                    "input_paths": [
                        str(data_dir / "train.json"),
                        str(data_dir / "val.json"),
                        str(data_dir / "test.json"),
                    ],
                    "output_dir": str(output_dir),
                },
                "candidate_settings": {
                    "max_candidates": 20,
                    "max_samples_per_candidate": 3,
                    "en_max_ngram": 3,
                    "min_count_for_llm": 1,
                    "min_hate_count_for_llm": 1,
                },
                "web_settings": {"backend": "disabled"},
                "llm_settings": {"backend": "disabled"},
                "inclusion": {"confidence_threshold": 0.65, "single_mention_confidence": 0.85, "min_count": 2},
                "runtime_settings": {"show_progress": False},
            }

            build_lexicon(
                "hatexplain",
                config,
                judge_client=FakeJudgeClient({"paki migrants"}),
                web_searcher=FakeWebSearcher(),
            )

            lexicon = json.loads((output_dir / "lexicon.json").read_text(encoding="utf-8"))
            self.assertEqual(lexicon["language"], "en")
            terms = {term["term"]: term for term in lexicon["terms"]}
            self.assertIn("paki migrants", terms)
            self.assertIn("categories", terms["paki migrants"])
            self.assertIn("nonhateful_meaning", terms["paki migrants"])
            self.assertIn("variants", terms["paki migrants"])

    def test_debug_mode_writes_search_trace_for_each_query(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = root / "cold"
            output_dir = root / "generated"
            write_json(data_dir / "train.json", [make_cold_record("train_1", "黑火星人", "Racism")])
            write_json(data_dir / "val.json", [])
            write_json(data_dir / "test.json", [make_cold_record("test_1", "黑火星人", "Racism")])

            config = {
                "data_paths": {
                    "input_paths": [
                        str(data_dir / "train.json"),
                        str(data_dir / "val.json"),
                        str(data_dir / "test.json"),
                    ],
                    "output_dir": str(output_dir),
                },
                "candidate_settings": {
                    "max_candidates": 1,
                    "max_samples_per_candidate": 3,
                    "zh_min_ngram": 3,
                    "zh_max_ngram": 4,
                    "min_count_for_llm": 1,
                    "min_hate_count_for_llm": 1,
                },
                "web_settings": {"backend": "disabled"},
                "llm_settings": {"backend": "disabled"},
                "inclusion": {"confidence_threshold": 0.65, "single_mention_confidence": 0.85, "min_count": 2},
                "runtime_settings": {"show_progress": False, "debug": True},
            }

            build_lexicon(
                "cold",
                config,
                judge_client=FakeJudgeClient({"黑火星人"}),
                web_searcher=FakeWebSearcher(),
            )

            search_log = output_dir / "debug" / "search_calls.jsonl"
            self.assertTrue(search_log.exists())
            rows = [json.loads(line) for line in search_log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["term"] == "黑火星人" for row in rows))
            self.assertIn("Mock evidence", rows[0]["results"][0]["title"])

    def test_llm_error_rejects_candidate_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = root / "cold"
            output_dir = root / "generated"
            write_json(data_dir / "train.json", [make_cold_record("train_1", "黑火星人", "Racism")])
            write_json(data_dir / "val.json", [])
            write_json(data_dir / "test.json", [make_cold_record("test_1", "黑火星人", "Racism")])

            config = {
                "data_paths": {
                    "input_paths": [
                        str(data_dir / "train.json"),
                        str(data_dir / "val.json"),
                        str(data_dir / "test.json"),
                    ],
                    "output_dir": str(output_dir),
                },
                "candidate_settings": {
                    "max_candidates": 1,
                    "max_samples_per_candidate": 3,
                    "zh_min_ngram": 3,
                    "zh_max_ngram": 4,
                    "min_count_for_llm": 1,
                    "min_hate_count_for_llm": 1,
                },
                "web_settings": {"backend": "disabled"},
                "llm_settings": {"backend": "disabled"},
                "inclusion": {"confidence_threshold": 0.65, "single_mention_confidence": 0.85, "min_count": 2},
                "runtime_settings": {
                    "show_progress": False,
                    "llm_failure_policy": "reject_candidate",
                    "max_llm_failures": 3,
                    "max_consecutive_llm_failures": 3,
                },
            }

            result = build_lexicon(
                "cold",
                config,
                judge_client=FailingWebJudgeClient({"黑火星人"}),
                web_searcher=FakeWebSearcher(),
            )

            self.assertEqual(result["included_count"], 0)
            self.assertEqual(result["llm_failure_count"], 1)
            judgement_rows = [
                json.loads(line)
                for line in (output_dir / "llm_judgements.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            rejected_rows = [
                json.loads(line)
                for line in (output_dir / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(judgement_rows[0]["llm_error"]["stage"], "web_evidence_judge")
            self.assertEqual(rejected_rows[0]["reject_reason"], "llm_error:web_evidence_judge")
            self.assertIn("mock API content filter", rejected_rows[0]["llm_error"]["message"])

    def test_resume_reuses_completed_judgement_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = root / "cold"
            output_dir = root / "generated"
            write_json(data_dir / "train.json", [make_cold_record("train_1", "黑火星人", "Racism")])
            write_json(data_dir / "val.json", [make_cold_record("val_1", "绿火星人", "Racism")])
            write_json(data_dir / "test.json", [make_cold_record("test_1", "绿火星人", "Racism")])
            output_dir.mkdir(parents=True, exist_ok=True)
            completed_judgement = {
                "rank": 1,
                "term": "黑火星人",
                "context_judge": {"supported": True, "confidence": 0.9, "reason": "done"},
                "web_evidence_judge": {"supported": True, "confidence": 0.9, "reason": "done", "evidence_ids": ["ev1"]},
                "final_lexicon_judge": {
                    "include": True,
                    "category": "Racism",
                    "categories": ["Racism"],
                    "definition": "already judged",
                    "nonhateful_meaning": "",
                    "variants": [],
                    "confidence": 0.9,
                    "reason": "already judged",
                    "evidence_ids": ["ev1"],
                },
            }
            write_jsonl([completed_judgement], output_dir / "llm_judgements.jsonl")
            write_jsonl(
                [{"rank": 1, "term": "黑火星人", "queries": [], "evidence": [{"id": "ev1"}]}],
                output_dir / "web_evidence.jsonl",
            )

            config = {
                "data_paths": {
                    "input_paths": [
                        str(data_dir / "train.json"),
                        str(data_dir / "val.json"),
                        str(data_dir / "test.json"),
                    ],
                    "output_dir": str(output_dir),
                },
                "candidate_settings": {
                    "max_candidates": 6,
                    "max_samples_per_candidate": 3,
                    "zh_min_ngram": 3,
                    "zh_max_ngram": 4,
                    "min_count_for_llm": 1,
                    "min_hate_count_for_llm": 1,
                },
                "web_settings": {"backend": "disabled"},
                "llm_settings": {"backend": "disabled"},
                "inclusion": {"confidence_threshold": 0.65, "single_mention_confidence": 0.85, "min_count": 1},
                "runtime_settings": {"show_progress": False, "resume": True},
            }
            judge = FakeJudgeClient({"绿火星人"})

            result = build_lexicon("cold", config, judge_client=judge, web_searcher=FakeWebSearcher())

            called_terms = {term for _stage, term in judge.calls}
            self.assertNotIn("黑火星人", called_terms)
            terms = {term["term"]: term for term in result["terms"]}
            self.assertIn("黑火星人", terms)
            state = json.loads((output_dir / "build_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["completed_count"], result["candidate_count"])


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeHTTPErrorResponse(FakeResponse):
    def __init__(self, status_code, payload):
        super().__init__(payload)
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def raise_for_status(self):
        raise requests.HTTPError(f"{self.status_code} Client Error", response=self)


class WebSearcherConfigTest(unittest.TestCase):
    def test_search_api_supports_google_custom_search_params(self):
        calls = []

        def fake_get(url, params, headers, timeout):
            calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
            return FakeResponse(
                {
                    "items": [
                        {
                            "title": "Example result",
                            "snippet": "A short result snippet.",
                            "link": "https://example.test",
                            "displayLink": "example.test",
                        }
                    ]
                }
            )

        searcher = WebSearcher(
            {
                "backend": "search_api",
                "api_base": "https://www.googleapis.com/customsearch/v1",
                "api_query_param": "q",
                "api_key": "KEY",
                "api_key_param": "key",
                "api_extra_params": {"cx": "CX", "num": 3},
                "api_results_path": "items",
                "cache_enabled": False,
            }
        )
        with patch("build_lex.web_search.requests.get", fake_get):
            evidence = searcher.search('"term" slur meaning')

        self.assertEqual(evidence[0]["source"], "example.test")
        self.assertEqual(calls[0]["params"]["q"], '"term" slur meaning')
        self.assertEqual(calls[0]["params"]["key"], "KEY")
        self.assertEqual(calls[0]["params"]["cx"], "CX")
        self.assertEqual(calls[0]["params"]["num"], 3)

    def test_search_api_supports_bearer_post_body(self):
        calls = []

        def fake_post(url, json, headers, timeout):
            calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
            return FakeResponse(
                {
                    "results": [
                        {
                            "title": "Tavily-style result",
                            "content": "A compact search snippet.",
                            "url": "https://example.test/tavily",
                        }
                    ]
                }
            )

        searcher = WebSearcher(
            {
                "backend": "search_api",
                "api_base": "https://api.tavily.com/search",
                "api_method": "POST",
                "api_query_param": "query",
                "api_key": "TVLY",
                "api_key_header": "Authorization",
                "api_key_header_prefix": "Bearer ",
                "api_extra_params": {"max_results": 3, "search_depth": "basic"},
                "api_results_path": "results",
                "cache_enabled": False,
            }
        )
        with patch("build_lex.web_search.requests.post", fake_post):
            evidence = searcher.search("term hate speech")

        self.assertEqual(evidence[0]["snippet"], "A compact search snippet.")
        self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer TVLY")
        self.assertEqual(calls[0]["json"]["query"], "term hate speech")
        self.assertEqual(calls[0]["json"]["max_results"], 3)

    def test_search_api_fails_fast_when_auth_key_is_missing(self):
        searcher = WebSearcher(
            {
                "backend": "search_api",
                "api_base": "https://api.tavily.com/search",
                "api_method": "POST",
                "api_query_param": "query",
                "api_key_env": "MISSING_TAVILY_KEY_FOR_TEST",
                "api_key_header": "Authorization",
                "api_key_header_prefix": "Bearer ",
                "cache_enabled": False,
            }
        )
        with self.assertRaisesRegex(ValueError, "MISSING_TAVILY_KEY_FOR_TEST"):
            searcher.search("term hate speech")


class DeepSeekJudgementClientTest(unittest.TestCase):
    def test_deepseek_backend_builds_official_chat_completion_payload(self):
        calls = []

        def fake_post(url, json, headers, timeout):
            calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
            return FakeResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": '{"supported": true, "confidence": 0.91, "reason": "ok"}',
                                "reasoning_content": "hidden reasoning",
                            },
                        }
                    ]
                }
            )

        client = create_judgement_client(
            {
                "backend": "deepseek",
                "api_key": "DSK",
                "model": "deepseek-v4-pro",
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
                "max_tokens": 777,
            }
        )
        self.assertIsInstance(client, OpenAICompatibleJudgementClient)
        with patch("build_lex.llm_lexicon_builder.requests.post", fake_post):
            result = client.complete_json("context_judge", {"candidate": {"term": "x"}})

        self.assertTrue(result["supported"])
        self.assertEqual(calls[0]["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer DSK")
        self.assertEqual(calls[0]["json"]["model"], "deepseek-v4-pro")
        self.assertEqual(calls[0]["json"]["response_format"], {"type": "json_object"})
        self.assertEqual(calls[0]["json"]["thinking"], {"type": "enabled"})
        self.assertEqual(calls[0]["json"]["reasoning_effort"], "high")
        self.assertEqual(calls[0]["json"]["max_tokens"], 777)
        self.assertFalse(calls[0]["json"]["stream"])
        self.assertNotIn("temperature", calls[0]["json"])

    def test_http_400_body_is_recorded_and_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            calls = []
            recorder = DebugRecorder(tmp_dir, True)

            def fake_post(url, json, headers, timeout):
                calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
                return FakeHTTPErrorResponse(400, {"error": {"message": "content filter triggered"}})

            client = create_judgement_client(
                {
                    "backend": "deepseek",
                    "api_key": "DSK",
                    "model": "deepseek-v4-pro",
                    "retries": 2,
                    "max_tokens": 200,
                },
                debug_recorder=recorder,
            )
            with patch("build_lex.llm_lexicon_builder.requests.post", fake_post):
                with self.assertRaises(LLMAPIError):
                    client.complete_json("context_judge", {"candidate": {"term": "blocked-term"}})

            self.assertEqual(len(calls), 1)
            rows = [
                json.loads(line)
                for line in (Path(tmp_dir) / "debug" / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[0]["raw_response"]["error"]["status_code"], 400)
            self.assertIn("content filter triggered", rows[0]["raw_response"]["error"]["response_body"])

    def test_debug_recorder_writes_llm_input_and_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorder = DebugRecorder(tmp_dir, True)

            def fake_post(url, json, headers, timeout):
                return FakeResponse(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": '{"supported": true, "confidence": 0.8, "reason": "ok"}'},
                            }
                        ]
                    }
                )

            client = create_judgement_client(
                {
                    "backend": "deepseek",
                    "api_key": "DSK",
                    "model": "deepseek-v4-pro",
                    "max_tokens": 200,
                },
                debug_recorder=recorder,
            )
            with patch("build_lex.llm_lexicon_builder.requests.post", fake_post):
                client.complete_json("context_judge", {"candidate": {"term": "debug-term"}})

            rows = [
                json.loads(line)
                for line in (Path(tmp_dir) / "debug" / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["term"], "debug-term")
            self.assertEqual(rows[0]["stage"], "context_judge")
            self.assertIn("messages", rows[0]["request_payload"])
            self.assertEqual(rows[0]["parsed_response"]["confidence"], 0.8)


def make_cold_record(record_id, content, group):
    return {
        "id": record_id,
        "content": content,
        "quadruples": [
            {
                "target": "NULL",
                "argument": content,
                "targeted_group": group,
                "hateful": "non-hate" if group == "non-hate" else "hate",
            }
        ],
    }


def make_full_record(record_id, content, target, group, hateful):
    return {
        "id": record_id,
        "content": content,
        "quadruples": [
            {
                "target": target,
                "argument": content,
                "targeted_group": group,
                "hateful": hateful,
            }
        ],
    }


def make_hatexplain_record(record_id, rationale):
    tokens = ["send", "back", *rationale.split()]
    return {
        "id": record_id,
        "content": " ".join(tokens),
        "tokens": tokens,
        "annotation": {
            "label": "offensive",
            "target_groups": ["Indian", "Refugee"],
            "rationales": [{"token_indices": [2, 3], "text": rationale}],
        },
    }


if __name__ == "__main__":
    unittest.main()
