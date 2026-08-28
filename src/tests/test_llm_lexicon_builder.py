import json
import runpy
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import requests

from build_lex.llm_lexicon_builder import (
    CandidateJudgementError,
    CandidateStats,
    CorpusCandidateResult,
    DebugRecorder,
    LLMAPIError,
    OpenAICompatibleJudgementClient,
    TERMINOLOGY_LIBRARY_ROLE,
    TERMINOLOGY_OBJECTIVE,
    TERMINOLOGY_SOURCE_POLICY,
    build_candidates,
    build_lexicon,
    create_judgement_client,
    default_config,
    judge_candidate,
    render_stage_prompt,
    score_candidates,
    select_candidates,
    should_include,
    write_jsonl,
)
from build_lex.formal_checkpoint import CheckpointSpec, FormalLexiconCheckpoint
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


def provider_checkpoint_spec(*, tavily_cap=3, deepseek_cap=3):
    return CheckpointSpec.build(
        intent={
            "authorization_sha256": "a" * 64,
            "builder_code_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "data_build_id": "data-fixture",
            "protocol_code_sha256s": {"fixture.py": "d" * 64},
            "train_records_sha256": "e" * 64,
        },
        provider_caps={
            "tavily": {"cap": tavily_cap, "scope_id": "tavily-key2/v1"},
            "deepseek": {"cap": deepseek_cap, "scope_id": "deepseek/v1"},
        },
        candidate_frame=[{"rank": 1, "term": "term"}],
        max_slot_attempts=3,
    )


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
    def test_duplicate_entrypoint_namespace_resolves_canonical_formal_builder(self):
        builder_module = sys.modules[build_lexicon.__module__]
        duplicate_namespace = runpy.run_path(
            str(Path(builder_module.__file__).resolve()),
            run_name="formal_entrypoint_probe",
        )

        self.assertIs(
            duplicate_namespace["_canonical_formal_builder"](),
            build_lexicon,
        )

    def test_duplicate_entrypoint_main_hands_off_canonical_formal_builder(self):
        builder_module = sys.modules[build_lexicon.__module__]
        duplicate_namespace = runpy.run_path(
            str(Path(builder_module.__file__).resolve()),
            run_name="formal_entrypoint_main_probe",
        )
        entrypoint_main = duplicate_namespace["main"]
        args = types.SimpleNamespace(
            dataset="full",
            config="config.json",
            debug=False,
            debug_dir=None,
            no_resume=False,
            data_ref="data_ref.json",
            train_partition_ref="train_partition_ref.json",
            train_records=None,
            split="train",
            target_root="lexicons",
            write_ref="lexicon_ref.json",
            env_file=".env",
            workspace_root=".",
        )
        entrypoint_main.__globals__["parse_args"] = lambda: args
        frozen_config = {"runtime_settings": {"log_level": "INFO", "resume": True}}
        preflight = types.SimpleNamespace(
            passed=True,
            activate_credentials=lambda: None,
            frozen_config=lambda: frozen_config,
            build_authorization=object(),
        )

        with (
            patch(
                "build_lex.stage1_preflight.preflight_formal_lexicon",
                return_value=preflight,
            ),
            patch(
                "build_lex.train_only.build_train_only_lexicon",
                return_value={"artifact_id": "lex-test", "terms": []},
            ) as build_mock,
            patch("requests.sessions.Session.request") as request_mock,
            patch("builtins.print"),
        ):
            entrypoint_main()

        self.assertIs(build_mock.call_args.kwargs["builder"], build_lexicon)
        self.assertEqual(
            build_mock.call_args.kwargs["builder"].__module__,
            "build_lex.llm_lexicon_builder",
        )
        request_mock.assert_not_called()

    def test_judgement_payload_separates_context_and_web_evidence_ids(self):
        candidate = CandidateStats(
            term="neutral-group-label",
            dataset="full",
            language="en",
            total_count=1,
            hate_count=1,
            category_counts=Counter({"LGBTQ": 1}),
            support_sample_ids=["train-sample-17"],
            sample_contexts=[
                {
                    "id": "train-sample-17",
                    "label": "hate",
                    "categories": ["LGBTQ"],
                    "source": "content",
                    "content": "A hateful sentence containing a neutral protected-identity label.",
                }
            ],
        )

        class CaptureClient:
            def __init__(self):
                self.payloads = {}

            def complete_json(self, stage, payload):
                self.payloads[stage] = payload
                if stage == "context_judge":
                    return {
                        "supported": False,
                        "category": "others",
                        "categories": ["others"],
                        "confidence": 0.95,
                        "reason": "The exact term is a neutral identity label.",
                    }
                if stage == "web_evidence_judge":
                    return {
                        "supported": False,
                        "confidence": 0.9,
                        "reason": "No independent derogatory sense is established.",
                        "evidence_ids": [],
                    }
                return {
                    "include": False,
                    "category": "others",
                    "categories": ["others"],
                    "definition": "",
                    "nonhateful_meaning": "sexual-orientation identity label",
                    "variants": [],
                    "confidence": 0.96,
                    "reason": "Surrounding toxicity is not attributable to the term.",
                    "evidence_ids": [],
                }

        client = CaptureClient()
        judge_candidate(
            candidate,
            [{"id": "web-1", "title": "Evidence", "snippet": "Neutral identity usage"}],
            client,
        )

        context_candidate = client.payloads["context_judge"]["candidate"]
        self.assertNotIn("support_sample_ids", context_candidate)
        self.assertNotIn("id", context_candidate["sample_contexts"][0])
        self.assertEqual(context_candidate["sample_contexts"][0]["example_ordinal"], 1)
        for stage in ("web_evidence_judge", "final_lexicon_judge"):
            contract = client.payloads[stage]["decision_contract"]
            self.assertEqual(contract["allowed_web_evidence_ids"], ["web-1"])
            self.assertIn("Training sample IDs", contract["citation_rule"])

    def test_stage_prompts_make_attribution_identity_and_category_contracts_explicit(self):
        payload = {
            "candidate": {"term": "neutral-group-label"},
            "decision_contract": {"allowed_web_evidence_ids": ["web-1"]},
        }
        context_prompt = render_stage_prompt("context_judge", payload)
        final_prompt = render_stage_prompt("final_lexicon_judge", payload)

        self.assertIn("positive example label is not evidence", context_prompt)
        self.assertIn("neutral protected-identity label", final_prompt)
        self.assertIn("categories=[\"others\"]", final_prompt)
        self.assertIn("allowed_web_evidence_ids", final_prompt)
        self.assertIn("Training sample IDs", final_prompt)
        self.assertIn("downstream model-visible evidence", final_prompt)
        self.assertIn("do not copy a canonical task-category label", final_prompt)

    def test_terminology_mode_is_category_free_and_label_independent(self):
        settings = {
            "objective": TERMINOLOGY_OBJECTIVE,
            "source_policy": TERMINOLOGY_SOURCE_POLICY,
            "zh_min_ngram": 2,
            "zh_max_ngram": 4,
            "zh_token_max_ngram": 4,
            "use_jieba": False,
            "keep_all_content_ngrams": True,
            "max_samples_per_candidate": 5,
        }
        hateful = {
            "id": "1",
            "content": "男同舔狗",
            "quadruples": [
                {
                    "target": "男同",
                    "argument": "男同舔狗",
                    "targeted_group": ["LGBTQ"],
                    "hateful": "hate",
                }
            ],
        }
        neutral = {
            **hateful,
            "quadruples": [
                {
                    "target": None,
                    "argument": "男同舔狗",
                    "targeted_group": ["non-hate"],
                    "hateful": "non-hate",
                }
            ],
        }
        first = build_candidates("full", [hateful], settings=settings)
        second = build_candidates("full", [neutral], settings=settings)

        self.assertEqual(
            [candidate.to_payload() for candidate in first.candidates],
            [candidate.to_payload() for candidate in second.candidates],
        )
        self.assertEqual(first.hate_records, 0)
        self.assertEqual(first.non_hate_records, 0)
        self.assertIn("男同", {candidate.term for candidate in first.candidates})
        for candidate in first.candidates:
            self.assertEqual(candidate.category_counts, Counter())
            for context in candidate.sample_contexts:
                self.assertNotIn("label", context)
                self.assertNotIn("categories", context)

        class TerminologyClient:
            def __init__(self):
                self.payloads = {}

            def complete_json(self, stage, payload):
                self.payloads[stage] = payload
                if stage == "context_judge":
                    return {
                        "supported": True,
                        "confidence": 0.9,
                        "reason": "该词有需解释的用法。",
                    }
                if stage == "web_evidence_judge":
                    return {
                        "supported": False,
                        "confidence": 0.9,
                        "reason": "没有提供联网证据。",
                        "evidence_ids": [],
                    }
                return {
                    "include": True,
                    "definition": "男同性恋者的常见简称。",
                    "usage_notes": "可作中性身份简称，态度取决于语境。",
                    "ambiguity_notes": "单独提及不代表攻击。",
                    "variants": [],
                    "confidence": 0.9,
                    "reason": "解释有助于理解。",
                    "evidence_ids": [],
                }

        client = TerminologyClient()
        judgement = judge_candidate(
            next(candidate for candidate in first.candidates if candidate.term == "男同"),
            [],
            client,
            resource_role=TERMINOLOGY_LIBRARY_ROLE,
        )
        self.assertNotIn("category", judgement["final_lexicon_judge"])
        for payload in client.payloads.values():
            candidate_payload = payload["candidate"]
            self.assertTrue(
                {
                    "category",
                    "categories",
                    "hate_count",
                    "hate_precision",
                    "primary_category",
                }.isdisjoint(candidate_payload)
            )

        prompt_payload = {
            "candidate": {"term": "男同"},
            "decision_contract": {
                "resource_role": TERMINOLOGY_LIBRARY_ROLE,
                "allowed_web_evidence_ids": [],
            },
        }
        final_prompt = render_stage_prompt(
            "final_lexicon_judge", prompt_payload, output_language="zh"
        )
        self.assertIn("category-free terminology-understanding", final_prompt)
        self.assertIn("Neutral identity terms", final_prompt)
        self.assertIn("Never emit category", final_prompt)
        self.assertNotIn('categories=["others"]', final_prompt)

    def test_terminology_judgement_rejects_extra_response_fields(self):
        class LeakyClient:
            def complete_json(self, stage, _payload):
                if stage == "context_judge":
                    return {
                        "supported": True,
                        "confidence": 0.9,
                        "reason": "fixture",
                        "metadata": {"primary_category": "LGBTQ"},
                    }
                raise AssertionError("later stages must not run")

        with self.assertRaisesRegex(
            CandidateJudgementError, "context_judge"
        ):
            judge_candidate(
                CandidateStats(
                    term="男同",
                    dataset="full",
                    language="zh",
                    total_count=1,
                    source_counts=Counter({"content": 1}),
                    support_sample_ids=["1"],
                    sample_contexts=[
                        {"id": "1", "source": "content", "content": "男同"}
                    ],
                ),
                [],
                LeakyClient(),
                resource_role=TERMINOLOGY_LIBRARY_ROLE,
            )

    def test_configured_neutral_identity_policy_is_a_deterministic_safety_net(self):
        candidate = CandidateStats(
            term="identity-label-fixture",
            dataset="full",
            language="en",
            total_count=5,
            hate_count=5,
            annotation_count=2,
        )
        include, reason = should_include(
            candidate,
            {
                "include": True,
                "confidence": 0.99,
                "nonhateful_meaning": "neutral identity reference",
            },
            {
                "confidence_threshold": 0.65,
                "min_count": 2,
                "neutral_identity_terms": ["IDENTITY-LABEL-FIXTURE"],
            },
        )

        self.assertFalse(include)
        self.assertIn("configured neutral protected-identity", reason)

    def test_judge_candidate_rejects_training_id_as_web_citation(self):
        candidate = CandidateStats(
            term="fixture-term",
            dataset="full",
            language="en",
        )

        class InvalidCitationClient:
            def complete_json(self, stage, _payload):
                if stage == "context_judge":
                    return {
                        "supported": False,
                        "category": "others",
                        "categories": ["others"],
                        "confidence": 0.9,
                        "reason": "fixture",
                    }
                return {
                    "supported": True,
                    "confidence": 0.9,
                    "reason": "fixture",
                    "evidence_ids": ["train-sample-id"],
                }

        with self.assertRaises(CandidateJudgementError) as raised:
            judge_candidate(
                candidate,
                [{"id": "web-1", "title": "fixture", "snippet": "fixture"}],
                InvalidCitationClient(),
            )
        self.assertEqual(raised.exception.stage, "web_evidence_judge")

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

    def test_mixed_zh_corpus_mines_english_phrases_and_suppresses_attributed_substrings(self):
        records = [
            make_full_record(
                "hate_1",
                "easy girl可不是国男喊的",
                "easy girl",
                "Sexism",
                "hate",
            ),
            make_full_record(
                "hate_2",
                "在easy girl眼里这也比国男好得多",
                "easy girl",
                "Sexism",
                "hate",
            ),
        ]
        settings = {
            "max_candidates": 30,
            "max_samples_per_candidate": 3,
            "zh_min_ngram": 2,
            "zh_max_ngram": 4,
            "zh_token_max_ngram": 4,
            "en_max_ngram": 3,
            "use_jieba": False,
            "min_count_for_llm": 1,
            "min_hate_count_for_llm": 1,
            "suppressed_reject_hints": ["substring_fragment"],
        }

        corpus = build_candidates("full", records, settings=settings, show_progress=False)
        all_terms = {candidate.term: candidate for candidate in corpus.candidates}
        selected_terms = {candidate.term for candidate in select_candidates(corpus, settings)}

        self.assertIn("easy girl", all_terms)
        self.assertEqual(all_terms["easy"].language, "en")
        self.assertEqual(all_terms["easy"].substring_of, "easy girl")
        self.assertEqual(all_terms["easy"].reject_hint, "substring_fragment")
        self.assertNotIn("easy", selected_terms)
        self.assertIn("easy girl", selected_terms)

    def test_case_study_boundary_fragments_are_filtered_before_provider_selection(self):
        contents = [
            "一些女拳不讲道理",
            "有些女拳都这样",
            "这些女拳自称清醒",
            "女拳在网上骂人",
            "都让黑蛆喝了吧",
        ]
        records = [
            make_full_record(
                f"hate_{index}",
                content,
                "女拳" if "女拳" in content else "黑蛆",
                "Sexism" if "女拳" in content else "Racism",
                "hate",
            )
            for index, content in enumerate(contents, start=1)
        ]
        settings = {
            "max_candidates": 100,
            "max_samples_per_candidate": 5,
            "zh_min_ngram": 2,
            "zh_max_ngram": 4,
            "zh_token_max_ngram": 4,
            "use_jieba": False,
            "min_count_for_llm": 1,
            "min_hate_count_for_llm": 1,
            "suppressed_reject_hints": [
                "broken_fragment",
                "singleton_ngram",
                "substring_fragment",
            ],
        }

        corpus = build_candidates("full", records, settings=settings, show_progress=False)
        all_terms = {candidate.term: candidate for candidate in corpus.candidates}
        selected_terms = {candidate.term for candidate in select_candidates(corpus, settings)}

        for fragment in ("些女拳", "女拳不", "女拳都", "女拳自", "拳在"):
            self.assertIn(fragment, all_terms)
            self.assertEqual(all_terms[fragment].reject_hint, "substring_fragment")
            self.assertNotIn(fragment, selected_terms)
        self.assertNotIn("都让黑蛆喝了吧", all_terms)

    def test_web_support_requires_a_cited_supplied_evidence_record(self):
        candidate = CandidateStats(
            term="黑蛆",
            dataset="full",
            language="zh",
            total_count=2,
            hate_count=2,
            category_counts=Counter({"Racism": 2}),
        )

        class EmptyEvidenceSupportClient:
            def complete_json(self, stage, _payload):
                if stage == "context_judge":
                    return {
                        "supported": True,
                        "category": "Racism",
                        "categories": ["Racism"],
                        "confidence": 0.95,
                        "reason": "dataset support",
                    }
                if stage == "web_evidence_judge":
                    return {
                        "supported": True,
                        "confidence": 0.95,
                        "reason": "claimed Web support",
                        "evidence_ids": [],
                    }
                raise AssertionError(stage)

        with self.assertRaises(CandidateJudgementError) as raised:
            judge_candidate(candidate, [], EmptyEvidenceSupportClient())
        self.assertEqual(raised.exception.stage, "web_evidence_judge")

    def test_final_include_requires_at_least_one_supporting_evidence_judge(self):
        candidate = CandidateStats(
            term="德行",
            dataset="full",
            language="zh",
            total_count=2,
            hate_count=2,
            category_counts=Counter({"others": 2}),
        )

        class UnsupportedIncludeClient:
            def complete_json(self, stage, _payload):
                if stage == "context_judge":
                    return {
                        "supported": False,
                        "category": "others",
                        "categories": ["others"],
                        "confidence": 0.9,
                        "reason": "generic word",
                    }
                if stage == "web_evidence_judge":
                    return {
                        "supported": False,
                        "confidence": 0.9,
                        "reason": "no lexical support",
                        "evidence_ids": [],
                    }
                return {
                    "include": True,
                    "category": "others",
                    "categories": ["others"],
                    "definition": "generic contextual criticism",
                    "nonhateful_meaning": "ordinary word for conduct",
                    "variants": [],
                    "confidence": 0.9,
                    "reason": "incorrect override",
                    "evidence_ids": [],
                }

        with self.assertRaises(CandidateJudgementError) as raised:
            judge_candidate(candidate, [], UnsupportedIncludeClient())
        self.assertEqual(raised.exception.stage, "final_lexicon_judge")

    def test_directional_coverage_filters_zh_fragments_without_suppressing_complete_terms(self):
        def candidate(term, record_ids, track="ngram_backoff"):
            record_ids = [str(record_id) for record_id in record_ids]
            item = CandidateStats(term=term, dataset="full", language="zh")
            item.support_sample_ids = record_ids
            item.total_count = len(record_ids)
            item.hate_count = len(record_ids)
            item.track_counts[track] = len(record_ids)
            item.category_counts["Racism"] = len(record_ids)
            return item

        truncated = candidate("德绑架", range(1, 8))
        full_phrase = candidate("道德绑架", range(1, 7))

        dangling = candidate("天被", range(11, 18))
        dangling_extensions = [
            candidate("天被骂", range(11, 13)),
            candidate("天被家", range(13, 15)),
            candidate("天被黑", [15]),
            candidate("天被麻", [16]),
            candidate("天被世", [17]),
        ]

        # Every occurrence of this complete word has some left extension, but
        # no single extension dominates and it has no dangling suffix.
        complete_word = candidate("绑架", range(21, 31))
        varied_extensions = [
            candidate("甲绑架", range(21, 28)),
            candidate("乙绑架", [28]),
            candidate("丙绑架", [29]),
            candidate("丁绑架", [30]),
        ]

        # Direct annotation evidence protects a complete phrase even when its
        # corpus usage happens to have one dominant modifier.
        annotated_word = candidate("仙女", range(31, 41), track="annotation_anchor")
        modified_word = candidate("小仙女", range(31, 40))

        # ``被`` is also a noun character.  A dominant *left* modifier does
        # not prove that a term ending in it is a dangling passive particle.
        homographic_noun = candidate("棉被", range(41, 51))
        modified_noun = candidate("旧棉被", range(41, 50))

        candidates = [
            truncated,
            full_phrase,
            dangling,
            *dangling_extensions,
            complete_word,
            *varied_extensions,
            annotated_word,
            modified_word,
            homographic_noun,
            modified_noun,
        ]
        score_candidates(candidates, hate_records=50, non_hate_records=1)
        corpus = CorpusCandidateResult(
            candidates=candidates,
            total_records=50,
            hate_records=50,
            non_hate_records=0,
            input_paths=[],
        )
        selected = {
            item.term
            for item in select_candidates(
                corpus,
                {
                    "max_candidates": 100,
                    "min_count_for_llm": 1,
                    "min_hate_count_for_llm": 1,
                },
            )
        }

        self.assertEqual(truncated.reject_hint, "substring_fragment")
        self.assertEqual(truncated.substring_of, "道德绑架")
        self.assertEqual(dangling.reject_hint, "substring_fragment")
        self.assertNotIn("德绑架", selected)
        self.assertNotIn("天被", selected)
        self.assertIn("绑架", selected)
        self.assertIn("仙女", selected)
        self.assertIn("棉被", selected)
        self.assertNotEqual(complete_word.reject_hint, "substring_fragment")
        self.assertNotEqual(annotated_word.reject_hint, "substring_fragment")
        self.assertNotEqual(homographic_noun.reject_hint, "substring_fragment")

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
        self.status_code = 200

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
    def test_normalized_evidence_is_bounded_before_direct_match_and_identity(self):
        settings = {
            "backend": "disabled",
            "require_direct_term_match": True,
            "dedupe_by_url": True,
            "max_title_chars": 10,
            "max_snippet_chars": 20,
            "max_url_chars": 30,
            "max_source_chars": 5,
        }
        searcher = WebSearcher(settings)
        evidence = searcher._normalize_search_results(
            {
                "results": [
                    {
                        "title": "neutral",
                        "content": "x" * 25 + " term",
                        "url": "https://example.test/filtered",
                        "source": "source",
                    },
                    {
                        "title": "term-" + "x" * 30,
                        "content": "term " + "y" * 40,
                        "url": "https://example.test/" + "z" * 80,
                        "source": "source-name",
                    },
                ]
            },
            '"term" slur meaning',
        )

        self.assertEqual(len(evidence), 1)
        self.assertEqual(len(evidence[0]["title"]), 10)
        self.assertTrue(evidence[0]["title"].endswith("…"))
        self.assertEqual(len(evidence[0]["snippet"]), 20)
        self.assertEqual(evidence[0]["url"], "")
        self.assertEqual(len(evidence[0]["source"]), 5)

        different_limits = dict(settings)
        different_limits["max_snippet_chars"] = 21
        self.assertNotEqual(
            searcher._cache_key('"term" slur meaning'),
            WebSearcher(different_limits)._cache_key('"term" slur meaning'),
        )

    def test_direct_term_policy_filters_generic_results_and_normalizes_urls(self):
        searcher = WebSearcher(
            {
                "backend": "disabled",
                "require_direct_term_match": True,
                "dedupe_by_url": True,
            }
        )
        evidence = searcher._normalize_search_results(
            {
                "results": [
                    {
                        "title": "Generic hate speech overview",
                        "content": "A page about discrimination in general.",
                        "url": "https://example.test/generic?tracking=1#top",
                    },
                    {
                        "title": "Why 偷井盖 became a regional stereotype",
                        "content": "The phrase 偷井盖 is discussed directly.",
                        "url": "HTTPS://EXAMPLE.TEST/direct/?tracking=1#top",
                    },
                    {
                        "title": "Duplicate 偷井盖 result",
                        "content": "偷井盖 again.",
                        "url": "https://example.test/direct#other",
                    },
                ]
            },
            '"偷井盖" 歧视 用语',
        )

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["url"], "https://example.test/direct")

    def test_direct_term_policy_uses_word_boundaries_for_latin_terms(self):
        searcher = WebSearcher(
            {
                "backend": "disabled",
                "require_direct_term_match": True,
                "dedupe_by_url": True,
            }
        )
        evidence = searcher._normalize_search_results(
            {
                "results": [
                    {
                        "title": "Gateway documentation",
                        "content": "A gateway is not the requested identity word.",
                        "url": "https://example.test/gateway",
                    },
                    {
                        "title": "Gay identity terminology",
                        "content": "Gay is ordinarily an identity label.",
                        "url": "https://example.test/gay",
                    },
                ]
            },
            '"gay" offensive term protected group',
        )

        self.assertEqual([item["title"] for item in evidence], ["Gay identity terminology"])

    def test_normalized_url_evidence_id_is_stable_across_queries(self):
        searcher = WebSearcher(
            {
                "backend": "disabled",
                "require_direct_term_match": True,
                "dedupe_by_url": True,
            }
        )
        payload = {
            "results": [
                {
                    "title": "偷井盖 stereotype",
                    "content": "偷井盖 is discussed directly.",
                    "url": "https://example.test/article?from=search#section",
                }
            ]
        }

        first = searcher._normalize_search_results(payload, '"偷井盖" 侮辱性 含义')
        second = searcher._normalize_search_results(payload, '"偷井盖" 歧视 用语')

        self.assertEqual(first[0]["id"], second[0]["id"])
        self.assertEqual(first[0]["url"], "https://example.test/article")

    def test_semantic_url_query_is_preserved_while_tracking_is_removed(self):
        searcher = WebSearcher(
            {
                "backend": "disabled",
                "require_direct_term_match": True,
                "dedupe_by_url": True,
            }
        )
        evidence = searcher._normalize_search_results(
            {
                "results": [
                    {
                        "title": "偷井盖 article one",
                        "url": "https://example.test/article?id=1&utm_source=search#top",
                    },
                    {
                        "title": "偷井盖 article two",
                        "url": "https://example.test/article?id=2&utm_source=search#top",
                    },
                ]
            },
            '"偷井盖" 侮辱性 含义',
        )

        self.assertEqual(
            [item["url"] for item in evidence],
            [
                "https://example.test/article?id=1",
                "https://example.test/article?id=2",
            ],
        )
        self.assertEqual(len({item["id"] for item in evidence}), 2)

    def test_search_api_supports_google_custom_search_params(self):
        calls = []

        def fake_get(url, params, headers, timeout, allow_redirects):
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

        def fake_post(url, json, headers, timeout, allow_redirects):
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

    def test_tavily_transient_retry_is_narrow_and_bounded(self):
        calls = []

        def fake_post(*_args, **_kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise requests.exceptions.SSLError("unexpected EOF")
            return FakeResponse({"results": []})

        searcher = WebSearcher(
            {
                "backend": "search_api",
                "api_base": "https://api.tavily.com/search",
                "api_method": "POST",
                "api_query_param": "query",
                "api_key": "TVLY",
                "api_key_header": "Authorization",
                "cache_enabled": False,
                "transport_retry_policy": {
                    "id": "tavily-transient/v1",
                    "retries": 2,
                    "base_sleep_seconds": 0,
                },
            }
        )
        with patch("build_lex.web_search.requests.post", side_effect=fake_post):
            self.assertEqual(searcher.search("term hate speech"), [])
        self.assertEqual(len(calls), 2)

    def test_checkpointed_tavily_redirect_is_terminal_and_never_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            checkpoint = FormalLexiconCheckpoint.create(
                root, provider_checkpoint_spec(tavily_cap=3)
            )
            searcher = WebSearcher(
                {
                    "backend": "search_api",
                    "api_base": "https://api.tavily.com/search",
                    "api_method": "POST",
                    "api_query_param": "query",
                    "api_key": "TVLY",
                    "api_key_header": "Authorization",
                    "cache_enabled": False,
                    "transport_retry_policy": {
                        "id": "tavily-transient/v1",
                        "retries": 2,
                        "base_sleep_seconds": 0,
                    },
                },
                formal_checkpoint=checkpoint,
            )
            redirect = FakeResponse({"results": []})
            redirect.status_code = 302
            with patch(
                "build_lex.web_search.requests.post", return_value=redirect
            ) as post:
                with self.assertRaises(requests.HTTPError):
                    searcher.search(
                        "term hate speech",
                        checkpoint_rank=1,
                        checkpoint_slot=1,
                        checkpoint_term="term",
                    )
            self.assertEqual(post.call_count, 1)
            self.assertIs(post.call_args.kwargs["allow_redirects"], False)
            self.assertEqual(
                checkpoint.summary()["provider_attempt_counts"]["tavily"], 1
            )
            self.assertFalse(checkpoint.can_retry("tavily", 1, "query_1"))
            checkpoint.close()

    def test_checkpointed_tavily_counts_retry_and_reuses_success_after_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            spec = provider_checkpoint_spec(tavily_cap=3)
            checkpoint = FormalLexiconCheckpoint.create(root, spec)
            settings = {
                "backend": "search_api",
                "api_base": "https://api.tavily.com/search",
                "api_method": "POST",
                "api_query_param": "query",
                "api_key": "TVLY",
                "api_key_header": "Authorization",
                "api_extra_params": {"max_results": 3, "search_depth": "basic"},
                "api_results_path": "results",
                "cache_enabled": False,
                "transport_retry_policy": {
                    "id": "tavily-transient/v1",
                    "retries": 2,
                    "base_sleep_seconds": 0,
                },
            }
            searcher = WebSearcher(settings, formal_checkpoint=checkpoint)
            responses = [
                requests.exceptions.SSLError("unexpected EOF"),
                FakeResponse(
                    {
                        "results": [
                            {
                                "title": "term evidence",
                                "content": "term is discussed",
                                "url": "https://example.test/term",
                            }
                        ]
                    }
                ),
            ]
            with patch(
                "build_lex.web_search.requests.post", side_effect=responses
            ) as post:
                first = searcher.search(
                    "term hate speech",
                    checkpoint_rank=1,
                    checkpoint_slot=1,
                    checkpoint_term="term",
                )
            self.assertEqual(post.call_count, 2)
            self.assertEqual(checkpoint.summary()["provider_attempt_counts"]["tavily"], 2)
            checkpoint.close()

            resumed = FormalLexiconCheckpoint.resume(root, spec)
            resumed_searcher = WebSearcher(settings, formal_checkpoint=resumed)
            with patch("build_lex.web_search.requests.post") as post:
                second = resumed_searcher.search(
                    "term hate speech",
                    checkpoint_rank=1,
                    checkpoint_slot=1,
                    checkpoint_term="term",
                )
            post.assert_not_called()
            self.assertEqual(first, second)
            resumed.close()

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
    def test_default_output_language_tracks_dataset(self):
        self.assertEqual(default_config("full")["llm_settings"]["output_language"], "zh")
        self.assertEqual(default_config("state")["llm_settings"]["output_language"], "zh")
        self.assertEqual(default_config("toxicn")["llm_settings"]["output_language"], "zh")
        self.assertEqual(default_config("cold")["llm_settings"]["output_language"], "zh")
        self.assertEqual(default_config("hatexplain")["llm_settings"]["output_language"], "en")

    def test_deepseek_backend_builds_official_chat_completion_payload(self):
        calls = []

        def fake_post(url, json, headers, timeout, allow_redirects):
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

    def test_checkpointed_deepseek_retries_once_and_reuses_success_after_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            spec = provider_checkpoint_spec(deepseek_cap=3)
            checkpoint = FormalLexiconCheckpoint.create(root, spec)
            settings = {
                "provider": "deepseek",
                "api_base": "https://api.deepseek.com",
                "api_key": "DSK",
                "model": "deepseek-v4-flash",
                "thinking": {"type": "disabled"},
                "reasoning_effort": None,
                "max_tokens": 1024,
                "retries": 2,
                "retry_sleep": 0,
                "strict_provider_audit": True,
            }
            client = OpenAICompatibleJudgementClient(
                settings, formal_checkpoint=checkpoint
            )
            parsed = {
                "supported": True,
                "category": "Racism",
                "categories": ["Racism"],
                "confidence": 0.9,
                "reason": "fixture",
            }
            response = FakeResponse(
                {
                    "model": "deepseek-v4-flash-0731",
                    "usage": {
                        "prompt_tokens": 10,
                        "prompt_cache_hit_tokens": 2,
                        "prompt_cache_miss_tokens": 8,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(parsed)},
                        }
                    ],
                }
            )
            payload = {"candidate": {"term": "term"}}
            with patch(
                "build_lex.llm_lexicon_builder.requests.post",
                side_effect=[requests.ConnectionError("EOF"), response],
            ) as post:
                first = client.complete_json(
                    "context_judge", payload, checkpoint_rank=1
                )
            self.assertEqual(post.call_count, 2)
            self.assertEqual(
                checkpoint.summary()["provider_attempt_counts"]["deepseek"], 2
            )
            checkpoint.close()

            resumed = FormalLexiconCheckpoint.resume(root, spec)
            resumed_client = OpenAICompatibleJudgementClient(
                settings, formal_checkpoint=resumed
            )
            with patch("build_lex.llm_lexicon_builder.requests.post") as post:
                second = resumed_client.complete_json(
                    "context_judge", payload, checkpoint_rank=1
                )
            post.assert_not_called()
            self.assertEqual(first, second)
            resumed.close()

    def test_checkpointed_deepseek_semantic_invalid_response_retries_before_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            spec = provider_checkpoint_spec(deepseek_cap=3)
            checkpoint = FormalLexiconCheckpoint.create(root, spec)
            settings = {
                "provider": "deepseek",
                "api_base": "https://api.deepseek.com",
                "api_key": "DSK",
                "model": "deepseek-v4-flash",
                "strict_provider_audit": True,
                "retries": 2,
                "retry_sleep": 0,
                "max_tokens": 1024,
                "thinking": {"type": "disabled"},
            }
            client = OpenAICompatibleJudgementClient(
                settings, formal_checkpoint=checkpoint
            )

            def response(parsed):
                return FakeResponse(
                    {
                        "model": "deepseek-v4-flash-0731",
                        "usage": {
                            "prompt_tokens": 10,
                            "prompt_cache_hit_tokens": 0,
                            "prompt_cache_miss_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": json.dumps(parsed)},
                            }
                        ],
                    }
                )

            invalid = {
                "supported": True,
                "confidence": 0.9,
                "reason": "invalid citation",
                "evidence_ids": ["invented-id"],
            }
            valid = dict(invalid, evidence_ids=["web-1"])
            payload = {
                "candidate": {"term": "term"},
                "decision_contract": {
                    "allowed_web_evidence_ids": ["web-1"]
                },
            }
            with patch(
                "build_lex.llm_lexicon_builder.requests.post",
                side_effect=[response(invalid), response(valid)],
            ) as post:
                result = client.complete_json(
                    "web_evidence_judge", payload, checkpoint_rank=1
                )
            self.assertEqual(result, valid)
            self.assertEqual(post.call_count, 2)
            self.assertTrue(
                all(
                    call.kwargs["allow_redirects"] is False
                    for call in post.call_args_list
                )
            )
            self.assertEqual(
                checkpoint.summary()["provider_attempt_counts"]["deepseek"], 2
            )
            checkpoint.close()

            resumed = FormalLexiconCheckpoint.resume(root, spec)
            resumed_client = OpenAICompatibleJudgementClient(
                settings, formal_checkpoint=resumed
            )
            with patch("build_lex.llm_lexicon_builder.requests.post") as post:
                self.assertEqual(
                    resumed_client.complete_json(
                        "web_evidence_judge", payload, checkpoint_rank=1
                    ),
                    valid,
                )
            post.assert_not_called()
            resumed.close()

    def test_checkpointed_deepseek_redirect_is_terminal_and_never_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = FormalLexiconCheckpoint.create(
                Path(tmp) / "checkpoint",
                provider_checkpoint_spec(deepseek_cap=3),
            )
            client = OpenAICompatibleJudgementClient(
                {
                    "provider": "deepseek",
                    "api_base": "https://api.deepseek.com",
                    "api_key": "DSK",
                    "model": "deepseek-v4-flash",
                    "retries": 2,
                    "retry_sleep": 0,
                    "max_tokens": 1024,
                },
                formal_checkpoint=checkpoint,
            )
            redirect = FakeResponse({"redirect": True})
            redirect.status_code = 307
            with patch(
                "build_lex.llm_lexicon_builder.requests.post",
                return_value=redirect,
            ) as post:
                with self.assertRaises(LLMAPIError):
                    client.complete_json(
                        "context_judge",
                        {"candidate": {"term": "term"}},
                        checkpoint_rank=1,
                    )
            self.assertEqual(post.call_count, 1)
            self.assertIs(post.call_args.kwargs["allow_redirects"], False)
            self.assertEqual(
                checkpoint.summary()["provider_attempt_counts"]["deepseek"], 1
            )
            self.assertFalse(
                checkpoint.can_retry("deepseek", 1, "context_judge")
            )
            checkpoint.close()

    def test_output_language_is_injected_into_prompt(self):
        calls = []

        def fake_post(url, json, headers, timeout, allow_redirects):
            calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
            return FakeResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json_module.dumps(
                                    {
                                        "include": True,
                                        "category": "Racism",
                                        "categories": ["Racism"],
                                        "definition": "中文解释",
                                        "nonhateful_meaning": "",
                                        "variants": [],
                                        "confidence": 0.9,
                                        "reason": "中文原因",
                                        "evidence_ids": [],
                                    },
                                    ensure_ascii=False,
                                )
                            },
                        }
                    ]
                }
            )

        import json as json_module

        client = create_judgement_client(
            {
                "backend": "deepseek",
                "api_key": "DSK",
                "model": "deepseek-v4-pro",
                "output_language": "zh",
                "max_tokens": 200,
            }
        )
        with patch("build_lex.llm_lexicon_builder.requests.post", fake_post):
            client.complete_json(
                "final_lexicon_judge",
                {"candidate": {"term": "测试词", "primary_category": "Racism"}},
            )

        user_prompt = calls[0]["json"]["messages"][1]["content"]
        self.assertIn("必须使用简体中文", user_prompt)
        self.assertIn("reason、definition、nonhateful_meaning", user_prompt)
        self.assertIn("category 标签保持不变", user_prompt)

    def test_http_400_body_is_recorded_and_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            calls = []
            recorder = DebugRecorder(tmp_dir, True)

            def fake_post(url, json, headers, timeout, allow_redirects):
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
                with self.assertRaises(LLMAPIError) as raised:
                    client.complete_json("context_judge", {"candidate": {"term": "blocked-term"}})

            self.assertEqual(len(calls), 1)
            self.assertTrue(raised.exception.provider_abstention)
            rows = [
                json.loads(line)
                for line in (Path(tmp_dir) / "debug" / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[0]["raw_response"]["error"]["status_code"], 400)
            self.assertTrue(rows[0]["raw_response"]["error"]["provider_abstention"])
            self.assertIn("content filter triggered", rows[0]["raw_response"]["error"]["response_body"])

    def test_retryable_provider_response_preserves_usage_in_billing_ledger(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ledger_path = Path(tmp_dir) / "billing.jsonl"
            recorder = DebugRecorder(
                tmp_dir,
                {"enabled": True, "billing_ledger_path": str(ledger_path)},
            )
            responses = [
                {
                    "model": "deepseek-v4-flash-0731",
                    "usage": {
                        "prompt_tokens": 10,
                        "prompt_cache_hit_tokens": 0,
                        "prompt_cache_miss_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                    "choices": [
                        {
                            "finish_reason": "insufficient_system_resource",
                            "message": {"content": "{}"},
                        }
                    ],
                },
                {
                    "model": "deepseek-v4-flash-0731",
                    "usage": {
                        "prompt_tokens": 10,
                        "prompt_cache_hit_tokens": 10,
                        "prompt_cache_miss_tokens": 0,
                        "completion_tokens": 8,
                        "total_tokens": 18,
                    },
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": '{"supported":true,"confidence":0.8,"reason":"ok"}'
                            },
                        }
                    ],
                },
            ]

            def fake_post(url, json, headers, timeout, allow_redirects):
                return FakeResponse(responses.pop(0))

            client = create_judgement_client(
                {
                    "backend": "deepseek",
                    "api_key": "DSK",
                    "model": "deepseek-v4-flash",
                    "retries": 1,
                    "retry_sleep": 0,
                    "max_tokens": 200,
                },
                debug_recorder=recorder,
            )
            with (
                patch("build_lex.llm_lexicon_builder.requests.post", fake_post),
                patch("build_lex.llm_lexicon_builder.time.sleep"),
            ):
                result = client.complete_json(
                    "context_judge", {"candidate": {"term": "retry-term"}}
                )

            self.assertTrue(result["supported"])
            rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            self.assertEqual([row["failed"] for row in rows], [True, False])
            self.assertEqual(
                [row["usage"]["total_tokens"] for row in rows], [12, 18]
            )

    def test_strict_provider_audit_fails_on_first_mismatched_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            calls = []
            ledger_path = Path(tmp_dir) / "billing.jsonl"
            recorder = DebugRecorder(
                tmp_dir,
                {"enabled": True, "billing_ledger_path": str(ledger_path)},
            )

            def fake_post(url, json, headers, timeout, allow_redirects):
                calls.append(json)
                return FakeResponse(
                    {
                        "model": "deepseek-v4-flash-pro",
                        "usage": {
                            "prompt_tokens": 10,
                            "prompt_cache_hit_tokens": 0,
                            "prompt_cache_miss_tokens": 10,
                            "completion_tokens": 8,
                            "total_tokens": 18,
                        },
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": '{"supported":true,"confidence":0.8,"reason":"ok"}'
                                },
                            }
                        ],
                    }
                )

            client = create_judgement_client(
                {
                    "backend": "deepseek",
                    "api_key": "DSK",
                    "model": "deepseek-v4-flash",
                    "strict_provider_audit": True,
                    "retries": 2,
                    "max_tokens": 1024,
                },
                debug_recorder=recorder,
            )
            with patch("build_lex.llm_lexicon_builder.requests.post", fake_post):
                with self.assertRaisesRegex(LLMAPIError, "unauthorized model"):
                    client.complete_json(
                        "context_judge", {"candidate": {"term": "strict-term"}}
                    )

            self.assertEqual(len(calls), 1)
            debug_rows = [
                json.loads(line)
                for line in (Path(tmp_dir) / "debug" / "llm_calls.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                debug_rows[0]["raw_response"]["model"],
                "deepseek-v4-flash-pro",
            )
            ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            self.assertTrue(ledger[0]["failed"])

    def test_debug_recorder_writes_llm_input_and_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorder = DebugRecorder(tmp_dir, True)

            def fake_post(url, json, headers, timeout, allow_redirects):
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
