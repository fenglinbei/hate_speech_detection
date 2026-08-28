import copy
import hashlib
import json
import unittest

import numpy as np

from data.retrieval_bundle import (
    LEXICON_EVIDENCE_RENDER_POLICY,
    LEXICON_TASK_LABEL_VISIBILITY,
    RetrievalBundleError,
    build_lexicon_catalog,
    cosine_score_matrices,
    prepare_context_bundle_from_scores,
)


ORDER = ["non-hate", "Region", "Racism", "Sexism", "LGBTQ", "others"]
QUOTA = {"non-hate": 4, "Region": 1, "Racism": 1, "Sexism": 2, "LGBTQ": 1, "others": 1}


def canonical_bundle_hash(value):
    unhashed = dict(value)
    unhashed.pop("bundle_sha256", None)
    return hashlib.sha256(
        json.dumps(
            unhashed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def record(index, group):
    hateful = "non-hate" if group == "non-hate" else "hate"
    return {
        "id": str(index),
        "content": f"训练文本{index}-{group}",
        "quadruples": [
            {
                "target": None if group == "non-hate" else group,
                "argument": f"观点{index}",
                "targeted_group": [group],
                "hateful": hateful,
            }
        ],
    }


def train():
    groups = ["non-hate"] * 4 + ["Region", "Racism"] + ["Sexism"] * 2 + ["LGBTQ", "others"]
    return [record(index + 1, group) for index, group in enumerate(groups)]


def terms():
    return [
        {
            "term": "种族词",
            "definition": "一个需要结合上下文理解的测试术语",
            "usage_notes": "可能出现在群体指称语境中",
            "ambiguity_notes": "词本身不决定说话者立场",
            "variants": [],
        },
        {
            "term": "性别词",
            "definition": "另一个用于检索测试的术语",
            "usage_notes": "需结合整句判断",
            "ambiguity_notes": "可能是提及而非攻击",
            "variants": ["别称"],
        },
    ]


def config():
    return {
        "source_class_order": ORDER,
        "allocated_class_top_k": QUOTA,
        "candidate_multiplier": 3,
        "similarity_threshold": 0.0,
        "lex_exact_top_k": 5,
        "lex_semantic_top_k": 5,
    }


class BundleTests(unittest.TestCase):
    def test_terminology_evidence_is_category_free(self):
        entry = build_lexicon_catalog(
            [
                {
                    "term": "男同",
                    "definition": "男同性恋的简称，词本身可用于中性指称。",
                    "usage_notes": "常作身份简称，具体态度由句子语境决定。",
                    "ambiguity_notes": "提及该词不等于针对该群体表达仇恨。",
                    "variants": ["男性同性恋者"],
                }
            ]
        )[0]

        self.assertRegex(entry["lexicon_id"], r"^lex:v2:[0-9a-f]{64}$")
        self.assertNotIn("category", entry)
        self.assertNotIn("categories", entry)
        self.assertNotIn("类别：", entry["rendered_block"])
        self.assertIn("词义说明：", entry["rendered_block"])
        self.assertIn("用法提示：", entry["rendered_block"])
        self.assertIn("歧义提示：", entry["rendered_block"])
        self.assertEqual(entry["evidence_kind"], "terminology")
        self.assertEqual(
            entry["render_policy"], LEXICON_EVIDENCE_RENDER_POLICY
        )
        self.assertEqual(
            entry["task_label_visibility"], LEXICON_TASK_LABEL_VISIBILITY
        )

        with self.assertRaisesRegex(RetrievalBundleError, "task-category"):
            build_lexicon_catalog(
                [
                    {
                        "term": "男同",
                        "category": "LGBTQ",
                        "definition": "不允许类别字段",
                    }
                ]
            )
        with self.assertRaisesRegex(RetrievalBundleError, "metadata.category"):
            build_lexicon_catalog(
                [
                    {
                        "term": "男同",
                        "definition": "嵌套字段也不允许绕过无类别契约",
                        "metadata": {"category": "LGBTQ"},
                    }
                ]
            )

    def test_duplicate_model_visible_lexicon_evidence_is_rejected(self):
        with self.assertRaisesRegex(RetrievalBundleError, "IDs are not unique"):
            build_lexicon_catalog(
                [
                    {
                        "term": "同一词",
                        "definition": "同一释义",
                        "variants": [],
                    },
                    {
                        "term": "同一词",
                        "definition": "同一释义",
                        "variants": [],
                    },
                ]
            )

    def test_partition_filter_precedes_scoring_and_keeps_full_query_frame(self):
        full = train()
        fit = [record for record in full if record["id"] != "1"]
        partition_config = config()
        partition_config["allocated_class_top_k"] = dict(
            partition_config["allocated_class_top_k"]
        )
        partition_config["allocated_class_top_k"]["non-hate"] = 3
        dependency = {
            "schema_version": "stage1-dependency-ref/v1",
            "artifact_kind": "train-partition",
            "artifact_id": "tpart-" + "a" * 64,
            "payload_manifest_sha256": "b" * 64,
            "logical_repo_path": "exps/fixture/train_partitions/tpart-" + "a" * 64,
        }
        locator = {
            "schema_version": "stage1-locator-ref/v1",
            "artifact_kind": "train-partition",
            "artifact_id": dependency["artifact_id"],
            "target_path": "/fixture/" + dependency["artifact_id"],
            "payload_manifest_sha256": dependency["payload_manifest_sha256"],
        }
        data_dependency = {
            "schema_version": "stage1-dependency-ref/v1",
            "artifact_kind": "data",
            "artifact_id": "data-" + "c" * 64,
            "payload_manifest_sha256": "d" * 64,
            "logical_repo_path": "exps/fixture/data/data-" + "c" * 64,
        }
        lexicon_dependency = {
            "schema_version": "stage1-dependency-ref/v1",
            "artifact_kind": "lexicon",
            "artifact_id": "lex-" + "e" * 64,
            "payload_manifest_sha256": "f" * 64,
            "logical_repo_path": "exps/fixture/lexicons/lex-" + "e" * 64,
        }
        bundle = prepare_context_bundle_from_scores(
            train_records=full,
            fit_demo_records=fit,
            calibration_ids=["1"],
            train_partition_dependency=dependency,
            train_partition_locator=locator,
            data_dependency=data_dependency,
            lexicon_dependency=lexicon_dependency,
            query_records=[record(100, "Racism")],
            lexicon_terms=terms(),
            demo_scores=np.zeros((1, 9)),
            lexicon_scores=np.zeros((1, 2)),
            split="dev",
            retrieval_config=partition_config,
        )
        self.assertEqual(len(bundle["train_query_pool"]), 10)
        self.assertEqual(len(bundle["demo_catalog"]), 9)
        self.assertNotIn(
            "1", {row["source_record_id"] for row in bundle["demo_catalog"]}
        )
        self.assertEqual(
            bundle["retrieval_provenance"]["train_partition_dependency"],
            dependency,
        )
        self.assertTrue(
            bundle["retrieval_provenance"]["calibration_demo_excluded"]
        )
        self.assertEqual(bundle["data_dependency"], data_dependency)
        self.assertEqual(bundle["lexicon_dependency"], lexicon_dependency)
        self.assertNotIn("train_partition_locator", bundle)
        self.assertNotIn("data_locator", bundle)
        self.assertNotIn("lexicon_locator", bundle)

        with self.assertRaisesRegex(RetrievalBundleError, "cover full train"):
            prepare_context_bundle_from_scores(
                train_records=full,
                fit_demo_records=fit[:-1],
                calibration_ids=["1"],
                train_partition_dependency=dependency,
                train_partition_locator=locator,
                data_dependency=data_dependency,
                lexicon_dependency=lexicon_dependency,
                query_records=[record(100, "Racism")],
                lexicon_terms=terms(),
                demo_scores=np.zeros((1, 8)),
                lexicon_scores=np.zeros((1, 2)),
                split="dev",
                retrieval_config=partition_config,
            )

    def test_runtime_locator_path_does_not_change_bundle_identity(self):
        full = train()
        fit = [record for record in full if record["id"] != "1"]
        retrieval = config()
        retrieval["allocated_class_top_k"] = dict(
            retrieval["allocated_class_top_k"]
        )
        retrieval["allocated_class_top_k"]["non-hate"] = 3
        dependencies = {
            "data_dependency": {
                "schema_version": "stage1-dependency-ref/v1",
                "artifact_kind": "data",
                "artifact_id": "data-" + "a" * 64,
                "payload_manifest_sha256": "b" * 64,
                "logical_repo_path": "artifacts/data/data-" + "a" * 64,
            },
            "lexicon_dependency": {
                "schema_version": "stage1-dependency-ref/v1",
                "artifact_kind": "lexicon",
                "artifact_id": "lex-" + "c" * 64,
                "payload_manifest_sha256": "d" * 64,
                "logical_repo_path": "artifacts/lexicons/lex-" + "c" * 64,
            },
            "train_partition_dependency": {
                "schema_version": "stage1-dependency-ref/v1",
                "artifact_kind": "train-partition",
                "artifact_id": "tpart-" + "e" * 64,
                "payload_manifest_sha256": "f" * 64,
                "logical_repo_path": "artifacts/partitions/tpart-" + "e" * 64,
            },
        }

        def make_bundle(prefix):
            locators = {}
            for key, dependency in dependencies.items():
                label = key.removesuffix("_dependency")
                locators[label + "_locator"] = {
                    "schema_version": "stage1-locator-ref/v1",
                    "artifact_kind": dependency["artifact_kind"],
                    "artifact_id": dependency["artifact_id"],
                    "target_path": prefix + "/" + dependency["artifact_id"],
                    "payload_manifest_sha256": dependency[
                        "payload_manifest_sha256"
                    ],
                }
            return prepare_context_bundle_from_scores(
                train_records=full,
                fit_demo_records=fit,
                calibration_ids=["1"],
                query_records=[record(100, "Racism")],
                lexicon_terms=terms(),
                demo_scores=np.zeros((1, 9)),
                lexicon_scores=np.zeros((1, 2)),
                split="dev",
                retrieval_config=retrieval,
                **dependencies,
                **locators,
            )

        first = make_bundle("/workspace-a")
        second = make_bundle("/workspace-b")
        self.assertEqual(first, second)

    def test_selector_and_complete_control_relevance(self):
        queries = [
            {
                "id": "100",
                "content": "这里含有种族词",
                "quadruples": [
                    {
                        "target": "种族词",
                        "argument": "这里含有种族词",
                        "targeted_group": ["Racism"],
                        "hateful": "hate",
                    }
                ],
            }
        ]
        demo_scores = np.linspace(0.1, 1.0, 10, dtype=np.float64).reshape(1, 10)
        lex_scores = np.array([[0.2, 0.9]])
        bundle = prepare_context_bundle_from_scores(
            train_records=train(),
            query_records=queries,
            lexicon_terms=terms(),
            demo_scores=demo_scores,
            lexicon_scores=lex_scores,
            split="dev",
            retrieval_config=config(),
        )
        self.assertEqual(
            [row["id"] for row in bundle["train_query_pool"]],
            [str(index) for index in range(1, 11)],
        )
        self.assertEqual([row["id"] for row in bundle["query_pool"]], ["100"])
        self.assertEqual(bundle["query_pool"][0]["source_split"], "dev")
        item = bundle["records"][0]
        self.assertEqual(len(item["selection"]["demos"]["selected_ids"]), 10)
        self.assertEqual(len(set(item["selection"]["demos"]["selected_ids"])), 10)
        self.assertEqual(len(item["control_relevance"]["demos"]), 10)
        self.assertEqual(len(item["control_relevance"]["lexicons"]), 2)
        self.assertTrue(bundle["retrieval_provenance"]["all_train_pool_relevance_complete"])
        self.assertIn(bundle["lexicon_catalog"][0]["lexicon_id"], item["selection"]["lexicons"]["selected_ids"])

    def test_query_gold_does_not_change_retrieval_selection(self):
        first_query = record(100, "Racism")
        first_query["content"] = "固定查询内容"
        second_query = copy.deepcopy(first_query)
        second_query["quadruples"][0]["targeted_group"] = ["Sexism"]
        scores = np.linspace(0.1, 1.0, 10).reshape(1, 10)
        kwargs = dict(
            train_records=train(),
            lexicon_terms=terms(),
            demo_scores=scores,
            lexicon_scores=np.array([[0.4, 0.3]]),
            split="dev",
            retrieval_config=config(),
        )
        first = prepare_context_bundle_from_scores(query_records=[first_query], **kwargs)
        second = prepare_context_bundle_from_scores(query_records=[second_query], **kwargs)
        self.assertEqual(
            first["records"][0]["selection"],
            second["records"][0]["selection"],
        )

    def test_score_shape_and_nonfinite_fail_closed(self):
        with self.assertRaisesRegex(RetrievalBundleError, "shape"):
            prepare_context_bundle_from_scores(
                train_records=train(),
                query_records=[record(100, "Racism")],
                lexicon_terms=terms(),
                demo_scores=np.zeros((1, 9)),
                lexicon_scores=np.zeros((1, 2)),
                split="dev",
                retrieval_config=config(),
            )

    def test_exact_score_and_selector_replay_rejects_fully_resealed_forgery(self):
        queries = [record(100, "Racism")]
        demo_scores = np.linspace(0.1, 1.0, 10, dtype=np.float64).reshape(1, 10)
        lexicon_scores = np.array([[0.4, 0.3]], dtype=np.float64)
        replay_config = config()
        replay_config["allocated_class_top_k"] = {
            label: (1 if label == "Sexism" else 0) for label in ORDER
        }
        kwargs = {
            "train_records": train(),
            "query_records": queries,
            "lexicon_terms": terms(),
            "split": "dev",
            "retrieval_config": replay_config,
        }
        original = prepare_context_bundle_from_scores(
            **kwargs,
            demo_scores=demo_scores,
            lexicon_scores=lexicon_scores,
        )

        injected = copy.deepcopy(original)
        demo_trace = injected["records"][0]["selection"]["demos"]
        injected_id = next(
            candidate["demo_id"]
            for candidate in demo_trace["candidates"]
            if candidate["demo_id"] not in demo_trace["selected_ids"]
        )
        for trace in (
            demo_trace,
            injected["records"][0]["retrieval"]["demo"],
        ):
            trace["selected_ids"].append(injected_id)
            trace["prompt_order"].append(injected_id)
        demo_trace["prompt_order_before_budget"] = list(
            demo_trace["prompt_order"]
        )
        injected["bundle_sha256"] = canonical_bundle_hash(injected)
        with self.assertRaisesRegex(RetrievalBundleError, "exact selector replay"):
            prepare_context_bundle_from_scores(
                **kwargs,
                demo_scores=demo_scores,
                lexicon_scores=lexicon_scores,
                score_evidence=injected["score_evidence"],
                expected_bundle=injected,
            )

        forged_scores = demo_scores.copy()
        forged_scores[0, 0] += 0.25
        score_resealed = prepare_context_bundle_from_scores(
            **kwargs,
            demo_scores=forged_scores,
            lexicon_scores=lexicon_scores,
        )
        with self.assertRaisesRegex(RetrievalBundleError, "score evidence differs"):
            prepare_context_bundle_from_scores(
                **kwargs,
                demo_scores=demo_scores,
                lexicon_scores=lexicon_scores,
                score_evidence=score_resealed["score_evidence"],
                expected_bundle=score_resealed,
            )

        malformed = copy.deepcopy(original["score_evidence"])
        malformed["demo_shape"] = [1, 9]
        with self.assertRaisesRegex(RetrievalBundleError, "shape"):
            prepare_context_bundle_from_scores(
                **kwargs,
                demo_scores=demo_scores,
                lexicon_scores=lexicon_scores,
                score_evidence=malformed,
            )


class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        return np.array([[len(text), sum(map(ord, text)) % 7 + 1] for text in texts], dtype=np.float32)


class EmbeddingTests(unittest.TestCase):
    def test_cosine_matrices_are_finite_and_sized(self):
        demo, lex = cosine_score_matrices(
            model=FakeEmbeddingModel(),
            train_texts=["a", "bb"],
            query_texts=["q"],
            lexicon_texts=["l", "ll", "lll"],
        )
        self.assertEqual(demo.shape, (1, 2))
        self.assertEqual(lex.shape, (1, 3))
        self.assertTrue(np.isfinite(demo).all())


if __name__ == "__main__":
    unittest.main()
