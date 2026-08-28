import hashlib
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import prompt as prompt_module
from data import build_context_manifest as context_builder_module

from data.build_context_manifest import (
    ContextBuildError,
    _replay_formal_retrieval_bundle,
    build_prepared_context_artifact,
    validate_context_ref,
)
from data.context_manifest import CONDITIONS, render_condition_item, text_sha256
from data.retrieval_bundle import (
    RETRIEVAL_BUNDLE_VERSION,
    build_demo_catalog,
    build_lexicon_catalog,
    build_query_pool,
    prepare_context_bundle_from_scores,
)
from data.train_partition import load_train_partition
from data.training_artifacts import (
    build_payload_manifest,
    portable_dependency,
    sha256_file,
    write_canonical_json,
)
from model.stage1_registry import ModelRegistryError, verified_model_source_lease
from scripts.stage1 import build_contexts as context_cli
from tests.stage1_semantic_fixtures import (
    make_embedding_model_fixture,
    make_formal_lexicon_artifact,
    make_semantic_data_artifact,
)


class FakeTokenizer:
    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt, enable_thinking=False):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return "\n".join(f"{row['role']}:{row['content']}" for row in conversation) + "\nassistant:"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(len(text)))


def deterministic_tokenizer_constructor(path):
    del path
    return FakeTokenizer()


def bundle():
    lex_id = "lex:v2:" + "1" * 64
    demo_id = "demo:v1:" + "2" * 64
    lex_block = (
        "术语：坏词\n词义说明：一种歧视词。\n"
        "用法提示：需结合整句理解。\n歧义提示：单独出现不等于攻击。"
    )
    demo_content = "训练例句"
    demo_block = "文本：训练例句\n输出：[]"
    train_quad = {
        "target": None,
        "argument": "训练例句",
        "targeted_group": ["non-hate"],
        "hateful": "non-hate",
    }
    dev_quad = {
        "target": "坏词",
        "argument": "包含坏词的测试句",
        "targeted_group": ["Racism"],
        "hateful": "hate",
    }
    value = {
        "schema_version": "stage1-prepared-context-bundle/v1",
        "split": "dev",
        "lexicon_catalog": [
            {
                "lexicon_id": lex_id,
                "term": "坏词",
                "definition": "一种歧视词。",
                "variants": [],
                "usage_notes": "需结合整句理解。",
                "ambiguity_notes": "单独出现不等于攻击。",
                "evidence_kind": "terminology",
                "render_policy": "category-free-terminology-evidence/v1",
                "task_label_visibility": "absent",
                "rendered_block": lex_block,
                "rendered_block_sha256": text_sha256(lex_block),
                "content_sha256": text_sha256(lex_block),
                "source_split": "train",
                "train_only": True,
            }
        ],
        "demo_catalog": [
            {
                "demo_id": demo_id,
                "source_record_id": "1",
                "content": demo_content,
                "output": '[{"target":null,"argument":"训练例句","targeted_group":["non-hate"],"hateful":"non-hate"}]',
                "content_sha256": text_sha256(demo_content),
                "gold_sha256": "a50a1704f22d97a23804965775bf15038806ef882ffc3546ee783ff882d3e0ef",
                "output_label": "non-hate",
                "rendered_block": demo_block,
                "source_split": "train",
                "train_only": True,
            }
        ],
        "train_query_pool": [
            {
                "id": "1",
                "content": demo_content,
                "quadruples": [train_quad],
                "content_sha256": text_sha256(demo_content),
                "gold_sha256": "a50a1704f22d97a23804965775bf15038806ef882ffc3546ee783ff882d3e0ef",
                "source_split": "train",
            }
        ],
        "query_pool": [
            {
                "id": "10",
                "content": "包含坏词的测试句",
                "quadruples": [dev_quad],
                "content_sha256": text_sha256("包含坏词的测试句"),
                "gold_sha256": "b03b69a9ca823211a1e17c72f20cf1e21732188b896b46f791d4ae5752fba8be",
                "source_split": "dev",
            }
        ],
        "records": [
            {
                "query": {
                    "id": "10",
                    "content": "包含坏词的测试句",
                    "content_sha256": text_sha256("包含坏词的测试句"),
                    "gold": [dev_quad],
                    "gold_sha256": "b03b69a9ca823211a1e17c72f20cf1e21732188b896b46f791d4ae5752fba8be",
                },
                "selection": {
                    "lexicons": {"prompt_order_before_budget": [lex_id]},
                    "demos": {"prompt_order_before_budget": [demo_id]},
                },
                "retrieval": {"policy": "synthetic-unit-test"},
                "control_relevance": {
                    "lexicons": [{"lexicon_id": lex_id, "written_similarity": 0.8}],
                    "demos": [{"demo_id": demo_id, "written_similarity": 0.7}],
                },
            }
        ],
        "retrieval_provenance": {
            "train_only_demo_pool": True,
            "train_only_lexicon_pool": True,
            "all_train_pool_relevance_complete": True,
            "saw_dev_test_labels_during_pool_build": False,
            "saw_model_predictions": False,
        },
    }
    value["bundle_sha256"] = canonical_bundle_sha256(value)
    return value


def canonical_bundle_sha256(value):
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


def canonical_value_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def deterministic_score_replayer(
    *,
    model_path,
    device_class,
    batch_size,
    train_texts,
    query_texts,
    lexicon_texts,
):
    del model_path, device_class, batch_size
    return (
        np.zeros((len(query_texts), len(train_texts)), dtype=np.float32),
        np.zeros((len(query_texts), len(lexicon_texts)), dtype=np.float32),
    )


def formal_semantic_bundle(root, *, split="dev"):
    data_ref, splits = make_semantic_data_artifact(root)
    lexicon_ref = make_formal_lexicon_artifact(root, data_ref=data_ref)
    partition_ref = root / "train_partition_ref.json"
    partition = load_train_partition(partition_ref, workspace_root=root)
    model_path, model_hash = make_embedding_model_fixture(root)
    config = json.loads(
        (Path(__file__).resolve().parents[2] / "config/stage1/context_factorial.json")
        .read_text(encoding="utf-8")
    )
    # The semantic fixture has one source class. Keep the production selector
    # contract while requesting one satisfiable Racism demo per query.
    config["retrieval"]["allocated_class_top_k"] = {
        label: (1 if label == "Racism" else 0)
        for label in config["retrieval"]["source_class_order"]
    }
    config["retrieval"]["demo_top_k"] = 1
    data_locator = json.loads(data_ref.read_text(encoding="utf-8"))
    lexicon_locator = json.loads(lexicon_ref.read_text(encoding="utf-8"))
    lexicon_document = json.loads(
        (Path(lexicon_locator["target_path"]) / "lexicon.json").read_text(
            encoding="utf-8"
        )
    )
    lexicon_catalog = build_lexicon_catalog(lexicon_document["terms"])
    demo_scores, lexicon_scores = deterministic_score_replayer(
        model_path=model_path,
        device_class="cpu",
        batch_size=8,
        train_texts=[record["content"] for record in partition.fit_records],
        query_texts=[record["content"] for record in splits[split]],
        lexicon_texts=[record["rendered_block"] for record in lexicon_catalog],
    )
    value = prepare_context_bundle_from_scores(
        train_records=splits["train"],
        query_records=splits[split],
        lexicon_terms=lexicon_document["terms"],
        demo_scores=demo_scores,
        lexicon_scores=lexicon_scores,
        split=split,
        retrieval_config=config["retrieval"],
        data_dependency=partition.data_dependency,
        lexicon_dependency=portable_dependency(
            lexicon_locator, Path(lexicon_locator["target_path"]), root
        ),
        scorer_provenance={
            "backend": "sentence-transformers-cosine/v1",
            "logical_model_path": model_path.relative_to(root).as_posix(),
            "model_file_tree_sha256": model_hash,
            "device_class": "cpu",
            "batch_size": 8,
        },
        fit_demo_records=partition.fit_records,
        calibration_ids=partition.calibration_ids,
        train_partition_dependency=partition.partition_dependency,
    )
    return value, config, data_ref, partition_ref, lexicon_ref


@patch(
    "data.build_context_manifest._construct_formal_tokenizer",
    new=deterministic_tokenizer_constructor,
)
@patch(
    "data.build_context_manifest._default_formal_score_replayer",
    new=deterministic_score_replayer,
)
class ContextArtifactTests(unittest.TestCase):
    def setUp(self):
        self.config = Path(__file__).resolve().parents[2] / "config/stage1/context_factorial.json"

    def test_engineering_artifact_round_trip_and_idempotence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref = root / "context_ref.json"
            first = build_prepared_context_artifact(
                prepared_bundle=bundle(),
                config=self.config,
                tokenizer=FakeTokenizer(),
                write_ref=ref,
                formal=False,
                target_root=root / "contexts",
            )
            second = build_prepared_context_artifact(
                prepared_bundle=bundle(),
                config=self.config,
                tokenizer=FakeTokenizer(),
                write_ref=ref,
                formal=False,
                target_root=root / "contexts",
            )
            self.assertEqual(first["artifact_id"], second["artifact_id"])
            meta = validate_context_ref(ref, tokenizer=FakeTokenizer())
            self.assertFalse(meta["scientific_eligible"])
            self.assertEqual(meta["record_count"], 1)
            target = Path(first["target_path"])
            self.assertTrue((target / "catalogs/query_pool.train.jsonl").is_file())
            self.assertTrue((target / "catalogs/query_pool.dev.jsonl").is_file())
            record = json.loads((target / "context_manifest.dev.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(set(record["conditions"]), {"C0", "CL", "CD", "CLD"})
            self.assertEqual(record["conditions"]["C0"]["lexicon_ids"], [])
            self.assertEqual(record["conditions"]["CLD"]["demo_ids"], [bundle()["demo_catalog"][0]["demo_id"]])

    def test_prompt_and_renderer_identity_changes_fail_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref = root / "context_ref.json"
            build_prepared_context_artifact(
                prepared_bundle=bundle(),
                config=self.config,
                tokenizer=FakeTokenizer(),
                write_ref=ref,
                formal=False,
                target_root=root / "contexts",
            )
            with patch.object(
                prompt_module,
                "STAGE1_QUAD_JSON_SYSTEM_PROMPT_V2",
                "changed system prompt",
            ):
                with self.assertRaisesRegex(ContextBuildError, "prompt/renderer"):
                    validate_context_ref(ref, tokenizer=FakeTokenizer())
            original = context_builder_module._module_source_sha256

            def changed_hash(module, *, label):
                value = original(module, label=label)
                return "0" * 64 if label == "context renderer" else value

            with patch.object(
                context_builder_module,
                "_module_source_sha256",
                side_effect=changed_hash,
            ):
                with self.assertRaisesRegex(ContextBuildError, "prompt/renderer"):
                    validate_context_ref(ref, tokenizer=FakeTokenizer())

    def test_scientific_build_rejects_public_backend_injection(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ContextBuildError, "forbids caller-injected"):
                build_prepared_context_artifact(
                    prepared_bundle=bundle(),
                    config=self.config,
                    tokenizer=FakeTokenizer(),
                    write_ref=None,
                    formal=True,
                    target_root=Path(directory),
                )

    def test_formal_cli_rejects_arbitrary_tokenizer_path(self):
        result = context_cli.main(
            [
                "build",
                "--config",
                "unused-config.json",
                "--prepared-bundle",
                "unused-bundle.json",
                "--tokenizer",
                "models/other-tokenizer",
                "--write-ref",
                "unused-ref.json",
            ]
        )
        self.assertEqual(result, 2)

    def test_full_tree_lease_rejects_extra_code_and_swap_restore_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_root = root / "models/tokenizer"
            scorer_root = root / "models/scorer"
            tokenizer_root.mkdir(parents=True)
            scorer_root.mkdir(parents=True)
            (tokenizer_root / "tokenizer.json").write_bytes(b"tokenizer-v1\n")
            (scorer_root / "weights.bin").write_bytes(b"scorer-v1\n")
            config = json.loads(self.config.read_text(encoding="utf-8"))
            config["budget"]["tokenizer_path"] = "models/tokenizer"
            provenance = {
                "scorer": {
                    "backend": "sentence-transformers-cosine/v1",
                    "logical_model_path": "models/scorer",
                    "model_file_tree_sha256": (
                        context_builder_module.embedding_model_file_tree_sha256(
                            scorer_root
                        )
                    ),
                }
            }
            _, contract = context_builder_module._runtime_source_identity(
                config=config,
                retrieval_provenance=provenance,
                workspace_root=root,
            )
            (tokenizer_root / "unexpected.py").write_text(
                "raise RuntimeError('must never load')\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ModelRegistryError, "inventory changed"):
                with verified_model_source_lease(
                    contract, source_names=("tokenizer",)
                ):
                    pass
            (tokenizer_root / "unexpected.py").unlink()
            original = (tokenizer_root / "tokenizer.json").read_bytes()
            with self.assertRaisesRegex(ModelRegistryError, "changed during"):
                with verified_model_source_lease(
                    contract, source_names=("tokenizer",)
                ):
                    (tokenizer_root / "tokenizer.json").write_bytes(b"swapped\n")
                    (tokenizer_root / "tokenizer.json").write_bytes(original)

    def test_query_pool_lineage_mismatch_fails_before_publish(self):
        value = bundle()
        value["query_pool"][0]["gold_sha256"] = "0" * 64
        value["bundle_sha256"] = canonical_bundle_sha256(value)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ContextBuildError, "non-canonical dev query-pool"):
                build_prepared_context_artifact(
                    prepared_bundle=value,
                    config=self.config,
                    tokenizer=FakeTokenizer(),
                    write_ref=None,
                    formal=False,
                    target_root=Path(directory),
                )

    def test_formal_bundle_requires_portable_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ContextBuildError, "canonical|portable data dependency"
            ):
                build_prepared_context_artifact(
                    prepared_bundle=bundle(),
                    config=self.config,
                    write_ref=None,
                    formal=True,
                    target_root=Path(directory),
                )

    def test_payload_tamper_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref = root / "context_ref.json"
            locator = build_prepared_context_artifact(
                prepared_bundle=bundle(),
                config=self.config,
                tokenizer=FakeTokenizer(),
                write_ref=ref,
                formal=False,
                target_root=root / "contexts",
            )
            path = Path(locator["target_path"]) / "config.resolved.json"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ContextBuildError, "payload"):
                validate_context_ref(ref, tokenizer=FakeTokenizer())

    def test_prepared_bundle_rejects_persisted_runtime_locator(self):
        value = bundle()
        value["data_locator"] = {
            "schema_version": "stage1-locator-ref/v1",
            "artifact_kind": "data",
            "artifact_id": "data-" + "1" * 64,
            "target_path": "/workspace-specific/data",
            "payload_manifest_sha256": "2" * 64,
        }
        value["bundle_sha256"] = canonical_bundle_sha256(value)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ContextBuildError, "must not persist"):
                build_prepared_context_artifact(
                    prepared_bundle=value,
                    config=self.config,
                    tokenizer=FakeTokenizer(),
                    write_ref=None,
                    formal=False,
                    target_root=Path(directory),
                )

    def test_formal_score_replay_rejects_resealed_selection_and_scores(self):
        config = json.loads(self.config.read_text(encoding="utf-8"))
        config["retrieval"]["allocated_class_top_k"] = {
            label: (1 if label == "non-hate" else 0)
            for label in config["retrieval"]["source_class_order"]
        }
        config["retrieval"]["demo_top_k"] = 1
        train_records = [
            {
                "id": str(index),
                "content": f"训练文本{index}",
                "quadruples": [
                    {
                        "target": None,
                        "argument": f"训练文本{index}",
                        "targeted_group": ["non-hate"],
                        "hateful": "non-hate",
                    }
                ],
            }
            for index in (1, 2)
        ]
        query_records = [
            {
                "id": "10",
                "content": "待判断文本",
                "quadruples": [
                    {
                        "target": None,
                        "argument": "待判断文本",
                        "targeted_group": ["non-hate"],
                        "hateful": "non-hate",
                    }
                ],
            }
        ]
        lexicon_terms = [
            {
                "term": "术语",
                "definition": "测试定义",
                "usage_notes": "",
                "ambiguity_notes": "",
                "variants": [],
            }
        ]
        dependencies = {
            "data_dependency": {
                "schema_version": "stage1-dependency-ref/v1",
                "artifact_kind": "data",
                "artifact_id": "data-" + "1" * 64,
                "payload_manifest_sha256": "2" * 64,
                "logical_repo_path": "artifacts/data/data-" + "1" * 64,
            },
            "lexicon_dependency": {
                "schema_version": "stage1-dependency-ref/v1",
                "artifact_kind": "lexicon",
                "artifact_id": "lex-" + "3" * 64,
                "payload_manifest_sha256": "4" * 64,
                "logical_repo_path": "artifacts/lexicons/lex-" + "3" * 64,
            },
            "train_partition_dependency": {
                "schema_version": "stage1-dependency-ref/v1",
                "artifact_kind": "train-partition",
                "artifact_id": "tpart-" + "5" * 64,
                "payload_manifest_sha256": "6" * 64,
                "logical_repo_path": "artifacts/partitions/tpart-" + "5" * 64,
            },
        }
        scorer = {
            "backend": "sentence-transformers-cosine/v1",
            "logical_model_path": "models/fixture",
            "model_file_tree_sha256": "7" * 64,
            "device_class": "cpu",
            "batch_size": 2,
        }

        def scores(value):
            return (
                np.array([[value]], dtype=np.float32),
                np.array([[0.0]], dtype=np.float32),
            )

        original = prepare_context_bundle_from_scores(
            train_records=train_records,
            fit_demo_records=[train_records[0]],
            calibration_ids=["2"],
            query_records=query_records,
            lexicon_terms=lexicon_terms,
            demo_scores=scores(0.0)[0],
            lexicon_scores=scores(0.0)[1],
            split="dev",
            retrieval_config=config["retrieval"],
            scorer_provenance=scorer,
            **dependencies,
        )
        snapshot = {
            "train_records": train_records,
            "fit_records": [train_records[0]],
            "split_records": query_records,
            "lexicon_terms": lexicon_terms,
            "lexicon_catalog": original["lexicon_catalog"],
            "calibration_ids": ["2"],
            "scorer": {"model_path": Path("/unused/test-model")},
        }

        def frozen_score_replayer(**kwargs):
            del kwargs
            return scores(0.0)

        _replay_formal_retrieval_bundle(
            bundle=original,
            config=config,
            snapshot=snapshot,
            score_replayer=frozen_score_replayer,
        )

        injected = copy.deepcopy(original)
        for trace in (
            injected["records"][0]["selection"]["demos"],
            injected["records"][0]["retrieval"]["demo"],
        ):
            trace["selected_ids"] = []
            trace["prompt_order"] = []
            trace["quota_assignments"] = []
        injected["records"][0]["selection"]["demos"][
            "prompt_order_before_budget"
        ] = []
        injected["bundle_sha256"] = canonical_bundle_sha256(injected)
        with self.assertRaisesRegex(ContextBuildError, "exact score replay"):
            _replay_formal_retrieval_bundle(
                bundle=injected,
                config=config,
                snapshot=snapshot,
                score_replayer=frozen_score_replayer,
            )

        score_resealed = prepare_context_bundle_from_scores(
            train_records=train_records,
            fit_demo_records=[train_records[0]],
            calibration_ids=["2"],
            query_records=query_records,
            lexicon_terms=lexicon_terms,
            demo_scores=scores(0.5)[0],
            lexicon_scores=scores(0.5)[1],
            split="dev",
            retrieval_config=config["retrieval"],
            scorer_provenance=scorer,
            **dependencies,
        )
        with self.assertRaisesRegex(ContextBuildError, "score evidence differs"):
            _replay_formal_retrieval_bundle(
                bundle=score_resealed,
                config=config,
                snapshot=snapshot,
                score_replayer=frozen_score_replayer,
            )

    def test_formal_context_id_is_stable_across_workspace_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory)
            builds = []
            bundle_hashes = []
            for name in ("workspace-a", "workspace-b"):
                root = outer / name
                value, config, data_ref, partition_ref, lexicon_ref = (
                    formal_semantic_bundle(root)
                )
                bundle_hashes.append(value["bundle_sha256"])
                builds.append(
                    build_prepared_context_artifact(
                        prepared_bundle=value,
                        config=config,
                        write_ref=root / "refs/context.json",
                        formal=True,
                        data_ref=data_ref,
                        train_partition_ref=partition_ref,
                        lexicon_ref=lexicon_ref,
                        target_root=root / "contexts",
                        workspace_root=root,
                    )
                )
                del value
            self.assertEqual(bundle_hashes[0], bundle_hashes[1])
            self.assertEqual(builds[0]["artifact_id"], builds[1]["artifact_id"])

    def test_formal_semantic_frame_rebuilds_every_upstream_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value, config, data_ref, partition_ref, lexicon_ref = formal_semantic_bundle(root)

            forged = dict(value)
            forged["demo_catalog"] = list(value["demo_catalog"])
            forged["demo_catalog"][0] = dict(value["demo_catalog"][0])
            forged["demo_catalog"][0]["content"] = "forged-demo-content"
            forged["bundle_sha256"] = canonical_bundle_sha256(forged)
            with self.assertRaisesRegex(ContextBuildError, "demo catalog differs"):
                build_prepared_context_artifact(
                    prepared_bundle=forged,
                    config=config,
                    write_ref=None,
                    formal=True,
                    data_ref=data_ref,
                    train_partition_ref=partition_ref,
                    lexicon_ref=lexicon_ref,
                    target_root=root / "forged-contexts",
                    workspace_root=root,
                )
            del forged

            calibration_injected = dict(value)
            calibration_injected["demo_catalog"] = list(value["demo_catalog"])
            partition = load_train_partition(partition_ref, workspace_root=root)
            calibration_demo, _ = build_demo_catalog(
                [partition.calibration_records[0]],
                source_class_order=config["retrieval"]["source_class_order"],
            )
            calibration_injected["demo_catalog"][-1] = calibration_demo[0]
            calibration_injected["bundle_sha256"] = canonical_bundle_sha256(
                calibration_injected
            )
            with self.assertRaisesRegex(ContextBuildError, "demo catalog differs"):
                build_prepared_context_artifact(
                    prepared_bundle=calibration_injected,
                    config=config,
                    write_ref=None,
                    formal=True,
                    data_ref=data_ref,
                    train_partition_ref=partition_ref,
                    lexicon_ref=lexicon_ref,
                    target_root=root / "calibration-injected-contexts",
                    workspace_root=root,
                )
            del calibration_injected

            wrong_scorer = dict(value)
            wrong_scorer["retrieval_provenance"] = copy.deepcopy(
                value["retrieval_provenance"]
            )
            wrong_scorer["retrieval_provenance"]["scorer"]["backend"] = (
                "synthetic-cosine/v1"
            )
            wrong_scorer["bundle_sha256"] = canonical_bundle_sha256(wrong_scorer)
            with self.assertRaisesRegex(ContextBuildError, "must use"):
                build_prepared_context_artifact(
                    prepared_bundle=wrong_scorer,
                    config=config,
                    write_ref=None,
                    formal=True,
                    data_ref=data_ref,
                    train_partition_ref=partition_ref,
                    lexicon_ref=lexicon_ref,
                    target_root=root / "wrong-scorer-contexts",
                    workspace_root=root,
                )
            del wrong_scorer

            model_symlink = root / "models" / "embedding-alias"
            model_symlink.symlink_to(root / value["retrieval_provenance"]["scorer"]["logical_model_path"])
            aliased_scorer = dict(value)
            aliased_scorer["retrieval_provenance"] = copy.deepcopy(
                value["retrieval_provenance"]
            )
            aliased_scorer["retrieval_provenance"]["scorer"][
                "logical_model_path"
            ] = model_symlink.relative_to(root).as_posix()
            aliased_scorer["bundle_sha256"] = canonical_bundle_sha256(
                aliased_scorer
            )
            with self.assertRaisesRegex(ContextBuildError, "symlinks"):
                build_prepared_context_artifact(
                    prepared_bundle=aliased_scorer,
                    config=config,
                    write_ref=None,
                    formal=True,
                    data_ref=data_ref,
                    train_partition_ref=partition_ref,
                    lexicon_ref=lexicon_ref,
                    target_root=root / "aliased-model-contexts",
                    workspace_root=root,
                )
            del aliased_scorer

            bad_hash = dict(value)
            bad_hash["bundle_sha256"] = "0" * 64
            with self.assertRaisesRegex(ContextBuildError, "bundle_sha256"):
                build_prepared_context_artifact(
                    prepared_bundle=bad_hash,
                    config=config,
                    write_ref=None,
                    formal=True,
                    data_ref=data_ref,
                    train_partition_ref=partition_ref,
                    lexicon_ref=lexicon_ref,
                    target_root=root / "bad-hash-contexts",
                    workspace_root=root,
                )

            ref = root / "refs/context.json"
            locator = build_prepared_context_artifact(
                prepared_bundle=value,
                config=config,
                write_ref=ref,
                formal=True,
                data_ref=data_ref,
                train_partition_ref=partition_ref,
                lexicon_ref=lexicon_ref,
                target_root=root / "contexts",
                workspace_root=root,
            )
            report = validate_context_ref(
                ref,
                workspace_root=root,
            )
            self.assertTrue(report["scientific_eligible"])
            self.assertEqual(report["record_count"], 643)
            target = Path(locator["target_path"])
            self.assertEqual(
                len((target / "catalogs/demo_pool.train.jsonl").read_text().splitlines()),
                len(
                    load_train_partition(
                        partition_ref, workspace_root=root
                    ).fit_ids
                ),
            )

            # Fully reseal a forged CLD prompt while retaining the exact same
            # context_build_id. Every context-local hash, runner adapter,
            # payload manifest, and locator is updated. Exact replay from the
            # frozen prepared bundle must still reject it.
            records_path = target / "context_manifest.dev.jsonl"
            records = [
                json.loads(line)
                for line in records_path.read_text(encoding="utf-8").splitlines()
            ]
            forged = records[0]
            forged_cld = forged["conditions"]["CLD"]
            forged_cld["messages"][1]["content"] += "\nFORGED-CONTEXT"
            rendered = FakeTokenizer().apply_chat_template(
                forged_cld["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            forged_cld["chat_prompt_tokens"] = len(rendered)
            forged_cld["chat_prompt_sha256"] = text_sha256(rendered)
            forged["budget"]["cld_chat_tokens_after"] = len(rendered)
            forged["record_sha256"] = canonical_value_sha256(
                {
                    key: value
                    for key, value in forged.items()
                    if key != "record_sha256"
                }
            )
            records_path.write_bytes(
                b"".join(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                    + b"\n"
                    for row in records
                )
            )
            meta_path = target / "context_manifest.dev.meta.json"
            forged_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            forged_meta["records_sha256"] = sha256_file(records_path)
            write_canonical_json(meta_path, forged_meta)
            for condition in CONDITIONS:
                write_canonical_json(
                    target
                    / "conditions"
                    / "runner"
                    / condition
                    / "dev.json",
                    [render_condition_item(row, condition) for row in records],
                )
            write_canonical_json(
                target / "payload_manifest.json", build_payload_manifest(target)
            )
            forged_locator = json.loads(ref.read_text(encoding="utf-8"))
            forged_locator["payload_manifest_sha256"] = sha256_file(
                target / "payload_manifest.json"
            )
            write_canonical_json(ref, forged_locator)
            with self.assertRaisesRegex(ContextBuildError, "exact frozen"):
                validate_context_ref(
                    ref,
                    workspace_root=root,
                )


if __name__ == "__main__":
    unittest.main()
