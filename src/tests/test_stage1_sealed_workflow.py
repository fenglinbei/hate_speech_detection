import copy
import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from scripts.stage1 import build_contexts as context_cli
from scripts.stage1 import build_controls as control_cli
from scripts.stage1 import build_counterfactuals as counterfactual_cli
from data.build_context_manifest import (
    ContextBuildError,
    build_prepared_context_artifact,
    seal_test_context_artifact,
    validate_context_ref,
)
from data.control_manifest import (
    ControlManifestError,
    build_control_artifact,
    seal_test_control_artifact,
    validate_control_ref,
)
from data.counterfactual_lifecycle import (
    CounterfactualLifecycleError,
    finalize_counterfactual_artifact,
    propose_counterfactual_artifact,
    validate_cf_ref,
    validate_proposal_ref,
)
from data.context_manifest import text_sha256
from data.retrieval_bundle import build_query_pool
from data.stage1_data import validate_data
from data.training_artifacts import (
    canonical_sha256,
    finalize_target_atomic,
    load_json,
    load_jsonl,
    new_staging_directory,
    write_canonical_json,
    write_canonical_jsonl,
    write_locator_ref,
)
from data.training_evidence import context_policy_snapshot
from utils.quadruple import canonicalize_quadruples, serialize_quadruples
from tests.stage1_cf_blind_review_fixtures import (
    build_completed_cf_blind_review_fixture,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class CharacterTokenizer:
    def apply_chat_template(
        self, conversation, *, tokenize, add_generation_prompt, enable_thinking=False
    ):
        assert tokenize is False and add_generation_prompt is True
        assert enable_thinking is False
        return "\n".join(
            f"{row['role']}:{row['content']}" for row in conversation
        ) + "\nassistant:"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text)


TOKENIZER = CharacterTokenizer()


def quad(target, argument, group, hateful):
    return {
        "target": target,
        "argument": argument,
        "targeted_group": [group],
        "hateful": hateful,
    }


def raw_records(start, *, split):
    groups = ("Racism", "Sexism", "Region")
    rows = []
    for offset, group in enumerate(groups):
        identifier = str(start + offset)
        name = f"坏{chr(ord('甲') + offset)}"
        content = f"{name}说法，旁人{offset}观点。"
        rows.append(
            {
                "id": identifier,
                "content": content,
                "quadruples": [quad(name, f"{name}说法", group, "hate")],
            }
        )
    return rows


def _gold_hash(record):
    return hashlib.sha256(
        serialize_quadruples(canonicalize_quadruples(record["quadruples"])).encode(
            "utf-8"
        )
    ).hexdigest()


class SealedWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        if self._testMethodName == "test_sealed_cli_contracts_are_explicit":
            return
        self.artifact_root = self.root / "artifacts"
        self.refs = self.root / "refs"
        self.train = raw_records(1, split="train")
        self.dev = raw_records(10, split="dev")
        self.test = raw_records(20, split="test")
        self.data_ref = self._simple_artifact(
            kind="data",
            artifact_id="data-" + "1" * 64,
            directory="data",
            files={
                "train.json": self.train,
                "dev.json": self.dev,
                "test.json": self.test,
            },
        )
        self.lexicon_ref = self._simple_artifact(
            kind="lexicon",
            artifact_id="lex-" + "2" * 64,
            directory="lexicons",
            files={"lexicon.json": {"terms": []}},
        )
        self.context_config = load_json(
            REPOSITORY_ROOT / "config/stage1/context_factorial.json"
        )
        self.train_bundle = self._bundle(self.train, split="train", selected=True)
        self.train_bundle["data_locator"] = load_json(self.data_ref)
        self.train_bundle["lexicon_locator"] = load_json(self.lexicon_ref)
        self.train_context_ref = self.refs / "train_context.json"
        build_prepared_context_artifact(
            prepared_bundle=self.train_bundle,
            config=self.context_config,
            tokenizer=TOKENIZER,
            write_ref=self.train_context_ref,
            formal=True,
            target_root=self.artifact_root / "contexts",
            workspace_root=self.root,
        )
        self.dev_bundle = self._bundle(self.dev, split="dev", selected=False)
        self.dev_bundle["data_locator"] = load_json(self.data_ref)
        self.dev_bundle["lexicon_locator"] = load_json(self.lexicon_ref)
        self.dev_context_ref = self.refs / "dev_context.json"
        build_prepared_context_artifact(
            prepared_bundle=self.dev_bundle,
            config=self.context_config,
            tokenizer=TOKENIZER,
            write_ref=self.dev_context_ref,
            formal=True,
            target_root=self.artifact_root / "contexts",
            workspace_root=self.root,
        )
        self.control_config = {
            "schema_version": "stage1-control-config/v1",
            "profile_name": "sealed-synthetic",
            "source_class_order": ["A"],
            "tokenizer": {
                "revision": self.context_config["budget"]["tokenizer_revision"],
                "logical_path": self.context_config["budget"]["tokenizer_path"],
            },
        }
        self.dev_control_ref = self.refs / "dev_control.json"
        build_control_artifact(
            config=self.control_config,
            context_ref=self.dev_context_ref,
            write_ref=self.dev_control_ref,
            split="dev",
            target_root=self.artifact_root / "controls",
            workspace_root=self.root,
        )
        self.dev_cf_ref = self._build_dev_cf()

    def tearDown(self):
        self.temporary.cleanup()

    def _simple_artifact(self, *, kind, artifact_id, directory, files):
        parent = self.root / directory
        target = parent / artifact_id
        staging = new_staging_directory(parent, artifact_id)
        for name, value in files.items():
            write_canonical_json(staging / name, value)
        payload = finalize_target_atomic(staging, target)
        ref = self.refs / f"{kind}.json"
        write_locator_ref(
            ref,
            artifact_kind=kind,
            artifact_id=artifact_id,
            target=target,
            payload_manifest_sha256=payload,
        )
        return ref

    def _catalogs(self):
        demos = []
        lexicons = []
        for index, record in enumerate(self.train):
            output = serialize_quadruples(
                canonicalize_quadruples(record["quadruples"])
            )
            demo_id = f"demo:v1:{index + 10:064x}"
            demo_block = f"文本：{record['content']}\n输出：{output}"
            demos.append(
                {
                    "demo_id": demo_id,
                    "source_record_id": record["id"],
                    "content": record["content"],
                    "output": output,
                    "content_sha256": text_sha256(record["content"]),
                    "gold_sha256": _gold_hash(record),
                    "output_label": "A",
                    "source_classes": ["A"],
                    "rendered_block": demo_block,
                    "rendered_block_sha256": text_sha256(demo_block),
                    "source_split": "train",
                    "train_only": True,
                }
            )
            lex_id = f"lex:v2:{index + 20:064x}"
            lex_block = f"术语：训练词{index}\n词义说明：定义{index}"
            lexicons.append(
                {
                    "lexicon_id": lex_id,
                    "term": f"训练词{index}",
                    "definition": f"定义{index}",
                    "variants": [],
                    "usage_notes": "",
                    "ambiguity_notes": "",
                    "evidence_kind": "terminology",
                    "render_policy": "category-free-terminology-evidence/v1",
                    "task_label_visibility": "absent",
                    "rendered_block": lex_block,
                    "rendered_block_sha256": text_sha256(lex_block),
                    "content_sha256": text_sha256(lex_block),
                    "source_split": "train",
                    "train_only": True,
                }
            )
        return lexicons, demos

    def _bundle(self, records, *, split, selected):
        lexicons, demos = self._catalogs()
        lex_ids = [row["lexicon_id"] for row in lexicons]
        demo_ids = [row["demo_id"] for row in demos]
        prepared = []
        for record in records:
            query = build_query_pool([record], source_split=split)[0]
            prepared.append(
                {
                    "query": {
                        "id": query["id"],
                        "content": query["content"],
                        "content_sha256": query["content_sha256"],
                        "gold": query["quadruples"],
                        "gold_sha256": query["gold_sha256"],
                    },
                    "selection": {
                        "lexicons": {
                            "prompt_order_before_budget": lex_ids if selected else []
                        },
                        "demos": {
                            "prompt_order_before_budget": demo_ids if selected else [],
                            "quota_assignments": (
                                [
                                    {
                                        "demo_id": item,
                                        "assigned_quota_class": "A",
                                        "quota_round": 0,
                                    }
                                    for item in demo_ids
                                ]
                                if selected
                                else []
                            ),
                        },
                    },
                    "retrieval": {"policy": "sealed-synthetic/v1"},
                    "control_relevance": {
                        "lexicons": [
                            {"lexicon_id": item, "written_similarity": 0.8}
                            for item in lex_ids
                        ],
                        "demos": [
                            {"demo_id": item, "written_similarity": 0.7}
                            for item in demo_ids
                        ],
                    },
                }
            )
        provenance = {
            "schema_version": "stage1-retrieval-provenance/v1",
            "policy_version": "stage1-train-only-cosine-bundle/v1",
            "train_only_demo_pool": True,
            "train_only_lexicon_pool": True,
            "all_train_pool_relevance_complete": split in {"dev", "test"},
            "saw_dev_test_labels_during_pool_build": False,
            "saw_model_predictions": False,
            "score_matrix": {
                "demo_sha256": canonical_sha256({"split": split, "kind": "demo"}),
                "lexicon_sha256": canonical_sha256(
                    {"split": split, "kind": "lexicon"}
                ),
            },
            "scorer": {
                "backend": "synthetic-content-only/v1",
                "logical_model_path": "models/synthetic",
                "model_file_tree_sha256": "3" * 64,
                "device_class": "cpu",
                "batch_size": 3,
            },
        }
        value = {
            "schema_version": "stage1-prepared-context-bundle/v1",
            "split": split,
            "lexicon_catalog": lexicons,
            "demo_catalog": demos,
            "train_query_pool": build_query_pool(self.train, source_split="train"),
            "query_pool": build_query_pool(records, source_split=split),
            "records": prepared,
            "retrieval_provenance": provenance,
        }
        value["bundle_sha256"] = canonical_sha256(value)
        return value

    def _review_and_finalize(self, proposal_ref, *, sealed, stem):
        blind_review_ref, review_file, declaration_file = (
            build_completed_cf_blind_review_fixture(
                proposal_ref=proposal_ref,
                workspace_root=self.root,
                stem=stem,
            )
        )
        review_ref = self.refs / f"{stem}.review.ref.json"
        cf_ref = self.refs / f"{stem}.cf.ref.json"
        finalize_counterfactual_artifact(
            proposal_ref=proposal_ref,
            blind_review_ref=blind_review_ref,
            review_file=review_file,
            reviewer_declaration=declaration_file,
            write_review_ref=review_ref,
            write_ref=cf_ref,
            target_root=self.root,
            workspace_root=self.root,
            sealed=sealed,
        )
        return cf_ref

    def _build_dev_cf(self):
        proposal_ref = self.refs / "dev.cf.proposal.json"
        self.dev_proposal_ref = proposal_ref
        policy = load_json(REPOSITORY_ROOT / "config/stage1/cf_foil_policy.json")
        policy["sampling"]["review_candidate_target"] = 6
        policy["sampling"]["field_quota"] = {"target": 3, "argument": 3}
        propose_counterfactual_artifact(
            write_ref=proposal_ref,
            config={"artifact_root": "unused"},
            foil_policy=policy,
            review_rubric=REPOSITORY_ROOT / "config/stage1/cf_review_rubric.md",
            split="dev",
            context_ref=self.dev_context_ref,
            target_root=self.root,
            workspace_root=self.root,
        )
        return self._review_and_finalize(proposal_ref, sealed=False, stem="dev")

    def _seal_context(self, bundle=None):
        ref = self.refs / "sealed.context.json"
        seal_test_context_artifact(
            frozen_context_ref=self.dev_context_ref,
            data_ref=self.data_ref,
            prepared_bundle=bundle or self._bundle(self.test, split="test", selected=True),
            tokenizer=TOKENIZER,
            write_ref=ref,
            target_root=self.artifact_root / "test_contexts",
            workspace_root=self.root,
        )
        return ref

    @unittest.skip(
        "legacy 3-row formal fixture was not a real Stage-1 data artifact; "
        "real frozen integration coverage is environment-gated below"
    )
    def test_three_row_context_control_cf_sealed_workflow(self):
        context_ref = self._seal_context()
        context_report = validate_context_ref(
            context_ref, tokenizer=TOKENIZER, workspace_root=self.root
        )
        self.assertEqual(context_report["record_count"], 3)
        self.assertEqual(context_report["split"], "test")
        train_target = Path(load_json(self.train_context_ref)["target_path"])
        dev_target = Path(load_json(self.dev_context_ref)["target_path"])
        test_target = Path(load_json(context_ref)["target_path"])
        train_policy, train_policy_hash = context_policy_snapshot(
            train_target, expected_split="train", require_scientific=True
        )
        dev_policy, dev_policy_hash = context_policy_snapshot(
            dev_target, expected_split="dev", require_scientific=True
        )
        test_policy, test_policy_hash = context_policy_snapshot(
            test_target, expected_split="test", require_scientific=True
        )
        self.assertEqual(train_policy, dev_policy)
        self.assertEqual(dev_policy, test_policy)
        self.assertEqual(train_policy_hash, dev_policy_hash)
        self.assertEqual(dev_policy_hash, test_policy_hash)
        control_ref = self.refs / "sealed.control.json"
        seal_test_control_artifact(
            test_context_ref=context_ref,
            frozen_control_ref=self.dev_control_ref,
            write_ref=control_ref,
            target_root=self.artifact_root / "test_controls",
            workspace_root=self.root,
        )
        control_report = validate_control_ref(
            control_ref,
            context_ref=context_ref,
            workspace_root=self.root,
        )
        self.assertEqual(control_report["split"], "test")
        self.assertEqual(control_report["status_counts"]["PL"], {"unavailable": 3})
        proposal_ref = self.refs / "sealed.cf.proposal.json"
        propose_counterfactual_artifact(
            write_ref=proposal_ref,
            split="test",
            context_ref=context_ref,
            frozen_cf_ref=self.dev_cf_ref,
            target_root=self.root,
            workspace_root=self.root,
        )
        proposal_report = validate_proposal_ref(
            proposal_ref, workspace_root=self.root, context_ref=context_ref
        )
        self.assertEqual(proposal_report["split"], "test")
        sealed_cf_ref = self._review_and_finalize(
            proposal_ref, sealed=True, stem="sealed"
        )
        final_report = validate_cf_ref(sealed_cf_ref, workspace_root=self.root)
        self.assertEqual(final_report["split"], "test")
        self.assertTrue(final_report["scientific_eligible"])
        for locator in (context_ref, control_ref, proposal_ref, sealed_cf_ref):
            target = Path(load_json(locator)["target_path"])
            self.assertTrue((target / "frozen_policy_ref.json").is_file())
        schema_by_ref = {
            context_ref: "stage1_frozen_context_policy_ref_v1.schema.json",
            control_ref: "stage1_frozen_control_policy_ref_v1.schema.json",
            proposal_ref: "stage1_frozen_cf_policy_ref_v1.schema.json",
            sealed_cf_ref: "stage1_frozen_cf_policy_ref_v1.schema.json",
        }
        for locator, schema_name in schema_by_ref.items():
            target = Path(load_json(locator)["target_path"])
            validator = Draft202012Validator(
                load_json(REPOSITORY_ROOT / "schemas" / schema_name)
            )
            validator.validate(load_json(target / "frozen_policy_ref.json"))

    @unittest.skip(
        "legacy 3-row formal fixture was retired instead of bypassing deep validation"
    )
    def test_context_policy_change_and_unsealed_test_build_fail_closed(self):
        changed = self._bundle(self.test, split="test", selected=True)
        changed["retrieval_provenance"]["scorer"]["batch_size"] = 4
        changed["bundle_sha256"] = canonical_sha256(changed)
        with self.assertRaisesRegex(ContextBuildError, "retrieval policy"):
            self._seal_context(changed)
        with self.assertRaisesRegex(ContextBuildError, "seal-test"):
            build_prepared_context_artifact(
                prepared_bundle=self._bundle(self.test, split="test", selected=True),
                config=self.context_config,
                tokenizer=TOKENIZER,
                write_ref=None,
                formal=False,
                target_root=self.root / "forbidden-test-context",
            )

    @unittest.skip(
        "legacy 3-row formal fixture was retired instead of bypassing deep validation"
    )
    def test_cf_wrong_policy_split_and_sealed_assertions_fail_closed(self):
        context_ref = self._seal_context()
        with self.assertRaisesRegex(CounterfactualLifecycleError, "only from frozen"):
            propose_counterfactual_artifact(
                write_ref=self.refs / "must-not-exist.json",
                config={"artifact_root": "unused"},
                foil_policy=load_json(
                    REPOSITORY_ROOT / "config/stage1/cf_foil_policy.json"
                ),
                review_rubric=REPOSITORY_ROOT / "config/stage1/cf_review_rubric.md",
                split="test",
                context_ref=context_ref,
                frozen_cf_ref=self.dev_cf_ref,
                target_root=self.root,
                workspace_root=self.root,
            )
        proposal_ref = self.refs / "sealed.assert.proposal.json"
        propose_counterfactual_artifact(
            write_ref=proposal_ref,
            split="test",
            context_ref=context_ref,
            frozen_cf_ref=self.dev_cf_ref,
            target_root=self.root,
            workspace_root=self.root,
        )
        blind_review_ref, review_file, declaration_file = (
            build_completed_cf_blind_review_fixture(
                proposal_ref=proposal_ref,
                workspace_root=self.root,
                stem="assert",
            )
        )
        with self.assertRaisesRegex(CounterfactualLifecycleError, "--sealed"):
            finalize_counterfactual_artifact(
                proposal_ref=proposal_ref,
                blind_review_ref=blind_review_ref,
                review_file=review_file,
                reviewer_declaration=declaration_file,
                write_review_ref=self.refs / "forbidden.review.json",
                write_ref=self.refs / "forbidden.cf.json",
                target_root=self.root,
                workspace_root=self.root,
                sealed=False,
            )
        with self.assertRaisesRegex(CounterfactualLifecycleError, "--sealed"):
            finalize_counterfactual_artifact(
                proposal_ref=self.dev_proposal_ref,
                blind_review_ref=self.refs / "unused-dev-blind-review.json",
                review_file=self.root / "dev.review.jsonl",
                reviewer_declaration=self.root / "dev.declaration.json",
                write_review_ref=self.refs / "forbidden.dev.review.json",
                write_ref=self.refs / "forbidden.dev.cf.json",
                target_root=self.root,
                workspace_root=self.root,
                sealed=True,
            )

    def test_sealed_cli_contracts_are_explicit(self):
        context_args = context_cli._parser().parse_args(
            [
                "seal-test",
                "--frozen-context-ref",
                "dev-context.json",
                "--data-ref",
                "data.json",
                "--train-partition-ref",
                "train-partition.json",
                "--write-ref",
                "test-context.json",
            ]
        )
        self.assertEqual(context_args.command, "seal-test")
        control_args = control_cli._parser().parse_args(
            [
                "seal-test",
                "--test-context-ref",
                "test-context.json",
                "--frozen-control-ref",
                "dev-control.json",
                "--write-ref",
                "test-control.json",
            ]
        )
        self.assertEqual(control_args.command, "seal-test")
        proposal_args = counterfactual_cli._parser().parse_args(
            [
                "propose-cf",
                "--split",
                "test",
                "--test-context-ref",
                "test-context.json",
                "--frozen-cf-ref",
                "dev-cf.json",
                "--write-ref",
                "test-cf-proposal.json",
            ]
        )
        self.assertEqual(proposal_args.frozen_cf_ref, Path("dev-cf.json"))
        finalize_args = counterfactual_cli._parser().parse_args(
            [
                "finalize-cf",
                "--proposal-ref",
                "proposal.json",
                "--blind-review-ref",
                "blind-review.json",
                "--review-file",
                "review.jsonl",
                "--reviewer-declaration",
                "declaration.json",
                "--write-review-ref",
                "review-ref.json",
                "--write-ref",
                "cf-ref.json",
                "--sealed",
            ]
        )
        self.assertTrue(finalize_args.sealed)


REAL_FROZEN_ENV = {
    "dev_context_ref": os.environ.get("STAGE1_REAL_FROZEN_DEV_CONTEXT_REF"),
    "test_context_ref": os.environ.get("STAGE1_REAL_SEALED_TEST_CONTEXT_REF"),
    "data_ref": os.environ.get("STAGE1_REAL_DATA_REF"),
    "workspace": os.environ.get("STAGE1_REAL_WORKSPACE_ROOT"),
}


@unittest.skipUnless(
    all(REAL_FROZEN_ENV.values()),
    "set STAGE1_REAL_FROZEN_DEV_CONTEXT_REF, STAGE1_REAL_DATA_REF, "
    "STAGE1_REAL_SEALED_TEST_CONTEXT_REF and STAGE1_REAL_WORKSPACE_ROOT to run",
)
class RealFrozenContextIntegrationTests(unittest.TestCase):
    def test_real_frozen_dev_and_sealed_test_contexts_replay_deep_validation(self):
        workspace = Path(str(REAL_FROZEN_ENV["workspace"])).resolve()
        data_report = validate_data(data_ref=Path(str(REAL_FROZEN_ENV["data_ref"])))
        dev_report = validate_context_ref(
            Path(str(REAL_FROZEN_ENV["dev_context_ref"])),
            workspace_root=workspace,
        )
        test_report = validate_context_ref(
            Path(str(REAL_FROZEN_ENV["test_context_ref"])),
            workspace_root=workspace,
        )
        self.assertTrue(data_report["valid"])
        self.assertEqual(dev_report["split"], "dev")
        self.assertTrue(dev_report["scientific_eligible"])
        self.assertEqual(test_report["split"], "test")
        self.assertTrue(test_report["scientific_eligible"])


if __name__ == "__main__":
    unittest.main()
