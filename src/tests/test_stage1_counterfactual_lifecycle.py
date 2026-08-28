import json
import tempfile
import unittest
from pathlib import Path

from data.counterfactual_lifecycle import (
    CounterfactualLifecycleError,
    _final_meta,
    completed_review_rows_sha256,
    finalize_counterfactual_artifact,
    propose_counterfactual_artifact,
    validate_cf_ref,
    validate_proposal_ref,
)
from data.counterfactual_manifest import (
    build_proposal_rows,
    finalize_rows,
    review_template,
)
from data.context_manifest import finalize_context_budget
from data.training_artifacts import (
    canonical_sha256,
    finalize_target_atomic,
    load_json,
    load_jsonl,
    new_staging_directory,
    write_bytes_atomic,
    write_canonical_json,
    write_canonical_jsonl,
    write_locator_ref,
)
from metrics.stage1_margin import replace_one_field
from scripts.stage1 import build_counterfactuals as counterfactual_cli
from tests.stage1_cf_blind_review_fixtures import (
    build_completed_cf_blind_review_fixture,
)
from utils.quadruple import canonicalize_quadruples


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def quad(target, argument, group, hateful):
    return {
        "target": target,
        "argument": argument,
        "targeted_group": [group],
        "hateful": hateful,
    }


def contexts():
    return [
        {
            "query": {
                "id": "10",
                "content": "坏甲说法，坏乙观点。",
                "gold": [
                    quad("坏甲", "坏甲说法", "Racism", "hate"),
                    quad("坏乙", "坏乙观点", "Sexism", "hate"),
                ],
            },
            "record_sha256": "a" * 64,
        }
    ]


def train_records():
    return [
        {"id": "1", "content": "x", "quadruples": [quad("x", "x", "Racism", "hate")]},
        {"id": "2", "content": "y", "quadruples": [quad("y", "y", "Sexism", "hate")]},
        {"id": "3", "content": "z", "quadruples": [quad(None, "z", "non-hate", "non-hate")]},
    ]


class FakeTokenizer:
    def apply_chat_template(
        self, conversation, *, tokenize, add_generation_prompt, enable_thinking=False
    ):
        self.assertions = (tokenize, add_generation_prompt, enable_thinking)
        return "\n".join(row["content"] for row in conversation) + "\nassistant:"

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


class CounterfactualCliWorkspaceDefaultsTests(unittest.TestCase):
    def test_target_root_never_redefines_default_workspace_root(self):
        target_root = Path("/tmp/stage1-cf-output-only")
        propose = counterfactual_cli._parser().parse_args(
            [
                "propose-cf",
                "--config", "config.json",
                "--review-rubric", "rubric.md",
                "--context-ref", "context.ref.json",
                "--write-ref", "proposal.ref.json",
                "--target-root", str(target_root),
            ]
        )
        finalize = counterfactual_cli._parser().parse_args(
            [
                "finalize-cf",
                "--proposal-ref", "proposal.ref.json",
                "--blind-review-ref", "blind-review.ref.json",
                "--review-file", "review.jsonl",
                "--reviewer-declaration", "declaration.json",
                "--write-review-ref", "review.ref.json",
                "--write-ref", "cf.ref.json",
                "--target-root", str(target_root),
            ]
        )
        self.assertEqual(propose.workspace_root, REPOSITORY_ROOT)
        self.assertEqual(finalize.workspace_root, REPOSITORY_ROOT)
        self.assertEqual(propose.target_root, target_root)
        self.assertEqual(finalize.target_root, target_root)
        self.assertEqual(
            finalize.blind_review_ref, Path("blind-review.ref.json")
        )


class CounterfactualLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.refs = self.root / "refs"
        self.proposal_ref = self.refs / "proposal.json"
        self.review_ref = self.refs / "review.json"
        self.cf_ref = self.refs / "cf.json"
        with (REPOSITORY_ROOT / "config/stage1/cf_foil_policy.json").open(
            "r", encoding="utf-8"
        ) as handle:
            self.policy = json.load(handle)
        self.policy["sampling"]["review_candidate_target"] = 4
        self.policy["sampling"]["field_quota"] = {"target": 2, "argument": 2}

    def tearDown(self):
        self.temporary.cleanup()

    def propose(self):
        return propose_counterfactual_artifact(
            config={"artifact_root": "unused-for-explicit-target"},
            foil_policy=self.policy,
            review_rubric=REPOSITORY_ROOT / "config/stage1/cf_review_rubric.md",
            write_ref=self.proposal_ref,
            split="dev",
            engineering_context_records=contexts(),
            engineering_train_records=train_records(),
            target_root=self.root,
            workspace_root=self.root,
        )

    def formal_context_ref(self):
        context_id = "ctx-" + "c" * 64
        base = {
            "context_build_id": context_id,
            "query": contexts()[0]["query"],
            "selection": {
                "lexicons": {"prompt_order_before_budget": []},
                "demos": {"prompt_order_before_budget": []},
            },
        }
        record = finalize_context_budget(
            base,
            lexicon_catalog={},
            demo_catalog={},
            system_prompt="system",
            user_prompt_template="L:{lexicons}\nD:{examples}\nQ:{text}",
            tokenizer=FakeTokenizer(),
            max_sequence_tokens=2048,
            completion_reserve_tokens=256,
        )
        parent = self.root / "contexts"
        target = parent / context_id
        staging = new_staging_directory(parent, context_id)
        write_canonical_json(
            staging / "context_manifest.dev.meta.json",
            {
                "schema_version": "stage1-context-manifest/v1",
                "context_build_id": context_id,
                "scientific_eligible": True,
            },
        )
        write_bytes_atomic(
            staging / "context_manifest.dev.jsonl",
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n",
        )
        write_canonical_jsonl(
            staging / "catalogs/query_pool.dev.jsonl",
            [
                {
                    "id": "10",
                    "content": contexts()[0]["query"]["content"],
                    "quadruples": contexts()[0]["query"]["gold"],
                }
            ],
            key="id",
            numeric_key=True,
        )
        write_canonical_jsonl(
            staging / "catalogs/query_pool.train.jsonl",
            train_records(),
            key="id",
            numeric_key=True,
        )
        payload_hash = finalize_target_atomic(staging, target)
        ref = self.refs / "context.json"
        write_locator_ref(
            ref,
            artifact_kind="context",
            artifact_id=context_id,
            target=target,
            payload_manifest_sha256=payload_hash,
        )
        return ref

    def completed_review(self, proposal):
        del proposal
        self.blind_review_ref, path, declaration_path = (
            build_completed_cf_blind_review_fixture(
                proposal_ref=self.proposal_ref,
                workspace_root=self.root,
                stem="lifecycle",
            )
        )
        return path, declaration_path

    def finalize(self, proposal):
        review_file, declaration = self.completed_review(proposal)
        locator = finalize_counterfactual_artifact(
            proposal_ref=self.proposal_ref,
            blind_review_ref=self.blind_review_ref,
            review_file=review_file,
            reviewer_declaration=declaration,
            write_review_ref=self.review_ref,
            write_ref=self.cf_ref,
            target_root=self.root,
            workspace_root=self.root,
        )
        return locator, review_file, declaration

    def test_engineering_fixture_is_explicitly_ineligible_and_content_addressed(self):
        proposal = self.propose()
        report = validate_proposal_ref(
            self.proposal_ref, workspace_root=self.root
        )
        self.assertRegex(proposal["artifact_id"], r"^cfp-[0-9a-f]{64}$")
        self.assertFalse(report["scientific_eligible"])
        target = Path(proposal["target_path"])
        self.assertEqual(
            load_json(target / "context_ref.json")["schema_version"],
            "stage1-engineering-context-fixture/v1",
        )
        self.assertTrue((target / "engineering_context.jsonl").is_file())

    def test_formal_context_ref_consumes_the_strict_frozen_query_pool_layout(self):
        context_ref = self.formal_context_ref()
        proposal = propose_counterfactual_artifact(
            config={"artifact_root": "unused-for-explicit-target"},
            foil_policy=self.policy,
            review_rubric=REPOSITORY_ROOT / "config/stage1/cf_review_rubric.md",
            write_ref=self.proposal_ref,
            split="dev",
            context_ref=context_ref,
            target_root=self.root,
            workspace_root=self.root,
        )
        report = validate_proposal_ref(
            self.proposal_ref,
            workspace_root=self.root,
            context_ref=context_ref,
        )
        self.assertTrue(report["scientific_eligible"])
        dependency = load_json(Path(proposal["target_path"]) / "context_ref.json")
        self.assertEqual(dependency["artifact_kind"], "context")
        self.assertNotIn("target_path", dependency)

    def test_candidates_and_final_rows_prove_exactly_one_field_change(self):
        proposal = self.propose()
        proposal_rows = load_jsonl(Path(proposal["target_path"]) / "candidates.dev.jsonl")
        for row in proposal_rows:
            before = canonicalize_quadruples(row["gold_quadruples"])
            after = replace_one_field(
                before,
                tuple_index=row["tuple_index"],
                field=row["field"],
                candidate_value=row["candidate_value"],
            )
            changed = [
                (index, field)
                for index, (gold, foil) in enumerate(zip(before, after, strict=True))
                for field in ("target", "argument", "targeted_group", "hateful")
                if getattr(gold, field) != getattr(foil, field)
            ]
            self.assertEqual(changed, [(row["tuple_index"], row["field"])])
        final, _, _ = self.finalize(proposal)
        report = validate_cf_ref(self.cf_ref, workspace_root=self.root)
        self.assertEqual(report["record_count"], 2 * 4)
        self.assertTrue(report["group_hate_coverage_gate"])
        self.assertFalse(report["scientific_eligible"])
        self.assertRegex(final["artifact_id"], r"^cf-[0-9a-f]{64}$")
        review_target = Path(load_json(self.review_ref)["target_path"])
        self.assertEqual(
            load_json(review_target / "cf_blind_review_ref.json")["artifact_id"],
            report["cf_blind_review_id"],
        )
        provenance = load_json(review_target / "review.provenance.json")
        self.assertEqual(
            provenance["cf_blind_review_dependency"],
            load_json(review_target / "cf_blind_review_ref.json"),
        )
        self.assertEqual(
            provenance["human_queue_rows_sha256"],
            load_json(review_target / "reviewer_declaration.json")[
                "human_queue_rows_sha256"
            ],
        )

    def test_review_candidate_set_must_be_exact(self):
        proposal = self.propose()
        review_file, declaration_path = self.completed_review(proposal)
        rows = load_jsonl(review_file)[:-1]
        write_canonical_jsonl(review_file, rows, key="candidate_id")
        with self.assertRaisesRegex(CounterfactualLifecycleError, "exactly"):
            finalize_counterfactual_artifact(
                proposal_ref=self.proposal_ref,
                blind_review_ref=self.blind_review_ref,
                review_file=review_file,
                reviewer_declaration=declaration_path,
                write_review_ref=self.review_ref,
                write_ref=self.cf_ref,
                target_root=self.root,
                workspace_root=self.root,
            )

    def test_blind_declaration_flags_fail_closed(self):
        proposal = self.propose()
        review_file, declaration_path = self.completed_review(proposal)
        declaration = load_json(declaration_path)
        declaration["saw_model_scores"] = True
        write_canonical_json(declaration_path, declaration)
        with self.assertRaises(CounterfactualLifecycleError):
            finalize_counterfactual_artifact(
                proposal_ref=self.proposal_ref,
                blind_review_ref=self.blind_review_ref,
                review_file=review_file,
                reviewer_declaration=declaration_path,
                write_review_ref=self.review_ref,
                write_ref=self.cf_ref,
                target_root=self.root,
                workspace_root=self.root,
            )

    def test_frozen_automatic_consensus_cannot_be_replaced_by_direct_review(self):
        proposal = self.propose()
        review_file, declaration_path = self.completed_review(proposal)
        blind_target = Path(load_json(self.blind_review_ref)["target_path"])
        auto_rows = load_jsonl(blind_target / "auto_review.jsonl")
        self.assertTrue(auto_rows)
        rows = load_jsonl(review_file)
        auto_id = auto_rows[0]["candidate_id"]
        next(row for row in rows if row["candidate_id"] == auto_id)["note"] = (
            "directly rewritten automatic decision"
        )
        write_canonical_jsonl(review_file, rows, key="candidate_id")
        declaration = load_json(declaration_path)
        declaration["completed_review_rows_sha256"] = completed_review_rows_sha256(
            rows
        )
        write_canonical_json(declaration_path, declaration)
        with self.assertRaisesRegex(
            CounterfactualLifecycleError, "automatic consensus"
        ):
            finalize_counterfactual_artifact(
                proposal_ref=self.proposal_ref,
                blind_review_ref=self.blind_review_ref,
                review_file=review_file,
                reviewer_declaration=declaration_path,
                write_review_ref=self.review_ref,
                write_ref=self.cf_ref,
                target_root=self.root,
                workspace_root=self.root,
            )

    def test_blind_review_from_sibling_proposal_is_rejected(self):
        proposal = self.propose()
        review_file, declaration_path = self.completed_review(proposal)
        sibling_policy = json.loads(json.dumps(self.policy))
        sibling_policy["sampling"]["stable_hash_seed"] += 1
        sibling_ref = self.refs / "sibling.proposal.json"
        propose_counterfactual_artifact(
            config={"artifact_root": "unused-for-explicit-target"},
            foil_policy=sibling_policy,
            review_rubric=REPOSITORY_ROOT / "config/stage1/cf_review_rubric.md",
            write_ref=sibling_ref,
            split="dev",
            engineering_context_records=contexts(),
            engineering_train_records=train_records(),
            target_root=self.root,
            workspace_root=self.root,
        )
        sibling_blind_ref, _, _ = build_completed_cf_blind_review_fixture(
            proposal_ref=sibling_ref,
            workspace_root=self.root,
            stem="sibling",
        )
        with self.assertRaisesRegex(
            CounterfactualLifecycleError, "different proposal dependency"
        ):
            finalize_counterfactual_artifact(
                proposal_ref=self.proposal_ref,
                blind_review_ref=sibling_blind_ref,
                review_file=review_file,
                reviewer_declaration=declaration_path,
                write_review_ref=self.review_ref,
                write_ref=self.cf_ref,
                target_root=self.root,
                workspace_root=self.root,
            )

    def test_payload_tamper_is_detected_before_replay(self):
        proposal = self.propose()
        target = Path(proposal["target_path"])
        rows = load_jsonl(target / "candidates.dev.jsonl")
        rows[0]["source_id"] = "tampered"
        write_canonical_jsonl(target / "candidates.dev.jsonl", rows, key="candidate_id")
        with self.assertRaisesRegex(CounterfactualLifecycleError, "payload manifest"):
            validate_proposal_ref(self.proposal_ref, workspace_root=self.root)

    def test_group_hate_gate_uses_all_gold_tuple_denominators(self):
        proposals, _ = build_proposal_rows(
            cf_proposal_id="cfp-test",
            context_records=contexts(),
            train_records=train_records(),
            field_quota={"target": 2, "argument": 2},
        )
        proposals = [row for row in proposals if row["field"] != "targeted_group"]
        reviews = review_template(proposals, reviewer_id="panel")
        for row in reviews:
            if row["decision"] == "":
                row["decision"] = "pass"
                row["reason_code"] = "valid-local-foil"
        _, summary = finalize_rows(
            proposals, reviews, cf_build_id="cf-test", review_id="review-test"
        )
        self.assertEqual(summary["field_construction"]["targeted_group"]["denominator"], 2)
        self.assertEqual(summary["field_construction"]["targeted_group"]["constructed"], 0)
        self.assertFalse(summary["group_hate_coverage_gate"])

    def test_complete_case_requires_every_tuple_for_the_field(self):
        rows = [
            {
                "query_id": "10",
                "tuple_index": tuple_index,
                "field": field,
                "construction_status": (
                    "no-valid-candidate"
                    if field == "target" and tuple_index == 1
                    else "ok"
                ),
                "record_sha256": canonical_sha256(
                    {"tuple_index": tuple_index, "field": field}
                ),
            }
            for tuple_index in (0, 1)
            for field in ("target", "argument", "targeted_group", "hateful")
        ]
        meta = _final_meta(
            cf_build_id="cf-test",
            id_inputs={},
            rows=rows,
            summary={
                "field_construction": {},
                "group_hate_coverage_gate": True,
            },
            proposal_id="cfp-test",
            review_id="cfr-test",
            split="dev",
            scientific_eligible=False,
        )
        self.assertNotIn("10", meta["complete_case_query_ids"]["target"])
        self.assertIn("10", meta["complete_case_query_ids"]["argument"])
        self.assertIn("10", meta["complete_case_query_ids"]["targeted_group"])
        self.assertIn("10", meta["complete_case_query_ids"]["hateful"])

    def test_same_inputs_are_idempotent_and_never_overwrite_different_payload(self):
        first_proposal = self.propose()
        first_payload = first_proposal["payload_manifest_sha256"]
        second_proposal = self.propose()
        self.assertEqual(first_proposal["artifact_id"], second_proposal["artifact_id"])
        self.assertEqual(first_payload, second_proposal["payload_manifest_sha256"])
        first_final, review_file, declaration = self.finalize(second_proposal)
        second_final = finalize_counterfactual_artifact(
            proposal_ref=self.proposal_ref,
            blind_review_ref=self.blind_review_ref,
            review_file=review_file,
            reviewer_declaration=declaration,
            write_review_ref=self.review_ref,
            write_ref=self.cf_ref,
            target_root=self.root,
            workspace_root=self.root,
        )
        self.assertEqual(first_final["artifact_id"], second_final["artifact_id"])
        self.assertEqual(
            first_final["payload_manifest_sha256"],
            second_final["payload_manifest_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
