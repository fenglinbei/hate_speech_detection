from __future__ import annotations

import copy
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft202012Validator

from build_lex.annotated_lexicon_repair import (
    LexiconRepairConflict, LexiconRepairError, REPAIR_SOURCE_SHA256,
    canonical_sha256, read_json, write_json,
)
from build_lex.annotated_lexicon_operation_review import (
    OperationReviewStore, build_operation_frame, entry_id, render_legacy_entry,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LEXICON_PATH = REPOSITORY_ROOT / "data/lexicon/annotated_lexicon.json"
SPAN_FRAME_PATH = REPOSITORY_ROOT / "exps/causal_context/annotated_lexicon_repair_v1/span_gold/frame.json"
REAL_OPERATION_FRAME_PATH = REPOSITORY_ROOT / "exps/causal_context/annotated_lexicon_repair_v1/repair_operation/frame.json"


def fixture_inputs(root: Path) -> dict[str, Path]:
    """Use the committed 39-row inventory, not a mutable human review session."""

    frame = read_json(SPAN_FRAME_PATH)
    rows = []
    for item in frame["items"]:
        expected = [
            {"surface": candidate["surface"], "span": candidate["span"], "source_candidate_id": candidate["candidate_id"]}
            for candidate in item["candidates"] if candidate["surface"] == "畒勾"
        ]
        rows.append({
            "item_id": item["item_id"], "source_item_id": item["source_item_id"],
            "cohort": item["cohort"], "content_sha256": item["content_sha256"],
            "query_content": item["query_content"], "expected_spans": expected,
            "review_notes": "synthetic fixture only; not reviewer judgments",
        })
    identity = {
        "schema_version": "annotated-lexicon-span-gold-reference/v1",
        "frame_id": frame["manifest"]["frame_id"], "frame_sha256": canonical_sha256(frame),
        "review_session_revision": "a" * 64, "reviewer_id": "fixture-reviewer",
        "source_lexicon_sha256": REPAIR_SOURCE_SHA256, "item_count": len(rows),
        "rows_sha256": canonical_sha256(rows), "development_only": True,
    }
    reference = {**identity, "reference_id": "span-gold-ref-" + canonical_sha256(identity), "rows": rows}
    reference_path = root / "span-reference.json"
    write_json(reference_path, reference)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for index, original in enumerate(read_json(LEXICON_PATH)["terms"]):
        grouped[original["term"]].append({"source_row_index": index, "lexicon_id": f"lex-{index:04d}", **original})
    required = set(frame["manifest"]["affected_terms"]) | {term for term, values in grouped.items() if len(values) > 1} | {"畒勾"}
    proposals = []
    for term in sorted(required):
        lexicon_id = entry_id(term, grouped.get(term, []))
        proposals.append({
            "term": term,
            "proposed_entry": {
                "lexicon_id": lexicon_id, "term": term, "variants": [],
                "senses": [{"sense_id": lexicon_id + "-s01", "definition": "仅测试用释义：" + term, "categories": ["others"]}],
                "match_policy": {"require_any": [], "exclude_any": []},
            },
            "rationale": "synthetic fixture only", "open_questions": [], "flags": [], "external_evidence": [],
        })
    proposals_path = root / "proposals.json"
    write_json(proposals_path, {"schema_version": "annotated-lexicon-operation-proposals/v1", "items": proposals})
    return {"lexicon_path": LEXICON_PATH, "span_frame_path": SPAN_FRAME_PATH,
            "span_reference_path": reference_path, "proposals_path": proposals_path,
            "output_path": root / "frame.json"}


class OperationReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.arguments = fixture_inputs(self.root)
        self.frame = build_operation_frame(**self.arguments)
        self.store = OperationReviewStore(
            frame_path=self.arguments["output_path"], session_path=self.root / "operation-session.json",
            reviewer_id="fixture-reviewer",
        )
        self.item = self.frame["items"][0]
        self.item_id = self.item["item_id"]

    def decision(self, *, resolution="approve", entry=None, notes="fixture checked") -> dict:
        return {"resolution": resolution, "entry": copy.deepcopy(self.item["proposed_entry"] if entry is None else entry), "notes": notes}

    def save(self, decision=None, *, confirm=True, expected_revision=None) -> dict:
        return self.store.save(
            expected_revision=self.store.bootstrap()["revision"] if expected_revision is None else expected_revision,
            item_id=self.item_id, decision=self.decision() if decision is None else decision, confirm=confirm,
        )

    def validate_schemas(self) -> None:
        for kind, payload in (("frame", self.frame), ("session", read_json(self.store.session_path))):
            schema = read_json(REPOSITORY_ROOT / f"schemas/annotated_lexicon_operation_{kind}_v1.schema.json")
            Draft202012Validator.check_schema(schema)
            Draft202012Validator(schema).validate(payload)

    def test_independent_empty_session_preserves_closed_inventory_and_schema(self) -> None:
        state = self.store.bootstrap()
        self.assertEqual(state["status"]["confirmed_count"], 0)
        self.assertEqual(state["status"]["deferred_count"], 0)
        self.assertEqual(state["status"]["open_count"], len(self.frame["items"]))
        self.assertNotEqual(state["frame_id"], read_json(SPAN_FRAME_PATH)["manifest"]["frame_id"])
        self.assertEqual(sum(item["operation_kind"] == "duplicate_merge" for item in self.frame["items"]), 11)
        self.assertEqual([item["term"] for item in self.frame["items"] if item["operation_kind"] == "new_entry"], ["畒勾"])
        self.assertTrue(all(row["resolution"] is None for row in state["items"]))
        self.validate_schemas()

    def test_cas_save_reopen_and_amendment_history(self) -> None:
        original = self.store.bootstrap()["revision"]
        saved = self.save(expected_revision=original)
        self.assertEqual(saved["decision"]["status"], "confirmed")
        self.assertEqual(saved["status"]["confirmed_count"], 1)
        with self.assertRaises(LexiconRepairConflict):
            self.save(expected_revision=original)
        with self.assertRaisesRegex(ValueError, "重新打开"):
            self.save(expected_revision=saved["revision"])
        reopened = self.store.reopen(expected_revision=saved["revision"], item_id=self.item_id, reason="修正说明")
        self.assertEqual(reopened["decision"]["status"], "draft")
        self.assertEqual(reopened["status"]["amendment_count"], 1)
        self.assertEqual(reopened["status"]["confirmed_count"], 0)
        amendment = self.store.snapshot()["session"]["amendments"][0]
        self.assertEqual(amendment["prior_revision"], saved["revision"])
        self.validate_schemas()

    def test_defer_remains_open_and_requires_reopen(self) -> None:
        saved = self.save(self.decision(resolution="defer", notes="需要更多语境"))
        self.assertEqual(saved["decision"]["status"], "deferred")
        self.assertEqual(saved["status"]["confirmed_count"], 0)
        self.assertEqual(saved["status"]["deferred_count"], 1)
        self.assertEqual(saved["status"]["open_count"], len(self.frame["items"]))
        with self.assertRaisesRegex(ValueError, "重新打开"):
            self.save()
        self.validate_schemas()

    def test_confirmation_requires_decision_and_reason_for_reject_or_defer(self) -> None:
        for resolution in (None, "reject", "defer"):
            with self.subTest(resolution=resolution), self.assertRaises(ValueError):
                self.save(self.decision(resolution=resolution, notes=""))
        self.assertEqual(self.store.bootstrap()["status"]["confirmed_count"], 0)

    def test_unknown_variant_and_equivalent_surface_cannot_be_adopted(self) -> None:
        for variant in ("不在本轮审核范围内的新词面", self.item["term"]):
            entry = copy.deepcopy(self.item["proposed_entry"])
            entry["variants"] = [variant]
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                self.save(self.decision(resolution="revise", entry=entry))

    def test_empty_senses_categories_or_definition_are_draft_only(self) -> None:
        variants = []
        entry = copy.deepcopy(self.item["proposed_entry"])
        entry["senses"] = []
        variants.append(entry)
        entry = copy.deepcopy(self.item["proposed_entry"])
        entry["senses"][0]["categories"] = []
        variants.append(entry)
        entry = copy.deepcopy(self.item["proposed_entry"])
        entry["senses"][0]["definition"] = " "
        variants.append(entry)
        for entry in variants:
            with self.subTest(entry=entry):
                decision = self.decision(resolution="revise", entry=entry)
                self.assertEqual(self.save(decision, confirm=False)["decision"]["status"], "draft")
                with self.assertRaises(ValueError):
                    self.save(decision)
        self.assertEqual(self.store.bootstrap()["status"]["confirmed_count"], 0)

    def test_term_and_stable_id_cannot_change(self) -> None:
        for field, changed in (("term", "范围外"), ("lexicon_id", "lex-9999")):
            entry = copy.deepcopy(self.item["proposed_entry"])
            entry[field] = changed
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "稳定"):
                self.save(self.decision(resolution="revise", entry=entry), confirm=False)

    def test_revised_content_requires_revise_and_legacy_render_is_deterministic(self) -> None:
        entry = copy.deepcopy(self.item["proposed_entry"])
        entry["senses"][0]["definition"] = "修订后的语用描述"
        entry["senses"][0]["categories"] = ["others", "Racism"]
        with self.assertRaisesRegex(ValueError, "修订采用"):
            self.save(self.decision(entry=entry))
        saved = self.save(self.decision(resolution="revise", entry=entry))
        self.assertEqual(saved["decision"]["resolution"], "revise")
        legacy = render_legacy_entry(entry)
        self.assertEqual(legacy["definition"], "[1] 修订后的语用描述")
        self.assertEqual(legacy["category"], "Racism, others")
        self.validate_schemas()

    def test_regex_compilation_failure_cannot_be_confirmed(self) -> None:
        entry = copy.deepcopy(self.item["proposed_entry"])
        entry["match_policy"]["require_any"] = [{"rule_id": "invalid", "target": "left", "pattern": "(?P<"}]
        decision = self.decision(resolution="revise", entry=entry)
        self.save(decision, confirm=False)
        with self.assertRaisesRegex(LexiconRepairError, "compilation failed"):
            self.save(decision)

    def test_snapshot_is_independent_and_hash_bound(self) -> None:
        self.save()
        snapshot = self.store.snapshot()
        self.assertEqual(snapshot["checksums"]["frame"], canonical_sha256(snapshot["frame"]))
        self.assertEqual(snapshot["checksums"]["session"], canonical_sha256(snapshot["session"]))
        snapshot["frame"]["items"][0]["term"] = "outside mutation"
        self.assertEqual(self.store.item_state(self.item_id)["item"]["term"], self.item["term"])

    def test_frame_content_and_identity_tampering_are_rejected(self) -> None:
        original = read_json(self.arguments["output_path"])
        for field in ("items", "manifest"):
            mutated = copy.deepcopy(original)
            if field == "items":
                mutated["items"][0]["rationale"] = "tampered"
            else:
                mutated["manifest"]["span_reference_id"] = "span-gold-ref-" + "0" * 64
            write_json(self.arguments["output_path"], mutated)
            with self.subTest(field=field), self.assertRaises(LexiconRepairError):
                OperationReviewStore(frame_path=self.arguments["output_path"], session_path=self.root / "other-session.json", reviewer_id="fixture-reviewer")

    def test_session_revision_tampering_wrong_reviewer_and_cross_frame_rejected(self) -> None:
        with self.assertRaisesRegex(LexiconRepairError, "reviewer"):
            OperationReviewStore(frame_path=self.arguments["output_path"], session_path=self.store.session_path, reviewer_id="someone-else")
        original = read_json(self.store.session_path)
        tampered = copy.deepcopy(original)
        tampered["decisions"][self.item_id]["notes"] = "tampered"
        write_json(self.store.session_path, tampered)
        with self.assertRaisesRegex(LexiconRepairError, "revision"):
            self.store.bootstrap()
        tampered = copy.deepcopy(original)
        tampered["frame_sha256"] = "0" * 64
        tampered["revision"] = canonical_sha256({key: value for key, value in tampered.items() if key != "revision"})
        write_json(self.store.session_path, tampered)
        with self.assertRaisesRegex(LexiconRepairError, "frame"):
            self.store.bootstrap()

    def test_session_symlink_is_rejected(self) -> None:
        linked = self.root / "linked-session.json"
        linked.symlink_to(self.store.session_path)
        with self.assertRaisesRegex(LexiconRepairError, "symlink"):
            OperationReviewStore(frame_path=self.arguments["output_path"], session_path=linked, reviewer_id="fixture-reviewer")

    def test_frame_build_is_idempotent_and_stable_ids_use_oldest_row(self) -> None:
        self.assertEqual(build_operation_frame(**self.arguments), self.frame)
        for item in self.frame["items"]:
            self.assertEqual(item["proposed_entry"]["lexicon_id"], entry_id(item["term"], item["source_entries"]))
            if item["source_entries"]:
                self.assertEqual(item["proposed_entry"]["lexicon_id"], min(item["source_entries"], key=lambda row: row["source_row_index"])["lexicon_id"])
        self.assertEqual(entry_id("new-term", []), entry_id("new-term", []))

    def test_source_and_proposal_inventory_changes_are_rejected(self) -> None:
        proposals = read_json(self.arguments["proposals_path"])
        proposals["items"].pop()
        write_json(self.arguments["proposals_path"], proposals)
        with self.assertRaisesRegex(LexiconRepairError, "inventory"):
            build_operation_frame(**self.arguments)
        changed_source = self.root / "changed-lexicon.json"
        write_json(changed_source, {"terms": []})
        with self.assertRaisesRegex(LexiconRepairError, "source lexicon"):
            build_operation_frame(**{**self.arguments, "lexicon_path": changed_source})

    def test_span_reference_rows_and_identity_are_checked(self) -> None:
        original = read_json(self.arguments["span_reference_path"])
        for mutation in ("rows", "identity"):
            changed = copy.deepcopy(original)
            if mutation == "rows":
                changed["rows"][0]["review_notes"] = "tampered"
            else:
                changed["reviewer_id"] = "changed-reviewer"
            write_json(self.arguments["span_reference_path"], changed)
            with self.subTest(mutation=mutation), self.assertRaises(LexiconRepairError):
                build_operation_frame(**self.arguments)

    def test_noncanonical_proposal_policy_is_normalized_before_approval(self) -> None:
        proposals = read_json(self.arguments["proposals_path"])
        proposals["items"][0]["proposed_entry"]["match_policy"] = {}
        write_json(self.arguments["proposals_path"], proposals)
        output_path = self.root / "canonical-frame.json"
        frame = build_operation_frame(**{**self.arguments, "output_path": output_path})
        item = next(row for row in frame["items"] if row["term"] == proposals["items"][0]["term"])
        self.assertEqual(item["proposed_entry"]["match_policy"], {"require_any": [], "exclude_any": []})
        store = OperationReviewStore(frame_path=output_path, session_path=self.root / "canonical-session.json", reviewer_id="fixture-reviewer")
        saved = store.save(expected_revision=store.bootstrap()["revision"], item_id=item["item_id"],
                           decision={"resolution": "approve", "entry": item["proposed_entry"], "notes": "fixture"}, confirm=True)
        self.assertEqual(saved["decision"]["status"], "confirmed")

    def test_rehashed_invalid_confirmed_entry_is_rejected_without_compilation(self) -> None:
        session = read_json(self.store.session_path)
        decision = session["decisions"][self.item_id]
        decision["status"] = "confirmed"
        decision["resolution"] = "revise"
        decision["entry"]["senses"] = []
        session["revision"] = canonical_sha256({key: value for key, value in session.items() if key != "revision"})
        write_json(self.store.session_path, session)
        with patch("build_lex.annotated_lexicon_operation_review.validate_match_policy") as compile_policy:
            with self.assertRaisesRegex(LexiconRepairError, "persisted operation decision"):
                self.store.bootstrap()
            compile_policy.assert_not_called()

    def test_real_56_card_frame_and_new_session_schemas(self) -> None:
        real_frame = read_json(REAL_OPERATION_FRAME_PATH)
        self.assertEqual(real_frame["manifest"]["item_count"], 56)
        self.assertEqual(real_frame["manifest"]["duplicate_term_group_count"], 11)
        real_store = OperationReviewStore(frame_path=REAL_OPERATION_FRAME_PATH,
                                          session_path=self.root / "real-frame-session.json", reviewer_id="fixture-reviewer")
        self.assertEqual(real_store.bootstrap()["status"]["open_count"], 56)
        for kind, value in (("frame", real_frame), ("session", real_store.snapshot()["session"])):
            schema = read_json(REPOSITORY_ROOT / f"schemas/annotated_lexicon_operation_{kind}_v1.schema.json")
            Draft202012Validator(schema).validate(value)

    def test_policy_unresolved_requires_policy_revision_not_just_notes(self) -> None:
        real_frame = read_json(REAL_OPERATION_FRAME_PATH)
        item = next(row for row in real_frame["items"] if "policy_unresolved" in row["flags"])
        store = OperationReviewStore(frame_path=REAL_OPERATION_FRAME_PATH,
                                     session_path=self.root / "unresolved-session.json", reviewer_id="fixture-reviewer")
        revision = store.bootstrap()["revision"]
        for resolution in ("approve", "revise"):
            decision = {"resolution": resolution, "entry": copy.deepcopy(item["proposed_entry"]), "notes": "补写备注不能代替策略修复"}
            with self.subTest(resolution=resolution), self.assertRaisesRegex(ValueError, "策略尚未解决"):
                store.save(expected_revision=revision, item_id=item["item_id"], decision=decision, confirm=True)
        entry = copy.deepcopy(item["proposed_entry"])
        entry["senses"][0]["definition"] += "；仅修改释义仍不能越过候选资格门槛。"
        with self.assertRaisesRegex(ValueError, "策略尚未解决"):
            store.save(expected_revision=revision, item_id=item["item_id"],
                       decision={"resolution": "revise", "entry": entry, "notes": "仅解释改动"}, confirm=True)
        result = store.save(expected_revision=revision, item_id=item["item_id"],
                            decision={"resolution": "defer", "entry": None, "notes": "需要判断上下文资格"}, confirm=True)
        self.assertEqual(result["decision"]["status"], "deferred")
        self.assertEqual(result["status"]["confirmed_count"], 0)


if __name__ == "__main__":
    unittest.main()
