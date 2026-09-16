"""Isolated evidence review tests; never uses a human or production session."""

from __future__ import annotations

import copy
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from build_lex.annotated_lexicon_repair import canonical_sha256, file_sha256, write_json
from tools.general_model_paired_review_ui.evidence_schema import empty_values
from tools.general_model_paired_review_ui.evidence_store import BUNDLE_SCHEMA, EvidenceReviewStore, ReviewConflict, ReviewError


def text_values(text: str, *, demo: bool = False) -> dict:
    return {**empty_values("demo" if demo else "query"), "hate": "non-hate", "group": [],
            "hate_reason": "no_attack", "group_reason": "no_group_target", "stance": "neutral",
            "target_types": ["none"], "expression_types": ["neutral_or_laughter"],
            "evidence": [{"source": "text", "start": 0, "end": len(text), "text": text}],
            **({"hate_original_status": "accepted", "group_original_status": "accepted"} if demo else {})}


def fixture(root: Path) -> Path:
    bundle = {"schema_version": BUNDLE_SCHEMA, "policy": {"version": "test-policy/v1", "sha256": "f" * 64},
              "source_identity": {"discovery": "test-fixture", "prior_session_sha256": "d" * 64},
              "order": ["one", "two"], "cases": {}, "objects": {}}

    def add(oid: str, kind: str, source: dict, values: dict, dependencies: list[str] | None = None) -> None:
        bundle["objects"][oid] = {"id": oid, "kind": kind, "version": canonical_sha256(source), "source": source,
                                  "depends_on": dependencies or [], "ai_draft": {"version": "ai-test/v1", "review_kind": "ai_note", "values": values,
                                  "provenance": {"generator": "synthetic-test-only", "input_sha256": canonical_sha256(source)}}}

    demo_text = "共享测试示例🙂"
    add("d1", "demo", {"text": demo_text, "original_answer": {"hate": "non-hate", "group": []}}, text_values(demo_text, demo=True))
    add("def1", "definition", {"text": "一个固定词义"}, {**empty_values("definition"), "definition_verdict": "reasonable", "issues": ["valid_sense"]})
    for key in ("one", "two"):
        text = "查询🙂 " + key
        qid, rid, hid = "q-" + key, "r-" + key, "h-" + key
        add(qid, "query", {"text": text}, text_values(text))
        add(rid, "relation", {"texts": {"query": text, "demo": demo_text}},
            {**empty_values("relation"), **dict.fromkeys(("topic_hate", "topic_group", "rule_hate", "rule_group"), "none"), "lexicon_risk": "no"}, [qid, "d1"])
        add(hid, "hit", {"texts": {"query": text, "definition": "一个固定词义"}},
            {**empty_values("hit"), "source_fit": "uncertain", "query_fit": "uncertain", "issues": ["provenance_mismatch"]}, [qid, "def1"])
        bundle["cases"][key] = {"query_id": key, "query_object_id": qid, "split": "discovery", "object_ids": [qid, "d1", "def1", rid, hid],
                                "comparison": {"gold": {"hate": "non-hate", "group": [], "sentinel": "SECRET_QUERY_GOLD"},
                                               "trajectories": {"SECRET_CONDITION": "SECRET_PREDICTION"}, "bucket": "SECRET_BUCKET",
                                               "ai_explanation": {"status": "awaiting_material_review"},
                                               "explanation_proposal": "SECRET_EXPLANATION"}}
    path = root / "experiment/results/draft/bundle.json"
    write_json(path, bundle)
    return path


class EvidenceStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="evidence-review-isolated-")
        self.root = Path(self.temporary.name)
        self.bundle = fixture(self.root)
        self.session = self.root / "experiment/reviews/automated-deployment-test/session.json"
        self.store = self.new_store()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def new_store(self, reviewer: str = "automated-deployment-test") -> EvidenceReviewStore:
        return EvidenceReviewStore(bundle_path=self.bundle, session_path=self.session, reviewer_id=reviewer)

    def obj(self, oid: str, key: str = "one") -> dict:
        return next(obj for obj in self.store.item_state(key)["objects"] if obj["id"] == oid)

    def action(self, action: str, key: str = "one", **kwargs) -> dict:
        return self.store.mutate(expected_revision=self.store.bootstrap()["revision"], item_id=key, action=action, **kwargs)

    def confirm_object(self, oid: str, key: str = "one", **kwargs) -> dict:
        obj = self.obj(oid, key)
        return self.action("confirm_object", key=key, object_id=oid, object_version=obj["review"]["version"], values=kwargs.get("values", obj["effective_values"]))

    def materials(self, key: str = "one") -> None:
        for obj in self.store.item_state(key)["objects"]:
            if obj["review"]["status"] != "confirmed" or obj["stale"]:
                self.confirm_object(obj["id"], key)
        self.action("reveal", key=key)

    @staticmethod
    def assessment() -> dict:
        return {"hate": "non-hate", "group": [], "hate_original_status": "accepted", "group_original_status": "accepted",
                "hate_use": "reference_analysis", "group_use": "reference_analysis", "explanation_choice": "defer", "note": ""}

    def test_ai_import_and_reads_leave_zero_human_and_hide_results(self) -> None:
        original_bytes = self.session.read_bytes()
        item = self.store.item_state("one")
        serialized = json.dumps(item)
        for secret in ("SECRET_QUERY_GOLD", "SECRET_PREDICTION", "SECRET_BUCKET", "SECRET_EXPLANATION"):
            self.assertNotIn(secret, serialized)
        self.assertEqual(item["objects"][1]["source"]["original_answer"], {"hate": "non-hate", "group": []})
        self.assertIsNone(item["comparison"])
        self.assertEqual(item["objects"][0]["effective_values"]["hate"], "non-hate")
        self.assertIsNone(item["objects"][0]["review"]["values"])
        self.assertEqual(item["bootstrap"]["status"]["confirmed_object_count"], 0)
        self.assertEqual(self.session.read_bytes(), original_bytes)
        exported = json.dumps(self.store.snapshot(item["revision"]))
        self.assertNotIn("SECRET_QUERY_GOLD", exported)
        with self.assertRaises(ReviewError):
            self.store.item_state("reserve")

    def test_absent_mechanism_proposal_cannot_be_adopted(self) -> None:
        self.materials()
        before = self.session.read_bytes()
        assessment = self.assessment()
        assessment['explanation_choice'] = 'adopt'
        with self.assertRaises(ReviewError):
            self.action('confirm', assessment=assessment)
        self.assertEqual(self.session.read_bytes(), before)
        assessment['explanation_choice'] = 'defer'
        self.action('confirm', assessment=assessment)

    def test_save_resume_and_conflict_keep_human_draft(self) -> None:
        initial = self.store.bootstrap()["revision"]
        values = self.obj("q-one")["effective_values"]
        values["note"] = "本地草稿🙂"
        self.action("save_object", object_id="q-one", object_version=0, values=values)
        restarted = self.new_store()
        obj = next(obj for obj in restarted.item_state("one")["objects"] if obj["id"] == "q-one")
        self.assertEqual(obj["review"]["values"]["note"], "本地草稿🙂")
        self.assertEqual(restarted.bootstrap()["status"]["confirmed_object_count"], 0)
        before = self.session.read_bytes()
        with self.assertRaises(ReviewConflict):
            self.store.mutate(expected_revision=initial, item_id="one", action="save_object", object_id="q-one", object_version=0, values={})
        self.assertEqual(self.session.read_bytes(), before)
        with self.assertRaises(ReviewConflict):
            self.action("confirm_object", object_id="q-one", object_version=0, values=values)

    def test_material_gate_explicit_confirmation_and_preserved_snapshot(self) -> None:
        before = self.session.read_bytes()
        for action in ("reveal", "confirm", "save_assessment"):
            with self.assertRaises(ReviewError):
                self.action(action, assessment=self.assessment())
            self.assertEqual(self.session.read_bytes(), before)
        self.materials()
        item = self.store.item_state("one")
        self.assertEqual(item["comparison"]["gold"]["sentinel"], "SECRET_QUERY_GOLD")
        self.assertEqual(item["review"]["status"], "draft")
        self.assertEqual(self.store.bootstrap()["status"]["confirmed_count"], 0)
        snapshot = copy.deepcopy(item["review"]["material_snapshots"][0])
        self.action("confirm", assessment=self.assessment())
        self.assertEqual(self.new_store().bootstrap()["status"]["confirmed_count"], 1)
        self.action("reopen", reason="misclick")
        self.assertEqual(self.store.item_state("one")["review"]["material_snapshots"][0], snapshot)
        export = self.store.snapshot(self.store.bootstrap()["revision"])
        previous = export["session"]["events"][-1]["previous_record"]
        self.assertEqual(previous["status"], "confirmed")
        self.assertEqual(previous["assessment"], self.assessment())

    def test_batch_affects_only_explicit_ids_and_failure_is_atomic(self) -> None:
        self.action("confirm_batch", objects=[{"object_id": "q-one", "object_version": 0}])
        self.assertEqual(self.obj("q-one")["review"]["status"], "confirmed")
        self.assertEqual(self.obj("d1")["review"]["status"], "unreviewed")
        self.assertEqual(self.store.bootstrap()["status"]["confirmed_object_count"], 1)
        with self.assertRaises(ReviewError):
            self.action("reveal")
        before = self.session.read_bytes()
        for selected in (
            [{"object_id": "d1", "object_version": 0}, {"object_id": "q-two", "object_version": 0}],
            [{"object_id": "d1", "object_version": 0}, {"object_id": "d1", "object_version": 0}],
            [{"object_id": "d1", "object_version": 0}, {"object_id": "def1", "object_version": 99}],
        ):
            with self.assertRaises(ReviewError):
                self.action("confirm_batch", objects=selected)
            self.assertEqual(self.session.read_bytes(), before)

    def test_shared_amendment_invalidates_references_and_can_be_reconfirmed(self) -> None:
        self.materials("one")
        self.action("confirm", assessment=self.assessment())
        self.materials("two")
        self.action("confirm", key="two", assessment=self.assessment())
        old_snapshot = copy.deepcopy(self.store.item_state("two")["review"]["material_snapshots"][0])
        demo = self.obj("d1")
        self.action("reopen_object", object_id="d1", object_version=demo["review"]["version"], reason="new_evidence")
        for key in ("one", "two"):
            self.assertTrue(self.store.item_state(key)["stale"])
            self.assertTrue(self.obj("r-" + key, key)["stale"])
        self.assertEqual(self.store.bootstrap()["status"]["confirmed_count"], 0)
        with self.assertRaises(ReviewError):
            self.action("confirm", assessment=self.assessment())
        self.confirm_object("d1")
        self.confirm_object("r-two", "two")
        self.action("reveal", key="two")
        self.action("confirm", key="two", assessment=self.assessment())
        updated = self.store.item_state("two")
        self.assertFalse(updated["stale"])
        self.assertEqual(len(updated["review"]["material_snapshots"]), 2)
        self.assertEqual(updated["review"]["material_snapshots"][0], old_snapshot)
        self.assertTrue(self.store.item_state("one")["stale"])
        self.assertEqual(self.new_store().bootstrap()["status"]["confirmed_count"], 1)

    def test_independent_unresolved_and_empty_group_are_preserved(self) -> None:
        values = self.obj("q-one")["effective_values"]
        values.update(hate=None, hate_reason="individual_scope", group=[])
        self.confirm_object("q-one", values=values)
        self.materials()
        final = self.assessment()
        final.update(hate=None, hate_original_status="policy_ambiguous", hate_use="verify_first")
        self.action("confirm", assessment=final)
        snapshot = self.store.snapshot(self.store.bootstrap()["revision"])
        self.assertEqual(snapshot["case_eligibility"]["one"]["reference_tasks"], ["group"])
        rows = list(csv.DictReader(io.StringIO(self.store.export_csv(snapshot).decode("utf-8-sig"))))
        case = next(row for row in rows if row["case_id"] == "one")
        self.assertEqual(case["hate"], "null")
        self.assertEqual(case["group"], "[]")
        self.assertEqual(snapshot["session"]["objects"]["q-one"]["adoption"], "adopted_modified")

    def test_choice_validation_and_unicode_evidence_fail_without_writes(self) -> None:
        values = self.obj("q-one")["effective_values"]
        invalids = []
        for field, value in (("hate", "maybe"), ("group", ["none"]), ("group", ["Racism", "Racism"]), ("group", "[]"), ("hate_reason", "invented"), ("target_types", ["none", "group"])):
            changed = copy.deepcopy(values)
            changed[field] = value
            invalids.append(changed)
        changed = copy.deepcopy(values)
        changed["evidence"][0]["end"] += 1
        invalids.append(changed)
        changed = copy.deepcopy(values)
        changed["hate"] = "hate"
        invalids.append(changed)
        for value in invalids:
            with self.subTest(value=value):
                before = self.session.read_bytes()
                with self.assertRaises(ReviewError):
                    self.confirm_object("q-one", values=value)
                self.assertEqual(self.session.read_bytes(), before)

    def test_original_status_does_not_invent_reference(self) -> None:
        demo = self.obj("d1")["effective_values"]
        demo.update(hate="hate", hate_reason="agreed_attack")
        with self.assertRaises(ReviewError):
            self.confirm_object("d1", values=demo)
        demo["hate_original_status"] = "suspected_error"
        self.confirm_object("d1", values=demo)
        self.materials()
        final = self.assessment()
        final["hate_original_status"] = "policy_ambiguous"
        with self.assertRaises(ReviewError):
            self.action("confirm", assessment=final)
        final["hate"] = None
        with self.assertRaises(ReviewError):
            self.action("confirm", assessment=final)
        final["hate_use"] = "verify_first"
        self.action("confirm", assessment=final)

    def test_source_identity_and_original_bytes_remain_protected(self) -> None:
        before = file_sha256(self.bundle)
        self.confirm_object("q-one")
        self.assertEqual(file_sha256(self.bundle), before)
        with self.assertRaises(ReviewError):
            self.new_store(reviewer="different-human")
        with self.assertRaises(ReviewError):
            EvidenceReviewStore(bundle_path=self.bundle, session_path=self.bundle, reviewer_id="automated-deployment-test")
        content = json.loads(self.bundle.read_text())
        content["source_identity"]["changed"] = True
        write_json(self.bundle, content)
        with self.assertRaises(ReviewError):
            self.new_store()

    def test_reserve_or_gold_contamination_is_rejected(self) -> None:
        baseline = json.loads(self.bundle.read_text())
        variations = []
        contaminated = copy.deepcopy(baseline)
        contaminated["objects"]["q-one"]["source"]["gold"] = "SECRET_QUERY_GOLD"
        variations.append(contaminated)
        reserve = copy.deepcopy(baseline)
        reserve["cases"]["one"]["split"] = "reserve"
        variations.append(reserve)
        orphan = copy.deepcopy(baseline)
        orphan["objects"]["reserve-query"] = copy.deepcopy(orphan["objects"]["q-one"])
        variations.append(orphan)
        for value in variations:
            write_json(self.bundle, value)
            with self.assertRaises(ReviewError):
                self.new_store()

    def test_exposure_and_export_revision(self) -> None:
        exposure = {"gold": "seen", "resources": "seen", "results": "seen", "ai": "seen"}
        old_revision = self.store.bootstrap()["revision"]
        self.action("save_exposure", prior_exposure=exposure)
        self.assertEqual(self.new_store().item_state("one")["review"]["prior_exposure"], exposure)
        with self.assertRaises(ReviewConflict):
            self.store.snapshot(old_revision)
        self.assertEqual(self.store.bootstrap()["status"]["confirmed_object_count"], 0)

    def test_prior_read_only_reference_preserves_known_exposure(self) -> None:
        data = json.loads(self.bundle.read_text())
        reference = {"source_session_sha256": "a" * 64, "record_sha256": "b" * 64, "status": "confirmed",
                     "known_exposure": {"resources": "seen", "gold": "seen", "results": "seen", "ai": "unsure"}}
        data["cases"]["one"]["prior_review_reference"] = reference
        write_json(self.bundle, data)
        self.session = self.root / "experiment/reviews/known-exposure-test/session.json"
        self.store = self.new_store()
        item = self.store.item_state("one")
        self.assertEqual(item["prior_review_reference"], reference)
        self.assertEqual(item["review"]["prior_exposure"]["gold"], "seen")
        self.assertEqual(item["review"]["status"], "unreviewed")
        self.assertIsNone(item["comparison"])
        exposure = copy.deepcopy(item["review"]["prior_exposure"])
        exposure["gold"] = "not_seen"
        before = self.session.read_bytes()
        with self.assertRaises(ReviewError):
            self.action("save_exposure", prior_exposure=exposure)
        self.assertEqual(self.session.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
