"""Policy migration and task-scoped review tests, using synthetic records only."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import unittest
from pathlib import Path

from tools.general_model_paired_review_ui.test_evidence_store import EvidenceStoreTests
from tools.general_model_paired_review_ui.evidence_policy import AMENDMENT_SCHEMA, migrate_policy_session, policy_ref
from tools.general_model_paired_review_ui.evidence_schema import normalize_assessment, normalize_values
from tools.general_model_paired_review_ui.evidence_store import EvidenceReviewStore, ReviewError, ReviewConflict
from build_lex.annotated_lexicon_repair import canonical_sha256, file_sha256, read_json, write_json


class EvidencePolicyTests(unittest.TestCase):
    setUp = EvidenceStoreTests.setUp
    tearDown = EvidenceStoreTests.tearDown
    new_store = EvidenceStoreTests.new_store
    obj = EvidenceStoreTests.obj
    action = EvidenceStoreTests.action
    confirm_object = EvidenceStoreTests.confirm_object
    materials = EvidenceStoreTests.materials
    assessment = staticmethod(EvidenceStoreTests.assessment)

    def policy_fixture(self) -> Path:
        bundle = read_json(self.bundle)
        text = "Synthetic group policy v2. Hate semantics remain unchanged.\n"
        policy = {"schema_version": AMENDMENT_SCHEMA, "version": "test-policy/v2",
                  "document_text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "bundle_sha256": file_sha256(self.bundle), "parent_policy": policy_ref(bundle["policy"]),
                  "approval": {"reviewer_id": "test-policy-owner", "confirmed_at": "2026-09-09T00:00:00Z"},
                  "impact": {"tasks": ["group"], "object_ids": [oid for oid, obj in bundle["objects"].items() if obj["kind"] in {"query", "demo", "relation"}],
                             "case_ids": list(bundle["order"]), "reason": "Synthetic fixture, never human data"}}
        path = self.root / "policy/v2.json"
        write_json(path, policy)
        return path

    def migrate(self) -> dict:
        self.policy_path = self.policy_fixture()
        result = migrate_policy_session(bundle_path=self.bundle, policy_path=self.policy_path, session_path=self.session,
                                        reviewer_id="automated-deployment-test", expected_revision=self.store.bootstrap()["revision"],
                                        expected_session_sha256=file_sha256(self.session))
        self.store = EvidenceReviewStore(bundle_path=self.bundle, session_path=self.session, reviewer_id="automated-deployment-test", policy_path=self.policy_path)
        return result

    def confirmed_case(self) -> None:
        self.materials()
        self.action("confirm", assessment=self.assessment())

    def confirm_current_group_materials(self) -> None:
        for oid in ("q-one", "d1", "r-one"):
            self.confirm_object(oid)

    def test_explicit_migration_preserves_every_existing_field_and_exact_backup(self) -> None:
        self.confirmed_case()
        values = self.obj("q-two", "two")["effective_values"]
        values["note"] = "Synthetic latest unsent conclusion draft"
        self.action("save_object", key="two", object_id="q-two", object_version=0, values=values)
        old_bytes, bundle_bytes = self.session.read_bytes(), self.bundle.read_bytes()
        before = read_json(self.session)
        result = self.migrate()
        after = read_json(self.session)
        self.assertEqual(Path(result["backup_path"]).read_bytes(), old_bytes)
        self.assertEqual(self.bundle.read_bytes(), bundle_bytes)
        self.assertEqual(after["policy_history"][0]["session"], before)
        self.assertEqual(after["policy_history"][0]["snapshot_sha256"], canonical_sha256(before))
        for kind in ("objects", "records"):
            for oid, row in before[kind].items():
                for field, value in row.items():
                    self.assertEqual(after[kind][oid][field], value, (kind, oid, field))
        for oid in ("def1", "h-one", "h-two"):
            self.assertEqual(after["objects"][oid], before["objects"][oid])
        self.assertEqual(after["events"][:-1], before["events"])
        item = self.store.item_state("one")
        self.assertTrue(item["comparison_previously_revealed"])
        self.assertIsNotNone(item["comparison"])
        self.assertEqual(item["review"]["prior_exposure"]["gold"], "seen")
        self.assertIsNone(self.store.item_state("two")["comparison"])
        self.assertEqual(self.obj("q-two", "two")["effective_values"], values)
        self.assertEqual(self.obj("q-one")["ai_draft"]["policy"], before["policy"])
        self.assertEqual(self.obj("q-one")["review"]["task_reviews"]["hate"]["status"], "confirmed")
        exported = self.store.snapshot(self.store.bootstrap()["revision"])
        self.assertEqual(exported["case_eligibility"]["one"]["reference_tasks"], ["hate"])
        self.assertEqual(result["human_confirmations_added"], 0)

    def test_startup_dry_run_stale_revision_and_old_writer_fail_closed(self) -> None:
        policy = self.policy_fixture()
        old_store = self.store
        raw, revision = self.session.read_bytes(), self.store.bootstrap()["revision"]
        with self.assertRaises(ReviewError):
            EvidenceReviewStore(bundle_path=self.bundle, session_path=self.session, reviewer_id="automated-deployment-test", policy_path=policy)
        with self.assertRaises(ReviewError):
            EvidenceReviewStore(bundle_path=self.bundle, session_path=self.session.parent / "missing.json", reviewer_id="automated-deployment-test", policy_path=policy)
        kwargs = dict(bundle_path=self.bundle, policy_path=policy, session_path=self.session, reviewer_id="automated-deployment-test")
        preview = migrate_policy_session(**kwargs, expected_revision=revision, dry_run=True)
        self.assertFalse(Path(preview["backup_path"]).exists())
        self.assertEqual(self.session.read_bytes(), raw)
        for mismatch in ({"expected_revision": "stale"}, {"expected_revision": revision, "expected_session_sha256": "0" * 64}):
            with self.assertRaises(ReviewConflict):
                migrate_policy_session(**kwargs, **mismatch)
        self.assertEqual(self.session.read_bytes(), raw)
        self.migrate()
        with self.assertRaises(ReviewError):
            old_store.bootstrap()
        with self.assertRaises(ReviewError):
            migrate_policy_session(**kwargs, expected_revision=revision)
        self.assertEqual(self.store.bootstrap()["status"]["confirmed_object_count"], 0)
        self.assertEqual([item["status"] for item in self.store.bootstrap()["items"]], ["unreviewed", "unreviewed"])

    def test_group_draft_confirm_and_export_preserve_hate_and_hits(self) -> None:
        self.confirmed_case()
        old = read_json(self.session)
        self.migrate()
        q = self.obj("q-one")
        values = copy.deepcopy(q["effective_values"])
        values.update(group=["others"], group_reason="identity_target")
        self.action("save_object", object_id="q-one", object_version=q["review"]["version"], values=values)
        q = self.obj("q-one")
        self.assertTrue(q["needs_group_recheck"])
        self.assertEqual(q["review"]["task_reviews"]["group"]["status"], "draft")
        self.assertFalse(self.obj("h-one")["stale"])
        self.assertEqual(q["review"]["non_group_version"], old["objects"]["q-one"]["version"])
        self.confirm_object("q-one")
        self.assertFalse(self.obj("q-one")["needs_group_recheck"])
        self.assertFalse(self.obj("h-one")["stale"])
        self.assertTrue(self.obj("r-one")["stale"])
        self.confirm_object("d1")
        self.confirm_object("r-one")
        self.assertFalse(self.obj("h-one")["stale"])
        self.assertTrue(self.store.item_state("one")["stale"])
        self.action("reveal")
        current = self.store.item_state("one")
        self.assertEqual(current["review"]["material_snapshots"][0], old["records"]["one"]["material_snapshots"][0])
        self.assertTrue(current["group_only_recheck"])
        self.assertEqual(current["task_reviews"]["hate"]["status"], "confirmed")
        with self.assertRaises(ReviewError):
            self.action("save_exposure", prior_exposure=dict.fromkeys(("gold", "resources", "results", "ai"), "not_seen"))
        final = self.assessment()
        final.update(group=["others"], group_original_status="policy_changed")
        self.action("save_assessment", assessment=final)
        interim = self.store.snapshot(self.store.bootstrap()["revision"])
        self.assertEqual(interim["case_eligibility"]["one"]["reference_tasks"], ["hate"])
        self.action("confirm", assessment=final)
        exported = self.store.snapshot(self.store.bootstrap()["revision"])
        tasks = exported["case_eligibility"]["one"]
        self.assertEqual(tasks["reference_tasks"], ["hate", "group"])
        self.assertEqual(tasks["tasks"]["hate"]["policy"], policy_ref(old["policy"]))
        self.assertEqual(tasks["tasks"]["group"]["policy"]["version"], "test-policy/v2")
        self.assertEqual(tasks["tasks"]["hate"]["confirmed_at"], old["records"]["one"]["confirmed_at"])
        rows = list(csv.DictReader(io.StringIO(self.store.export_csv(exported).decode("utf-8-sig"))))
        case = next(row for row in rows if row["case_id"] == "one")
        self.assertEqual(case["hate_policy_version"], "test-policy/v1")
        self.assertEqual(case["group_policy_version"], "test-policy/v2")
        self.assertEqual(case["group_reference_eligible"], "True")
        self.assertEqual(EvidenceReviewStore(bundle_path=self.bundle, session_path=self.session, reviewer_id="automated-deployment-test", policy_path=self.policy_path).bootstrap()["status"]["confirmed_count"], 1)

    def test_group_only_lock_requires_explicit_reopen_for_shared_fields(self) -> None:
        self.confirmed_case()
        self.migrate()
        q = self.obj("q-one")
        before = self.session.read_bytes()
        for field, value in (("hate", "hate"), ("note", "different shared basis"), ("stance", "opposes")):
            values = copy.deepcopy(q["effective_values"])
            values[field] = value
            with self.assertRaises(ReviewError):
                self.action("save_object", object_id="q-one", object_version=q["review"]["version"], values=values)
            self.assertEqual(self.session.read_bytes(), before)
        self.confirm_current_group_materials()
        self.action("reveal")
        self.assertEqual(self.store.item_state("one")["review"]["status"], "draft")
        final = self.assessment()
        final["note"] = "Shared rationale changed after explicit reopening"
        with self.assertRaises(ReviewError):
            self.action("save_assessment", assessment=final)
        self.action("reopen", reason="new_evidence")
        self.assertFalse(self.store.item_state("one")["group_only_recheck"])
        self.action("save_assessment", assessment=final)
        self.assertEqual(self.store.snapshot(self.store.bootstrap()["revision"])["case_eligibility"]["one"]["reference_tasks"], [])

    def test_new_material_snapshot_cannot_restore_a_stale_hate_confirmation(self) -> None:
        self.confirmed_case()
        self.migrate()
        q = self.obj("q-one")
        self.action("reopen_object", object_id="q-one", object_version=q["review"]["version"], reason="new_evidence")
        values = copy.deepcopy(self.obj("q-one")["effective_values"])
        values.update(hate="hate", hate_reason="agreed_attack")
        self.confirm_object("q-one", values=values)
        self.assertTrue(self.obj("h-one")["stale"])
        stale_export = self.store.snapshot(self.store.bootstrap()["revision"])
        self.assertTrue(stale_export["object_staleness"]["h-one"])
        exported_rows = list(csv.DictReader(io.StringIO(self.store.export_csv(stale_export).decode("utf-8-sig"))))
        self.assertEqual(next(row for row in exported_rows if row["object_id"] == "h-one")["stale"], "True")
        self.confirm_object("d1")
        self.confirm_object("r-one")
        self.confirm_object("h-one")
        before = self.store.snapshot(self.store.bootstrap()["revision"])
        self.assertFalse(before["case_eligibility"]["one"]["tasks"]["hate"]["reference_eligible"])
        self.action("reveal")
        case = self.store.item_state("one")
        self.assertFalse(case["group_only_recheck"])
        self.assertEqual(case["task_reviews"]["hate"]["status"], "draft")
        exported = self.store.snapshot(self.store.bootstrap()["revision"])
        self.assertFalse(exported["case_eligibility"]["one"]["tasks"]["hate"]["reference_eligible"])
        final = self.assessment()
        final.update(hate="hate", hate_original_status="suspected_error")
        self.action("confirm", assessment=final)
        self.assertEqual(self.store.snapshot(self.store.bootstrap()["revision"])["case_eligibility"]["one"]["reference_tasks"], ["hate", "group"])

    def test_policy_changed_is_group_only_resolved_and_requires_active_amendment(self) -> None:
        final = self.assessment()
        final.update(group=["others"], group_original_status="policy_changed")
        with self.assertRaises(ReviewError):
            normalize_assessment(final, self.assessment(), required=True)
        self.assertEqual(normalize_assessment(final, self.assessment(), required=True, allow_policy_changed=True)["group"], ["others"])
        for changed in ({"group": None}, {"hate_original_status": "policy_changed"}, {"group_original_status": "policy_ambiguous"}):
            with self.assertRaises(ReviewError):
                normalize_assessment({**final, **changed}, self.assessment(), required=True, allow_policy_changed=True)
        demo = self.obj("d1")
        values = {**demo["effective_values"], "group": ["others"], "group_reason": "identity_target", "group_original_status": "policy_changed"}
        with self.assertRaises(ReviewError):
            normalize_values("demo", values, demo["source"], required=True)
        self.assertEqual(normalize_values("demo", values, demo["source"], required=True, allow_policy_changed=True)["group_original_status"], "policy_changed")

    def test_invalid_policy_text_parent_impact_and_missing_relation_are_rejected(self) -> None:
        path = self.policy_fixture()
        baseline = read_json(path)
        changes = [lambda p: p.update(document_text="tampered"),
                   lambda p: p.update(parent_policy={"version": "wrong", "sha256": "f" * 64}),
                   lambda p: p["impact"]["object_ids"].append("def1"),
                   lambda p: p["impact"]["object_ids"].remove("r-two"),
                   lambda p: p["impact"]["case_ids"].remove("two")]
        before = self.session.read_bytes()
        for change in changes:
            policy = copy.deepcopy(baseline)
            change(policy)
            write_json(path, policy)
            with self.assertRaises(ReviewError):
                migrate_policy_session(bundle_path=self.bundle, policy_path=path, session_path=self.session,
                                       reviewer_id="automated-deployment-test", expected_revision=self.store.bootstrap()["revision"])
            self.assertEqual(self.session.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
