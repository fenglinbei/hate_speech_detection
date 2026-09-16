import copy
import csv
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.review.finalize_evidence_annotations import (
    ROOT, RUN, BUNDLE, POLICY, EvidenceReviewStore, apply, sha,
    prepare_finalized_session, read_json,
)
from tools.general_model_paired_review_ui.evidence_finalization import current_final
from build_lex.annotated_lexicon_repair import LexiconRepairError, _with_revision


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="hsd-final-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.session = self.root / "session.json"
        shutil.copyfile(ROOT / "exps/causal_context/general_model_ld_nolabel_paired_cases_v1/reviews/paired-cases-02/runtime/ai-v2-reannotation-parent-20260911.json", self.session)
        self.store = EvidenceReviewStore(bundle_path=BUNDLE, policy_path=POLICY,
                                         session_path=self.session, reviewer_id="liaozijie")
        self.before = self.session.read_bytes()
        self.artifact = RUN / "final-results-v1/final-result.json"

    def test_complete_resume_export_and_reopen(self):
        apply(self.store, self.artifact, self.root / "backup.json", sha(self.before))
        self.assertEqual((self.root / "backup.json").read_bytes(), self.before)
        store = EvidenceReviewStore(bundle_path=BUNDLE, policy_path=POLICY,
                                    session_path=self.session, reviewer_id="liaozijie")
        saved = self.session.read_bytes()
        result = apply(store, self.artifact, self.root / "backup.json", sha(self.before))
        self.assertTrue(result["already_applied"])
        self.assertEqual(saved, self.session.read_bytes())
        snapshot = store.snapshot(result["revision"])
        self.assertEqual(snapshot["status"]["confirmed_object_count"], 1072)
        self.assertEqual(snapshot["status"]["confirmed_count"], 3)
        self.assertFalse(any(snapshot["object_staleness"].values()))
        previous = json.loads(self.before)
        self.assertEqual(snapshot["session"]["records"], previous["records"])
        kept = [oid for oid, row in previous["objects"].items() if row["status"] == "confirmed"]
        self.assertEqual(len(kept), 118)
        self.assertTrue(all(snapshot["session"]["objects"][oid] == previous["objects"][oid] for oid in kept))
        rows = {r["object_id"]: r for r in csv.DictReader(io.StringIO(store.export_csv(snapshot).decode("utf-8-sig"))) if r["record_type"] == "object"}
        self.assertEqual(rows["demo:5998"]["attack_severity"], "1")
        self.assertEqual(json.loads(rows["demo:5998"]["group"]), ["Sexism", "others"])
        self.assertIn("感染HIV", rows["definition:lex-0182:sense:01"]["adopted_definition"])
        hits = [r for oid, r in snapshot["session"]["objects"].items() if oid.startswith("hit:") and r.get("final_annotation")]
        self.assertEqual(sum(r["values"]["source_fit"] == "uncertain" for r in hits), 5)
        key = next(key for key, case in store.cases.items() if "demo:5998" in case["object_ids"])
        store.mutate(expected_revision=result["revision"], item_id=key, action="reopen_object",
                     object_id="demo:5998", object_version=snapshot["session"]["objects"]["demo:5998"]["version"],
                     reason="isolated finalization reopening test")
        reopened = store._read()
        self.assertIsNone(current_final(reopened["objects"]["demo:5998"]))
        self.assertIsNotNone(reopened["objects"]["demo:5998"]["final_annotation"])

    def test_conflict_and_invalid_payload_never_write(self):
        with self.assertRaises(LexiconRepairError):
            apply(self.store, self.artifact, self.root / "backup.json", "0" * 64)
        self.assertEqual(self.before, self.session.read_bytes())
        self.assertFalse((self.root / "backup.json").exists())
        for change in ("scope", "source", "authorization"):
            artifact = read_json(self.artifact)
            if change == "scope":
                artifact["scope_ids"] = []
            elif change == "source":
                artifact["records"][0]["source_sha256"] = "0" * 64
            else:
                artifact["authorization_text"] = ""
            with self.assertRaises(LexiconRepairError):
                prepare_finalized_session(self.store, json.loads(self.before), artifact, "1" * 64)
        self.assertEqual(self.before, self.session.read_bytes())

    def test_receipt_tampering_rejected(self):
        artifact = read_json(self.artifact)
        candidate = prepare_finalized_session(self.store, json.loads(self.before), artifact, sha(self.artifact.read_bytes()))
        candidate["objects"]["demo:5998"]["final_annotation"]["values"]["attack_severity"] = 4
        with self.assertRaises(LexiconRepairError):
            self.store._validate_session(_with_revision(candidate))


if __name__ == "__main__":
    unittest.main()
