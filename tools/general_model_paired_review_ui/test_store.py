"""Exercise phase boundaries and persistence with an isolated synthetic session."""

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

from build_lex.annotated_lexicon_repair import file_sha256
from tools.general_model_paired_review_ui.store import (
    CONDITIONS, PairedReviewStore, ReviewConflict, ReviewError, empty_notes,
)


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def make_fixture(root: Path) -> Path:
    data = root / "experiment/results/run"
    index = []
    for position, key in enumerate(("one", "two")):
        prefix = str(position).zfill(2) + "-" + key
        index.append({
            "query_id": key, "focus_task": "hate",
            "resources_card": "cases/cards/" + prefix + "-1-resources.md",
        })
        task_row = {
            "prediction": {"labels": ["hate"], "top_score_gap": 0.5, "within_two_epsilon": False, "tied_top_count": 1},
            "correct": True, "readouts": {"answer_sum/gold/best_nongold_margin": 0.5},
            "score_mode_sensitive": False, "context": {"prompt_tokens": 20},
        }
        card = {
            "selection": {"query_id": key, "focus_task": "hate", "split": "discovery", "primary_bucket": "SECRET_BUCKET", "candidate_labels": ["SECRET_BUCKET"]},
            "query": {"id": key, "content": "测试查询 " + key, "projection": {
                "hate": "hate", "group": ["others"], "extraction": [{"target": "SECRET_GOLD_TARGET"}],
            }},
            "profile": {
                "resources": {"lq_ids": [], "demo_ids": ["demo"], "demo_match_ids": {"demo": []}},
                "conditions": {condition: {"hate": task_row, "group": task_row} for condition in CONDITIONS},
                "core_mask": {"hate": "1111", "group": "1111"},
                "six_condition_mask": {"hate": "111111", "group": "111111"},
            },
            "lexicon_entries": [],
            "demonstrations": [{"id": "demo", "content": "可读的固定示例", "projection": {"hate": "non-hate", "group": []}}],
            "contexts": [{"condition": "CD", "task": "hate", "prompt_text": "SECRET_FULL_PROMPT", "prompt_tokens": 20}],
        }
        write(data / ("cases/card_data/" + prefix + ".json"), card)
    write(data / "cases/cards_index.json", index)
    write(data / "cases/initial_review_batch.json", index[:1])
    ai = data / "cases/ai_review.csv"
    ai.write_text("query_id,candidate_explanation\none,SECRET_AI_EXPLANATION\n", encoding="utf-8")
    hashes = {path.relative_to(data).as_posix(): file_sha256(path) for path in data.rglob("*") if path.is_file()}
    write(data / "manifest.json", {"status": "complete", "identity": "test-only-paired-frame", "artifacts": hashes})
    return data


def resource_notes() -> dict:
    notes = empty_notes()
    for key in ("ambiguity_stance", "definition_fit", "category_relation", "demo_correspondence"):
        notes["resources"][key] = "测试观察：" + key
    return notes


def assessment_notes() -> dict:
    notes = resource_notes()
    notes["assessment"].update({
        "gold_verdict": "agree", "stage2_candidate_explanation": "测试候选解释",
        "alternative_explanation": "测试替代解释", "falsifiable_followup": "同位置等长的测试对照",
        "disposition": "input_control",
    })
    return notes


class ReviewStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="paired-human-review-test-")
        self.root = Path(self.temporary.name)
        self.data = make_fixture(self.root)
        self.session = self.root / "state/session.json"
        self.store = self.new_store()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def new_store(self, **kwargs) -> PairedReviewStore:
        return PairedReviewStore(data_dir=self.data, session_path=self.session, reviewer_id=kwargs.get("reviewer", "automated-test"))

    def action(self, action: str, notes: dict | None = None, **kwargs) -> dict:
        return self.store.mutate(
            expected_revision=self.store.bootstrap()["revision"], item_id="one",
            action=action, notes=notes, **kwargs,
        )

    def test_resource_only_api_and_scope(self) -> None:
        item = self.store.item_state("one")
        wire = json.dumps(item)
        for hidden in ("SECRET_GOLD_TARGET", "SECRET_FULL_PROMPT", "SECRET_AI_EXPLANATION", "SECRET_BUCKET"):
            self.assertNotIn(hidden, wire)
        self.assertEqual(item["demonstrations"][0]["answer"], "non-hate")
        self.assertIsNone(item["trajectory"])
        self.assertEqual(self.store.bootstrap()["initial_ids"], ["one"])
        with self.assertRaises(ReviewError):
            self.store.item_state("reserve")
        with self.assertRaises(ReviewError):
            self.store.prompt("one", "CD", "hate")

    def test_invalid_reveal_is_atomic_and_drafts_do_not_unlock(self) -> None:
        before = self.session.read_bytes()
        with self.assertRaises(ReviewError):
            self.action("reveal", empty_notes())
        self.assertEqual(self.session.read_bytes(), before)
        self.action("save", resource_notes())
        self.assertIsNone(self.store.item_state("one")["trajectory"])
        self.assertEqual(self.new_store().item_state("one")["review"]["resources"], resource_notes()["resources"])

    def test_reveal_freezes_first_notes_and_ai_requires_human_assessment(self) -> None:
        with self.assertRaises(ReviewError):
            self.action("confirm", assessment_notes())
        self.action("reveal", resource_notes())
        item = self.store.item_state("one")
        self.assertEqual(item["trajectory"]["gold"]["extraction"][0]["target"], "SECRET_GOLD_TARGET")
        self.assertEqual(self.store.prompt("one", "CD", "hate")["prompt_text"], "SECRET_FULL_PROMPT")
        changed = resource_notes()
        changed["resources"]["definition_fit"] = "改写最初判断"
        with self.assertRaises(ReviewError):
            self.action("save", changed)
        with self.assertRaises(ReviewError):
            self.action("reveal_ai", resource_notes())
        self.assertIsNone(self.store.item_state("one")["ai_review"])
        self.action("reveal_ai", assessment_notes())
        row = self.store.item_state("one")
        self.assertEqual(row["ai_review"]["candidate_explanation"], "SECRET_AI_EXPLANATION")
        self.assertEqual(row["review"]["pre_ai_assessment"], assessment_notes()["assessment"])
        self.assertEqual(row["review"]["status"], "draft")
        self.assertEqual(self.store.bootstrap()["status"]["confirmed_count"], 0)

    def test_confirm_reopen_and_exact_export_preserve_input_and_history(self) -> None:
        hashes = {p: file_sha256(p) for p in self.data.rglob("*") if p.is_file()}
        self.action("reveal", resource_notes())
        notes = assessment_notes()
        notes["assessment"]["gold_dispute"] = "=SENSITIVE_FORMULA"
        self.action("confirm", notes)
        row = self.store.item_state("one")["review"]
        with self.assertRaises(ReviewConflict):
            self.action("save", notes)
        snapshot = self.store.snapshot(self.store.bootstrap()["revision"])
        self.assertEqual(snapshot["status"]["initial_confirmed_count"], 1)
        exported = list(csv.DictReader(io.StringIO(self.store.export_csv(snapshot).decode("utf-8-sig"))))
        self.assertEqual(exported[0]["review_status"], "confirmed")
        self.assertEqual(exported[0]["gold_dispute"], "'=SENSITIVE_FORMULA")
        self.assertEqual(snapshot["session"]["records"]["one"]["assessment"]["gold_dispute"], "=SENSITIVE_FORMULA")
        self.assertEqual(exported[1]["primary_bucket"], "")
        self.action("reopen", reason="补充测试核验")
        reopened = self.store.snapshot(self.store.bootstrap()["revision"])
        self.assertEqual(reopened["session"]["events"][-1]["previous_record"], row)
        self.assertEqual(reopened["session"]["records"]["one"]["resources_sha256"], row["resources_sha256"])
        self.assertEqual({p: file_sha256(p) for p in hashes}, hashes)

    def test_stale_tab_is_rejected_without_overwriting_saved_notes(self) -> None:
        stale = self.store.bootstrap()["revision"]
        notes = resource_notes()
        self.action("save", notes)
        other = self.new_store()
        with self.assertRaises(ReviewConflict):
            other.mutate(expected_revision=stale, item_id="one", action="save", notes=empty_notes())
        self.assertEqual(other.item_state("one")["review"]["resources"], notes["resources"])

    def test_gold_dispute_and_deferred_use_require_reasons(self) -> None:
        self.action("reveal", resource_notes())
        notes = assessment_notes()
        notes["assessment"].update(gold_verdict="uncertain", disposition="verify_first", falsifiable_followup="")
        with self.assertRaises(ReviewError):
            self.action("confirm", notes)
        notes["assessment"]["gold_dispute"] = "上下文不足"
        with self.assertRaises(ReviewError):
            self.action("confirm", notes)
        notes["assessment"]["patching_defer_reason"] = "先核对文本上下文"
        self.action("confirm", notes)
        self.assertEqual(self.store.bootstrap()["status"]["input_control_count"], 0)

    def test_other_reviewer_tampering_and_frozen_output_paths_are_rejected(self) -> None:
        with self.assertRaises(ReviewError):
            self.new_store(reviewer="different-reviewer")
        with self.assertRaises(ReviewError):
            PairedReviewStore(data_dir=self.data, session_path=self.data / "reviews.json", reviewer_id="automated-test")
        changed = json.loads(self.session.read_text())
        changed["records"]["one"]["resources"]["definition_fit"] = "unaudited edit"
        write(self.session, changed)
        with self.assertRaises(ReviewError):
            self.new_store()

    def test_corrupted_case_cannot_be_used_with_existing_receipt(self) -> None:
        path = self.data / "cases/card_data/00-one.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaises(ReviewError):
            self.new_store()


if __name__ == "__main__":
    unittest.main()
