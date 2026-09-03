from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from build_lex.annotated_lexicon_repair import (
    LexiconRepairConflict,
    LexiconRepairError,
    SpanGoldReviewStore,
    build_span_gold_frame,
    finalize_span_gold,
    read_json,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PILOT_ROOT = REPOSITORY_ROOT / "exps/causal_context/stage1_exploratory_qwen3_ld_v0"


class AnnotatedLexiconRepairTests(unittest.TestCase):
    def build_store(self, root: Path) -> tuple[dict, SpanGoldReviewStore]:
        frame_path = root / "frame.json"
        frame = build_span_gold_frame(
            candidate_frame_path=PILOT_ROOT / "candidate_frame.json",
            audit_session_path=PILOT_ROOT / "audit/input_quality_session.json",
            output_path=frame_path,
        )
        store = SpanGoldReviewStore(
            frame_path=frame_path,
            session_path=root / "session.json",
            reviewer_id="fixture-reviewer",
        )
        return frame, store

    def test_builds_closed_39_item_frame_with_known_omissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            frame, _store = self.build_store(Path(temporary))

        self.assertEqual(
            frame["manifest"]["cohort_counts"],
            {
                "accepted_control": 3,
                "approved_adjudication": 2,
                "repair_target": 34,
            },
        )
        by_source = {row["source_item_id"]: row for row in frame["items"]}
        omission_surfaces = {
            row["surface"]
            for row in by_source["blind-564b9e2044612908"]["candidates"]
            if "known_omission" in row["source_types"]
        }
        self.assertEqual(omission_surfaces, {"畒勾", "蝲蝲蛄"})
        duplicate_base = [
            row
            for row in by_source["blind-f97a87fe31c5e383"]["candidates"]
            if row["surface"] == "基"
        ]
        self.assertEqual(len(duplicate_base), 1)
        self.assertEqual(len(duplicate_base[0]["source_hits"]), 2)

    def test_review_save_is_cas_bound_and_reopen_records_amendment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _frame, store = self.build_store(Path(temporary))
            bootstrap = store.bootstrap()
            item_id = bootstrap["items"][0]["item_id"]
            state = store.item_state(item_id)
            decision = {
                "candidate_actions": {
                    candidate_id: "drop"
                    for candidate_id in state["decision"]["candidate_actions"]
                },
                "additional_spans": [],
                "notes": "fixture",
            }
            saved = store.save(
                expected_revision=bootstrap["revision"],
                item_id=item_id,
                decision=decision,
                confirm=True,
            )
            self.assertEqual(saved["decision"]["status"], "confirmed")
            with self.assertRaises(LexiconRepairConflict):
                store.save(
                    expected_revision=bootstrap["revision"],
                    item_id=item_id,
                    decision=decision,
                    confirm=False,
                )
            reopened = store.reopen(
                expected_revision=saved["revision"],
                item_id=item_id,
                reason="fixture correction",
            )
            self.assertEqual(reopened["decision"]["status"], "draft")
            self.assertEqual(reopened["status"]["amendment_count"], 1)

    def test_additional_span_uses_unicode_codepoint_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _frame, store = self.build_store(Path(temporary))
            state = next(
                store.item_state(row["item_id"])
                for row in store.bootstrap()["items"]
                if "褐兰州" in row["surfaces"]
            )
            item = state["item"]
            actions = {
                candidate_id: "drop"
                for candidate_id in state["decision"]["candidate_actions"]
            }
            start = item["query_content"].index("哪里")
            with self.assertRaisesRegex(ValueError, "surface"):
                store.save(
                    expected_revision=state["revision"],
                    item_id=item["item_id"],
                    decision={
                        "candidate_actions": actions,
                        "additional_spans": [
                            {"start": start, "end": start + 2, "surface": "错误", "reason": "fixture"}
                        ],
                        "notes": "",
                    },
                    confirm=True,
                )

    def test_finalize_requires_all_items_and_locks_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame, store = self.build_store(root)
            with self.assertRaisesRegex(LexiconRepairError, "every"):
                finalize_span_gold(
                    frame_path=store.frame_path,
                    session_path=store.session_path,
                    output_path=root / "reference.json",
                )
            revision = store.bootstrap()["revision"]
            for item in frame["items"]:
                item_id = item["item_id"]
                decision = {
                    "candidate_actions": {
                        candidate["candidate_id"]: "drop" for candidate in item["candidates"]
                    },
                    "additional_spans": [],
                    "notes": "fixture",
                }
                saved = store.save(
                    expected_revision=revision,
                    item_id=item_id,
                    decision=decision,
                    confirm=True,
                )
                revision = saved["revision"]
            output = root / "reference.json"
            reference = finalize_span_gold(
                frame_path=store.frame_path,
                session_path=store.session_path,
                output_path=output,
            )
            self.assertEqual(reference["item_count"], 39)
            self.assertEqual(read_json(output)["reference_id"], reference["reference_id"])
            with self.assertRaisesRegex(ValueError, "finalized"):
                store.reopen(
                    expected_revision=store.bootstrap()["revision"],
                    item_id=frame["items"][0]["item_id"],
                    reason="too late",
                )


if __name__ == "__main__":
    unittest.main()
