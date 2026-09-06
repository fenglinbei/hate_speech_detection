from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_lex.terminology_candidate_review import (  # noqa: E402
    CandidateReviewConflict,
    CandidateReviewError,
    _canonical_sha,
    _normalize_mentions,
    _read_session,
    _tree_sha256,
    _with_session_revision,
    build_development_report,
    create_review_session,
    finalize_development_gold,
    lock_raw_phase,
    reopen_review_item,
    save_raw_annotation,
    validate_development_frame,
    validate_development_gold,
    validate_development_report,
)
from data.training_artifacts import (  # noqa: E402
    TrainingArtifactError,
    load_json,
    resolve_locator_ref,
    validate_payload_manifest,
    write_canonical_json,
)


CONFIG_PATH = REPOSITORY_ROOT / "config/stage1/wp3_candidate_generators_v1.json"
FRAME_REF = (
    REPOSITORY_ROOT
    / "exps/causal_context/stage1_p0/wp3_candidate_generators_v2/refs/development_frame_ref.json"
)


@unittest.skipUnless(FRAME_REF.is_file(), "materialized S2.1 frame is unavailable")
class DevelopmentFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.frame_dir = resolve_locator_ref(
            FRAME_REF, "wp3-s21-development-frame"
        )
        cls.frame = validate_development_frame(
            cls.frame_dir,
            workspace_root=REPOSITORY_ROOT,
            generator_config_path=CONFIG_PATH,
        )

    def test_frozen_counts_fit_replay_and_human_projection(self) -> None:
        counts = self.frame["manifest"]["historical_sources"]["selection_counts"]
        self.assertEqual(
            counts,
            {
                "a1_unique": 200,
                "dual_model_unique": 240,
                "intersection": 16,
                "union": 424,
                "dual_model_repeat_pages_excluded": 48,
            },
        )
        self.assertEqual(len(self.frame["cases"]), 424)
        self.assertTrue(
            all(
                set(case) == {"case_id", "blind_alias", "content"}
                for case in self.frame["cases"]
            )
        )
        wire = json.dumps(self.frame["cases"], ensure_ascii=False)
        for forbidden in (
            '"label"',
            '"category"',
            '"targeted_group"',
            '"hateful"',
            '"record_id"',
        ):
            self.assertNotIn(forbidden, wire)

    def test_historical_packages_remain_byte_identical_to_bound_trees(self) -> None:
        source = self.frame["manifest"]["historical_sources"]
        review_root = REPOSITORY_ROOT / "exps/causal_context/stage1_p0/review_packages"
        self.assertEqual(
            _tree_sha256(review_root / "termspan-review-4ed4230bd227ab2744be"),
            source["a1"]["tree_sha256"],
        )
        self.assertEqual(
            _tree_sha256(review_root / "dmspan-review-30583732ec1adb2f"),
            source["dual_model"]["tree_sha256"],
        )

    def test_payload_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / self.frame_dir.name
            shutil.copytree(self.frame_dir, copied)
            cases_path = copied / "cases.json"
            cases_path.write_bytes(cases_path.read_bytes() + b" ")
            with self.assertRaises(TrainingArtifactError):
                validate_payload_manifest(copied)

    def test_occurrence_nested_overlap_and_route_contract(self) -> None:
        rows = _normalize_mentions(
            "ababa",
            [
                {
                    "surface": "aba",
                    "occurrence_ordinal": 1,
                    "provisional_route": "A_candidate",
                    "reason_codes": ["stable_core_candidate"],
                    "notes": "",
                },
                {
                    "surface": "bab",
                    "occurrence_ordinal": 1,
                    "provisional_route": "C_candidate",
                    "reason_codes": ["context_required"],
                    "notes": "",
                },
                {
                    "surface": "aba",
                    "occurrence_ordinal": 2,
                    "provisional_route": "B_candidate",
                    "reason_codes": ["evidence_required"],
                    "notes": "",
                },
            ],
            require_route=True,
        )
        self.assertEqual([(row["start"], row["end"]) for row in rows], [(0, 3), (1, 4), (2, 5)])
        with self.assertRaisesRegex(CandidateReviewError, "duplicated"):
            _normalize_mentions(
                "ababa",
                [
                    {
                        "surface": "aba",
                        "occurrence_ordinal": 1,
                        "provisional_route": "A_candidate",
                        "reason_codes": ["stable_core_candidate"],
                        "notes": "",
                    },
                    {
                        "surface": "aba",
                        "occurrence_ordinal": 1,
                        "provisional_route": "B_candidate",
                        "reason_codes": ["context_required"],
                        "notes": "",
                    },
                ],
                require_route=True,
            )

    def test_raw_state_cas_lock_reopen_and_offset_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session_path = Path(directory) / "session.json"
            session = create_review_session(
                frame_dir=self.frame_dir,
                session_path=session_path,
                reviewer_id="reviewer-test",
                workspace_root=REPOSITORY_ROOT,
                generator_config_path=CONFIG_PATH,
            )
            with self.assertRaisesRegex(CandidateReviewError, "incomplete"):
                lock_raw_phase(
                    session_path=session_path,
                    expected_revision=session["revision"],
                )
            case = self.frame["cases"][0]
            surface = next(char for char in case["content"] if not char.isspace())
            annotation = {
                "needs_explanation": True,
                "mentions": [
                    {
                        "surface": surface,
                        "occurrence_ordinal": 1,
                        "provisional_route": "A_candidate",
                        "reason_codes": ["stable_core_candidate"],
                        "notes": "",
                    }
                ],
                "notes": "test",
            }
            saved = save_raw_annotation(
                frame_dir=self.frame_dir,
                session_path=session_path,
                case_id=case["case_id"],
                annotation=annotation,
                confirm=True,
                expected_revision=session["revision"],
            )
            mention = saved["raw_annotations"][case["case_id"]]["mentions"][0]
            self.assertEqual(case["content"][mention["start"] : mention["end"]], surface)
            with self.assertRaisesRegex(CandidateReviewError, "reopened"):
                save_raw_annotation(
                    frame_dir=self.frame_dir,
                    session_path=session_path,
                    case_id=case["case_id"],
                    annotation=annotation,
                    confirm=False,
                    expected_revision=saved["revision"],
                )
            with self.assertRaisesRegex(CandidateReviewError, "unknown"):
                save_raw_annotation(
                    frame_dir=self.frame_dir,
                    session_path=session_path,
                    case_id="S21-UNKNOWN",
                    annotation={"needs_explanation": False, "mentions": [], "notes": ""},
                    confirm=False,
                    expected_revision=saved["revision"],
                )
            with self.assertRaises(CandidateReviewConflict):
                save_raw_annotation(
                    frame_dir=self.frame_dir,
                    session_path=session_path,
                    case_id=self.frame["cases"][1]["case_id"],
                    annotation={"needs_explanation": False, "mentions": [], "notes": ""},
                    confirm=False,
                    expected_revision=session["revision"],
                )
            with self.assertRaisesRegex(CandidateReviewError, "fields"):
                save_raw_annotation(
                    frame_dir=self.frame_dir,
                    session_path=session_path,
                    case_id=self.frame["cases"][1]["case_id"],
                    annotation={
                        "needs_explanation": False,
                        "mentions": [],
                        "notes": "",
                        "start": 0,
                    },
                    confirm=False,
                    expected_revision=saved["revision"],
                )
            reopened = reopen_review_item(
                session_path=session_path,
                scope="raw",
                case_id=case["case_id"],
                proposal_id=None,
                reason="fix an occurrence",
                expected_revision=saved["revision"],
            )
            self.assertEqual(reopened["raw_annotations"][case["case_id"]]["status"], "draft")
            self.assertEqual(len(reopened["amendments"]), 1)

    def _complete_session(self, session_path: Path) -> dict:
        session = create_review_session(
            frame_dir=self.frame_dir,
            session_path=session_path,
            reviewer_id="reviewer-e2e",
            workspace_root=REPOSITORY_ROOT,
            generator_config_path=CONFIG_PATH,
        )
        for row in session["raw_annotations"].values():
            row["status"] = "confirmed"
        session["phase"] = "diagnostic"
        session["raw_lock"] = {
            "policy": "raw-lock-before-unattributed-proposal-reveal/v1",
            "raw_annotations_sha256": _canonical_sha(session["raw_annotations"]),
            "locked_at": "2026-08-29T00:00:00+00:00",
        }
        decisions = {}
        for proposal_row in self.frame["proposal_rows"]:
            case_id = proposal_row["case_id"]
            for proposal in proposal_row["proposals"]:
                key = f"{case_id}:{proposal['proposal_id']}"
                decisions[key] = {
                    "case_id": case_id,
                    "proposal_id": proposal["proposal_id"],
                    "status": "confirmed",
                    "action": "accept",
                    "result_mentions": [
                        {
                            "surface": proposal["surface"],
                            "occurrence_ordinal": proposal["occurrence_ordinal"],
                            "start": proposal["start"],
                            "end": proposal["end"],
                        }
                    ],
                    "reason_codes": ["stable_core_candidate"],
                    "notes": "",
                }
        session["diagnostic_decisions"] = decisions
        session["updated_at"] = "2026-08-29T00:00:00+00:00"
        completed = _with_session_revision(session)
        write_canonical_json(session_path, completed)
        return completed

    def test_end_to_end_finalize_report_validate_and_immutability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session_path = root / "session.json"
            self._complete_session(session_path)
            gold = finalize_development_gold(
                frame_dir=self.frame_dir,
                session_path=session_path,
                workspace_root=REPOSITORY_ROOT,
                generator_config_path=CONFIG_PATH,
                output_root=root / "gold",
                reviewer_id="reviewer-e2e",
            )
            self.assertEqual(
                validate_development_gold(gold["target"], frame_dir=self.frame_dir)["gold_id"],
                gold["gold_id"],
            )
            report = build_development_report(
                frame_dir=self.frame_dir,
                gold_dir=gold["target"],
                output_root=root / "reports",
            )
            validated = validate_development_report(
                report["target"],
                frame_dir=self.frame_dir,
                gold_dir=gold["target"],
            )
            self.assertEqual(validated["report_id"], report["report_id"])
            self.assertEqual(validated["report"]["scope"], "development-only")
            finalized = _read_session(session_path)
            with self.assertRaisesRegex(CandidateReviewError, "immutable"):
                save_raw_annotation(
                    frame_dir=self.frame_dir,
                    session_path=session_path,
                    case_id=self.frame["cases"][0]["case_id"],
                    annotation={"needs_explanation": False, "mentions": [], "notes": ""},
                    confirm=False,
                    expected_revision=finalized["revision"],
                )


if __name__ == "__main__":
    unittest.main()
