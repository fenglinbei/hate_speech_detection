from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_lex.terminology_candidate_generators_v2 import (  # noqa: E402
    CandidateGeneratorError,
    generate_g3_observations,
)
from build_lex.terminology_g3_development_run import (  # noqa: E402
    FAMILY_ORDER,
    G3DevelopmentRunError,
    PUBLIC_FRAME_ARTIFACT_KIND,
    REFERENCE_ARTIFACT_KIND,
    _build_from_validated_inputs,
    _derive_completion_receipt,
    _validate_public_frame_payload,
    _validate_run_payload,
)
from build_lex.terminology_candidate_successor_run import (  # noqa: E402
    validate_g3_run_ref_for_planning as validate_runner_projection,
)
from data.training_artifacts import (  # noqa: E402
    canonical_sha256,
    finalize_target_atomic,
    load_json,
    new_staging_directory,
    write_canonical_json,
)
from scripts.stage1 import wp3_g3_dev_run as run_cli  # noqa: E402


PROFILE_PATH = REPOSITORY_ROOT / "config/stage1/wp3_g3_profile_full_v2.json"


def fake_romanizer(text: str) -> list[str]:
    return ["x" for _ in text]


def fixture_reference() -> dict:
    return {
        "schema_version": "wp3-g3-form-reference/v1",
        "reference_role": "form-only-label-free-non-lexicon",
        "scope": "development-only",
        "scientific_eligible": False,
        "sealed": False,
        "rows": [
            {
                "canonical": "和谐",
                "pinyin": ["he", "xie"],
                "initials": "hx",
                "phonetic_scan_enabled": True,
                "variants": [
                    {
                        "surface": "河蟹",
                        "family": "phonetic_variant",
                        "evidence_ids": ["g3ev-public-a"],
                    }
                ],
            }
        ],
    }


def fixture_cases() -> list[dict[str, str]]:
    return [
        {"case_id": "CASE-001", "blind_alias": "公开-001", "content": "河蟹 A股😀"},
        {"case_id": "CASE-002", "blind_alias": "公开-002", "content": "全角ａｂ"},
    ]


def dependency(kind: str, artifact_id: str, fill: str) -> dict:
    return {
        "schema_version": "stage1-dependency-ref/v1",
        "artifact_kind": kind,
        "artifact_id": artifact_id,
        "payload_manifest_sha256": fill * 64,
        "logical_repo_path": f"test-fixtures/{artifact_id}",
    }


class G3DevelopmentRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = load_json(PROFILE_PATH)
        self.reference = fixture_reference()
        self.cases = fixture_cases()
        self.frame_dependency = dependency(
            PUBLIC_FRAME_ARTIFACT_KIND, "wp3devframe-" + "a" * 64, "b"
        )
        self.reference_dependency = dependency(
            REFERENCE_ARTIFACT_KIND, "wp3g3formref-" + "c" * 64, "d"
        )
        self.reference_id = self.reference_dependency["artifact_id"]
        self.reference_sha256 = canonical_sha256(self.reference)

    def build(self, output: Path, *, generator=generate_g3_observations) -> dict:
        return _build_from_validated_inputs(
            workspace_root=REPOSITORY_ROOT,
            cases=self.cases,
            frame_dependency=self.frame_dependency,
            reference=self.reference,
            reference_dependency=self.reference_dependency,
            reference_id=self.reference_id,
            reference_sha256=self.reference_sha256,
            profile=self.profile,
            romanizer=fake_romanizer,
            output_root=output,
            write_ref=None,
            expected_case_count=2,
            generator=generator,
        )

    def validate(self, result: dict, *, generator=generate_g3_observations) -> dict:
        return _validate_run_payload(
            Path(result["target"]),
            cases=self.cases,
            reference=self.reference,
            profile=self.profile,
            romanizer=fake_romanizer,
            frame_dependency=self.frame_dependency,
            reference_dependency=self.reference_dependency,
            reference_id=self.reference_id,
            reference_sha256=self.reference_sha256,
            expected_case_count=2,
            generator=generator,
            require_current_implementation=True,
        )

    def test_small_public_frame_run_is_complete_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = self.build(Path(temporary))
            second = self.build(Path(temporary))
            self.assertTrue(first["complete"])
            self.assertEqual(first["artifact_id"], second["artifact_id"])
            self.assertEqual(
                first["payload_manifest_sha256"], second["payload_manifest_sha256"]
            )
            receipt = first["completion_receipt"]
            self.assertEqual(receipt["enabled_families"], list(FAMILY_ORDER))
            self.assertEqual(receipt["successful_record_count"], 2)
            self.assertEqual(receipt["terminal_failure_count"], 0)
            self.assertEqual(
                receipt["offset_replay_count"], receipt["offset_occurrence_count"]
            )
            for family in FAMILY_ORDER:
                self.assertEqual(
                    receipt["families"][family]["record_completion_count"], 2
                )
                self.assertEqual(receipt["families"][family]["record_error_count"], 0)

    def test_424_synthetic_artifact_matches_successor_runner_projection(self) -> None:
        cases = [
            {
                "case_id": f"SYN-{index:03d}",
                "blind_alias": f"合成-{index:03d}",
                "content": f"合成文本{index}",
            }
            for index in range(424)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            run_ref = Path(temporary) / "g3_run_ref.json"
            result = _build_from_validated_inputs(
                workspace_root=REPOSITORY_ROOT,
                cases=cases,
                frame_dependency=self.frame_dependency,
                reference=self.reference,
                reference_dependency=self.reference_dependency,
                reference_id=self.reference_id,
                reference_sha256=self.reference_sha256,
                profile=self.profile,
                romanizer=fake_romanizer,
                output_root=Path(temporary) / "runs",
                write_ref=run_ref,
                expected_case_count=424,
            )
            manifest = result["manifest"]
            replayed_projection = {
                "artifact_kind": manifest["artifact_kind"],
                "artifact_id": result["artifact_id"],
                "payload_manifest_sha256": result["payload_manifest_sha256"],
                "profile_id": manifest["profile_id"],
                "profile_sha256": manifest["profile_sha256"],
                "reference_id": manifest["reference_id"],
                "reference_sha256": manifest["reference_sha256"],
                "romanizer_backend_id": manifest["romanizer_backend_id"],
                "case_count": manifest["case_count"],
                "enabled_families": manifest["enabled_families"],
                "terminal_failure_count": manifest["terminal_failure_count"],
                "offset_replay_count": manifest["offset_replay_count"],
                "offset_occurrence_count": manifest["offset_occurrence_count"],
                "offset_replay_rate": manifest["offset_replay_rate"],
                "complete": result["completion_receipt"]["complete"],
            }
            with patch(
                "build_lex.terminology_g3_development_run."
                "validate_g3_run_ref_for_planning",
                return_value=replayed_projection,
            ) as strong_validator:
                projection = validate_runner_projection(run_ref)
            strong_validator.assert_called_once_with(
                run_ref,
                workspace_root=REPOSITORY_ROOT,
            )
            self.assertTrue(result["complete"])
            self.assertEqual(projection["artifact_id"], result["artifact_id"])
            self.assertEqual(projection["case_count"], 424)
            self.assertEqual(projection["enabled_families"], list(FAMILY_ORDER))

    def test_record_failure_is_published_as_incomplete_not_caller_overridden(self) -> None:
        def failing_generator(**kwargs):
            if kwargs["record_id"] == "CASE-002":
                raise CandidateGeneratorError("synthetic deterministic failure")
            return generate_g3_observations(**kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            result = self.build(Path(temporary), generator=failing_generator)
            self.assertFalse(result["complete"])
            receipt = result["completion_receipt"]
            self.assertEqual(receipt["successful_record_count"], 1)
            self.assertEqual(receipt["terminal_failure_count"], 1)
            self.assertTrue(
                all(
                    receipt["families"][family]["record_error_count"] == 1
                    for family in FAMILY_ORDER
                )
            )

    def test_disabled_family_fails_before_execution(self) -> None:
        profile = copy.deepcopy(self.profile)
        profile["families"]["emoji"] = False
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(G3DevelopmentRunError):
                _build_from_validated_inputs(
                    workspace_root=REPOSITORY_ROOT,
                    cases=self.cases,
                    frame_dependency=self.frame_dependency,
                    reference=self.reference,
                    reference_dependency=self.reference_dependency,
                    reference_id=self.reference_id,
                    reference_sha256=self.reference_sha256,
                    profile=profile,
                    romanizer=fake_romanizer,
                    output_root=temporary,
                    write_ref=None,
                    expected_case_count=2,
                )

    def test_wrong_offset_or_occurrence_cannot_form_a_receipt(self) -> None:
        rows = generate_g3_observations(
            record_id=self.cases[0]["case_id"],
            content=self.cases[0]["content"],
            reference_document=self.reference,
            romanizer=fake_romanizer,
            profile=self.profile,
            max_surface_chars=self.profile["limits"]["max_surface_chars"],
            mixed_script_max_chars=self.profile["limits"]["mixed_script_max_chars"],
            phonetic_max_distance=0,
        )
        self.assertTrue(rows)
        tampered = copy.deepcopy(rows[0])
        tampered["occurrence_ordinal"] += 1
        with self.assertRaises(G3DevelopmentRunError):
            _derive_completion_receipt(
                cases=self.cases,
                observations=[tampered],
                failures=[],
                frame_dependency=self.frame_dependency,
                reference_dependency=self.reference_dependency,
                reference_id=self.reference_id,
                reference_sha256=self.reference_sha256,
                profile=self.profile,
                expected_case_count=2,
            )

    def test_payload_tamper_fails_before_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.build(Path(temporary))
            observations = Path(result["target"]) / "observations.jsonl"
            observations.write_bytes(observations.read_bytes() + b"\n")
            with self.assertRaises(G3DevelopmentRunError):
                self.validate(result)

    def test_public_frame_fixture_is_validated_without_private_or_gold_files(self) -> None:
        cases = self.cases
        proposals = [
            {"case_id": case["case_id"], "proposals": []} for case in cases
        ]
        base = {
            "schema_version": "wp3-s21-development-frame/v1",
            "artifact_kind": PUBLIC_FRAME_ARTIFACT_KIND,
            "case_count": 2,
            "cases_sha256": canonical_sha256(cases),
            "diagnostic_proposals_sha256": canonical_sha256(proposals),
            "source_policy": "fit-content-only-no-task-fields/v1",
            "task_label_contribution_count": 0,
            "calibration_dev_test_intersection_count": 0,
            "scientific_eligible": False,
            "sealed": False,
            "human_visible_phase_a_fields": ["case_id", "blind_alias", "content"],
            "proposal_reveal_policy": "raw-lock-before-unattributed-proposal-reveal/v1",
            "legacy_proposals_are_diagnostic_only": True,
        }
        identity = {
            key: value
            for key, value in base.items()
            if key
            not in {
                "human_visible_phase_a_fields",
                "proposal_reveal_policy",
                "legacy_proposals_are_diagnostic_only",
            }
        }
        frame_id = "wp3devframe-" + canonical_sha256(identity)
        manifest = {**base, "frame_id": frame_id}
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            target = parent / frame_id
            staging = new_staging_directory(parent, frame_id)
            write_canonical_json(staging / "manifest.json", manifest)
            write_canonical_json(staging / "cases.json", cases)
            write_canonical_json(staging / "diagnostic_proposals.json", proposals)
            finalize_target_atomic(staging, target)
            validated = _validate_public_frame_payload(target, expected_case_count=2)
            self.assertEqual(validated["cases"], cases)
            self.assertEqual(
                {path.name for path in target.iterdir()},
                {
                    "manifest.json",
                    "cases.json",
                    "diagnostic_proposals.json",
                    "payload_manifest.json",
                },
            )

            leaked_base = {
                **base,
                "public_metadata": {"nested": {"gold": "forbidden"}},
            }
            leaked_identity = {
                key: value
                for key, value in leaked_base.items()
                if key
                not in {
                    "human_visible_phase_a_fields",
                    "proposal_reveal_policy",
                    "legacy_proposals_are_diagnostic_only",
                }
            }
            leaked_frame_id = "wp3devframe-" + canonical_sha256(leaked_identity)
            leaked_manifest = {**leaked_base, "frame_id": leaked_frame_id}
            leaked_target = parent / leaked_frame_id
            leaked_staging = new_staging_directory(parent, leaked_frame_id)
            write_canonical_json(leaked_staging / "manifest.json", leaked_manifest)
            write_canonical_json(leaked_staging / "cases.json", cases)
            write_canonical_json(
                leaked_staging / "diagnostic_proposals.json", proposals
            )
            finalize_target_atomic(leaked_staging, leaked_target)
            with self.assertRaisesRegex(
                G3DevelopmentRunError,
                r"manifest\.public_metadata\.nested\.gold",
            ):
                _validate_public_frame_payload(
                    leaked_target, expected_case_count=2
                )

    def test_cli_has_no_raw_gold_or_fit_input_surface(self) -> None:
        parser = run_cli.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["run-g3-dev", "--raw-gold", "/tmp/forbidden"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["run-g3-dev", "--fit", "/tmp/forbidden"])


if __name__ == "__main__":
    unittest.main()
