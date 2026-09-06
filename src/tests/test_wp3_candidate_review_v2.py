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

from build_lex.terminology_candidate_development_v2 import (  # noqa: E402
    CandidateDevelopmentV2Error,
    LEGACY_PHASE_B_STATUS,
    _validate_locked_raw_payload,
    build_locked_raw_gold,
    load_s21b_protocol,
    validate_locked_raw_gold,
    validate_locked_raw_session,
)
from build_lex.terminology_candidate_generators import resolve_exact_span  # noqa: E402
from build_lex.terminology_candidate_review import (  # noqa: E402
    _canonical_sha,
    _with_session_revision,
    create_review_session,
    validate_development_frame,
)
from data.training_artifacts import (  # noqa: E402
    canonical_sha256,
    resolve_locator_ref,
    write_canonical_json,
)


LEGACY_CONFIG = REPOSITORY_ROOT / "config/stage1/wp3_candidate_generators_v1.json"
PROTOCOL = REPOSITORY_ROOT / "config/stage1/wp3_candidate_generators_s21b_v1.json"
RAW_GOLD_SCHEMA = REPOSITORY_ROOT / "schemas/wp3_s21_locked_raw_gold_v1.schema.json"
FRAME_REF = (
    REPOSITORY_ROOT
    / "exps/causal_context/stage1_p0/wp3_candidate_generators_v2/refs/development_frame_ref.json"
)


@unittest.skipUnless(FRAME_REF.is_file(), "materialized S2.1 frame is unavailable")
class LockedRawGoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.frame_dir = resolve_locator_ref(
            FRAME_REF, "wp3-s21-development-frame"
        )
        cls.frame = validate_development_frame(
            cls.frame_dir,
            workspace_root=REPOSITORY_ROOT,
            generator_config_path=LEGACY_CONFIG,
            require_current_sources=True,
        )

    def _session(self, root: Path, *, locked: bool = True) -> Path:
        path = root / "session.json"
        session = create_review_session(
            frame_dir=self.frame_dir,
            session_path=path,
            reviewer_id="reviewer-s21b",
            workspace_root=REPOSITORY_ROOT,
            generator_config_path=LEGACY_CONFIG,
        )
        if not locked:
            return path
        for annotation in session["raw_annotations"].values():
            annotation["status"] = "confirmed"
        session["phase"] = "diagnostic"
        session["raw_lock"] = {
            "policy": "raw-lock-before-unattributed-proposal-reveal/v1",
            "raw_annotations_sha256": _canonical_sha(session["raw_annotations"]),
            "locked_at": "2026-08-30T00:00:00+00:00",
        }
        session["updated_at"] = "2026-08-30T00:00:00+00:00"
        write_canonical_json(path, _with_session_revision(session))
        return path

    def _build(self, root: Path, session: Path) -> dict:
        return build_locked_raw_gold(
            frame_dir=self.frame_dir,
            session_path=session,
            workspace_root=REPOSITORY_ROOT,
            legacy_generator_config_path=LEGACY_CONFIG,
            protocol_path=PROTOCOL,
            annotation_schema_path=RAW_GOLD_SCHEMA,
            output_root=root / "raw_gold",
            reviewer_id="reviewer-s21b",
        )

    def test_freeze_is_deterministic_and_does_not_mutate_legacy_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = self._session(root)
            before = session.read_bytes()
            first = self._build(root, session)
            second = self._build(root, session)
            self.assertEqual(first["raw_gold_id"], second["raw_gold_id"])
            self.assertEqual(first["payload_manifest_sha256"], second["payload_manifest_sha256"])
            self.assertEqual(session.read_bytes(), before)
            self.assertEqual(first["manifest"]["case_count"], 424)
            self.assertEqual(first["manifest"]["mention_count"], 0)
            self.assertEqual(first["manifest"]["legacy_phase_b_status"], LEGACY_PHASE_B_STATUS)
            self.assertEqual(
                first["manifest"]["legacy_diagnostic_decision_count_at_freeze"],
                0,
            )
            self.assertFalse(first["manifest"]["sealed"])

    def test_independent_validation_and_payload_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._build(root, self._session(root))
            validated = validate_locked_raw_gold(
                result["target"],
                frame_dir=self.frame_dir,
                workspace_root=REPOSITORY_ROOT,
                legacy_generator_config_path=LEGACY_CONFIG,
                protocol_path=PROTOCOL,
            )
            self.assertEqual(validated["raw_gold_id"], result["raw_gold_id"])

            copied_parent = root / "tampered"
            copied = copied_parent / result["raw_gold_id"]
            shutil.copytree(result["target"], copied)
            annotations = copied / "annotations.json"
            annotations.write_bytes(annotations.read_bytes() + b" ")
            with self.assertRaises(CandidateDevelopmentV2Error):
                validate_locked_raw_gold(
                    copied,
                    frame_dir=self.frame_dir,
                    workspace_root=REPOSITORY_ROOT,
                    legacy_generator_config_path=LEGACY_CONFIG,
                    protocol_path=PROTOCOL,
                )

    def test_prelock_hash_drift_and_wrong_reviewer_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prelock = self._session(root, locked=False)
            with self.assertRaisesRegex(
                CandidateDevelopmentV2Error, "unfinalized locked"
            ):
                self._build(root, prelock)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session_path = self._session(root)
            session = json.loads(session_path.read_text(encoding="utf-8"))
            session["raw_annotations"]["S21-001"]["notes"] = "drift"
            write_canonical_json(session_path, _with_session_revision(session))
            with self.assertRaisesRegex(
                CandidateDevelopmentV2Error, "raw lock does not bind"
            ):
                self._build(root, session_path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session_path = self._session(root)
            with self.assertRaisesRegex(CandidateDevelopmentV2Error, "does not own"):
                build_locked_raw_gold(
                    frame_dir=self.frame_dir,
                    session_path=session_path,
                    workspace_root=REPOSITORY_ROOT,
                    legacy_generator_config_path=LEGACY_CONFIG,
                    protocol_path=PROTOCOL,
                    annotation_schema_path=RAW_GOLD_SCHEMA,
                    output_root=root / "raw_gold",
                    reviewer_id="another-reviewer",
                )

    def test_protocol_tampering_cannot_authorize_execution(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        protocol["execution"]["model_calls_authorized"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            write_canonical_json(path, protocol)
            with self.assertRaisesRegex(
                CandidateDevelopmentV2Error, "must remain unauthorized"
            ):
                load_s21b_protocol(path)

    def test_protocol_phase_b_and_s22_fields_are_exact(self) -> None:
        mutations = []
        missing_census = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        del missing_census["phase_b_v2"]["required_census"]
        mutations.append(missing_census)
        changed_population = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        changed_population["phase_b_v2"]["sampled_population"] = "all-proposals"
        mutations.append(changed_population)
        extra_phase_b_authority = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        extra_phase_b_authority["phase_b_v2"]["model_calls_authorized"] = True
        mutations.append(extra_phase_b_authority)
        extra_s22_authority = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        extra_s22_authority["s22"]["materialization_authorized"] = True
        mutations.append(extra_s22_authority)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, protocol in enumerate(mutations):
                path = root / f"protocol-{index}.json"
                write_canonical_json(path, protocol)
                with self.subTest(index=index), self.assertRaises(
                    CandidateDevelopmentV2Error
                ):
                    load_s21b_protocol(path)

    def test_protocol_rejects_noncanonical_numbers_and_schema_cotamper(self) -> None:
        mutations = []
        float_case_count = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        float_case_count["current_generator_pilot"]["case_count"] = 424.0
        mutations.append(float_case_count)
        boolean_decision_count = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        boolean_decision_count["legacy_phase_b"]["required_decision_count"] = False
        mutations.append(boolean_decision_count)
        boolean_amendment_count = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        boolean_amendment_count["locked_raw_gold"][
            "allowed_post_reveal_raw_amendment_count"
        ] = False
        mutations.append(boolean_amendment_count)
        changed_date = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        changed_date["accepted_date"] = "2026-08-31"
        mutations.append(changed_date)
        schema_cotamper = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        permissive_schema = {
            "$id": schema_cotamper["locked_raw_gold"]["annotation_schema_id"]
        }
        schema_cotamper["locked_raw_gold"][
            "annotation_schema_sha256"
        ] = canonical_sha256(permissive_schema)
        mutations.append(schema_cotamper)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, protocol in enumerate(mutations):
                path = root / f"protocol-numeric-{index}.json"
                write_canonical_json(path, protocol)
                with self.subTest(index=index), self.assertRaises(
                    CandidateDevelopmentV2Error
                ):
                    load_s21b_protocol(path)

    def test_annotation_schema_is_protocol_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = self._session(root)
            permissive_schema = root / "schema.json"
            write_canonical_json(permissive_schema, {})
            with self.assertRaisesRegex(
                CandidateDevelopmentV2Error, "schema is not protocol-bound"
            ):
                build_locked_raw_gold(
                    frame_dir=self.frame_dir,
                    session_path=session,
                    workspace_root=REPOSITORY_ROOT,
                    legacy_generator_config_path=LEGACY_CONFIG,
                    protocol_path=PROTOCOL,
                    annotation_schema_path=permissive_schema,
                    output_root=root / "raw_gold",
                    reviewer_id="reviewer-s21b",
                )

    def test_post_reveal_and_malformed_amendments_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session_path = self._session(root)
            session = json.loads(session_path.read_text(encoding="utf-8"))
            session["amendments"].append(
                {
                    "scope": "raw",
                    "case_id": "S21-001",
                    "proposal_id": None,
                    "reason": "post-reveal change",
                    "post_reveal_raw_amendment": True,
                    "reopened_at": "2026-08-30T00:01:00+00:00",
                    "prior_revision": session["revision"],
                }
            )
            write_canonical_json(session_path, _with_session_revision(session))
            with self.assertRaisesRegex(
                CandidateDevelopmentV2Error, "amendment identity differs"
            ):
                self._build(root, session_path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session_path = self._session(root)
            session = json.loads(session_path.read_text(encoding="utf-8"))
            session["amendments"].append("not-an-object")
            write_canonical_json(session_path, _with_session_revision(session))
            with self.assertRaisesRegex(
                CandidateDevelopmentV2Error, "must be an object"
            ):
                self._build(root, session_path)

    def test_legacy_decision_keys_and_count_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session_path = self._session(root)
            session = json.loads(session_path.read_text(encoding="utf-8"))
            session["diagnostic_decisions"] = {"unknown:proposal": {}}
            write_canonical_json(session_path, _with_session_revision(session))
            with self.assertRaisesRegex(
                CandidateDevelopmentV2Error, "decision identities differ"
            ):
                self._build(root, session_path)

        with tempfile.TemporaryDirectory() as directory:
            session_path = self._session(Path(directory))
            session = json.loads(session_path.read_text(encoding="utf-8"))
            session.pop("diagnostic_decisions")
            with self.assertRaisesRegex(
                CandidateDevelopmentV2Error, "decision count differs"
            ):
                validate_locked_raw_session(
                    self.frame,
                    session,
                    legacy_decision_count_override=6307,
                )

    def test_mention_surface_cannot_have_edge_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session_path = self._session(root)
            session = json.loads(session_path.read_text(encoding="utf-8"))
            selected = None
            for case in self.frame["cases"]:
                content = str(case["content"])
                for index, character in enumerate(content[:-1]):
                    if character.isspace() and not content[index + 1].isspace():
                        selected = (str(case["case_id"]), content[index : index + 2])
                        break
                if selected is not None:
                    break
            self.assertIsNotNone(selected)
            case_id, surface = selected
            content = next(
                str(case["content"])
                for case in self.frame["cases"]
                if case["case_id"] == case_id
            )
            start, end = resolve_exact_span(content, surface, 1)
            identity = {
                "surface": surface,
                "occurrence_ordinal": 1,
                "start": start,
                "end": end,
            }
            session["raw_annotations"][case_id]["needs_explanation"] = True
            session["raw_annotations"][case_id]["mentions"] = [
                {
                    "mention_id": "s21mention-" + _canonical_sha(identity)[:32],
                    **identity,
                    "provisional_route": "A_candidate",
                    "reason_codes": ["stable_core_candidate"],
                    "notes": "",
                }
            ]
            session["raw_lock"]["raw_annotations_sha256"] = _canonical_sha(
                session["raw_annotations"]
            )
            write_canonical_json(session_path, _with_session_revision(session))
            with self.assertRaisesRegex(
                CandidateDevelopmentV2Error,
                "leading or trailing whitespace",
            ):
                self._build(root, session_path)

    def test_publication_paths_cannot_alias_legacy_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session_path = self._session(root)
            before = session_path.read_bytes()
            with self.assertRaisesRegex(
                CandidateDevelopmentV2Error, "protected session"
            ):
                build_locked_raw_gold(
                    frame_dir=self.frame_dir,
                    session_path=session_path,
                    workspace_root=REPOSITORY_ROOT,
                    legacy_generator_config_path=LEGACY_CONFIG,
                    protocol_path=PROTOCOL,
                    annotation_schema_path=RAW_GOLD_SCHEMA,
                    output_root=root / "raw_gold",
                    reviewer_id="reviewer-s21b",
                    write_ref=session_path,
                )
            self.assertEqual(session_path.read_bytes(), before)
            self.assertFalse((root / "raw_gold").exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session_path = self._session(root)
            frame_files_before = {
                path.relative_to(self.frame_dir)
                for path in self.frame_dir.rglob("*")
            }
            with self.assertRaisesRegex(
                CandidateDevelopmentV2Error, "protected legacy path"
            ):
                build_locked_raw_gold(
                    frame_dir=self.frame_dir,
                    session_path=session_path,
                    workspace_root=REPOSITORY_ROOT,
                    legacy_generator_config_path=LEGACY_CONFIG,
                    protocol_path=PROTOCOL,
                    annotation_schema_path=RAW_GOLD_SCHEMA,
                    output_root=self.frame_dir,
                    reviewer_id="reviewer-s21b",
                )
            self.assertEqual(
                {
                    path.relative_to(self.frame_dir)
                    for path in self.frame_dir.rglob("*")
                },
                frame_files_before,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session_path = self._session(root)
            frame_ref_before = FRAME_REF.read_bytes()
            with self.assertRaisesRegex(
                CandidateDevelopmentV2Error, "different artifact locator"
            ):
                build_locked_raw_gold(
                    frame_dir=self.frame_dir,
                    session_path=session_path,
                    workspace_root=REPOSITORY_ROOT,
                    legacy_generator_config_path=LEGACY_CONFIG,
                    protocol_path=PROTOCOL,
                    annotation_schema_path=RAW_GOLD_SCHEMA,
                    output_root=root / "raw_gold",
                    reviewer_id="reviewer-s21b",
                    write_ref=FRAME_REF,
                )
            self.assertEqual(FRAME_REF.read_bytes(), frame_ref_before)
            self.assertFalse((root / "raw_gold").exists())

    def test_manifest_and_declaration_counts_require_canonical_integers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._build(root, self._session(root))
            copied = root / "noncanonical" / result["raw_gold_id"]
            shutil.copytree(result["target"], copied)
            manifest_path = copied / "manifest.json"
            declaration_path = copied / "reviewer_declaration.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            declaration = json.loads(
                declaration_path.read_text(encoding="utf-8")
            )
            manifest["post_reveal_raw_amendment_count"] = False
            declaration["post_reveal_raw_amendment_count"] = False
            write_canonical_json(manifest_path, manifest)
            write_canonical_json(declaration_path, declaration)
            with self.assertRaisesRegex(
                CandidateDevelopmentV2Error, "counts are not canonical"
            ):
                _validate_locked_raw_payload(
                    copied,
                    frame=self.frame,
                    expected_protocol=load_s21b_protocol(PROTOCOL),
                    require_current_validator=False,
                )

    def test_bound_legacy_sources_remain_current_and_valid(self) -> None:
        validated = validate_development_frame(
            self.frame_dir,
            workspace_root=REPOSITORY_ROOT,
            generator_config_path=LEGACY_CONFIG,
            require_current_sources=True,
        )
        self.assertEqual(validated["frame_id"], self.frame["frame_id"])


if __name__ == "__main__":
    unittest.main()
