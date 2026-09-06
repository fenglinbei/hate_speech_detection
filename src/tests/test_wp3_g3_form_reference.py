from __future__ import annotations

import copy
import fcntl
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_lex.terminology_g3_form_reference import (  # noqa: E402
    FROZEN_SOURCE_URLS,
    G3FormReferenceError,
    G3FormReviewConflict,
    SYNC_RECEIPT_ID_PREFIX,
    _write_session_cas,
    build_form_review_frame,
    create_form_review_session,
    fetch_public_source_snapshots,
    finalize_form_reference,
    read_form_review_session,
    reopen_form_decision,
    save_form_decision,
    save_form_decision_from_validated_frame,
    sync_public_sources,
    validate_form_reference,
    validate_form_review_frame,
    validate_public_source_bundle,
)
from build_lex import terminology_g3_form_reference as form_reference_module  # noqa: E402
from data.training_artifacts import (  # noqa: E402
    TrainingArtifactError,
    build_payload_manifest,
    canonical_sha256,
    sha256_file,
    validate_payload_manifest,
    write_canonical_json,
)


CATALOG = REPOSITORY_ROOT / "config/stage1/wp3_g3_public_source_catalog_v1.json"
REFERENCE_SCHEMA = REPOSITORY_ROOT / "schemas/wp3_g3_form_reference_v1.schema.json"


class _Response:
    def __init__(
        self,
        url: str,
        body: bytes,
        *,
        status_code: int = 200,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.url = url
        self.status_code = status_code
        self.headers = {
            "Content-Type": "text/html; charset=utf-8",
            "Content-Length": str(len(body)),
            **(extra_headers or {}),
        }
        self._body = body
        self.closed = False

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield self._body

    def close(self):
        self.closed = True


class _Session:
    def __init__(self) -> None:
        self.trust_env = True
        self.calls: list[str] = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        self.last_kwargs = kwargs
        revision = "123456"
        body = (
            f'<html><script>{{"wgRevisionId":{revision}}}</script>'
            '<body><p>YYDS 永远的神 AWSL 啊我死了 雨女无瓜 与你无关</p></body></html>'
        ).encode()
        return _Response(url, body)

    def close(self):
        pass


class _DowngradeSession(_Session):
    def get(self, url, **kwargs):
        self.calls.append(url)
        self.last_kwargs = kwargs
        return _Response(
            url,
            b"redirect",
            status_code=302,
            extra_headers={"Location": url.replace("https://", "http://", 1)},
        )


class G3FormReferenceTests(unittest.TestCase):
    def _inputs(self, root: Path):
        snapshot_root = root / "snapshots"
        fetch_public_source_snapshots(
            workspace_root=REPOSITORY_ROOT,
            catalog_path=CATALOG,
            output_directory=snapshot_root,
            session_factory=lambda: _Session(),
        )
        extraction = {
            "schema_version": "wp3-g3-form-extraction/v1",
            "extractor": "human-prepared-form-relations/v1",
            "evidence": [
                {
                    "evidence_id": "g3ev-yyds-surface",
                    "source_id": "china-daily-yyds-nbcs-hhh",
                    "quote": "YYDS",
                    "occurrence_ordinal": 1,
                    "relation_note": "explicit surface",
                },
                {
                    "evidence_id": "g3ev-yyds-canonical",
                    "source_id": "china-daily-yyds-nbcs-hhh",
                    "quote": "永远的神",
                    "occurrence_ordinal": 1,
                    "relation_note": "explicit expansion",
                },
            ],
            "items": [
                {
                    "item_id": "g3form-yyds",
                    "surface": "YYDS",
                    "canonical": "永远的神",
                    "proposed_family": "phonetic_variant",
                    "phonetic_scan_enabled": False,
                    "evidence_ids": [
                        "g3ev-yyds-surface",
                        "g3ev-yyds-canonical",
                    ],
                }
            ],
        }
        index_path = root / "snapshot_index.json"
        receipt_path = snapshot_root / "sync_receipt.json"
        extraction_path = root / "extraction.json"
        shutil.copyfile(snapshot_root / "snapshot_index.json", index_path)
        write_canonical_json(extraction_path, extraction)
        return snapshot_root, index_path, receipt_path, extraction_path

    def _artifacts(self, root: Path):
        snapshot_root, index_path, receipt_path, extraction_path = self._inputs(root)
        source = sync_public_sources(
            workspace_root=REPOSITORY_ROOT,
            catalog_path=CATALOG,
            snapshot_root=snapshot_root,
            snapshot_index_path=index_path,
            sync_receipt_path=receipt_path,
            extraction_path=extraction_path,
            output_root=root / "source_artifacts",
        )
        frame = build_form_review_frame(
            source_bundle_dir=source["target"],
            workspace_root=REPOSITORY_ROOT,
            output_root=root / "frames",
        )
        return source, frame

    def test_closed_source_bundle_frame_and_payload_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, frame = self._artifacts(root)
            self.assertEqual(source["manifest"]["source_count"], 7)
            self.assertEqual(source["manifest"]["item_count"], 1)
            self.assertEqual(frame["manifest"]["item_count"], 1)
            self.assertEqual(
                source["manifest"]["sync_receipt_id"],
                source["sync_receipt"]["sync_receipt_id"],
            )
            schema_bindings = {
                "catalog_schema_sha256": "wp3_g3_public_source_catalog_v1.schema.json",
                "snapshot_index_schema_sha256": "wp3_g3_frozen_source_input_v1.schema.json",
                "sync_receipt_schema_sha256": "wp3_g3_public_source_sync_receipt_v1.schema.json",
                "extraction_schema_sha256": "wp3_g3_form_extraction_v1.schema.json",
            }
            for field, filename in schema_bindings.items():
                self.assertEqual(
                    source["manifest"][field],
                    sha256_file(REPOSITORY_ROOT / "schemas" / filename),
                )
            validate_public_source_bundle(
                source["target"], workspace_root=REPOSITORY_ROOT
            )
            validate_form_review_frame(
                frame["target"],
                source_bundle_dir=source["target"],
                workspace_root=REPOSITORY_ROOT,
            )
            copied = root / Path(frame["target"]).name
            shutil.copytree(frame["target"], copied)
            (copied / "items.json").write_bytes(
                (copied / "items.json").read_bytes() + b" "
            )
            with self.assertRaises(TrainingArtifactError):
                validate_payload_manifest(copied)

    def test_extraction_rejects_outside_source_and_forbidden_data_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_root, index_path, receipt_path, extraction_path = self._inputs(root)
            extraction = json.loads(extraction_path.read_text())
            extraction["items"][0]["record_id"] = "fit-1"
            write_canonical_json(extraction_path, extraction)
            with self.assertRaises(G3FormReferenceError):
                sync_public_sources(
                    workspace_root=REPOSITORY_ROOT,
                    catalog_path=CATALOG,
                    snapshot_root=snapshot_root,
                    snapshot_index_path=index_path,
                    sync_receipt_path=receipt_path,
                    extraction_path=extraction_path,
                    output_root=root / "out",
                )

    def test_cas_confirm_reopen_and_defer_finalize_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, frame = self._artifacts(root)
            session_path = root / "working" / "session.json"
            session = create_form_review_session(
                frame_dir=frame["target"],
                source_bundle_dir=source["target"],
                workspace_root=REPOSITORY_ROOT,
                session_path=session_path,
                reviewer_id="reviewer-test",
            )
            with self.assertRaisesRegex(G3FormReferenceError, "incomplete"):
                finalize_form_reference(
                    frame_dir=frame["target"],
                    source_bundle_dir=source["target"],
                    workspace_root=REPOSITORY_ROOT,
                    session_path=session_path,
                    reviewer_id="reviewer-test",
                    romanizer_profile_path=root / "missing-profile.json",
                    output_root=root / "references",
                    reference_schema_path=REFERENCE_SCHEMA,
                    romanizer=lambda text: ["x"] * len(text),
                    initials_builder=lambda values: "".join(v[0] for v in values),
                )
            decision = {
                "action": "accept",
                "surface": "YYDS",
                "canonical": "永远的神",
                "family": "phonetic_variant",
                "phonetic_scan_enabled": False,
                "evidence_ids": ["g3ev-yyds-surface", "g3ev-yyds-canonical"],
                "notes": "",
            }
            validated_frame = validate_form_review_frame(
                frame["target"],
                source_bundle_dir=source["target"],
                workspace_root=REPOSITORY_ROOT,
            )
            cached_saved = save_form_decision_from_validated_frame(
                frame=validated_frame,
                session_path=session_path,
                item_id="g3form-yyds",
                decision=decision,
                confirm=False,
                expected_revision=session["revision"],
            )
            self.assertEqual(
                cached_saved["decisions"]["g3form-yyds"]["status"], "draft"
            )
            saved = save_form_decision(
                frame_dir=frame["target"],
                source_bundle_dir=source["target"],
                workspace_root=REPOSITORY_ROOT,
                session_path=session_path,
                item_id="g3form-yyds",
                decision=decision,
                confirm=True,
                expected_revision=cached_saved["revision"],
            )
            with self.assertRaises(G3FormReviewConflict):
                reopen_form_decision(
                    session_path=session_path,
                    item_id="g3form-yyds",
                    reason="stale",
                    expected_revision=session["revision"],
                )
            reopened = reopen_form_decision(
                session_path=session_path,
                item_id="g3form-yyds",
                reason="check evidence again",
                expected_revision=saved["revision"],
            )
            self.assertEqual(len(reopened["amendments"]), 1)
            self.assertEqual(
                read_form_review_session(session_path)["decisions"]["g3form-yyds"]["status"],
                "draft",
            )

    def test_session_cas_waits_for_cross_process_advisory_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, frame = self._artifacts(root)
            session_path = root / "working" / "session.json"
            session = create_form_review_session(
                frame_dir=frame["target"],
                source_bundle_dir=source["target"],
                workspace_root=REPOSITORY_ROOT,
                session_path=session_path,
                reviewer_id="reviewer-test",
            )
            child_code = """
import sys
from build_lex.terminology_g3_form_reference import _form_review_session_lock
with _form_review_session_lock(sys.argv[1], exclusive=True):
    print("locked", flush=True)
    sys.stdin.readline()
"""
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(SRC_ROOT)
            process = subprocess.Popen(
                [sys.executable, "-c", child_code, str(session_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
            )
            self.assertIsNotNone(process.stdout)
            self.assertIsNotNone(process.stdin)
            marker = process.stdout.readline().strip()
            if marker != "locked":
                _, stderr = process.communicate(input="\n", timeout=5)
                process.stdin.close()
                process.stdout.close()
                process.stderr.close()
                self.fail(f"lock holder did not start: {marker!r} {stderr}")

            started = threading.Event()
            completed = threading.Event()
            failures: list[BaseException] = []
            updated = copy.deepcopy(session)
            updated["updated_at"] = "cross-process-lock-test"

            def mutate() -> None:
                started.set()
                try:
                    _write_session_cas(
                        session_path,
                        updated,
                        expected_revision=session["revision"],
                    )
                except BaseException as exc:  # surfaced in the main test thread
                    failures.append(exc)
                finally:
                    completed.set()

            worker = threading.Thread(target=mutate, daemon=True)
            worker.start()
            self.assertTrue(started.wait(2))
            self.assertFalse(completed.wait(0.2))
            process.stdin.write("\n")
            process.stdin.flush()
            returncode = process.wait(timeout=5)
            process.stdin.close()
            process.stdout.close()
            child_stderr = process.stderr.read()
            process.stderr.close()
            self.assertEqual(returncode, 0, child_stderr)
            self.assertTrue(completed.wait(5))
            worker.join(timeout=1)
            self.assertEqual(failures, [])
            self.assertEqual(
                read_form_review_session(session_path)["updated_at"],
                "cross-process-lock-test",
            )
            lock_path = session_path.with_name(session_path.name + ".lock")
            self.assertEqual(stat.S_IMODE(lock_path.stat().st_mode), 0o600)

    def test_finalize_holds_exclusive_lock_before_snapshot_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session_path = root / "session.json"

            def probe(**_kwargs):
                lock_path = session_path.with_name(session_path.name + ".lock")
                descriptor = os.open(lock_path, os.O_RDWR)
                try:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(
                            descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB
                        )
                finally:
                    os.close(descriptor)
                return {"lock_held": True}

            with patch.object(
                form_reference_module,
                "_finalize_form_reference_locked",
                side_effect=probe,
            ):
                result = finalize_form_reference(
                    frame_dir=root / "unused-frame",
                    source_bundle_dir=root / "unused-sources",
                    workspace_root=REPOSITORY_ROOT,
                    session_path=session_path,
                    reviewer_id="reviewer-test",
                    romanizer_profile_path=root / "unused-profile.json",
                    output_root=root / "unused-output",
                    reference_schema_path=REFERENCE_SCHEMA,
                )
            self.assertEqual(result, {"lock_held": True})

    def test_finalize_builds_evidence_conditioned_initials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, frame = self._artifacts(root)
            session_path = root / "working" / "session.json"
            session = create_form_review_session(
                frame_dir=frame["target"],
                source_bundle_dir=source["target"],
                workspace_root=REPOSITORY_ROOT,
                session_path=session_path,
                reviewer_id="reviewer-test",
            )
            decision = {
                "action": "accept",
                "surface": "YYDS",
                "canonical": "永远的神",
                "family": "phonetic_variant",
                "phonetic_scan_enabled": False,
                "evidence_ids": ["g3ev-yyds-surface", "g3ev-yyds-canonical"],
                "notes": "public form evidence",
            }
            save_form_decision(
                frame_dir=frame["target"],
                source_bundle_dir=source["target"],
                workspace_root=REPOSITORY_ROOT,
                session_path=session_path,
                item_id="g3form-yyds",
                decision=decision,
                confirm=True,
                expected_revision=session["revision"],
            )
            profile = {"profile_version": "wp3-g3-profile/full-v2"}
            profile_path = root / "profile.json"
            write_canonical_json(profile_path, profile)
            romanizer = lambda text: ["yong", "yuan", "de", "shen"]
            initials = lambda values: "".join(value[0] for value in values)
            result = finalize_form_reference(
                frame_dir=frame["target"],
                source_bundle_dir=source["target"],
                workspace_root=REPOSITORY_ROOT,
                session_path=session_path,
                reviewer_id="reviewer-test",
                romanizer_profile_path=profile_path,
                output_root=root / "references",
                reference_schema_path=REFERENCE_SCHEMA,
                romanizer=romanizer,
                initials_builder=initials,
            )
            row = result["reference"]["rows"][0]
            self.assertEqual(row["initials"], "yyds")
            self.assertEqual(
                row["variants"][0]["evidence_ids"],
                ["g3ev-yyds-canonical", "g3ev-yyds-surface"],
            )
            self.assertFalse(result["reference"]["scientific_eligible"])
            self.assertFalse(result["reference"]["sealed"])
            validate_form_reference(
                result["target"],
                frame_dir=frame["target"],
                source_bundle_dir=source["target"],
                workspace_root=REPOSITORY_ROOT,
                romanizer_profile_path=profile_path,
                reference_schema_path=REFERENCE_SCHEMA,
                romanizer=romanizer,
                initials_builder=initials,
            )
            with self.assertRaisesRegex(G3FormReferenceError, "immutable"):
                save_form_decision(
                    frame_dir=frame["target"],
                    source_bundle_dir=source["target"],
                    workspace_root=REPOSITORY_ROOT,
                    session_path=session_path,
                    item_id="g3form-yyds",
                    decision=decision,
                    confirm=False,
                    expected_revision=read_form_review_session(session_path)["revision"],
                )

    def test_network_synchronizer_is_exact_bounded_and_revision_locked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = _Session()
            result = fetch_public_source_snapshots(
                workspace_root=REPOSITORY_ROOT,
                catalog_path=CATALOG,
                output_directory=root / "downloaded",
                session_factory=lambda: session,
            )
            self.assertEqual(result["source_count"], 7)
            self.assertFalse(session.trust_env)
            self.assertEqual(len(session.calls), 9)
            self.assertTrue(
                all(call in FROZEN_SOURCE_URLS.values() or "?oldid=123456" in call for call in session.calls)
            )
            index = json.loads((root / "downloaded/snapshot_index.json").read_text())
            receipt = json.loads((root / "downloaded/sync_receipt.json").read_text())
            wiki = [row for row in index["snapshots"] if row["wikimedia_revision_id"]]
            self.assertEqual(len(wiki), 2)
            self.assertTrue(all(row["final_url"].endswith("?oldid=123456") for row in wiki))
            self.assertTrue(all(row["media_type"] == "text/plain" for row in index["snapshots"]))
            self.assertTrue(receipt["sync_receipt_id"].startswith(SYNC_RECEIPT_ID_PREFIX))
            self.assertEqual(receipt["snapshot_index_sha256"], canonical_sha256(index))
            self.assertEqual(receipt["source_count"], 7)
            self.assertTrue(receipt["network_access_performed"])
            self.assertTrue(receipt["complete"])
            self.assertEqual(receipt["source_ids"], sorted(FROZEN_SOURCE_URLS))
            self.assertEqual(
                receipt["fetch_implementation_sha256"],
                sha256_file(form_reference_module.__file__),
            )
            self.assertEqual(result["sync_receipt_id"], receipt["sync_receipt_id"])
            with self.assertRaisesRegex(G3FormReferenceError, "overwrite"):
                fetch_public_source_snapshots(
                    workspace_root=REPOSITORY_ROOT,
                    catalog_path=CATALOG,
                    output_directory=root / "downloaded",
                    session_factory=lambda: _Session(),
                )

    def test_frozen_bundle_replays_receipt_after_payload_manifest_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = self._artifacts(root)
            copied = root / "tampered-source" / Path(source["target"]).name
            copied.parent.mkdir()
            shutil.copytree(source["target"], copied)
            receipt_path = copied / "sync_receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["fetch_implementation_sha256"] = "0" * 64
            identity = {
                key: value for key, value in receipt.items() if key != "sync_receipt_id"
            }
            receipt["sync_receipt_id"] = SYNC_RECEIPT_ID_PREFIX + canonical_sha256(
                identity
            )
            write_canonical_json(receipt_path, receipt)
            write_canonical_json(
                copied / "payload_manifest.json", build_payload_manifest(copied)
            )
            with self.assertRaisesRegex(G3FormReferenceError, "does not replay"):
                validate_public_source_bundle(
                    copied,
                    workspace_root=REPOSITORY_ROOT,
                    require_current_implementation=True,
                )

    def test_freeze_rejects_missing_or_forged_sync_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_root, index_path, receipt_path, extraction_path = self._inputs(root)
            receipt_path.unlink()
            with self.assertRaisesRegex(G3FormReferenceError, "receipt is missing"):
                sync_public_sources(
                    workspace_root=REPOSITORY_ROOT,
                    catalog_path=CATALOG,
                    snapshot_root=snapshot_root,
                    snapshot_index_path=index_path,
                    sync_receipt_path=receipt_path,
                    extraction_path=extraction_path,
                    output_root=root / "missing-receipt-output",
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_root, index_path, receipt_path, extraction_path = self._inputs(root)
            receipt = json.loads(receipt_path.read_text())
            receipt["fetch_implementation_sha256"] = "0" * 64
            identity = {
                key: value for key, value in receipt.items() if key != "sync_receipt_id"
            }
            receipt["sync_receipt_id"] = SYNC_RECEIPT_ID_PREFIX + canonical_sha256(
                identity
            )
            write_canonical_json(receipt_path, receipt)
            with self.assertRaisesRegex(G3FormReferenceError, "does not replay"):
                sync_public_sources(
                    workspace_root=REPOSITORY_ROOT,
                    catalog_path=CATALOG,
                    snapshot_root=snapshot_root,
                    snapshot_index_path=index_path,
                    sync_receipt_path=receipt_path,
                    extraction_path=extraction_path,
                    output_root=root / "forged-receipt-output",
                )

    def test_freeze_rejects_snapshot_index_drift_after_sync(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_root, index_path, receipt_path, extraction_path = self._inputs(root)
            index = json.loads(index_path.read_text())
            index["snapshots"][0]["fetched_at"] = "2099-01-01T00:00:00+00:00"
            write_canonical_json(index_path, index)
            with self.assertRaisesRegex(G3FormReferenceError, "does not replay"):
                sync_public_sources(
                    workspace_root=REPOSITORY_ROOT,
                    catalog_path=CATALOG,
                    snapshot_root=snapshot_root,
                    snapshot_index_path=index_path,
                    sync_receipt_path=receipt_path,
                    extraction_path=extraction_path,
                    output_root=root / "drifted-index-output",
                )

    def test_network_synchronizer_rejects_catalog_https_downgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "downloaded"
            with self.assertRaisesRegex(
                G3FormReferenceError,
                r"source moe-network-language-experts .*https://.* to http://",
            ):
                fetch_public_source_snapshots(
                    workspace_root=REPOSITORY_ROOT,
                    catalog_path=CATALOG,
                    output_directory=destination,
                    session_factory=lambda: _DowngradeSession(),
                )
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
