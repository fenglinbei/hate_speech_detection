from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

from build_lex import terminology_g3_source_catalog_v2 as lifecycle
from build_lex.terminology_g3_form_reference import load_source_catalog
from data.training_artifacts import canonical_sha256, load_json, sha256_file, write_canonical_json


CATALOG = REPOSITORY_ROOT / "config/stage1/wp3_g3_public_source_catalog_v2.json"
CATALOG_V1 = REPOSITORY_ROOT / "config/stage1/wp3_g3_public_source_catalog_v1.json"


class G3SourceCatalogV2Tests(unittest.TestCase):
    def _catalog(self) -> dict:
        return lifecycle.load_source_catalog_v2(
            CATALOG, workspace_root=REPOSITORY_ROOT
        )

    def _write_catalog(self, directory: Path, value: dict) -> Path:
        path = directory / "catalog.json"
        write_canonical_json(path, value)
        return path

    def test_catalog_is_closed_and_v1_remains_unchanged(self) -> None:
        catalog = self._catalog()
        self.assertEqual(len(catalog["targets"]), 42)
        self.assertEqual(
            canonical_sha256(sorted(row["requested_url"] for row in catalog["targets"])),
            lifecycle.FROZEN_TARGET_URLS_SHA256,
        )
        self.assertEqual(
            [row["target_id"] for row in catalog["targets"] if row["disposition"] == "rejected"],
            ["bupt-internet-slang-project"],
        )
        legacy = load_source_catalog(CATALOG_V1, workspace_root=REPOSITORY_ROOT)
        self.assertEqual(legacy["schema_version"], "wp3-g3-public-source-catalog/v1")
        self.assertEqual(len(legacy["external_sources"]), 7)

    def test_catalog_rejects_url_tamper_and_role_gate_bypass(self) -> None:
        original = self._catalog()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            changed_url = copy.deepcopy(original)
            changed_url["targets"][0]["requested_url"] += "changed"
            with self.assertRaisesRegex(
                lifecycle.G3SourceCatalogV2Error, "allowlist"
            ):
                lifecycle.load_source_catalog_v2(
                    self._write_catalog(root, changed_url),
                    workspace_root=REPOSITORY_ROOT,
                )
            role_bypass = copy.deepcopy(original)
            target = next(
                row
                for row in role_bypass["targets"]
                if row["source_role"] == "method_only"
            )
            target["disposition"] = "accepted"
            target["evidence_allowed"] = True
            with self.assertRaisesRegex(
                lifecycle.G3SourceCatalogV2Error, "direct/candidate"
            ):
                lifecycle.load_source_catalog_v2(
                    self._write_catalog(root, role_bypass),
                    workspace_root=REPOSITORY_ROOT,
                )

    def _fake_dependencies(
        self, root: Path, catalog: dict
    ) -> tuple[
        tuple[dict, Path, dict],
        tuple[dict, Path, dict, list[dict], list[dict]],
    ]:
        capture_root = root / "capture"
        capture_root.mkdir()
        manual_root = root / "manual"
        manual_root.mkdir()
        catalog_components = {
            component["component_id"]: (target, component)
            for target in catalog["targets"]
            for component in target["components"]
        }
        pages: list[dict] = []
        for target in catalog["targets"]:
            is_manual = any(
                component["component_id"] in lifecycle.MANUAL_COMPONENT_IDS
                for component in target["components"]
            )
            usable = target["disposition"] != "rejected"
            response_file = None
            projection_file = None
            response_sha = None
            projection_sha = None
            response_size = 0
            final_url = None
            status = "network_error" if is_manual else "downloaded_usable"
            if target["disposition"] == "rejected":
                status = "http_error"
            if usable and not is_manual:
                payload = f"normalized capture for {target['target_id']}\n".encode()
                response_file = f"responses/{target['target_id']}.html"
                projection_file = f"text/{target['target_id']}.txt"
                for logical in (response_file, projection_file):
                    path = capture_root / logical
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(payload)
                response_sha = hashlib.sha256(payload).hexdigest()
                projection_sha = response_sha
                response_size = len(payload)
                final_url = target["requested_url"]
            pages.append(
                {
                    "requested_url": target["requested_url"],
                    "fetch_status": status,
                    "final_url": final_url,
                    "status_code": 200 if final_url else None,
                    "response_file": response_file,
                    "response_size_bytes": response_size,
                    "response_sha256": response_sha,
                    "text_projection_file": projection_file,
                    "text_projection_sha256": projection_sha,
                    "content_analysis": {
                        "content_validation": "usable_static_text",
                        "media_kind": "html",
                    },
                    "wikimedia_revision_id": None,
                    "transport_downgrade": False,
                    "cross_host_redirect": False,
                    "redirect_chain": [],
                }
            )
        capture_dependency = {
            "artifact_kind": lifecycle.CAPTURE_ARTIFACT_KIND,
            "artifact_id": "wp3capture-" + "a" * 64,
            "payload_manifest_sha256": "b" * 64,
        }
        manual_components: list[dict] = []
        aliases: list[dict] = []
        for component_id in sorted(lifecycle.MANUAL_COMPONENT_IDS):
            target, component = catalog_components[component_id]
            source_id = target["target_id"]
            raw_sha = hashlib.sha256(f"raw:{component_id}".encode()).hexdigest()
            raw_size = 100 + len(component_id)
            normalized_file = None
            normalized_sha = None
            normalized_size = None
            candidate = None
            if target["disposition"] == "accepted":
                payload = f"normalized manual for {component_id}\n".encode()
                normalized_file = f"normalized/{component_id}.txt"
                normalized_path = manual_root / normalized_file
                normalized_path.parent.mkdir(exist_ok=True)
                normalized_path.write_bytes(payload)
                normalized_sha = hashlib.sha256(payload).hexdigest()
                normalized_size = len(payload)
            if component_id == "chime-license":
                raw_sha = lifecycle.CHIME_LICENSE_SHA256
            if component_id == "chime-data-json":
                rows = []
                for ordinal in range(1, 186):
                    type_en = "abbreviation" if ordinal <= 52 else "homophonic pun"
                    rows.append(
                        {
                            "source_row_ordinal": ordinal,
                            "meme": f"m{ordinal}",
                            "meaning": f"meaning {ordinal}",
                            "origin": None,
                            "type_cn": "缩写" if type_en == "abbreviation" else "谐音",
                            "type_en": type_en,
                        }
                    )
                projection = {
                    "schema_version": "wp3-g3-chime-form-candidate-projection/v1",
                    "source_component_id": component_id,
                    "raw_sha256": raw_sha,
                    "source_record_count": 1458,
                    "allowed_types": ["abbreviation", "homophonic pun"],
                    "candidate_count": 185,
                    "rows": rows,
                }
                candidate_file = f"derived/{component_id}-form-candidates.json"
                candidate_path = manual_root / candidate_file
                candidate_path.parent.mkdir(exist_ok=True)
                write_canonical_json(candidate_path, projection)
                candidate = {
                    "schema_version": "wp3-g3-chime-form-candidate-projection/v1",
                    "file": candidate_file,
                    "size_bytes": candidate_path.stat().st_size,
                    "sha256": sha256_file(candidate_path),
                    "count": 185,
                    "allowed_types": ["abbreviation", "homophonic pun"],
                }
            requested_url = target["requested_url"]
            snapshot_url = component["component_url"]
            alias_ids = [f"{component_id}-upload"]
            if component_id == "people-weibo-neologisms-page-1":
                alias_ids.append(f"{component_id}-duplicate-upload")
            manual_components.append(
                {
                    "source_id": source_id,
                    "component_id": component_id,
                    "requested_url": requested_url if component["page_ordinal"] != 2 else snapshot_url,
                    "final_url": snapshot_url,
                    "snapshot_url": snapshot_url,
                    "acquisition_mode": (
                        "publisher_pdf"
                        if component["component_kind"] == "paper"
                        else "user_supplied_archive"
                    ),
                    "disposition": "candidate_only",
                    "aliases": alias_ids,
                    "raw_size_bytes": raw_size,
                    "raw_sha256": raw_sha,
                    "normalized_text_file": normalized_file,
                    "normalized_text_size_bytes": normalized_size,
                    "normalized_text_sha256": normalized_sha,
                    "candidate_projection": candidate,
                    "format_metadata": {},
                    "wikimedia_revision_id": component["wikimedia_oldid"],
                }
            )
            for alias_id in alias_ids:
                aliases.append(
                    {
                        "alias_id": alias_id,
                        "source_id": source_id,
                        "component_id": component_id,
                        "attachment_locator": f"attachments/{alias_id}",
                        "expected_size_bytes": raw_size,
                        "expected_sha256": raw_sha,
                    }
                )
        manual_manifest = {
            "companion_contracts": [
                {
                    "archive_component_id": "chime-repository-archive",
                    "data_component_id": "chime-data-json",
                    "license_component_id": "chime-license",
                    "repository_commit": lifecycle.CHIME_COMMIT,
                }
            ]
        }
        manual_dependency = {
            "artifact_kind": lifecycle.MANUAL_INTAKE_ARTIFACT_KIND,
            "artifact_id": "wp3manual-" + "c" * 64,
            "payload_manifest_sha256": "d" * 64,
        }
        return (
            (capture_dependency, capture_root, {"pages": pages}),
            (
                manual_dependency,
                manual_root,
                manual_manifest,
                manual_components,
                aliases,
            ),
        )

    def test_bundle_is_deterministic_and_payload_tamper_fails(self) -> None:
        catalog = self._catalog()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, manual = self._fake_dependencies(root, catalog)
            output = root / "bundles"
            with mock.patch.object(
                lifecycle, "_load_capture_dependency", return_value=capture
            ), mock.patch.object(
                lifecycle, "_load_manual_dependency", return_value=manual
            ):
                first = lifecycle.build_public_source_bundle_v2(
                    workspace_root=REPOSITORY_ROOT,
                    catalog_path=CATALOG,
                    capture_directory=root / "unused-capture",
                    manual_intake_directory=root / "unused-manual",
                    output_root=output,
                )
                second = lifecycle.build_public_source_bundle_v2(
                    workspace_root=REPOSITORY_ROOT,
                    catalog_path=CATALOG,
                    capture_directory=root / "unused-capture",
                    manual_intake_directory=root / "unused-manual",
                    output_root=output,
                )
            self.assertEqual(first["source_bundle_id"], second["source_bundle_id"])
            self.assertEqual(
                first["payload_manifest_sha256"], second["payload_manifest_sha256"]
            )
            self.assertEqual(first["acquisition_receipt"]["target_count"], 42)
            self.assertEqual(first["acquisition_receipt"]["manual_alias_count"], 13)
            failed_capture = next(
                row
                for row in first["acquisition_receipt"]["target_coverage"]
                if row["target_id"] == "wikipedia-internet-language"
            )
            self.assertEqual(failed_capture["capture_audit"]["fetch_status"], "network_error")
            self.assertTrue(failed_capture["selected_component_ids"])
            component_path = Path(first["target"]) / first["components"][0]["normalized_file"]
            component_path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(
                lifecycle.G3SourceCatalogV2Error, "payload manifest"
            ):
                lifecycle.validate_public_source_bundle_v2(
                    first["target"], workspace_root=REPOSITORY_ROOT
                )

    def test_receipt_rejects_rehashed_coverage_role_tamper(self) -> None:
        catalog = self._catalog()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, manual = self._fake_dependencies(root, catalog)
            with mock.patch.object(
                lifecycle, "_load_capture_dependency", return_value=capture
            ), mock.patch.object(
                lifecycle, "_load_manual_dependency", return_value=manual
            ):
                result = lifecycle.build_public_source_bundle_v2(
                    workspace_root=REPOSITORY_ROOT,
                    catalog_path=CATALOG,
                    capture_directory=root / "unused-capture",
                    manual_intake_directory=root / "unused-manual",
                    output_root=root / "bundles",
                )
            receipt = copy.deepcopy(result["acquisition_receipt"])
            tampered_row = next(
                row
                for row in receipt["target_coverage"]
                if row["source_role"] == "method_only"
            )
            tampered_row["source_role"] = "candidate_pool"
            receipt["target_coverage_sha256"] = canonical_sha256(
                receipt["target_coverage"]
            )
            identity = {
                key: value
                for key, value in receipt.items()
                if key != "acquisition_receipt_id"
            }
            receipt["acquisition_receipt_id"] = (
                lifecycle.ACQUISITION_RECEIPT_ID_PREFIX + canonical_sha256(identity)
            )
            with self.assertRaisesRegex(
                lifecycle.G3SourceCatalogV2Error, "coverage differs"
            ):
                lifecycle._validate_acquisition_receipt(
                    receipt,
                    catalog=result["catalog"],
                    components=result["components"],
                    workspace_root=REPOSITORY_ROOT,
                )


if __name__ == "__main__":
    unittest.main()
