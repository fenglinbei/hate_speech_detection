from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import socket
import sys
import tempfile
import unittest
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from unittest import mock
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_lex.terminology_g3_manual_source_intake import (  # noqa: E402
    CHIME_PROJECTION_SCHEMA_VERSION,
    ManualSourceIntakeError,
    _analyze_mhtml,
    _analyze_zip,
    _strict_json_bytes,
    build_manual_source_intake,
    load_manual_source_map,
    load_normalized_components,
    normalize_publisher_pdf_bytes,
    validate_manual_source_intake,
)


COMMIT = "865ef186a0e797ec5ac242524a3c45b30a429542"
ACQUIRED_AT = "2026-08-30T21:22:50+08:00"
MHTML_URL = "https://example.com/archive"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _chime_rows() -> list[dict[str, object]]:
    return [
        {
            "meme": "xswl",
            "meaning": "xswl 是笑死我了的拼音首字母缩写。",
            "origin": None,
            "examples": ["看到这个我 xswl。"],
            "profanity": False,
            "offense": False,
            "type_cn": "缩写",
            "type_en": "abbreviation",
        },
        {
            "meme": "蚌埠住了",
            "meaning": "蚌埠住了是绷不住了的谐音写法。",
            "origin": "来自近音替代。",
            "examples": ["真的蚌埠住了。"],
            "profanity": False,
            "offense": False,
            "type_cn": "谐音",
            "type_en": "homophonic pun",
        },
    ]


def _mhtml_bytes(*, select_root_with_start: bool = False) -> bytes:
    message = MIMEMultipart("related", type="text/html")
    message["From"] = '"Saved by Blink"'
    message["Snapshot-Content-Location"] = MHTML_URL
    message["Subject"] = "Archived page"
    message["Date"] = "Sun, 30 Aug 2026 21:22:50 +0800"
    if select_root_with_start:
        message.set_param("start", "<root@mhtml.test>", header="Content-Type")
        decoy = MIMEText(
            "<html><body>first body part is not the declared root</body></html>",
            "html",
            "utf-8",
        )
        decoy["Content-Location"] = "https://example.com/decoy"
        decoy["Content-ID"] = "<decoy@mhtml.test>"
        message.attach(decoy)
    html = (
        "<!doctype html><html><head><title>Archived page</title>"
        "<style>ignored</style></head><body><h1>网络语言材料</h1>"
        "<p>xswl 是笑死我了的拼音首字母缩写，这段正文足够长，"
        "并且只由本地 MIME root 生成可重放的正文投影。"
        "归档验证还会固定根部件哈希、页面位置、保存日期和字符编码，"
        "后续同位置的子框架不得混入主页面证据。</p>"
        "<script>ignored()</script></body></html>"
    )
    root = MIMEText(html, "html", "utf-8")
    root["Content-Location"] = MHTML_URL
    root["Content-ID"] = "<root@mhtml.test>"
    message.attach(root)
    child = MIMEText(
        "<html><body>child frame text must not enter the root projection</body></html>",
        "html",
        "utf-8",
    )
    child["Content-Location"] = MHTML_URL
    child["Content-ID"] = "<child@mhtml.test>"
    message.attach(child)
    image = MIMEText("not visible", "plain", "utf-8")
    image["Content-Location"] = "https://example.com/resource.txt"
    message.attach(image)
    return message.as_bytes()


def _zip_bytes(data: bytes, license_payload: bytes, *, malicious: bool = False) -> bytes:
    output = io.BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.comment = COMMIT.encode("ascii")
        archive.writestr("chime-main/data/chime_full.json", data)
        archive.writestr("chime-main/LICENSE", license_payload)
        if malicious:
            archive.writestr("../escape.txt", b"unsafe")
    return output.getvalue()


def _row(
    *,
    alias_id: str,
    source_id: str,
    component_id: str,
    locator: str,
    payload: bytes,
    format_name: str,
    requested_url: str,
    snapshot_url: str | None = None,
    acquisition_mode: str,
    source_role: str,
    disposition: str,
    title: str | None = None,
    pagination: dict[str, int] | None = None,
    revision: str | None = None,
) -> dict[str, object]:
    return {
        "alias_id": alias_id,
        "source_id": source_id,
        "component_id": component_id,
        "attachment_locator": locator,
        "requested_url": requested_url,
        "final_url": snapshot_url or requested_url,
        "snapshot_url": snapshot_url or requested_url,
        "acquired_at": ACQUIRED_AT,
        "title": title,
        "acquisition_mode": acquisition_mode,
        "source_role": source_role,
        "format": format_name,
        "disposition": disposition,
        "expected_size_bytes": len(payload),
        "expected_sha256": _sha(payload),
        "pagination": pagination,
        "wikimedia_revision_id": revision,
    }


def _fixture(root: Path) -> tuple[Path, dict[str, bytes]]:
    data = json.dumps(_chime_rows(), ensure_ascii=False, separators=(",", ":")).encode()
    license_payload = b"MIT License\n\nPermission is hereby granted.\n"
    archive = _zip_bytes(data, license_payload)
    mhtml = _mhtml_bytes()
    payloads = {
        "repo/chime.zip": archive,
        "repo/chime.json": data,
        "repo/LICENSE": license_payload,
        "pages/page-a.mhtml": mhtml,
        "pages/page-a-copy.mhtml": mhtml,
    }
    for locator, payload in payloads.items():
        target = root / locator
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    archive_url = f"https://github.com/example/chime/archive/{COMMIT}.zip"
    data_url = f"https://raw.githubusercontent.com/example/chime/{COMMIT}/data/chime_full.json"
    license_url = f"https://raw.githubusercontent.com/example/chime/{COMMIT}/LICENSE"
    attachments = [
        _row(
            alias_id="chime-archive",
            source_id="chime-data",
            component_id="chime-repository-archive",
            locator="repo/chime.zip",
            payload=archive,
            format_name="zip",
            requested_url=archive_url,
            acquisition_mode="repository_archive",
            source_role="license_or_provenance",
            disposition="acquisition_only",
        ),
        _row(
            alias_id="chime-json",
            source_id="chime-data",
            component_id="chime-data-json",
            locator="repo/chime.json",
            payload=data,
            format_name="json",
            requested_url=data_url,
            acquisition_mode="repository_data",
            source_role="candidate_pool",
            disposition="candidate_only",
        ),
        _row(
            alias_id="chime-license",
            source_id="chime-data",
            component_id="chime-license",
            locator="repo/LICENSE",
            payload=license_payload,
            format_name="text",
            requested_url=license_url,
            acquisition_mode="license_file",
            source_role="license_or_provenance",
            disposition="acquisition_only",
        ),
        _row(
            alias_id="page-a-copy-one",
            source_id="page-source",
            component_id="page-source-page",
            locator="pages/page-a.mhtml",
            payload=mhtml,
            format_name="mhtml",
            requested_url=MHTML_URL,
            acquisition_mode="user_supplied_archive",
            source_role="direct_evidence",
            disposition="evidence_eligible",
            title="Archived page",
        ),
        _row(
            alias_id="page-a-copy-two",
            source_id="page-source",
            component_id="page-source-page",
            locator="pages/page-a-copy.mhtml",
            payload=mhtml,
            format_name="mhtml",
            requested_url=MHTML_URL,
            acquisition_mode="user_supplied_archive",
            source_role="direct_evidence",
            disposition="evidence_eligible",
            title="Archived page",
        ),
    ]
    mapping = {
        "schema_version": "wp3-g3-manual-source-map/v1",
        "map_id": "test-manual-map-v1",
        "scope": "development-only-form-only-label-free-non-lexicon",
        "network_access_allowed": False,
        "attachments": attachments,
        "companion_contracts": [
            {
                "contract_id": "chime-companions-v1",
                "repository_commit": COMMIT,
                "archive_component_id": "chime-repository-archive",
                "data_component_id": "chime-data-json",
                "license_component_id": "chime-license",
                "archive_comment": COMMIT,
                "member_bindings": [
                    {
                        "member_path": "chime-main/data/chime_full.json",
                        "component_id": "chime-data-json",
                    },
                    {
                        "member_path": "chime-main/LICENSE",
                        "component_id": "chime-license",
                    },
                ],
                "expected_record_count": 2,
                "expected_candidate_count": 2,
                "expected_candidate_type_counts": {
                    "abbreviation": 1,
                    "homophonic pun": 1,
                },
            }
        ],
    }
    mapping_path = root / "mapping.json"
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    return mapping_path, payloads


class ManualSourceIntakeTests(unittest.TestCase):
    def test_strict_json_rejects_duplicate_keys_and_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(ManualSourceIntakeError, "duplicate JSON key"):
            _strict_json_bytes(b'{"a":1,"a":2}', label="fixture")
        for token in (b"NaN", b"Infinity", b"-Infinity"):
            with self.assertRaisesRegex(ManualSourceIntakeError, "non-finite"):
                _strict_json_bytes(b'{"a":' + token + b"}", label="fixture")

    def test_mapping_rejects_absolute_locator_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping_path, _ = _fixture(root)
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            mapping["attachments"][0]["attachment_locator"] = "/tmp/archive.zip"
            mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
            with self.assertRaisesRegex(ManualSourceIntakeError, "normalized relative"):
                load_manual_source_map(mapping_path)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping_path, _ = _fixture(root)
            original = root / "repo/chime.json"
            original.unlink()
            os.symlink(root / "repo/LICENSE", original)
            with tempfile.TemporaryDirectory() as output:
                with self.assertRaisesRegex(ManualSourceIntakeError, "symlink"):
                    build_manual_source_intake(
                        attachment_root=root,
                        source_map_path=mapping_path,
                        output_root=output,
                    )

    def test_zip_traversal_and_companion_mismatch_fail_closed(self) -> None:
        data = json.dumps(_chime_rows(), ensure_ascii=False).encode()
        license_payload = b"MIT\n"
        contract = {
            "archive_component_id": "archive",
            "archive_comment": COMMIT,
            "repository_commit": COMMIT,
            "member_bindings": [
                {"member_path": "chime-main/data/chime_full.json", "component_id": "data"},
                {"member_path": "chime-main/LICENSE", "component_id": "license"},
            ],
        }
        with self.assertRaisesRegex(ManualSourceIntakeError, "unsafe"):
            _analyze_zip(
                _zip_bytes(data, license_payload, malicious=True),
                {"component_id": "archive"},
                companion_contract=contract,
                component_payloads={"data": data, "license": license_payload},
            )
        with self.assertRaisesRegex(ManualSourceIntakeError, "differs byte-for-byte"):
            _analyze_zip(
                _zip_bytes(data, license_payload),
                {"component_id": "archive"},
                companion_contract=contract,
                component_payloads={"data": b"[]", "license": license_payload},
            )

    def test_mhtml_related_start_selects_content_id_not_first_body_part(self) -> None:
        projection, metadata, candidate = _analyze_mhtml(
            _mhtml_bytes(select_root_with_start=True),
            {
                "snapshot_url": MHTML_URL,
                "acquired_at": ACQUIRED_AT,
                "title": "Archived page",
                "wikimedia_revision_id": None,
            },
        )
        self.assertIsNone(candidate)
        self.assertEqual(metadata["root_selection"], "content-type-start")
        self.assertEqual(metadata["related_start"], "<root@mhtml.test>")
        self.assertNotIn(b"first body part", projection)

    def test_build_replay_dedup_and_chime_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as output:
            root = Path(temporary)
            mapping_path, _ = _fixture(root)
            with mock.patch.object(
                socket,
                "create_connection",
                side_effect=AssertionError("network access is forbidden"),
            ):
                first = build_manual_source_intake(
                    attachment_root=root,
                    source_map_path=mapping_path,
                    output_root=output,
                )
                second = build_manual_source_intake(
                    attachment_root=root,
                    source_map_path=mapping_path,
                    output_root=output,
                )
            self.assertEqual(first["intake_id"], second["intake_id"])
            self.assertEqual(first["payload_manifest_sha256"], second["payload_manifest_sha256"])
            replay = validate_manual_source_intake(first["target"])
            self.assertEqual(replay["manifest"]["summary"]["alias_count"], 5)
            self.assertEqual(replay["manifest"]["summary"]["component_count"], 4)
            self.assertEqual(replay["manifest"]["summary"]["unique_raw_payload_count"], 4)
            self.assertEqual(replay["manifest"]["summary"]["deduplicated_alias_count"], 1)
            components = {
                row["component_id"]: row for row in load_normalized_components(first["target"])
            }
            page = components["page-source-page"]
            self.assertEqual(len(page["aliases"]), 2)
            self.assertEqual(page["format_metadata"]["snapshot_content_location"], MHTML_URL)
            self.assertEqual(page["format_metadata"]["date_iso8601"], ACQUIRED_AT)
            self.assertEqual(page["format_metadata"]["root_selection"], "first-body-part")
            self.assertEqual(page["format_metadata"]["duplicate_location_count"], 1)
            page_text = (
                Path(first["target"]) / page["normalized_text_file"]
            ).read_text(encoding="utf-8")
            self.assertNotIn("child frame text", page_text)
            chime = components["chime-data-json"]
            candidate = chime["candidate_projection"]
            self.assertEqual(candidate["schema_version"], CHIME_PROJECTION_SCHEMA_VERSION)
            self.assertEqual(candidate["count"], 2)
            projection = json.loads(
                (Path(first["target"]) / candidate["file"]).read_text(encoding="utf-8")
            )
            self.assertEqual([row["source_row_ordinal"] for row in projection["rows"]], [1, 2])
            self.assertNotIn("profanity", projection["rows"][0])
            self.assertNotIn("offense", projection["rows"][0])
            self.assertNotIn("examples", projection["rows"][0])
            self.assertFalse(replay["manifest"]["network_access_performed"])
            self.assertFalse(replay["manifest"]["promotion"]["g3_catalog_applied"])
            dependency = replay["manifest"]["dependency_binding"]
            for field in (
                "builder_implementation_sha256",
                "dependency_lock_sha256",
                "requirements_lock_sha256",
                "source_map_schema_sha256",
                "intake_schema_sha256",
            ):
                self.assertRegex(dependency[field], r"^[0-9a-f]{64}$")

    def test_payload_tamper_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as output:
            root = Path(temporary)
            mapping_path, _ = _fixture(root)
            report = build_manual_source_intake(
                attachment_root=root,
                source_map_path=mapping_path,
                output_root=output,
            )
            target = Path(report["target"])
            normalized = target / "normalized/page-source-page.txt"
            normalized.write_bytes(normalized.read_bytes() + b"tamper")
            with self.assertRaises(ManualSourceIntakeError):
                validate_manual_source_intake(target)

    @unittest.skipUnless(importlib.util.find_spec("pypdf"), "fixed pypdf environment unavailable")
    def test_fixed_pdf_backend_extracts_text_and_rejects_open_action(self) -> None:
        from pypdf import PdfWriter
        from pypdf.generic import (
            DecodedStreamObject,
            DictionaryObject,
            NameObject,
            TextStringObject,
        )

        writer = PdfWriter()
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 72 720 Td (Hello PDF) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
        output = io.BytesIO()
        writer.write(output)
        projection, metadata = normalize_publisher_pdf_bytes(
            output.getvalue(),
            component_id="test-pdf",
            requested_url="https://example.com/test.pdf",
            final_url="https://example.com/test.pdf",
        )
        self.assertEqual(projection, b"Hello PDF\n")
        self.assertEqual(metadata["page_count"], 1)
        self.assertEqual(metadata["pages"][0]["start_offset"], 0)
        self.assertEqual(metadata["pages"][0]["end_offset"], 9)

        writer.root_object[NameObject("/OpenAction")] = DictionaryObject(
            {
                NameObject("/S"): NameObject("/JavaScript"),
                NameObject("/JS"): TextStringObject("app.alert('x')"),
            }
        )
        dangerous = io.BytesIO()
        writer.write(dangerous)
        with self.assertRaisesRegex(ManualSourceIntakeError, "active|forbidden"):
            normalize_publisher_pdf_bytes(
                dangerous.getvalue(),
                component_id="test-pdf",
                requested_url="https://example.com/test.pdf",
                final_url="https://example.com/test.pdf",
            )

    def test_schemas_are_valid_json_and_cli_help_is_available(self) -> None:
        for schema in (
            ROOT / "schemas/wp3_g3_manual_source_map_v1.schema.json",
            ROOT / "schemas/wp3_g3_manual_source_intake_v1.schema.json",
        ):
            value = json.loads(schema.read_text(encoding="utf-8"))
            self.assertEqual(value["$schema"], "https://json-schema.org/draft/2020-12/schema")


if __name__ == "__main__":
    unittest.main()
