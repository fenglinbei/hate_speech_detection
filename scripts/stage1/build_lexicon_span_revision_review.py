#!/usr/bin/env python3
"""Build or validate the offline 49-case exact-span revision package."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.stage1.build_lexicon_candidate_review import (
    canonical_bytes,
    load_json,
    make_zip,
    sha256_bytes,
    sha256_file,
    validate_package as validate_parent_package,
    write_json,
)


SCHEMA_VERSION = "stage1-lexicon-span-revision-package/v1"
ANNOTATION_SCHEMA_VERSION = "stage1-lexicon-span-revision-annotation/v1"
EXPECTED_PARENT_PACKAGE_ID = (
    "lcgate-2bb8d8fa91a79500a5e2cce5cf5e968b47436842dc24eafb5717f858011142d6"
)
EXPECTED_CASE_COUNT = 49
SOURCE_SURFACE_FORMS = frozenset({"fragment", "sentence"})
SPAN_SCOPES = frozenset({"standalone_group_term", "productive_stem"})
INTERFACE_SOURCES = {
    "index.html": REPOSITORY_ROOT / "tools/lexicon_span_revision_review/index.html",
    "app.js": REPOSITORY_ROOT / "tools/lexicon_span_revision_review/app.js",
    "span.css": REPOSITORY_ROOT / "tools/lexicon_span_revision_review/span.css",
    "styles.css": REPOSITORY_ROOT / "tools/lexicon_candidate_review/styles.css",
}


class SpanRevisionPackageError(RuntimeError):
    """Raised when a span-revision package is invalid."""


def annotation_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": ANNOTATION_SCHEMA_VERSION,
        "type": "object",
        "required": [
            "schema_version",
            "package_id",
            "reviewer_id",
            "review_status",
            "annotations",
        ],
        "properties": {
            "schema_version": {"const": ANNOTATION_SCHEMA_VERSION},
            "package_id": {"type": "string", "pattern": "^lcsr-[0-9a-f]{64}$"},
            "reviewer_id": {"type": "string", "minLength": 1, "maxLength": 80},
            "review_status": {"enum": ["partial", "complete"]},
            "exported_at": {"type": "string"},
            "annotations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "case_id",
                        "source_case_id",
                        "original_term",
                        "decision",
                        "spans",
                        "notes",
                        "saved_at",
                    ],
                    "properties": {
                        "case_id": {"type": "string", "pattern": "^LSR49-[0-9]{3}$"},
                        "source_case_id": {"type": "string", "pattern": "^LCG80-[0-9]{3}$"},
                        "original_term": {"type": "string", "minLength": 1},
                        "decision": {"enum": ["no_valid_span", "corrected_spans"]},
                        "spans": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": [
                                    "surface",
                                    "scope",
                                    "context_index",
                                    "context_record_id",
                                ],
                                "properties": {
                                    "surface": {"type": "string", "minLength": 1, "maxLength": 80},
                                    "scope": {"enum": sorted(SPAN_SCOPES)},
                                    "context_index": {"type": "integer", "minimum": 0},
                                    "context_record_id": {"type": "string", "minLength": 1},
                                },
                                "additionalProperties": False,
                            },
                        },
                        "notes": {"type": "string", "maxLength": 1000},
                        "saved_at": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": True,
    }


def validate_source_annotations(
    annotations: Mapping[str, Any],
    *,
    parent_cases: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if (
        annotations.get("schema_version")
        != "stage1-lexicon-candidate-annotation/v1"
        or annotations.get("package_id") != EXPECTED_PARENT_PACKAGE_ID
        or annotations.get("review_status") != "complete"
        or annotations.get("progress") != {"complete": 80, "total": 80}
        or not isinstance(annotations.get("annotations"), list)
    ):
        raise SpanRevisionPackageError("source annotations are not the complete frozen 80-case result")
    source_by_id: dict[str, dict[str, Any]] = {}
    parent_by_id = {case["case_id"]: case for case in parent_cases}
    for row in annotations["annotations"]:
        if not isinstance(row, Mapping):
            raise SpanRevisionPackageError("source annotation row is not an object")
        case_id = row.get("case_id")
        parent = parent_by_id.get(case_id)
        if (
            not isinstance(case_id, str)
            or parent is None
            or case_id in source_by_id
            or row.get("term") != parent.get("term")
            or row.get("surface_form") not in {
                "complete",
                "fragment",
                "sentence",
                "uncertain",
            }
        ):
            raise SpanRevisionPackageError(f"invalid source annotation row: {case_id!r}")
        source_by_id[case_id] = dict(row)
    if set(source_by_id) != set(parent_by_id):
        raise SpanRevisionPackageError("source annotation case frame is incomplete")
    return source_by_id


def render_readme(manifest: Mapping[str, Any]) -> str:
    return f"""# 49-case 正确 span 人工修订包

本包完全离线运行。它只包含上一轮 80-case 中被标为碎片或句子的 49 条记录。

## 使用

1. 解压 ZIP，保持文件相对位置不变。
2. 直接打开 `index.html`。
3. 每条选择“无有效词条”，或选择上下文并填写一个或多个正确 span。
4. 正确 span 必须逐字、连续出现在所选上下文中；页面会在保存时校验。
5. 完成 49 条后导出 `lexicon_span_revision_annotations.json` 并交回。

正确 span 只收录 `独立群体词` 或 `可生产词干`。如果句中只有普通词、通用辱骂、
行为描述，或现有上下文不足以确定稳定词条，请选择“无有效词条”。

## 冻结信息

- package ID: `{manifest['package_id']}`
- parent package ID: `{manifest['source']['parent_package_id']}`
- source annotation SHA-256: `{manifest['source']['source_annotations_sha256']}`
- case count: `{manifest['sample']['case_count']}`
- source surface forms: fragment 28 / sentence 21

页面不展示候选分数、数据标签、旧模型判断或 Web 结果，不发起网络请求。
`manifest.json`、`cases.json`、`audit/source_annotations.json` 和 `SHA256SUMS`
用于复核来源与包完整性。
"""


def validate_package(package_dir: Path) -> dict[str, Any]:
    manifest = load_json(package_dir / "manifest.json")
    cases = load_json(package_dir / "cases.json")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("source", {}).get("parent_package_id")
        != EXPECTED_PARENT_PACKAGE_ID
    ):
        raise SpanRevisionPackageError("span-revision manifest is invalid")
    if (
        not isinstance(cases, list)
        or len(cases) != EXPECTED_CASE_COUNT
        or len({case.get("case_id") for case in cases}) != EXPECTED_CASE_COUNT
        or len({case.get("source_case_id") for case in cases}) != EXPECTED_CASE_COUNT
    ):
        raise SpanRevisionPackageError("span-revision package must contain 49 unique cases")
    surface_counts = Counter(case.get("source_surface_form") for case in cases)
    if surface_counts != Counter({"fragment": 28, "sentence": 21}):
        raise SpanRevisionPackageError("source fragment/sentence composition drifted")
    for case in cases:
        if not case.get("contexts"):
            raise SpanRevisionPackageError("every revision case needs at least one context")
        record_ids = [context.get("record_id") for context in case["contexts"]]
        if any(not isinstance(value, str) or not value for value in record_ids):
            raise SpanRevisionPackageError("revision context lacks its frozen record ID")
    checksum_rows = {}
    for line in (package_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checksum_rows[relative] = digest
    for relative, expected in checksum_rows.items():
        path = package_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise SpanRevisionPackageError(f"checksum mismatch: {relative}")
    cases_js = (package_dir / "cases.js").read_text(encoding="utf-8")
    if not cases_js.startswith("window.LEXICON_SPAN_REVISION_PACKAGE = "):
        raise SpanRevisionPackageError("cases.js does not expose the offline revision package")
    return {
        "status": "ok",
        "package_id": manifest["package_id"],
        "case_count": len(cases),
        "source_surface_counts": dict(sorted(surface_counts.items())),
        "checksum_file_count": len(checksum_rows),
        "external_provider_calls": 0,
    }


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    validate_parent_package(args.parent_package)
    parent_manifest = load_json(args.parent_package / "manifest.json")
    parent_cases = load_json(args.parent_package / "cases.json")
    if parent_manifest.get("package_id") != EXPECTED_PARENT_PACKAGE_ID:
        raise SpanRevisionPackageError("unexpected parent package ID")
    source_annotations = load_json(args.annotations)
    source_by_id = validate_source_annotations(
        source_annotations,
        parent_cases=parent_cases,
    )
    parent_by_id = {case["case_id"]: case for case in parent_cases}
    selected_source_ids = [
        case["case_id"]
        for case in parent_cases
        if source_by_id[case["case_id"]]["surface_form"] in SOURCE_SURFACE_FORMS
    ]
    if len(selected_source_ids) != EXPECTED_CASE_COUNT:
        raise SpanRevisionPackageError(
            f"expected 49 fragment/sentence rows, found {len(selected_source_ids)}"
        )

    cases: list[dict[str, Any]] = []
    for index, source_case_id in enumerate(selected_source_ids, start=1):
        parent = parent_by_id[source_case_id]
        source_row = source_by_id[source_case_id]
        cases.append(
            {
                "case_id": f"LSR49-{index:03d}",
                "source_case_id": source_case_id,
                "original_term": parent["term"],
                "source_surface_form": source_row["surface_form"],
                "contexts": parent["contexts"],
            }
        )

    interface_sha256s = {
        destination: sha256_file(source) for destination, source in INTERFACE_SOURCES.items()
    }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "parent_package_id": parent_manifest["package_id"],
            "parent_sample_sha256": parent_manifest["sample"]["sample_sha256"],
            "source_annotations_sha256": sha256_file(args.annotations),
            "source_annotation_schema": source_annotations["schema_version"],
            "source_review_status": "complete",
        },
        "sample": {
            "selection_rule": "surface_form in {fragment,sentence}/v1",
            "case_count": len(cases),
            "surface_form_counts": dict(
                sorted(Counter(case["source_surface_form"] for case in cases).items())
            ),
            "cases_sha256": sha256_bytes(canonical_bytes(cases)),
        },
        "annotation_protocol": {
            "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
            "annotation_schema_sha256": sha256_bytes(canonical_bytes(annotation_schema())),
            "builder_code_sha256": sha256_file(Path(__file__)),
            "interface_sha256s": interface_sha256s,
            "reviewer_count": 1,
            "span_must_be_exact_context_substring": True,
            "allowed_span_scopes": sorted(SPAN_SCOPES),
        },
        "audit": {
            "external_provider_calls": 0,
            "source_cases_missing": 0,
            "source_case_duplicates": 0,
        },
    }
    package_id = "lcsr-" + sha256_bytes(canonical_bytes(identity))
    manifest = {**identity, "package_id": package_id}
    package_dir = args.output_root / f"lexicon_span_revision_49_{package_id[5:17]}"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    (package_dir / "audit").mkdir(parents=True)
    for destination, source in INTERFACE_SOURCES.items():
        shutil.copyfile(source, package_dir / destination)
    write_json(package_dir / "manifest.json", manifest)
    write_json(package_dir / "cases.json", cases)
    write_json(package_dir / "annotation_results.schema.json", annotation_schema())
    write_json(package_dir / "audit/source_annotations.json", source_annotations)
    encoded = json.dumps(
        {"manifest": manifest, "cases": cases},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    (package_dir / "cases.js").write_text(
        f"window.LEXICON_SPAN_REVISION_PACKAGE = {encoded};\n",
        encoding="utf-8",
    )
    (package_dir / "README.md").write_text(render_readme(manifest), encoding="utf-8")
    checksum_paths = sorted(
        path
        for path in package_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (package_dir / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(package_dir).as_posix()}\n"
            for path in checksum_paths
        ),
        encoding="utf-8",
    )
    validation = validate_package(package_dir)
    zip_path = make_zip(package_dir)
    return {
        **validation,
        "package_dir": str(package_dir),
        "zip_path": str(zip_path),
        "zip_sha256": sha256_file(zip_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build the frozen span-revision package")
    build.add_argument("--annotations", required=True, type=Path)
    build.add_argument(
        "--parent-package",
        type=Path,
        default=REPOSITORY_ROOT
        / "exps/causal_context/stage1_p0/review_packages/lexicon_candidate_gate_80_2bb8d8fa91a7",
    )
    build.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "exps/causal_context/stage1_p0/review_packages",
    )
    check = subparsers.add_parser("check", help="validate an existing package directory")
    check.add_argument("package_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_package(args) if args.command == "build" else validate_package(args.package_dir)
    except (SpanRevisionPackageError, OSError, ValueError, KeyError) as exc:
        print(f"[lexicon-span-revision] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
