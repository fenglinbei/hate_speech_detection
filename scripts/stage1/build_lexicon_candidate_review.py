#!/usr/bin/env python3
"""Build or validate the offline 80-case lexicon-candidate review package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from build_lex.llm_lexicon_builder import build_candidates, select_candidates
from build_lex.train_only import resolve_train_input


SCHEMA_VERSION = "stage1-lexicon-candidate-review-package/v1"
ANNOTATION_SCHEMA_VERSION = "stage1-lexicon-candidate-annotation/v1"
SAMPLING_POLICY_ID = "track-quantile-hash-sample/v1"
SAMPLING_SEED = "stage1-p0-lexicon-candidate-gate-80/v1"
TRACK_QUOTAS = {
    "annotation_anchor": 13,
    "variant_slang": 39,
    "group_reference": 3,
    "ngram_backoff": 24,
    "contrastive_phrase": 1,
}
TEMPLATE_FILES = ("index.html", "app.js", "styles.css")


class ReviewPackageError(RuntimeError):
    """Raised when a review package cannot be built or validated."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def track_quantile_sample(
    selected: Sequence[Any],
    *,
    frame_sha256: str,
) -> list[tuple[int, Any]]:
    ranked = list(enumerate(selected, start=1))
    by_track: dict[str, list[tuple[int, Any]]] = {}
    for item in ranked:
        by_track.setdefault(item[1].track, []).append(item)
    if set(by_track) != set(TRACK_QUOTAS):
        raise ReviewPackageError(
            "selected candidate tracks drifted: "
            f"expected={sorted(TRACK_QUOTAS)} actual={sorted(by_track)}"
        )

    sampled: list[tuple[int, Any]] = []
    for track, quota in TRACK_QUOTAS.items():
        population = by_track[track]
        if len(population) < quota:
            raise ReviewPackageError(
                f"track {track!r} has {len(population)} candidates, below quota {quota}"
            )
        for bin_index in range(quota):
            start = bin_index * len(population) // quota
            end = (bin_index + 1) * len(population) // quota
            width = end - start
            if width <= 0:
                raise ReviewPackageError("sampling quantile bin is empty")
            selector = sha256_bytes(
                f"{SAMPLING_SEED}|{frame_sha256}|{track}|{bin_index}".encode("utf-8")
            )
            sampled.append(population[start + int(selector[:16], 16) % width])

    if len(sampled) != sum(TRACK_QUOTAS.values()):
        raise ReviewPackageError("sample count does not match the frozen quota")
    if len({candidate.term for _rank, candidate in sampled}) != len(sampled):
        raise ReviewPackageError("sampled candidate terms are not unique")
    sampled.sort(
        key=lambda item: sha256_bytes(
            f"{SAMPLING_SEED}|review-order|{frame_sha256}|{item[1].term}".encode(
                "utf-8"
            )
        )
    )
    return sampled


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
            "package_id": {"type": "string", "pattern": "^lcgate-[0-9a-f]{64}$"},
            "reviewer_id": {"type": "string", "minLength": 1, "maxLength": 80},
            "review_status": {"enum": ["partial", "complete"]},
            "exported_at": {"type": "string"},
            "annotations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "case_id",
                        "term",
                        "surface_form",
                        "term_scope",
                        "provider_eligible",
                        "confidence",
                        "saved_at",
                    ],
                    "properties": {
                        "case_id": {"type": "string", "pattern": "^LCG80-[0-9]{3}$"},
                        "term": {"type": "string", "minLength": 1},
                        "surface_form": {
                            "enum": ["complete", "fragment", "sentence", "uncertain"]
                        },
                        "term_scope": {
                            "enum": [
                                "standalone_group_term",
                                "productive_stem",
                                "phrase_only",
                                "generic_abuse",
                                "behavior_or_phenomenon",
                                "non_hateful_or_other",
                                "context_fragment",
                                "uncertain",
                            ]
                        },
                        "provider_eligible": {"enum": ["yes", "no", "uncertain"]},
                        "confidence": {"enum": ["high", "medium", "low"]},
                        "notes": {"type": "string", "maxLength": 1000},
                        "saved_at": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": True,
    }


def render_readme(manifest: Mapping[str, Any]) -> str:
    return f"""# 词典候选 80-case 离线人工门

本包完全离线运行，不需要安装依赖，也不会发起网络请求。

## 使用

1. 解压 ZIP，并保持包内文件相对位置不变。
2. 用 Chrome、Edge 或 Firefox 直接打开 `index.html`。
3. 标注会自动保存在当前浏览器；建议中途使用“导出备份”。
4. 完成 80 条后点击“导出完整结果”，得到 `lexicon_candidate_annotations.json`。
5. 如需换电脑或浏览器，可用页面的“导入结果/备份”恢复。

只判断 exact candidate，不要把整句标签、相邻辱骂词或主题类别投射到候选本身。
页面故意不显示候选排序、分数、语料标签、模型票据和 Web 结果。

## 判定口径

- `完整词/短语`：形式完整，可独立分析。
- `碎片或句子`：截断 ngram、语法残片或整句。
- 只有 `独立群体词` 和 `可生产词干` 会自动导出为 `provider_eligible=yes`。
- `仅完整短语成立`、`通用辱骂`、`行为/现象`、`非仇恨/其他`、`上下文碎片`均为 `no`。
- 无法可靠判断时选择“不确定”，不要勉强二选一。

## 冻结信息

- package ID: `{manifest['package_id']}`
- fit records: `{manifest['source']['fit_record_count']}`
- provider-bound candidate frame: `{manifest['candidate_frame']['selected_count']}`
- sample size: `{manifest['sample']['sample_size']}`
- sampling policy: `{manifest['sample']['policy_id']}`

`audit/candidate_sample.jsonl` 和 `manifest.json` 用于复核抽样与来源，不是页面标注输入。
`SHA256SUMS` 可用于检查包内文件是否被改动。
"""


def make_zip(package_dir: Path) -> Path:
    zip_path = package_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = Path(package_dir.name) / path.relative_to(package_dir)
            info = zipfile.ZipInfo(str(relative).replace("\\", "/"))
            info.date_time = (2026, 8, 26, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return zip_path


def validate_package(package_dir: Path) -> dict[str, Any]:
    manifest = load_json(package_dir / "manifest.json")
    cases = load_json(package_dir / "cases.json")
    checksums = {}
    for line in (package_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checksums[relative] = digest
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ReviewPackageError("package schema is invalid")
    if len(cases) != 80 or len({case.get("case_id") for case in cases}) != 80:
        raise ReviewPackageError("package must contain 80 unique cases")
    if len({case.get("term") for case in cases}) != 80:
        raise ReviewPackageError("package terms must be unique")
    if Counter(case.get("sampling", {}).get("track") for case in cases) != Counter(
        TRACK_QUOTAS
    ):
        raise ReviewPackageError("sample track quota drifted")
    if any(not case.get("contexts") for case in cases):
        raise ReviewPackageError("every case must have at least one fit context")
    for relative, expected in checksums.items():
        path = package_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ReviewPackageError(f"checksum mismatch: {relative}")
    cases_js = (package_dir / "cases.js").read_text(encoding="utf-8")
    if not cases_js.startswith("window.LEXICON_REVIEW_PACKAGE = "):
        raise ReviewPackageError("cases.js does not expose the offline package")
    if manifest.get("audit", {}).get("calibration_context_intersection_count") != 0:
        raise ReviewPackageError("calibration context leaked into the review package")
    return {
        "status": "ok",
        "package_id": manifest["package_id"],
        "case_count": len(cases),
        "track_counts": dict(
            sorted(Counter(case["sampling"]["track"] for case in cases).items())
        ),
        "checksum_file_count": len(checksums),
        "calibration_context_intersection_count": 0,
    }


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.config)
    candidate_settings = config.get("candidate_settings")
    if not isinstance(candidate_settings, Mapping):
        raise ReviewPackageError("config lacks candidate_settings")
    template_root = REPOSITORY_ROOT / "tools/lexicon_candidate_review"
    template_sha256s = {
        name: sha256_file(template_root / name) for name in TEMPLATE_FILES
    }
    train_input = resolve_train_input(
        data_ref=args.data_ref,
        train_partition_ref=args.train_partition_ref,
        workspace_root=args.workspace_root,
        formal=True,
    )
    corpus = build_candidates(
        "full",
        list(train_input.records),
        dict(candidate_settings),
        input_paths=[str(train_input.train_path)],
        show_progress=False,
    )
    selected = select_candidates(corpus, dict(candidate_settings))
    candidate_frame = [
        {"rank": rank, **candidate.to_payload()}
        for rank, candidate in enumerate(selected, start=1)
    ]
    frame_sha256 = sha256_bytes(canonical_bytes(candidate_frame))
    sampled = track_quantile_sample(selected, frame_sha256=frame_sha256)

    cases: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    sampled_context_ids: set[str] = set()
    for review_index, (source_rank, candidate) in enumerate(sampled, start=1):
        case_id = f"LCG80-{review_index:03d}"
        contexts = []
        for context in candidate.sample_contexts:
            record_id = str(context.get("id"))
            sampled_context_ids.add(record_id)
            contexts.append(
                {
                    "record_id": record_id,
                    "source": str(context.get("source") or "context"),
                    "content": str(context.get("content") or ""),
                }
            )
        cases.append(
            {
                "case_id": case_id,
                "term": candidate.term,
                "language": candidate.language,
                "contexts": contexts,
                "sampling": {
                    "track": candidate.track,
                    "source_rank": source_rank,
                },
            }
        )
        audit_rows.append(
            {
                "case_id": case_id,
                "source_rank": source_rank,
                **candidate.to_payload(),
            }
        )

    fit_ids = set(train_input.record_ids)
    calibration_ids = set(train_input.forbidden_ids)
    if not sampled_context_ids.issubset(fit_ids):
        raise ReviewPackageError("sample context is outside the frozen fit partition")
    calibration_overlap = sampled_context_ids.intersection(calibration_ids)
    if calibration_overlap:
        raise ReviewPackageError("sample context intersects the calibration partition")

    identity = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "data_dependency": train_input.data_ref,
            "train_partition_dependency": train_input.train_partition_dependency,
            "fit_record_count": len(train_input.records),
            "fit_ids_sha256": train_input.train_ids_sha256,
            "fit_data_sha256": train_input.train_data_sha256,
        },
        "candidate_frame": {
            "raw_count": len(corpus.candidates),
            "selected_count": len(selected),
            "selected_track_counts": dict(
                sorted(Counter(candidate.track for candidate in selected).items())
            ),
            "frame_sha256": frame_sha256,
            "candidate_config_sha256": sha256_bytes(canonical_bytes(candidate_settings)),
            "builder_code_sha256": sha256_file(
                REPOSITORY_ROOT / "src/build_lex/llm_lexicon_builder.py"
            ),
        },
        "sample": {
            "policy_id": SAMPLING_POLICY_ID,
            "seed": SAMPLING_SEED,
            "sample_size": len(cases),
            "track_quotas": TRACK_QUOTAS,
            "sample_sha256": sha256_bytes(canonical_bytes(cases)),
        },
        "annotation_protocol": {
            "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
            "annotation_schema_sha256": sha256_bytes(
                canonical_bytes(annotation_schema())
            ),
            "package_builder_code_sha256": sha256_file(Path(__file__)),
            "interface_sha256s": template_sha256s,
            "reviewer_count": 1,
        },
        "audit": {
            "sample_context_record_count": len(sampled_context_ids),
            "all_contexts_in_fit": True,
            "calibration_context_intersection_count": 0,
            "external_provider_calls": 0,
        },
    }
    package_id = "lcgate-" + sha256_bytes(canonical_bytes(identity))
    manifest = {**identity, "package_id": package_id}

    package_dir = args.output_root / f"lexicon_candidate_gate_80_{package_id[7:19]}"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    (package_dir / "audit").mkdir(parents=True)
    for name in TEMPLATE_FILES:
        shutil.copyfile(template_root / name, package_dir / name)
    write_json(package_dir / "manifest.json", manifest)
    write_json(package_dir / "cases.json", cases)
    write_json(package_dir / "annotation_results.schema.json", annotation_schema())
    package_payload = {"manifest": manifest, "cases": cases}
    encoded = json.dumps(package_payload, ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    (package_dir / "cases.js").write_text(
        f"window.LEXICON_REVIEW_PACKAGE = {encoded};\n", encoding="utf-8"
    )
    (package_dir / "README.md").write_text(render_readme(manifest), encoding="utf-8")
    with (package_dir / "audit/candidate_sample.jsonl").open("w", encoding="utf-8") as handle:
        for row in audit_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

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
        "fit_record_count": len(train_input.records),
        "raw_candidate_count": len(corpus.candidates),
        "selected_candidate_count": len(selected),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build the frozen offline review package")
    build.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "config/stage1/lexicon_train_only.json",
    )
    build.add_argument(
        "--data-ref",
        type=Path,
        default=REPOSITORY_ROOT / "exps/causal_context/stage1_p0/refs/data_ref.json",
    )
    build.add_argument(
        "--train-partition-ref",
        type=Path,
        default=REPOSITORY_ROOT
        / "exps/causal_context/stage1_p0/refs/train_partition_ref.json",
    )
    build.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    build.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT
        / "exps/causal_context/stage1_p0/review_packages",
    )
    check = subparsers.add_parser("check", help="validate an existing package directory")
    check.add_argument("package_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_package(args) if args.command == "build" else validate_package(args.package_dir)
    except (ReviewPackageError, OSError, ValueError, KeyError) as exc:
        print(f"[lexicon-candidate-review] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
