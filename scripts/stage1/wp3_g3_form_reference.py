#!/usr/bin/env python3
"""Build and operate the WP3 G3 public-form reference lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from build_lex.terminology_g3_form_reference import (  # noqa: E402
    FRAME_ARTIFACT_KIND,
    REFERENCE_ARTIFACT_KIND,
    SOURCE_BUNDLE_ARTIFACT_KIND,
    G3FormReferenceError,
    build_form_review_frame,
    fetch_public_source_snapshots,
    finalize_form_reference,
    sync_public_sources,
    validate_form_reference,
    validate_form_review_frame,
    validate_public_source_bundle,
)
from build_lex.terminology_g3_source_catalog_v2 import (  # noqa: E402
    G3SourceCatalogV2Error,
    build_public_source_bundle_v2,
)
from data.training_artifacts import (  # noqa: E402
    TrainingArtifactError,
    load_json,
    resolve_locator_ref,
)


DEFAULT_ROOT = Path(
    "exps/causal_context/stage1_p0/wp3_candidate_generators_v2/g3_form_reference"
)
DEFAULT_CATALOG = Path("config/stage1/wp3_g3_public_source_catalog_v1.json")
DEFAULT_CATALOG_V2 = Path("config/stage1/wp3_g3_public_source_catalog_v2.json")
DEFAULT_PROFILE = Path("config/stage1/wp3_g3_profile_full_v2.json")
DEFAULT_REFERENCE_SCHEMA = Path("schemas/wp3_g3_form_reference_v1.schema.json")
DEFAULT_SOURCE_REF = DEFAULT_ROOT / "refs/public_source_bundle_ref.json"
DEFAULT_FRAME_REF = DEFAULT_ROOT / "refs/form_review_frame_ref.json"
DEFAULT_REFERENCE_REF = DEFAULT_ROOT / "refs/form_reference_ref.json"
DEFAULT_V1_PAIR_SEEDS = Path("config/stage1/wp3_g3_form_extraction_v1.json")
DEFAULT_V2_PAIR_SEEDS = Path("config/stage1/wp3_g3_form_relation_seeds_v2.json")


def _in_workspace(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _resolve(
    *, root: Path, explicit: Path | None, locator: Path, expected_kind: str
) -> Path:
    if explicit is not None:
        return _in_workspace(root, explicit).resolve()
    _, target = resolve_locator_ref(_in_workspace(root, locator), expected_kind)
    return target


def _base(command: argparse.ArgumentParser) -> None:
    command.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)


def _source(command: argparse.ArgumentParser) -> None:
    source = command.add_mutually_exclusive_group()
    source.add_argument("--source-bundle-dir", type=Path)
    source.add_argument("--source-bundle-ref", type=Path, default=DEFAULT_SOURCE_REF)


def _frame(command: argparse.ArgumentParser) -> None:
    source = command.add_mutually_exclusive_group()
    source.add_argument("--frame-dir", type=Path)
    source.add_argument("--frame-ref", type=Path, default=DEFAULT_FRAME_REF)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    sync = commands.add_parser(
        "sync-g3-public-sources",
        help="Fetch only the fixed seven-URL allowlist into a new snapshot directory.",
    )
    _base(sync)
    sync.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    sync.add_argument("--output-directory", type=Path, required=True)

    freeze = commands.add_parser(
        "freeze-g3-public-sources",
        help=(
            "Freeze synchronizer-produced snapshots, their required receipt, "
            "and a controlled extraction."
        ),
    )
    _base(freeze)
    freeze.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    freeze.add_argument("--snapshot-root", type=Path, required=True)
    freeze.add_argument("--snapshot-index", type=Path)
    freeze.add_argument("--extraction", type=Path, required=True)
    freeze.add_argument(
        "--output-root", type=Path, default=DEFAULT_ROOT / "public_source_bundles"
    )
    freeze.add_argument("--write-ref", type=Path, default=DEFAULT_SOURCE_REF)

    freeze_v2 = commands.add_parser(
        "freeze-g3-public-sources-v2",
        help=(
            "Build the closed 42-target v2 source bundle offline from a "
            "validated zero-proxy capture and validated manual intake."
        ),
    )
    _base(freeze_v2)
    freeze_v2.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_V2)
    freeze_v2.add_argument("--capture-dir", type=Path, required=True)
    freeze_v2.add_argument("--manual-intake-dir", type=Path, required=True)
    freeze_v2.add_argument(
        "--output-root", type=Path, default=DEFAULT_ROOT / "public_source_bundles"
    )
    freeze_v2.add_argument("--write-ref", type=Path, default=DEFAULT_SOURCE_REF)

    validate_source = commands.add_parser(
        "validate-g3-public-sources", help="Validate an immutable public-source bundle."
    )
    _base(validate_source)
    _source(validate_source)
    validate_source.add_argument("--allow-implementation-drift", action="store_true")

    build = commands.add_parser(
        "build-g3-reference", help="Build the immutable non-empty form-review frame."
    )
    _base(build)
    _source(build)
    build.add_argument("--output-root", type=Path, default=DEFAULT_ROOT / "review_frames")
    build.add_argument("--write-ref", type=Path, default=DEFAULT_FRAME_REF)
    build.add_argument(
        "--pair-seed",
        action="append",
        type=Path,
        default=None,
        help=(
            "Pair-only seed document for v2 relocation. Repeat for additional "
            "seed registers; defaults to the existing v1 form pairs."
        ),
    )

    validate_frame = commands.add_parser(
        "validate-g3-reference-frame", help="Validate the form-review frame."
    )
    _base(validate_frame)
    _source(validate_frame)
    _frame(validate_frame)
    validate_frame.add_argument("--allow-implementation-drift", action="store_true")

    serve = commands.add_parser(
        "serve-g3-reference", help="Serve the CAS-protected loopback form reviewer."
    )
    _base(serve)
    _source(serve)
    _frame(serve)
    serve.add_argument("--reviewer-id", required=True)
    serve.add_argument("--session-file", type=Path, default=DEFAULT_ROOT / "review_working/session.json")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8767)
    serve.add_argument(
        "--public-origin",
        help=(
            "Exact trusted HTTPS reverse-proxy origin, for example "
            "https://hsd.example."
        ),
    )
    serve.add_argument("--check", action="store_true")

    finalize = commands.add_parser(
        "finalize-g3-reference", help="Finalize a complete, defer-free human review."
    )
    _base(finalize)
    _source(finalize)
    _frame(finalize)
    finalize.add_argument("--reviewer-id", required=True)
    finalize.add_argument("--session-file", type=Path, default=DEFAULT_ROOT / "review_working/session.json")
    finalize.add_argument("--romanizer-profile", type=Path, default=DEFAULT_PROFILE)
    finalize.add_argument("--reference-schema", type=Path, default=DEFAULT_REFERENCE_SCHEMA)
    finalize.add_argument("--output-root", type=Path, default=DEFAULT_ROOT / "references")
    finalize.add_argument("--write-ref", type=Path, default=DEFAULT_REFERENCE_REF)

    validate_reference = commands.add_parser(
        "validate-g3-reference", help="Independently validate a finalized form reference."
    )
    _base(validate_reference)
    _source(validate_reference)
    _frame(validate_reference)
    reference = validate_reference.add_mutually_exclusive_group()
    reference.add_argument("--reference-dir", type=Path)
    reference.add_argument("--reference-ref", type=Path, default=DEFAULT_REFERENCE_REF)
    validate_reference.add_argument("--romanizer-profile", type=Path, default=DEFAULT_PROFILE)
    validate_reference.add_argument("--reference-schema", type=Path, default=DEFAULT_REFERENCE_SCHEMA)
    validate_reference.add_argument("--allow-implementation-drift", action="store_true")
    return parser


def _safe_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "source_bundle_id",
            "frame_id",
            "reference_id",
            "target",
            "payload_manifest_sha256",
        )
        if key in result
    } | {
        "scope": "development-only",
        "reference_role": "form-only-label-free-non-lexicon",
        "scientific_eligible": False,
        "sealed": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.workspace_root.resolve()
    try:
        if args.command == "sync-g3-public-sources":
            output = fetch_public_source_snapshots(
                workspace_root=root,
                catalog_path=_in_workspace(root, args.catalog),
                output_directory=_in_workspace(root, args.output_directory),
            )
        elif args.command == "freeze-g3-public-sources":
            snapshot_root = _in_workspace(root, args.snapshot_root)
            snapshot_index = (
                _in_workspace(root, args.snapshot_index)
                if args.snapshot_index is not None
                else snapshot_root / "snapshot_index.json"
            )
            output = _safe_result(
                sync_public_sources(
                    workspace_root=root,
                    catalog_path=_in_workspace(root, args.catalog),
                    snapshot_root=snapshot_root,
                    snapshot_index_path=snapshot_index,
                    sync_receipt_path=snapshot_root / "sync_receipt.json",
                    extraction_path=_in_workspace(root, args.extraction),
                    output_root=_in_workspace(root, args.output_root),
                    write_ref=_in_workspace(root, args.write_ref),
                )
            )
        elif args.command == "freeze-g3-public-sources-v2":
            output = _safe_result(
                build_public_source_bundle_v2(
                    workspace_root=root,
                    catalog_path=_in_workspace(root, args.catalog),
                    capture_directory=_in_workspace(root, args.capture_dir),
                    manual_intake_directory=_in_workspace(
                        root, args.manual_intake_dir
                    ),
                    output_root=_in_workspace(root, args.output_root),
                    write_ref=_in_workspace(root, args.write_ref),
                )
            )
        else:
            source_bundle = _resolve(
                root=root,
                explicit=args.source_bundle_dir,
                locator=args.source_bundle_ref,
                expected_kind=SOURCE_BUNDLE_ARTIFACT_KIND,
            )
            if args.command == "validate-g3-public-sources":
                source_manifest = load_json(source_bundle / "manifest.json")
                if (
                    isinstance(source_manifest, dict)
                    and source_manifest.get("schema_version")
                    == "wp3-g3-public-source-bundle/v2"
                ):
                    from build_lex.terminology_g3_source_catalog_v2 import (
                        validate_public_source_bundle_v2,
                    )

                    output = _safe_result(
                        validate_public_source_bundle_v2(
                            source_bundle,
                            workspace_root=root,
                            require_current_implementation=(
                                not args.allow_implementation_drift
                            ),
                        )
                    )
                else:
                    output = _safe_result(
                        validate_public_source_bundle(
                            source_bundle,
                            workspace_root=root,
                            require_current_implementation=(
                                not args.allow_implementation_drift
                            ),
                        )
                    )
            elif args.command == "build-g3-reference":
                source_manifest = load_json(source_bundle / "manifest.json")
                if (
                    isinstance(source_manifest, dict)
                    and source_manifest.get("schema_version")
                    == "wp3-g3-public-source-bundle/v2"
                ):
                    from build_lex.terminology_g3_relation_review import (
                        build_form_relation_review_frame_v2,
                    )

                    seed_paths = args.pair_seed or [
                        DEFAULT_V1_PAIR_SEEDS,
                        DEFAULT_V2_PAIR_SEEDS,
                    ]
                    output = _safe_result(
                        build_form_relation_review_frame_v2(
                            source_bundle_dir=source_bundle,
                            workspace_root=root,
                            output_root=_in_workspace(root, args.output_root),
                            seed_paths=[_in_workspace(root, path) for path in seed_paths],
                            write_ref=_in_workspace(root, args.write_ref),
                        )
                    )
                else:
                    output = _safe_result(
                        build_form_review_frame(
                            source_bundle_dir=source_bundle,
                            workspace_root=root,
                            output_root=_in_workspace(root, args.output_root),
                            write_ref=_in_workspace(root, args.write_ref),
                        )
                    )
            else:
                frame = _resolve(
                    root=root,
                    explicit=args.frame_dir,
                    locator=args.frame_ref,
                    expected_kind=FRAME_ARTIFACT_KIND,
                )
                if args.command == "validate-g3-reference-frame":
                    output = _safe_result(
                        validate_form_review_frame(
                            frame,
                            source_bundle_dir=source_bundle,
                            workspace_root=root,
                            require_current_implementation=not args.allow_implementation_drift,
                        )
                    )
                elif args.command == "serve-g3-reference":
                    from tools.wp3_g3_form_review_ui.server import run_server

                    return run_server(
                        workspace_root=root,
                        source_bundle_dir=source_bundle,
                        frame_dir=frame,
                        session_path=_in_workspace(root, args.session_file),
                        reviewer_id=args.reviewer_id,
                        host=args.host,
                        port=args.port,
                        check=args.check,
                        public_origin=args.public_origin,
                    )
                elif args.command == "finalize-g3-reference":
                    output = _safe_result(
                        finalize_form_reference(
                            frame_dir=frame,
                            source_bundle_dir=source_bundle,
                            workspace_root=root,
                            session_path=_in_workspace(root, args.session_file),
                            reviewer_id=args.reviewer_id,
                            romanizer_profile_path=_in_workspace(root, args.romanizer_profile),
                            output_root=_in_workspace(root, args.output_root),
                            reference_schema_path=_in_workspace(root, args.reference_schema),
                            write_ref=_in_workspace(root, args.write_ref),
                        )
                    )
                else:
                    reference = _resolve(
                        root=root,
                        explicit=args.reference_dir,
                        locator=args.reference_ref,
                        expected_kind=REFERENCE_ARTIFACT_KIND,
                    )
                    output = _safe_result(
                        validate_form_reference(
                            reference,
                            frame_dir=frame,
                            source_bundle_dir=source_bundle,
                            workspace_root=root,
                            romanizer_profile_path=_in_workspace(root, args.romanizer_profile),
                            reference_schema_path=_in_workspace(root, args.reference_schema),
                            require_current_implementation=not args.allow_implementation_drift,
                        )
                    )
    except (
        G3FormReferenceError,
        G3SourceCatalogV2Error,
        TrainingArtifactError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
