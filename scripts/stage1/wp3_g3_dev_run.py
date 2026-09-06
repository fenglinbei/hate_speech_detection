#!/usr/bin/env python3
"""Run and independently validate WP3 full G3 on the public S2.1 frame."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from build_lex.terminology_g3_development_run import (  # noqa: E402
    G3DevelopmentRunError,
    RUN_ARTIFACT_KIND,
    build_g3_development_run,
    validate_g3_development_run,
)
from build_lex.terminology_g3_form_reference import (  # noqa: E402
    REFERENCE_ARTIFACT_KIND,
)
from data.training_artifacts import (  # noqa: E402
    TrainingArtifactError,
    resolve_locator_ref,
)


PUBLIC_FRAME_ARTIFACT_KIND = "wp3-s21-development-frame"
DEFAULT_GENERATOR_ROOT = Path(
    "exps/causal_context/stage1_p0/wp3_candidate_generators_v2"
)
DEFAULT_REFERENCE_ROOT = DEFAULT_GENERATOR_ROOT / "g3_form_reference"
DEFAULT_FRAME_REF = DEFAULT_GENERATOR_ROOT / "refs/development_frame_ref.json"
DEFAULT_REFERENCE_REF = DEFAULT_REFERENCE_ROOT / "refs/form_reference_ref.json"
DEFAULT_PROFILE = Path("config/stage1/wp3_g3_profile_full_v2.json")
DEFAULT_RUN_ROOT = DEFAULT_GENERATOR_ROOT / "g3_development_runs"
DEFAULT_RUN_REF = DEFAULT_GENERATOR_ROOT / "refs/g3_development_run_ref.json"


def _in_workspace(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _resolve_input(
    *,
    root: Path,
    explicit_dir: Path | None,
    locator_ref: Path,
    expected_kind: str,
) -> Path:
    if explicit_dir is not None:
        target = _in_workspace(root, explicit_dir)
        if not target.is_dir() or target.is_symlink():
            raise G3DevelopmentRunError("explicit artifact directory is invalid")
        return target
    _, target = resolve_locator_ref(
        _in_workspace(root, locator_ref), expected_kind=expected_kind
    )
    return target


def _add_workspace(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)


def _add_frame(parser: argparse.ArgumentParser) -> None:
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--frame-dir", type=Path)
    choice.add_argument("--frame-ref", type=Path, default=DEFAULT_FRAME_REF)


def _add_reference(parser: argparse.ArgumentParser) -> None:
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--reference-dir", type=Path)
    choice.add_argument("--reference-ref", type=Path, default=DEFAULT_REFERENCE_REF)


def _add_run(parser: argparse.ArgumentParser) -> None:
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--run-dir", type=Path)
    choice.add_argument("--run-ref", type=Path, default=DEFAULT_RUN_REF)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run-g3-dev",
        help="Run full-v2 G3 over exactly the public 424-case development frame.",
    )
    _add_workspace(run)
    _add_frame(run)
    _add_reference(run)
    run.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    run.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    run.add_argument("--write-ref", type=Path, default=DEFAULT_RUN_REF)

    validate = commands.add_parser(
        "validate-g3-dev",
        help="Independently rerun and validate a G3 development credential.",
    )
    _add_workspace(validate)
    _add_frame(validate)
    _add_reference(validate)
    _add_run(validate)
    validate.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    return parser


def _safe_result(result: Mapping[str, Any]) -> dict[str, Any]:
    receipt = result.get("completion_receipt")
    return {
        "artifact_kind": RUN_ARTIFACT_KIND,
        "artifact_id": result.get("artifact_id"),
        "target": result.get("target"),
        "payload_manifest_sha256": result.get("payload_manifest_sha256"),
        "complete": result.get("complete"),
        "completion_receipt": receipt,
        "scope": "development-only",
        "scientific_eligible": False,
        "sealed": False,
    }
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.workspace_root.resolve()
    try:
        frame = _resolve_input(
            root=root,
            explicit_dir=args.frame_dir,
            locator_ref=args.frame_ref,
            expected_kind=PUBLIC_FRAME_ARTIFACT_KIND,
        )
        reference = _resolve_input(
            root=root,
            explicit_dir=args.reference_dir,
            locator_ref=args.reference_ref,
            expected_kind=REFERENCE_ARTIFACT_KIND,
        )
        profile = _in_workspace(root, args.profile)
        if args.command == "run-g3-dev":
            result = build_g3_development_run(
                workspace_root=root,
                reference_dir=reference,
                frame_dir=frame,
                profile_path=profile,
                output_root=_in_workspace(root, args.output_root),
                write_ref=_in_workspace(root, args.write_ref),
            )
        else:
            run = _resolve_input(
                root=root,
                explicit_dir=args.run_dir,
                locator_ref=args.run_ref,
                expected_kind=RUN_ARTIFACT_KIND,
            )
            result = validate_g3_development_run(
                run,
                workspace_root=root,
                reference_dir=reference,
                frame_dir=frame,
                profile_path=profile,
            )
    except (G3DevelopmentRunError, TrainingArtifactError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_safe_result(result), ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("complete") is True else 3


if __name__ == "__main__":
    raise SystemExit(main())
