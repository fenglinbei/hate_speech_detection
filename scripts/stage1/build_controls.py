#!/usr/bin/env python3
"""Build and validate immutable Stage 1 PL/PD control artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from data.control_manifest import (  # noqa: E402
    ControlManifestError,
    build_control_artifact,
    seal_test_control_artifact,
    validate_control_ref,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build frozen train-only PL/PD placebo controls."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-controls")
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--context-ref", required=True, type=Path)
    build.add_argument("--write-ref", required=True, type=Path)
    build.add_argument("--split", required=True, choices=("train", "dev"))
    build.add_argument("--target-root", type=Path)
    build.add_argument("--tokenizer-path", type=Path)
    build.add_argument("--tokenizer-revision")
    build.add_argument(
        "--allow-unavailable",
        action="store_true",
        help="Freeze unavailable masks (sealed-test/engineering only; formal dev defaults to fail closed).",
    )

    validate = subparsers.add_parser("validate-controls")
    validate.add_argument("--control-ref", required=True, type=Path)
    validate.add_argument("--context-ref", type=Path)
    validate.add_argument("--tokenizer-path", type=Path)
    validate.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    seal = subparsers.add_parser(
        "seal-test",
        help="Build test PL/PD from exact frozen dev config/code/tier policy.",
    )
    seal.add_argument("--test-context-ref", required=True, type=Path)
    seal.add_argument("--frozen-control-ref", required=True, type=Path)
    seal.add_argument("--write-ref", required=True, type=Path)
    seal.add_argument("--tokenizer-path", type=Path)
    seal.add_argument("--target-root", type=Path)
    seal.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    return parser


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_tokenizer(path: Path) -> Any:
    if not path.is_dir():
        raise ControlManifestError(f"tokenizer path is not a local directory: {path}")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise ControlManifestError("transformers is required to load the frozen tokenizer") from exc
    return AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=False,
    )


def _context_is_scientific(context_ref: Path, split: str) -> bool:
    locator = _read_json(context_ref)
    target = Path(str(locator.get("target_path", "")))
    meta = _read_json(target / f"context_manifest.{split}.meta.json")
    return isinstance(meta, Mapping) and meta.get("scientific_eligible") is True


def _control_is_scientific(control_ref: Path) -> bool:
    locator = _read_json(control_ref)
    target = Path(str(locator.get("target_path", "")))
    meta = _read_json(target / "control_manifest.meta.json")
    if not isinstance(meta, Mapping):
        raise ControlManifestError("stored control meta is not an object")
    split = meta.get("split")
    dependency = _read_json(target / "context_ref.json")
    context_target = target.parent.parent / str(dependency.get("logical_repo_path", ""))
    context_meta = _read_json(
        context_target / f"context_manifest.{split}.meta.json"
    )
    return split == "test" or (
        isinstance(context_meta, Mapping)
        and context_meta.get("scientific_eligible") is True
    )


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build-controls":
            config = _read_json(args.config)
            if not isinstance(config, Mapping):
                raise ControlManifestError("control config root must be an object")
            if _context_is_scientific(args.context_ref, args.split) and (
                args.tokenizer_path is not None or args.tokenizer_revision is not None
            ):
                raise ControlManifestError(
                    "scientific controls forbid CLI tokenizer path/revision overrides"
                )
            tokenizer = (
                _load_tokenizer(args.tokenizer_path.resolve())
                if args.tokenizer_path is not None
                else None
            )
            result = build_control_artifact(
                config=args.config,
                context_ref=args.context_ref,
                write_ref=args.write_ref,
                split=args.split,
                tokenizer=tokenizer,
                tokenizer_revision=args.tokenizer_revision,
                target_root=args.target_root,
                allow_unavailable=True if args.allow_unavailable else None,
            )
        elif args.command == "seal-test":
            if args.tokenizer_path is not None:
                raise ControlManifestError(
                    "seal-test forbids CLI tokenizer path overrides"
                )
            result = seal_test_control_artifact(
                test_context_ref=args.test_context_ref,
                frozen_control_ref=args.frozen_control_ref,
                write_ref=args.write_ref,
                target_root=args.target_root,
                workspace_root=args.workspace_root,
            )
        else:
            if _control_is_scientific(args.control_ref) and args.tokenizer_path is not None:
                raise ControlManifestError(
                    "scientific validation forbids CLI tokenizer path overrides"
                )
            tokenizer = (
                _load_tokenizer(args.tokenizer_path.resolve())
                if args.tokenizer_path is not None
                else None
            )
            result = validate_control_ref(
                args.control_ref,
                tokenizer=tokenizer,
                context_ref=args.context_ref,
                workspace_root=args.workspace_root,
            )
    except (ControlManifestError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
