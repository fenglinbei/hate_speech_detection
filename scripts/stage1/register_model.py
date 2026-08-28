#!/usr/bin/env python3
"""Register and resolve immutable Stage 1 base/checkpoint model artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from data.training_artifacts import TrainingArtifactError  # noqa: E402
from model.stage1_registry import (  # noqa: E402
    finalize_model_registry,
    register_base_model,
    register_legacy_model,
    register_trained_model,
    register_training_receipt,
    resolve_registered_model,
    validate_model_artifact,
    validate_model_registry,
    validate_training_receipt_artifact,
)


def _binding(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("binding must be MODEL_KEY=MODEL_REF")
    key, path = value.split("=", 1)
    if not key or not path:
        raise argparse.ArgumentTypeError("binding must be MODEL_KEY=MODEL_REF")
    return key, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    base = subparsers.add_parser("register-base", help="register a local base model")
    base.add_argument("--model-dir", required=True, type=Path)
    base.add_argument("--tokenizer-dir", type=Path)
    base.add_argument("--model-name", required=True)
    base.add_argument("--tokenizer-revision", required=True)
    base.add_argument("--environment-ref", required=True, type=Path)
    base.add_argument("--write-ref", required=True, type=Path)
    base.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    base.add_argument("--target-root", type=Path)

    legacy = subparsers.add_parser(
        "register-legacy",
        help="register the immutable Qwen2-era engineering smoke checkpoint",
    )
    legacy.add_argument("--checkpoint", required=True, type=Path)
    legacy.add_argument(
        "--composition",
        default="auto",
        choices=("auto", "full"),
        help="auto-detect or require a self-contained full checkpoint",
    )
    legacy.add_argument(
        "--tokenizer-root",
        default="same",
        help="tokenizer directory or the literal 'same' (default: checkpoint)",
    )
    legacy.add_argument(
        "--chat-template-source",
        default="auto",
        choices=("auto", "tokenizer-config", "chat-template-jinja"),
        help="assert the actual source of the tokenizer chat template",
    )
    legacy.add_argument("--environment-ref", required=True, type=Path)
    legacy.add_argument("--write-ref", required=True, type=Path)
    legacy.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    legacy.add_argument("--target-root", type=Path)

    receipt = subparsers.add_parser(
        "register-receipt", help="wrap a training receipt as a portable artifact"
    )
    receipt.add_argument("--receipt-json", required=True, type=Path)
    receipt.add_argument("--training-plan-ref", required=True, type=Path)
    receipt.add_argument("--schedule-ref", required=True, type=Path)
    receipt.add_argument("--base-model-ref", required=True, type=Path)
    receipt.add_argument("--environment-ref", required=True, type=Path)
    receipt.add_argument("--write-ref", required=True, type=Path)
    receipt.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    receipt.add_argument("--target-root", type=Path)

    trained = subparsers.add_parser(
        "register-formal", help="register one selected full/adapter checkpoint"
    )
    trained.add_argument("--checkpoint-dir", required=True, type=Path)
    trained.add_argument("--tokenizer-dir", type=Path)
    trained.add_argument("--checkpoint-format", required=True, choices=("full", "adapter"))
    trained.add_argument("--model-key", required=True)
    trained.add_argument("--training-plan-ref", required=True, type=Path)
    trained.add_argument("--schedule-ref", required=True, type=Path)
    trained.add_argument("--training-receipt-ref", required=True, type=Path)
    trained.add_argument("--base-model-ref", required=True, type=Path)
    trained.add_argument("--environment-ref", required=True, type=Path)
    trained.add_argument("--write-ref", required=True, type=Path)
    trained.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    trained.add_argument("--target-root", type=Path)

    finalize = subparsers.add_parser(
        "finalize-registry", help="freeze an exact plan-slot/model bijection"
    )
    finalize.add_argument(
        "--scope", required=True, choices=("engineering-smoke", "pilot", "formal")
    )
    finalize.add_argument("--training-plan-ref", required=True, type=Path)
    finalize.add_argument(
        "--binding",
        "--bind",
        dest="binding",
        action="append",
        required=True,
        type=_binding,
        metavar="KEY=REF",
    )
    finalize.add_argument("--write-ref", required=True, type=Path)
    finalize.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    finalize.add_argument("--target-root", type=Path)

    validate_model = subparsers.add_parser("validate-model")
    validate_model.add_argument("--model-ref", required=True, type=Path)
    validate_model.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)

    validate_receipt = subparsers.add_parser("validate-receipt")
    validate_receipt.add_argument("--receipt-ref", required=True, type=Path)
    validate_receipt.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)

    validate_registry = subparsers.add_parser("validate-registry")
    validate_registry.add_argument(
        "--registry-ref",
        "--model-registry-ref",
        dest="registry_ref",
        required=True,
        type=Path,
    )
    validate_registry.add_argument(
        "--scope", choices=("engineering-smoke", "pilot", "formal")
    )
    validate_registry.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)

    resolve = subparsers.add_parser(
        "resolve", help="re-hash and resolve one registered runtime model"
    )
    resolve.add_argument(
        "--registry-ref",
        "--model-registry-ref",
        dest="registry_ref",
        required=True,
        type=Path,
    )
    resolve.add_argument("--model-key", required=True)
    resolve.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "register-base":
            result = register_base_model(
                model_dir=args.model_dir,
                tokenizer_dir=args.tokenizer_dir,
                model_name=args.model_name,
                tokenizer_revision=args.tokenizer_revision,
                environment_ref=args.environment_ref,
                write_ref=args.write_ref,
                workspace_root=args.workspace_root,
                target_root=args.target_root,
            )
        elif args.command == "register-legacy":
            result = register_legacy_model(
                checkpoint=args.checkpoint,
                composition=args.composition,
                tokenizer_root=args.tokenizer_root,
                chat_template_source=args.chat_template_source,
                environment_ref=args.environment_ref,
                write_ref=args.write_ref,
                workspace_root=args.workspace_root,
                target_root=args.target_root,
            )
        elif args.command == "register-receipt":
            result = register_training_receipt(
                receipt_json=args.receipt_json,
                training_plan_ref=args.training_plan_ref,
                schedule_ref=args.schedule_ref,
                base_model_ref=args.base_model_ref,
                environment_ref=args.environment_ref,
                write_ref=args.write_ref,
                workspace_root=args.workspace_root,
                target_root=args.target_root,
            )
        elif args.command == "register-formal":
            result = register_trained_model(
                checkpoint_dir=args.checkpoint_dir,
                tokenizer_dir=args.tokenizer_dir,
                checkpoint_format=args.checkpoint_format,
                model_key=args.model_key,
                training_plan_ref=args.training_plan_ref,
                schedule_ref=args.schedule_ref,
                training_receipt_ref=args.training_receipt_ref,
                base_model_ref=args.base_model_ref,
                environment_ref=args.environment_ref,
                write_ref=args.write_ref,
                workspace_root=args.workspace_root,
                target_root=args.target_root,
            )
        elif args.command == "finalize-registry":
            result = finalize_model_registry(
                scope=args.scope,
                training_plan_ref=args.training_plan_ref,
                model_bindings=args.binding,
                write_ref=args.write_ref,
                workspace_root=args.workspace_root,
                target_root=args.target_root,
            )
        elif args.command == "validate-model":
            result = validate_model_artifact(
                args.model_ref, workspace_root=args.workspace_root
            )
        elif args.command == "validate-receipt":
            result = validate_training_receipt_artifact(
                args.receipt_ref, workspace_root=args.workspace_root
            )
        elif args.command == "validate-registry":
            result = validate_model_registry(
                args.registry_ref, workspace_root=args.workspace_root
            )
            if args.scope is not None and result["scope"] != args.scope:
                raise TrainingArtifactError(
                    f"registry scope {result['scope']!r} differs from requested {args.scope!r}"
                )
        else:
            resolved = resolve_registered_model(
                registry_ref=args.registry_ref,
                model_key=args.model_key,
                workspace_root=args.workspace_root,
            )
            result = {
                "schema_version": "stage1-runtime-model-resolution/v1",
                "registry_id": resolved.registry_id,
                "model_key": resolved.model_key,
                "role": resolved.role,
                "seed": resolved.seed,
                "checkpoint_format": resolved.checkpoint_format,
                "checkpoint_path": str(resolved.checkpoint_path),
                "tokenizer_path": str(resolved.tokenizer_path),
                "base_model_path": str(resolved.base_model_path),
                "tokenizer_revision": resolved.tokenizer_revision,
                "tokenizer_content_revision": resolved.tokenizer_content_revision,
                "model_artifact_id": resolved.model_artifact_id,
                "scientific_eligible": resolved.scientific_eligible,
            }
    except (TrainingArtifactError, OSError, ValueError) as exc:
        print(f"[stage1-model-registry] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
