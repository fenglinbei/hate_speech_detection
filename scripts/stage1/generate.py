#!/usr/bin/env python3
"""Preflight, execute, and validate immutable Stage 1 generation runs.

No command accepts a checkpoint or tokenizer path.  Both are resolved through
``--model-registry-ref`` plus ``--model-key``.  ``run-hf`` is the only command
that can execute real model inference and requires an additional acknowledgement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from data.generation_lifecycle import (  # noqa: E402
    FixtureExecutor,
    GenerationLifecycleError,
    LocalHFExecutor,
    build_generation_artifact,
    preflight_generation,
    prepare_generation,
    validate_generation_ref,
)
from data.training_artifacts import (  # noqa: E402
    TrainingArtifactError,
    load_json,
    load_jsonl,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--context-ref", required=True, type=Path)
    parser.add_argument("--control-ref", type=Path)
    parser.add_argument(
        "--cf-ref",
        type=Path,
        help=(
            "Final counterfactual ref. Required for sealed test generation and "
            "forbidden for engineering publication."
        ),
    )
    parser.add_argument("--model-registry-ref", required=True, type=Path)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--sealed",
        action="store_true",
        default=None,
        help="Assert that the validated context lineage is sealed test (never selects it).",
    )
    parser.add_argument(
        "--determinism-repetitions",
        type=int,
        choices=(1, 2),
        default=1,
        help="Execute the complete frozen frame once or twice before publication.",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Immutable complete-block Stage 1 generation lifecycle"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight", help="Validate and freeze traversal without running a model"
    )
    _common(preflight)
    preflight.add_argument("--engineering", action="store_true")

    fixture = subparsers.add_parser(
        "build-fixture",
        help="Publish an engineering-only artifact from exact synthetic outputs",
    )
    _common(fixture)
    fixture.add_argument("--fixture-jsonl", required=True, type=Path)
    fixture.add_argument("--target-root", required=True, type=Path)
    fixture.add_argument("--write-ref", required=True, type=Path)

    run_hf = subparsers.add_parser(
        "run-hf",
        help="Explicitly run local Hugging Face inference from a registered model",
    )
    _common(run_hf)
    run_hf.add_argument("--target-root", required=True, type=Path)
    run_hf.add_argument("--write-ref", required=True, type=Path)
    run_hf.add_argument("--engineering", action="store_true")
    run_hf.add_argument(
        "--execute-real-inference",
        action="store_true",
        help="Required acknowledgement; this command may load a large local model.",
    )

    validate = subparsers.add_parser(
        "validate", help="Read-only deep validation of a generation locator"
    )
    validate.add_argument("--generation-ref", required=True, type=Path)
    validate.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    validate.add_argument("--require-scientific", action="store_true")
    return parser


def _profile(path: Path) -> Mapping[str, Any]:
    value = load_json(path)
    if not isinstance(value, Mapping):
        raise GenerationLifecycleError("generation profile root must be an object")
    return value


def _preflight_descriptor(profile: Mapping[str, Any], engineering: bool) -> dict[str, Any]:
    backend = str(profile.get("backend", ""))
    if backend == "transformers":
        return {
            "executor_id": "hf-local-transformers/v1",
            "executor_revision": "stage1-generation-hf-executor/v1",
            "backend": "transformers",
            "scientific_eligible": not engineering,
        }
    if backend == "vllm" and not engineering:
        return {
            "executor_id": "vllm-local-registered/v1",
            "executor_revision": "stage1-generation-vllm-executor/v1",
            "backend": "vllm",
            "scientific_eligible": True,
        }
    return {
        "executor_id": "preflight-intent/v1",
        "executor_revision": "stage1-generation-preflight/v1",
        "backend": "preflight" if engineering else backend,
        "scientific_eligible": not engineering,
    }


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            profile = _profile(args.profile)
            result = preflight_generation(
                profile=profile,
                context_ref=args.context_ref,
                control_ref=args.control_ref,
                cf_ref=args.cf_ref,
                model_registry_ref=args.model_registry_ref,
                model_key=args.model_key,
                workspace_root=args.workspace_root,
                scope="engineering" if args.engineering else "formal",
                executor_descriptor=_preflight_descriptor(profile, args.engineering),
                sealed=args.sealed,
                determinism_repetitions=args.determinism_repetitions,
            )
        elif args.command == "build-fixture":
            executor = FixtureExecutor.from_rows(load_jsonl(args.fixture_jsonl))
            result = build_generation_artifact(
                profile=args.profile,
                context_ref=args.context_ref,
                control_ref=args.control_ref,
                cf_ref=args.cf_ref,
                model_registry_ref=args.model_registry_ref,
                model_key=args.model_key,
                workspace_root=args.workspace_root,
                target_root=args.target_root,
                write_ref=args.write_ref,
                scope="engineering",
                executor=executor,
                sealed=args.sealed,
                determinism_repetitions=args.determinism_repetitions,
            )
        elif args.command == "run-hf":
            if not args.execute_real_inference:
                raise GenerationLifecycleError(
                    "run-hf requires --execute-real-inference"
                )
            profile = _profile(args.profile)
            if args.engineering:
                descriptor = {
                    "executor_id": "hf-local-transformers/v1",
                    "executor_revision": "stage1-generation-hf-executor/v1",
                    "backend": "transformers",
                    "scientific_eligible": True,
                }
                prepared = prepare_generation(
                    profile=profile,
                    context_ref=args.context_ref,
                    control_ref=args.control_ref,
                    cf_ref=args.cf_ref,
                    model_registry_ref=args.model_registry_ref,
                    model_key=args.model_key,
                    workspace_root=args.workspace_root,
                    scope="engineering",
                    executor_descriptor=descriptor,
                    sealed=args.sealed,
                    determinism_repetitions=args.determinism_repetitions,
                )
                executor = LocalHFExecutor.from_registered_model(
                    prepared.model, prepared.profile
                )
                # Engineering smoke may reuse the exact already-rehashed
                # handle. Formal publication never accepts this injection.
                direct_resolver = lambda **_: prepared.model
                dependency_resolver = lambda **_: prepared.model
                result = build_generation_artifact(
                    profile=profile,
                    context_ref=args.context_ref,
                    control_ref=args.control_ref,
                    cf_ref=args.cf_ref,
                    model_registry_ref=args.model_registry_ref,
                    model_key=args.model_key,
                    workspace_root=args.workspace_root,
                    target_root=args.target_root,
                    write_ref=args.write_ref,
                    scope="engineering",
                    executor=executor,
                    sealed=args.sealed,
                    determinism_repetitions=args.determinism_repetitions,
                    model_resolver=direct_resolver,
                    model_dependency_resolver=dependency_resolver,
                )
            else:
                # The formal lifecycle resolves, re-hashes, loads, and binds
                # the executor internally. No caller-owned model object can
                # reach formal publication.
                result = build_generation_artifact(
                    profile=profile,
                    context_ref=args.context_ref,
                    control_ref=args.control_ref,
                    cf_ref=args.cf_ref,
                    model_registry_ref=args.model_registry_ref,
                    model_key=args.model_key,
                    workspace_root=args.workspace_root,
                    target_root=args.target_root,
                    write_ref=args.write_ref,
                    scope="formal",
                    sealed=args.sealed,
                    determinism_repetitions=args.determinism_repetitions,
                )
        else:
            result = validate_generation_ref(
                args.generation_ref,
                workspace_root=args.workspace_root,
                require_scientific=args.require_scientific,
            )
    except (GenerationLifecycleError, TrainingArtifactError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
