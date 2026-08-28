#!/usr/bin/env python3
"""CLI for the blind 240+48 dual-model span audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_lex.dual_model_span_audit import (  # noqa: E402
    DEFAULT_SEED,
    EXPECTED_TASKS,
    ProviderConfig,
    ResultStore,
    analyze_results,
    build_review_package,
    export_results,
    freeze_sampling_frame,
    load_dotenv_value,
    run_provider,
    validate_frame,
)


DEFAULT_DATA_REF = REPOSITORY_ROOT / "exps/causal_context/stage1_p0/refs/data_ref.json"
DEFAULT_PARTITION_REF = (
    REPOSITORY_ROOT
    / "exps/causal_context/stage1_p0/refs/train_partition_ref.json"
)
DEFAULT_FRAME_ROOT = (
    REPOSITORY_ROOT / "exps/causal_context/stage1_p0/dual_model_span_audit/frames"
)


def _json_print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _provider_config(args: argparse.Namespace) -> ProviderConfig:
    if args.provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            api_key = load_dotenv_value(args.env_file, "DEEPSEEK_API_KEY")
    else:
        api_key = args.api_key or "EMPTY"
    return ProviderConfig(
        provider=args.provider,
        model=args.model,
        api_base=args.api_base,
        api_key=api_key,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout,
        max_attempts_per_task=args.max_attempts,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )


def command_freeze(args: argparse.Namespace) -> int:
    target = freeze_sampling_frame(
        data_ref=args.data_ref,
        train_partition_ref=args.train_partition_ref,
        workspace_root=REPOSITORY_ROOT,
        output_root=args.output_root,
        seed=args.seed,
    )
    report = validate_frame(target)
    _json_print(
        {
            "frame_dir": str(target),
            "frame_id": report["frame_id"],
            "counts": report["manifest"]["counts"],
            "observed_minimum_repeat_gap": report["report"][
                "observed_minimum_repeat_gap"
            ],
        }
    )
    return 0


def command_validate(args: argparse.Namespace) -> int:
    report = validate_frame(args.frame_dir)
    _json_print(
        {
            "valid": True,
            "frame_id": report["frame_id"],
            "counts": report["manifest"]["counts"],
        }
    )
    return 0


def command_run(args: argparse.Namespace) -> int:
    config = _provider_config(args)
    result = run_provider(
        frame_dir=args.frame_dir,
        checkpoint_path=args.checkpoint,
        config=config,
        limit=args.limit,
    )
    result["config"] = config.public_dict()
    _json_print(result)
    if result["failures"]:
        return 2
    if args.require_complete and result["completed_total"] != EXPECTED_TASKS:
        return 3
    return 0


def command_status(args: argparse.Namespace) -> int:
    frame = validate_frame(args.frame_dir)
    store = ResultStore(args.checkpoint, frame_id=frame["frame_id"])
    _json_print(store.status())
    return 0


def command_export(args: argparse.Namespace) -> int:
    _json_print(
        export_results(
            frame_dir=args.frame_dir,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
        )
    )
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    _json_print(
        analyze_results(
            frame_dir=args.frame_dir,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
        )
    )
    return 0


def command_package(args: argparse.Namespace) -> int:
    _json_print(
        build_review_package(
            frame_dir=args.frame_dir,
            checkpoint_path=args.checkpoint,
            templates_dir=args.templates_dir,
            output_root=args.output_root,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze-frame")
    freeze.add_argument("--data-ref", type=Path, default=DEFAULT_DATA_REF)
    freeze.add_argument(
        "--train-partition-ref", type=Path, default=DEFAULT_PARTITION_REF
    )
    freeze.add_argument("--output-root", type=Path, default=DEFAULT_FRAME_ROOT)
    freeze.add_argument("--seed", default=DEFAULT_SEED)
    freeze.set_defaults(function=command_freeze)

    validate = subparsers.add_parser("validate-frame")
    validate.add_argument("--frame-dir", type=Path, required=True)
    validate.set_defaults(function=command_validate)

    run = subparsers.add_parser("run-provider")
    run.add_argument("--frame-dir", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--provider", choices=("qwen", "deepseek"), required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--api-base", required=True)
    run.add_argument("--api-key")
    run.add_argument("--env-file", type=Path, default=REPOSITORY_ROOT / ".env")
    run.add_argument("--concurrency", type=int, default=8)
    run.add_argument("--timeout", type=int, default=180)
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--temperature", type=float, default=0.2)
    run.add_argument("--top-p", type=float, default=0.8)
    run.add_argument("--max-tokens", type=int, default=768)
    run.add_argument("--limit", type=int)
    run.add_argument("--require-complete", action="store_true")
    run.set_defaults(function=command_run)

    status = subparsers.add_parser("status")
    status.add_argument("--frame-dir", type=Path, required=True)
    status.add_argument("--checkpoint", type=Path, required=True)
    status.set_defaults(function=command_status)

    export = subparsers.add_parser("export")
    export.add_argument("--frame-dir", type=Path, required=True)
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.set_defaults(function=command_export)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--frame-dir", type=Path, required=True)
    analyze.add_argument("--checkpoint", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.set_defaults(function=command_analyze)

    package = subparsers.add_parser("build-review-package")
    package.add_argument("--frame-dir", type=Path, required=True)
    package.add_argument("--checkpoint", type=Path, required=True)
    package.add_argument(
        "--templates-dir",
        type=Path,
        default=REPOSITORY_ROOT / "tools/dual_model_span_review",
    )
    package.add_argument("--output-root", type=Path, required=True)
    package.set_defaults(function=command_package)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
