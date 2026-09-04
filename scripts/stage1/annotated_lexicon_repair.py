#!/usr/bin/env python3
"""Lifecycle CLI for the bounded annotated-lexicon repair v1."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from build_lex.annotated_lexicon_repair import (  # noqa: E402
    LexiconRepairError,
    build_span_gold_frame,
    finalize_span_gold,
)


DEFAULT_PILOT_ROOT = REPOSITORY_ROOT / "exps/causal_context/stage1_exploratory_qwen3_ld_v0"
DEFAULT_REPAIR_ROOT = REPOSITORY_ROOT / "exps/causal_context/annotated_lexicon_repair_v1"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-span-gold", help="build the closed span review frame")
    build.add_argument(
        "--candidate-frame",
        type=Path,
        default=DEFAULT_PILOT_ROOT / "candidate_frame.json",
    )
    build.add_argument(
        "--audit-session",
        type=Path,
        default=DEFAULT_PILOT_ROOT / "audit/input_quality_session.json",
    )
    build.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPAIR_ROOT / "span_gold/frame.json",
    )

    serve = subparsers.add_parser("serve-span-gold", help="serve the adapted span review UI")
    serve.add_argument("--frame", type=Path, default=DEFAULT_REPAIR_ROOT / "span_gold/frame.json")
    serve.add_argument(
        "--session",
        type=Path,
        default=DEFAULT_REPAIR_ROOT / "span_gold/session.json",
    )
    serve.add_argument("--reviewer-id", default="liaozijie")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8769)
    serve.add_argument("--public-origin")
    serve.add_argument("--check", action="store_true")

    finalize = subparsers.add_parser("finalize-span-gold", help="freeze confirmed exact spans")
    finalize.add_argument("--frame", type=Path, default=DEFAULT_REPAIR_ROOT / "span_gold/frame.json")
    finalize.add_argument(
        "--session",
        type=Path,
        default=DEFAULT_REPAIR_ROOT / "span_gold/session.json",
    )
    finalize.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPAIR_ROOT / "span_gold/reference.json",
    )

    operation_build = subparsers.add_parser(
        "build-repair-operation", help="build the independent entry-operation review frame"
    )
    operation_build.add_argument(
        "--lexicon", type=Path, default=REPOSITORY_ROOT / "data/lexicon/annotated_lexicon.json"
    )
    operation_build.add_argument(
        "--span-frame", type=Path, default=DEFAULT_REPAIR_ROOT / "span_gold/frame.json"
    )
    operation_build.add_argument(
        "--span-reference", type=Path, default=DEFAULT_REPAIR_ROOT / "span_gold/reference.json"
    )
    operation_build.add_argument(
        "--proposals",
        type=Path,
        default=REPOSITORY_ROOT / "config/stage1/annotated_lexicon_repair_operation_proposals_v1.json",
    )
    operation_build.add_argument(
        "--output", type=Path, default=DEFAULT_REPAIR_ROOT / "repair_operation/frame.json"
    )

    operation_serve = subparsers.add_parser(
        "serve-repair-operation", help="serve the independent entry-operation review UI"
    )
    operation_serve.add_argument(
        "--frame", type=Path, default=DEFAULT_REPAIR_ROOT / "repair_operation/frame.json"
    )
    operation_serve.add_argument(
        "--session", type=Path, default=DEFAULT_REPAIR_ROOT / "repair_operation/session.json"
    )
    operation_serve.add_argument("--reviewer-id", default="liaozijie")
    operation_serve.add_argument("--host", default="127.0.0.1")
    operation_serve.add_argument("--port", type=int, default=8769)
    operation_serve.add_argument("--public-origin")
    operation_serve.add_argument("--check", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "build-span-gold":
            frame = build_span_gold_frame(
                candidate_frame_path=arguments.candidate_frame,
                audit_session_path=arguments.audit_session,
                output_path=arguments.output,
            )
            print(
                json.dumps(
                    {
                        "frame_id": frame["manifest"]["frame_id"],
                        "item_count": frame["manifest"]["item_count"],
                        "cohort_counts": frame["manifest"]["cohort_counts"],
                        "output": str(arguments.output.resolve()),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if arguments.command == "serve-span-gold":
            ui_root = REPOSITORY_ROOT / "tools/annotated_lexicon_repair_review_ui"
            if str(ui_root) not in sys.path:
                sys.path.insert(0, str(ui_root))
            from server import run_server

            return run_server(
                frame_path=arguments.frame,
                session_path=arguments.session,
                reviewer_id=arguments.reviewer_id,
                host=arguments.host,
                port=arguments.port,
                check=arguments.check,
                public_origin=arguments.public_origin,
            )
        if arguments.command == "finalize-span-gold":
            reference = finalize_span_gold(
                frame_path=arguments.frame,
                session_path=arguments.session,
                output_path=arguments.output,
            )
            print(
                json.dumps(
                    {
                        "reference_id": reference["reference_id"],
                        "item_count": reference["item_count"],
                        "output": str(arguments.output.resolve()),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if arguments.command == "build-repair-operation":
            from build_lex.annotated_lexicon_operation_review import build_operation_frame

            frame = build_operation_frame(
                lexicon_path=arguments.lexicon,
                span_frame_path=arguments.span_frame,
                span_reference_path=arguments.span_reference,
                proposals_path=arguments.proposals,
                output_path=arguments.output,
            )
            print(
                json.dumps(
                    {
                        "frame_id": frame["manifest"]["frame_id"],
                        "item_count": frame["manifest"]["item_count"],
                        "output": str(arguments.output.resolve()),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if arguments.command == "serve-repair-operation":
            # A stage-specific module name avoids reusing the span server when
            # callers invoke main() more than once in the same Python process.
            module_path = REPOSITORY_ROOT / "tools/annotated_lexicon_operation_review_ui/server.py"
            spec = importlib.util.spec_from_file_location("annotated_lexicon_operation_server", module_path)
            if spec is None or spec.loader is None:
                raise LexiconRepairError("repair-operation web service module is missing")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module.run_server(
                frame_path=arguments.frame,
                session_path=arguments.session,
                reviewer_id=arguments.reviewer_id,
                host=arguments.host,
                port=arguments.port,
                check=arguments.check,
                public_origin=arguments.public_origin,
            )
        raise AssertionError(arguments.command)
    except (LexiconRepairError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
