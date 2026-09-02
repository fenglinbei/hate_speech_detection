#!/usr/bin/env python3
"""CLI for the development-only Qwen3 L/D mechanism pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from diagnostics.exploratory_qwen3_ld_v0 import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    PilotError,
    build_candidate_frame,
    build_context_grid,
    evaluate_generation_grid,
    freeze_frame,
    run_full_generation,
    run_margin_scoring,
    run_preflight,
    serve_audit,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run the frozen development-only Qwen3 lexicon/demo mechanism pilot."
    )
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    subparsers = result.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build-frame")
    audit = subparsers.add_parser("serve-audit")
    audit.add_argument("--reviewer-id", default="liaozijie")
    audit.add_argument("--host", default="127.0.0.1")
    audit.add_argument("--port", type=int, default=8766)
    audit.add_argument("--check", action="store_true")
    audit.add_argument("--frame-file", type=Path, default=None)
    audit.add_argument("--session-file", type=Path, default=None)
    audit.add_argument("--public-origin", default=None)
    freeze = subparsers.add_parser("freeze-frame")
    freeze.add_argument("--session-file", type=Path, default=None)
    contexts = subparsers.add_parser("build-contexts")
    contexts.add_argument("--device", default=None)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--device", default=None)
    preflight.add_argument("--stop-at-local-time", default=None)
    full = subparsers.add_parser("run")
    full.add_argument("--device", default=None)
    full.add_argument("--stop-at-local-time", default=None)
    subparsers.add_parser("evaluate")
    margins = subparsers.add_parser("margins")
    margins.add_argument("--device", default=None)
    margins.add_argument("--stop-at-local-time", default=None)
    margins.add_argument("--batch-size", type=int, default=None)
    return result


def main() -> int:
    args = parser().parse_args()
    common = {"config_path": args.config, "output_root": args.output_root}
    if args.command == "build-frame":
        result = build_candidate_frame(**common)
    elif args.command == "serve-audit":
        result = serve_audit(
            output_root=args.output_root,
            reviewer_id=args.reviewer_id,
            host=args.host,
            port=args.port,
            check=args.check,
            frame_path=args.frame_file,
            session_path=args.session_file,
            public_origin=args.public_origin,
        )
    elif args.command == "freeze-frame":
        result = freeze_frame(**common, session_path=args.session_file)
    elif args.command == "build-contexts":
        result = build_context_grid(**common, device=args.device)
    elif args.command == "preflight":
        result = run_preflight(
            **common, device=args.device, stop_at_local_time=args.stop_at_local_time
        )
    elif args.command == "run":
        result = run_full_generation(
            **common, device=args.device, stop_at_local_time=args.stop_at_local_time
        )
    elif args.command == "evaluate":
        result = evaluate_generation_grid(**common)
    elif args.command == "margins":
        result = run_margin_scoring(
            **common,
            device=args.device,
            stop_at_local_time=args.stop_at_local_time,
            batch_size=args.batch_size,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PilotError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
