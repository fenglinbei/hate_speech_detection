#!/usr/bin/env python3
"""Build, inspect, validate and execute the development-only L/D run package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diagnostics.general_model_package import (  # noqa: E402
    DEFAULT_CONFIG, DEFAULT_OUTPUT, build_dev, read_json, resolve_package, validate_package,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-dev")
    build.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    build.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    for command in ("inspect", "validate", "run-preflight", "run-dev", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--package", type=Path, default=DEFAULT_OUTPUT / "package_ref.json")
        if command == "validate":
            sub.add_argument("--replay", action="store_true")
        elif command in {"run-preflight", "run-dev"}:
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--device", default="cuda:0")
            if command == "run-dev":
                sub.add_argument("--preflight", type=Path, required=True)
                sub.add_argument("--tasks", nargs="+", choices=("hate", "group", "extraction"))
                sub.add_argument("--conditions", nargs="+")
        elif command == "evaluate":
            sub.add_argument("--run", type=Path, required=True)
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build-dev":
        result = build_dev(args.config, args.output_root.resolve())
    elif args.command == "validate":
        result = validate_package(args.package, replay=args.replay)
    elif args.command == "inspect":
        target = resolve_package(args.package)
        result = {"path": str(target), "manifest": read_json(target / "manifest.json"),
                  "readiness": read_json(target / "readiness.json")}
    else:
        from diagnostics.general_model_runtime import evaluate_run, run_local
        if args.command == "evaluate":
            result = evaluate_run(args.package, args.run, args.output)
        else:
            result = run_local(
                args.package, args.output, phase="preflight" if args.command == "run-preflight" else "dev",
                device=args.device, tasks=getattr(args, "tasks", None),
                conditions=getattr(args, "conditions", None), preflight=getattr(args, "preflight", None),
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
