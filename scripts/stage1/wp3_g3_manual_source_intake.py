#!/usr/bin/env python3
"""Build or replay a zero-network WP3 G3 manual-source intake artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from build_lex.terminology_g3_manual_source_intake import (  # noqa: E402
    ManualSourceIntakeError,
    build_manual_source_intake,
    load_manual_source_map,
    validate_manual_source_intake,
)


DEFAULT_OUTPUT_ROOT = Path(
    "exps/causal_context/stage1_p0/wp3_candidate_generators_v2/"
    "g3_manual_source_intakes"
)


def _workspace_path(value: Path) -> Path:
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    check_map = commands.add_parser("validate-map")
    check_map.add_argument("--source-map", type=Path, required=True)

    build = commands.add_parser("build")
    build.add_argument(
        "--attachment-root",
        type=Path,
        required=True,
        help="Absolute read-only root used to resolve mapping attachment_locator values.",
    )
    build.add_argument("--source-map", type=Path, required=True)
    build.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    validate = commands.add_parser("validate")
    validate.add_argument("--intake-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-map":
            mapping = load_manual_source_map(_workspace_path(args.source_map))
            report = {
                "map_id": mapping["map_id"],
                "alias_count": len(mapping["attachments"]),
                "component_count": len(
                    {row["component_id"] for row in mapping["attachments"]}
                ),
                "network_access_allowed": mapping["network_access_allowed"],
            }
        elif args.command == "build":
            if not args.attachment_root.is_absolute():
                raise ManualSourceIntakeError("--attachment-root must be absolute")
            report = build_manual_source_intake(
                attachment_root=args.attachment_root,
                source_map_path=_workspace_path(args.source_map),
                output_root=_workspace_path(args.output_root),
            )
        else:
            report = validate_manual_source_intake(
                _workspace_path(args.intake_dir)
            )
    except ManualSourceIntakeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
