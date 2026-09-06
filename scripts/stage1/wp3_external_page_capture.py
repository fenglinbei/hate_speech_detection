#!/usr/bin/env python3
"""Capture or validate proposed WP3 public pages without environment proxies."""

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

from build_lex.terminology_external_page_capture import (  # noqa: E402
    ExternalPageCaptureError,
    capture_incomplete_pages_with_browser,
    capture_registered_pages,
    derive_capture_excluding_urls,
    validate_browser_render_capture,
    validate_external_page_capture,
)


DEFAULT_REGISTER = Path(
    "docs/research/experiment-plans/wp3-g3-source-policy-and-candidate-register.md"
)
DEFAULT_OUTPUT_ROOT = Path(
    "exps/causal_context/stage1_p0/wp3_candidate_generators_v2/"
    "g3_source_candidate_downloads"
)
DEFAULT_CHROME_BINARY = Path(
    "/root/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome"
)


def _workspace_path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    capture = commands.add_parser("capture")
    capture.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    capture.add_argument("--register", type=Path, default=DEFAULT_REGISTER)
    capture.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    capture.add_argument("--workers", type=int, default=4)

    validate = commands.add_parser("validate")
    validate.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    validate.add_argument("--capture-dir", type=Path, required=True)

    derive = commands.add_parser("derive-without-url")
    derive.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    derive.add_argument("--capture-dir", type=Path, required=True)
    derive.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    derive.add_argument("--exclude-url", action="append", required=True)

    render = commands.add_parser("render-incomplete")
    render.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    render.add_argument("--capture-dir", type=Path, required=True)
    render.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "browser_renders"
    )
    render.add_argument("--chrome-binary", type=Path, default=DEFAULT_CHROME_BINARY)

    validate_render = commands.add_parser("validate-render")
    validate_render.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    validate_render.add_argument("--render-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.workspace_root.resolve()
    try:
        if args.command == "capture":
            report = capture_registered_pages(
                workspace_root=root,
                register_path=_workspace_path(root, args.register),
                output_root=_workspace_path(root, args.output_root),
                workers=args.workers,
            )
        elif args.command == "validate":
            report = validate_external_page_capture(
                _workspace_path(root, args.capture_dir)
            )
        elif args.command == "derive-without-url":
            report = derive_capture_excluding_urls(
                parent_capture_directory=_workspace_path(root, args.capture_dir),
                output_root=_workspace_path(root, args.output_root),
                excluded_urls=args.exclude_url,
            )
        elif args.command == "render-incomplete":
            report = capture_incomplete_pages_with_browser(
                parent_capture_directory=_workspace_path(root, args.capture_dir),
                output_root=_workspace_path(root, args.output_root),
                chrome_binary=args.chrome_binary,
            )
        else:
            report = validate_browser_render_capture(
                _workspace_path(root, args.render_dir)
            )
    except ExternalPageCaptureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
