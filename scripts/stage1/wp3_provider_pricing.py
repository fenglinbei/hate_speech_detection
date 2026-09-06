#!/usr/bin/env python3
"""Verify frozen official pricing snapshots without network access."""

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

from build_lex.terminology_provider_pricing import (  # noqa: E402
    ProviderPricingError,
    validate_provider_pricing_bundle,
    verify_provider_pricing_evidence,
    write_verified_pricing_bundle,
)


def _path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify-pricing")
    verify.add_argument("--evidence", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser("validate-pricing")
    validate.add_argument("--evidence", type=Path, required=True)
    validate.add_argument("--bundle", type=Path, required=True)
    return parser


def _summary(bundle: dict) -> dict:
    receipt = bundle["verification_receipt"]
    return {
        "schema_version": bundle["schema_version"],
        "scope": bundle["scope"],
        "scientific_eligible": bundle["scientific_eligible"],
        "sealed": bundle["sealed"],
        "verification_id": receipt["verification_id"],
        "verification_receipt_sha256": bundle["verification_receipt_sha256"],
        "pricing_projection_sha256": receipt["pricing_projection_sha256"],
        "provider_count": len(receipt["source_replays"]),
        "network_access_performed": receipt["network_access_performed"],
        "verified": receipt["verified"],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.workspace_root.resolve()
    try:
        evidence = _path(root, args.evidence)
        if args.command == "verify-pricing":
            bundle = verify_provider_pricing_evidence(
                evidence, workspace_root=root
            )
            write_verified_pricing_bundle(_path(root, args.output), bundle)
        else:
            bundle = validate_provider_pricing_bundle(
                _path(root, args.bundle),
                evidence=evidence,
                workspace_root=root,
            )
    except (ProviderPricingError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_summary(bundle), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
