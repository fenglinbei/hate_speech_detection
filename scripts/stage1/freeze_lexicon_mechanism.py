#!/usr/bin/env python3
"""Build the user-authorized mechanism resource snapshot without review/GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from build_lex.annotated_lexicon_repair import LexiconRepairError  # noqa: E402
from build_lex.lexicon_mechanism_freeze import build_mechanism_freeze, verify_freeze, write_freeze  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repair = ROOT / "exps/causal_context/annotated_lexicon_repair_v1"
    records = ROOT / "exps/causal_context/lexicon_mechanism_frozen_v1"
    parser.add_argument("--lexicon", type=Path, default=ROOT / "data/lexicon/annotated_lexicon.json")
    parser.add_argument("--frame", type=Path, default=repair / "repair_operation/frame.json")
    parser.add_argument("--session", type=Path, default=records / "review_session_snapshot.json")
    parser.add_argument("--span-reference", type=Path, default=repair / "span_gold/reference.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/lexicon/annotated_lexicon_mechanism_frozen_v1.json")
    parser.add_argument("--record-root", type=Path, default=records)
    parser.add_argument("--check", action="store_true", help="rebuild in memory and verify all existing frozen files without writing")
    args = parser.parse_args()
    try:
        bundle = build_mechanism_freeze(
            lexicon_path=args.lexicon, frame_path=args.frame, session_path=args.session,
            span_reference_path=args.span_reference,
        )
        action = verify_freeze if args.check else write_freeze
        paths = action(bundle, output_path=args.output, record_root=args.record_root)
        print(json.dumps({"lexicon_build_id": bundle["lexicon"]["lexicon_build_id"],
                          "sha256": bundle["manifest"]["artifact_sha256"]["lexicon"],
                          "counts": bundle["manifest"]["counts"], "paths": paths}, ensure_ascii=False, sort_keys=True))
        return 0
    except (LexiconRepairError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
