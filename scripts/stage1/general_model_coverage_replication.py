#!/usr/bin/env python3
"""CLI for frozen 14B/27B merged-L replication."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from diagnostics.general_model_coverage_replication import main

if __name__ == "__main__":
    raise SystemExit(main())
