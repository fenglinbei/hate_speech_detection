#!/usr/bin/env python3
"""Run only the CPU posthoc paired analysis; no model execution entrypoints."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from diagnostics.general_model_nolabel_paired_cases import main


if __name__ == "__main__":
    raise SystemExit(main())
