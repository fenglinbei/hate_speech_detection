#!/usr/bin/env python3
"""Entrypoint for the independently frozen dictionary-category removal experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from diagnostics.general_model_nolabel import main


if __name__ == "__main__":
    raise SystemExit(main())
