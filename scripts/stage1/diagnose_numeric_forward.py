#!/usr/bin/env python3
"""Run gold-free forward diagnostics without changing the numerical registration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from diagnostics.numeric_forward_diagnostic import main


if __name__ == "__main__":
    raise SystemExit(main())
