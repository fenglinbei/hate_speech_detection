#!/usr/bin/env python3
"""Entry point for the registered general-model numerical measurement."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from diagnostics.general_model_numeric import main


if __name__ == "__main__":
    raise SystemExit(main())
