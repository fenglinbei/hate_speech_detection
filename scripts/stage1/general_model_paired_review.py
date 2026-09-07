#!/usr/bin/env python3
"""Start the reusable three-column paired-case human review workbench."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools.general_model_paired_review_ui.server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
