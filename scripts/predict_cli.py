#!/usr/bin/env python3
"""
Launch the interactive MAL prediction shell.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mal_pred.interfaces.interactive import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())


