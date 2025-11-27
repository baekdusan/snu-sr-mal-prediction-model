#!/usr/bin/env python3
"""
CLI wrapper for the MAL data augmentation pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mal_pred.pipelines.augment import main  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MAL data augmentation pipeline.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing artifacts and restart from scratch.",
    )
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    main(resume=not args.fresh)


if __name__ == "__main__":
    cli()


