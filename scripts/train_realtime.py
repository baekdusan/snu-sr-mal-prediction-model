#!/usr/bin/env python3
"""
Train the real-time MAL regressor and persist the best model artifact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mal_pred.training.realtime import train_realtime_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/retrain the MAL regressor.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional override for the processed dataset.",
    )
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    kwargs = {}
    if args.data_path:
        kwargs["data_path"] = args.data_path
    summary = train_realtime_model(**kwargs)
    print("\nTraining summary:", summary)


if __name__ == "__main__":
    cli()


