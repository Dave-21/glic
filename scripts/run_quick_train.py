#!/usr/bin/env python3
"""Runs a 1-epoch CPU training loop to validate the pipeline end-to-end."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train


def main() -> None:
    train.NUM_EPOCHS = 1
    train.BATCH_SIZE = min(train.BATCH_SIZE, 2)
    train.VAL_BATCH_SIZE = 1
    train.NUM_WORKERS = 0
    train.DEVICE = "cpu"
    train.train()


if __name__ == "__main__":
    main()
