"""Run from repository root: python scripts/run_pipeline.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path("src")))
from rul.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
