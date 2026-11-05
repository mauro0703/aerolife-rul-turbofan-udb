"""Verify bundled NASA bytes; --force re-downloads the NASA-linked archive."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src")))
from rul.data import download_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(download_data(force=args.force))
