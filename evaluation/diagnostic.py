"""Compatibility wrapper for the ASCII-safe diagnostic script."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.diagnostic_simple import run_full_diagnostic


if __name__ == "__main__":
    data_path = "data/data.txt"
    if len(sys.argv) > 1:
        data_path = sys.argv[1]

    run_full_diagnostic(data_path)
