"""Thin entrypoint for fetching price CSVs. Logic lives in market_intel.data.fetch.

Usage:
    python src/preprocess.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market_intel.data.fetch import main  # noqa: E402

if __name__ == "__main__":
    main()
