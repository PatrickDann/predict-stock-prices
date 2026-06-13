"""Thin CLI for the leakage-free LSTM. Real logic lives in market_intel.models.lstm.

Usage:
    python src/multi_feature_model.py AAPL
    python src/multi_feature_model.py MSFT --dataset tech --plot
    python src/multi_feature_model.py AAPL --walk-forward 5 --epochs 20
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running as a loose script (`python src/multi_feature_model.py`) without install.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market_intel.models.lstm import run  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Leakage-free LSTM stock-price forecasting")
    p.add_argument("ticker", help="e.g. AAPL")
    p.add_argument("--dataset", default=None, help="file stem, e.g. 'tech' (default: <ticker>)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--time-step", type=int, default=60)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--walk-forward", type=int, default=0, metavar="N", help="N walk-forward folds")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-path", default=None, help="save plot to file instead of showing")
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args(argv)

    run(
        ticker=args.ticker,
        dataset=args.dataset,
        data_dir=args.data_dir,
        time_step=args.time_step,
        epochs=args.epochs,
        batch_size=args.batch_size,
        plot=args.plot,
        plot_path=args.plot_path,
        walk_forward=args.walk_forward,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
