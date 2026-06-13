"""Modeling layer.

``windowing`` is TensorFlow-free and safe to import anywhere. Import the LSTM
explicitly (``from market_intel.models.lstm import run``) so TensorFlow only
loads when you actually train.
"""

from market_intel.models.windowing import (
    SplitData,
    WalkForwardFold,
    prepare_splits,
    walk_forward_splits,
)

__all__ = ["SplitData", "WalkForwardFold", "prepare_splits", "walk_forward_splits"]
