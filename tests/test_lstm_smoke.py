"""End-to-end smoke test of the training path (tiny data, 1 epoch).

Imports TensorFlow, so it's a little slow; kept minimal. Run just this file
with: pytest tests/test_lstm_smoke.py
"""

import numpy as np
import pandas as pd

from market_intel.models.windowing import prepare_splits


def _synthetic_frame(n=260, seed=1):
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 1, n)) + 100
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Close": base,
            "High": base + 1,
            "Low": base - 1,
            "Open": base,
            "Volume": rng.integers(1e6, 2e6, n).astype(float),
        },
        index=idx,
    )


def test_train_evaluate_runs():
    from market_intel.models.lstm import train_evaluate

    df = _synthetic_frame(260)
    split = prepare_splits(df, time_step=20)
    result = train_evaluate(split, epochs=1, batch_size=16, verbose=0)

    assert "test_rmse" in result.metrics
    assert np.isfinite(result.metrics["test_rmse"])
    assert len(result.y_test_pred) == len(result.y_test_actual) > 0
    assert len(result.test_index) == len(result.y_test_pred)


def test_walk_forward_runs():
    from market_intel.models.lstm import walk_forward_eval

    df = _synthetic_frame(300)
    wf = walk_forward_eval(df, time_step=20, n_splits=2, epochs=1)
    assert wf["folds"]
    assert np.isfinite(wf["mean_rmse"])
