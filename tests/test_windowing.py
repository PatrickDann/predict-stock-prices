"""Regression tests for the leakage fix — the core correctness guarantee.

These are written to FAIL if anyone reintroduces a leak: full-dataset feature
*or* target scaling, pre-target-row windowing (off-by-one), or cross-fold
leakage in walk-forward. TensorFlow-free, so they run fast.
"""

import numpy as np
import pandas as pd
import pytest

from market_intel.models.windowing import (
    DEFAULT_FIELDS,
    chronological_split_indices,
    prepare_splits,
    walk_forward_splits,
)

FIELDS = list(DEFAULT_FIELDS)


def _synthetic_frame(n=500, seed=0):
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 1, n)) + 100
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Close": base,
            "High": base + rng.uniform(0, 2, n),
            "Low": base - rng.uniform(0, 2, n),
            "Open": base + rng.normal(0, 0.5, n),
            "Volume": rng.integers(1e6, 5e6, n).astype(float),
        },
        index=idx,
    )


def _ramp_frame(n=500):
    """Strictly increasing — global max is guaranteed to land in the test slice."""
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    ramp = np.arange(n, dtype="float64")
    return pd.DataFrame(
        {"Close": ramp, "High": ramp + 1, "Low": ramp - 1, "Open": ramp, "Volume": ramp + 10},
        index=idx,
    )


# --- split index math ---


def test_split_indices():
    assert chronological_split_indices(1000, 0.7, 0.15) == (700, 850)


@pytest.mark.parametrize(
    "train_frac,val_frac",
    [(0.0, 0.1), (1.0, 0.0), (0.7, 0.3), (0.7, 0.35), (-0.1, 0.1), (0.5, -0.1)],
)
def test_split_indices_invalid_fracs(train_frac, val_frac):
    with pytest.raises(ValueError):
        chronological_split_indices(1000, train_frac, val_frac)


# --- boundary / coverage ---


def test_no_target_crosses_split_boundary():
    """Train targets in train; val in val; test in test — no leakage."""
    df = _synthetic_frame(500)
    s = prepare_splits(df, time_step=60)

    assert s.train_pos.min() == s.time_step  # full look-back exists for the first window
    assert s.train_pos.max() < s.train_end
    assert s.val_pos.min() >= s.train_end and s.val_pos.max() < s.val_end
    assert s.test_pos.min() >= s.val_end and s.test_pos.max() < len(df)

    # every valid target row [time_step, n) is windowed exactly once, no gaps
    all_pos = np.concatenate([s.train_pos, s.val_pos, s.test_pos])
    np.testing.assert_array_equal(np.sort(all_pos), np.arange(s.time_step, len(df)))


# --- the central no-leakage claim: windows are strictly causal ---


def test_windows_exclude_target_and_future_rows():
    """X[k] must be exactly the `time_step` rows BEFORE its target (not incl. it)."""
    df = _synthetic_frame(300)
    s = prepare_splits(df, time_step=30)
    feat_scaled = s.feature_scaler.transform(df[FIELDS].to_numpy(dtype="float64"))

    for X, pos in ((s.X_train, s.train_pos), (s.X_test, s.test_pos)):
        i = int(pos[0])
        assert np.allclose(X[0], feat_scaled[i - s.time_step : i])  # exact window contents
        assert np.allclose(X[0][-1], feat_scaled[i - 1])  # last row is i-1 ...
        assert not np.allclose(X[0][-1], feat_scaled[i])  # ... NOT the target row i


def test_target_value_alignment():
    df = _synthetic_frame(300)
    s = prepare_splits(df, time_step=30)
    target_scaled = s.target_scaler.transform(df[["Close"]].to_numpy(dtype="float64")).ravel()
    i = int(s.test_pos[0])
    assert np.isclose(s.y_test[0], target_scaled[i])


# --- scalers fit on train only (BOTH of them) ---


def test_scalers_fit_on_train_only():
    """Feature AND target scalers must reflect train-slice stats, not the full set.

    The ramp guarantees the global max is in the held-out test slice, so a leaked
    (full-fit) scaler would have data_max_ == global max and fail these asserts.
    """
    df = _ramp_frame(500)
    s = prepare_splits(df, time_step=60)
    values = df[FIELDS].to_numpy(dtype="float64")

    assert np.allclose(s.feature_scaler.data_max_, values[: s.train_end].max(axis=0))
    assert not np.allclose(s.feature_scaler.data_max_, values.max(axis=0))

    assert np.allclose(s.target_scaler.data_max_, values[: s.train_end, 0].max())
    assert not np.allclose(s.target_scaler.data_max_, values[:, 0].max())


# --- shapes & guards ---


def test_shapes_consistent():
    df = _synthetic_frame(400)
    s = prepare_splits(df, time_step=40)
    assert s.X_train.shape[0] == len(s.y_train) == len(s.train_pos)
    assert s.X_test.shape[1:] == (40, len(FIELDS))


def test_val_frac_zero():
    df = _synthetic_frame(400)
    s = prepare_splits(df, time_step=40, train_frac=0.85, val_frac=0.0)
    assert s.X_val.shape == (0, 40, len(FIELDS))
    assert len(s.val_pos) == 0
    assert s.train_end == s.val_end


def test_too_little_data_raises():
    with pytest.raises(ValueError):
        prepare_splits(_synthetic_frame(50), time_step=60)


def test_train_slice_too_small_raises():
    # enough total rows, but the train fraction is too small for a full window
    with pytest.raises(ValueError, match="train slice"):
        prepare_splits(_synthetic_frame(200), time_step=60, train_frac=0.2)


def test_target_col_not_in_features_raises():
    with pytest.raises(ValueError):
        prepare_splits(_synthetic_frame(300), feature_cols=("High", "Low"), target_col="Close")


# --- walk-forward CV is leakage-free (the path that previously had no asserts) ---


def test_walk_forward_scalers_fit_on_train_only():
    df = _ramp_frame(600)
    folds = walk_forward_splits(df, time_step=30, n_splits=4)
    assert folds
    values = df[FIELDS].to_numpy(dtype="float64")
    for f in folds:
        assert np.allclose(f.feature_scaler.data_max_, values[: f.train_end].max(axis=0))
        assert np.allclose(f.target_scaler.data_max_, values[: f.train_end, 0].max())
    # a fold that doesn't see the whole series proves it isn't a full-data fit
    assert not np.allclose(folds[0].feature_scaler.data_max_, values.max(axis=0))


def test_walk_forward_folds_causal():
    df = _synthetic_frame(400)
    folds = walk_forward_splits(df, time_step=20, n_splits=4)
    assert folds
    values = df[FIELDS].to_numpy(dtype="float64")
    for f in folds:
        # test targets are exactly the rows [train_end, stop) — contiguous, forward
        assert len(f.y_test) == f.stop - f.train_end
        feat_scaled = f.feature_scaler.transform(values)
        # first test window = the `time_step` rows ending just before train_end
        assert np.allclose(f.X_test[0], feat_scaled[f.train_end - 20 : f.train_end])
        assert np.allclose(f.X_test[0][-1], feat_scaled[f.train_end - 1])
