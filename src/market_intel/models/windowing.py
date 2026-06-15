"""Leakage-free sequence construction for time-series models.

The original pipeline fit the scaler on the *entire* dataset and built sliding
windows *before* splitting — both leak future information into training and
inflate reported accuracy. This module enforces the correct discipline:

1. Split chronologically *first* (by row position).
2. Fit scalers on the **train slice only**, then transform val/test with them.
3. Build windows per split so a window's target never crosses into a future
   split. Earlier rows are allowed only as *past* look-back context, which is
   information genuinely available at prediction time.

Both ``prepare_splits`` (fixed 3-way split) and ``walk_forward_splits``
(expanding-window CV) share ``_fit_train_scalers`` and ``_windows`` so the
leakage-free discipline cannot drift between the two code paths. See the tests
in tests/test_windowing.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

DEFAULT_FIELDS = ("Close", "High", "Low", "Open", "Volume")


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler
    target_col: str
    time_step: int
    train_end: int
    val_end: int
    # target row positions for each split (for index alignment / leakage checks)
    train_pos: np.ndarray
    val_pos: np.ndarray
    test_pos: np.ndarray
    index: pd.Index

    @property
    def test_index(self) -> pd.Index:
        return self.index[self.test_pos]

    def inverse_target(self, scaled: np.ndarray) -> np.ndarray:
        """Map model output (target-scaler space) back to price units."""
        scaled = np.asarray(scaled).reshape(-1, 1)
        return self.target_scaler.inverse_transform(scaled).ravel()


@dataclass
class WalkForwardFold:
    fold: int
    train_end: int
    stop: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler

    def inverse_target(self, scaled: np.ndarray) -> np.ndarray:
        scaled = np.asarray(scaled).reshape(-1, 1)
        return self.target_scaler.inverse_transform(scaled).ravel()


def chronological_split_indices(
    n: int, train_frac: float = 0.70, val_frac: float = 0.15
) -> tuple[int, int]:
    """Return ``(train_end, val_end)`` row positions for a chronological split."""
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("require 0<train_frac<1, 0<=val_frac, train_frac+val_frac<1")
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return train_end, val_end


def _fit_train_scalers(
    values: np.ndarray, train_end: int, target_col_idx: int
) -> tuple[MinMaxScaler, MinMaxScaler, np.ndarray, np.ndarray]:
    """Fit feature + target scalers on the TRAIN slice only, then transform all.

    The single place scalers are fit. ``fit`` sees only ``values[:train_end]``;
    transforming later rows with those train-derived min/max is legitimate
    inference behavior, not leakage.
    """
    target = values[:, target_col_idx : target_col_idx + 1]
    feature_scaler = MinMaxScaler((0, 1)).fit(values[:train_end])
    target_scaler = MinMaxScaler((0, 1)).fit(target[:train_end])
    return (
        feature_scaler,
        target_scaler,
        feature_scaler.transform(values),
        target_scaler.transform(target),
    )


def _windows(
    feat_scaled: np.ndarray,
    target_scaled: np.ndarray,
    lo: int,
    hi: int,
    time_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windows whose *target* row ``i`` lies in ``[max(lo, time_step), hi)``.

    The input window is ``feat_scaled[i-time_step:i]`` (strictly past rows, the
    half-open slice excludes ``i``), so no window ever peeks at or beyond its own
    target.
    """
    start = max(lo, time_step)
    xs, ys, pos = [], [], []
    for i in range(start, hi):
        xs.append(feat_scaled[i - time_step : i, :])
        ys.append(target_scaled[i, 0])
        pos.append(i)
    if not xs:
        n_features = feat_scaled.shape[1]
        empty_x = np.empty((0, time_step, n_features))
        return empty_x, np.empty((0,)), np.empty((0,), dtype=int)
    return np.asarray(xs), np.asarray(ys), np.asarray(pos, dtype=int)


def prepare_splits(
    df: pd.DataFrame,
    feature_cols: tuple[str, ...] = DEFAULT_FIELDS,
    target_col: str = "Close",
    time_step: int = 60,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> SplitData:
    """Build leakage-free train/val/test sequences from a price frame."""
    if target_col not in feature_cols:
        raise ValueError(f"target_col {target_col!r} must be in feature_cols")

    values = df[list(feature_cols)].to_numpy(dtype="float64")
    n = len(values)
    if n <= time_step + 2:
        raise ValueError(f"need more than time_step+2={time_step + 2} rows, got {n}")

    train_end, val_end = chronological_split_indices(n, train_frac, val_frac)
    if train_end <= time_step:
        raise ValueError(
            f"train slice ({train_end}) must exceed time_step ({time_step}); "
            "use a smaller time_step or more data"
        )

    tcol = feature_cols.index(target_col)
    feature_scaler, target_scaler, feat_scaled, target_scaled = _fit_train_scalers(
        values, train_end, tcol
    )

    X_train, y_train, train_pos = _windows(feat_scaled, target_scaled, 0, train_end, time_step)
    X_val, y_val, val_pos = _windows(feat_scaled, target_scaled, train_end, val_end, time_step)
    X_test, y_test, test_pos = _windows(feat_scaled, target_scaled, val_end, n, time_step)

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        target_col=target_col,
        time_step=time_step,
        train_end=train_end,
        val_end=val_end,
        train_pos=train_pos,
        val_pos=val_pos,
        test_pos=test_pos,
        index=df.index,
    )


def walk_forward_splits(
    df: pd.DataFrame,
    feature_cols: tuple[str, ...] = DEFAULT_FIELDS,
    target_col: str = "Close",
    time_step: int = 60,
    n_splits: int = 5,
) -> list[WalkForwardFold]:
    """Build leakage-free expanding-window folds (sklearn ``TimeSeriesSplit``).

    TensorFlow-free so the leakage discipline can be unit-tested without training.
    Each fold fits scalers on its own train slice (past only) and windows the
    immediately-following test rows. Folds whose train slice is shorter than
    ``time_step`` are skipped.
    """
    if target_col not in feature_cols:
        raise ValueError(f"target_col {target_col!r} must be in feature_cols")

    values = df[list(feature_cols)].to_numpy(dtype="float64")
    tcol = feature_cols.index(target_col)
    n = len(values)

    folds: list[WalkForwardFold] = []
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.arange(n))):
        train_end = int(train_idx[-1]) + 1
        stop = int(test_idx[-1]) + 1
        if train_end <= time_step:
            continue

        feat_scaler, tgt_scaler, feat_s, tgt_s = _fit_train_scalers(values, train_end, tcol)
        X_tr, y_tr, _ = _windows(feat_s, tgt_s, 0, train_end, time_step)
        X_te, y_te, _ = _windows(feat_s, tgt_s, train_end, stop, time_step)
        if not len(X_tr) or not len(X_te):
            continue

        folds.append(
            WalkForwardFold(
                fold=fold,
                train_end=train_end,
                stop=stop,
                X_train=X_tr,
                y_train=y_tr,
                X_test=X_te,
                y_test=y_te,
                feature_scaler=feat_scaler,
                target_scaler=tgt_scaler,
            )
        )
    return folds
