"""LSTM price forecasting — leakage-free training, evaluation, and walk-forward CV.

Same model architecture as the original ``multi_feature_model.py`` (2x LSTM(50)
+ Dropout + Dense), but fed by ``windowing.prepare_splits`` so the scaler is fit
on train only and windows never cross split boundaries. Adds a walk-forward
(expanding-window) evaluator, the realistic way to score a time-series model.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from market_intel.data.loaders import load_prices
from market_intel.models.windowing import (
    DEFAULT_FIELDS,
    SplitData,
    prepare_splits,
    walk_forward_splits,
)


def set_seeds(seed: int = 42) -> None:
    """Best-effort reproducibility (CPU)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


def build_model(time_step: int, n_features: int, units: int = 50, dropout: float = 0.2):
    """Build and compile the LSTM. Imports TensorFlow lazily."""
    from tensorflow.keras import Input, Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential(
        [
            Input(shape=(time_step, n_features)),
            LSTM(units, return_sequences=True),
            Dropout(dropout),
            LSTM(units, return_sequences=False),
            Dropout(dropout),
            Dense(25, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@dataclass
class TrainResult:
    metrics: dict[str, float]
    test_index: pd.Index
    y_test_actual: np.ndarray
    y_test_pred: np.ndarray
    model: object = field(repr=False, default=None)


def train_evaluate(
    split: SplitData,
    epochs: int = 20,
    batch_size: int = 32,
    verbose: int = 0,
    seed: int = 42,
    early_stopping_patience: int = 5,
) -> TrainResult:
    """Train on the prepared split and return price-unit RMSE/MAE per partition.

    When a validation split exists, EarlyStopping(restore_best_weights=True)
    uses it for model selection. This is leakage-free: EarlyStopping only reads
    ``val_loss`` and never folds val data into weights or scaling.
    """
    set_seeds(seed)
    model = build_model(split.time_step, split.X_train.shape[2])

    validation_data = (split.X_val, split.y_val) if len(split.X_val) else None
    callbacks = []
    if validation_data is not None and early_stopping_patience:
        from tensorflow.keras.callbacks import EarlyStopping

        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
            )
        )
    model.fit(
        split.X_train,
        split.y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
    )

    metrics: dict[str, float] = {}
    partitions = {"train": (split.X_train, split.y_train), "test": (split.X_test, split.y_test)}
    if validation_data is not None:
        partitions["val"] = (split.X_val, split.y_val)

    test_actual = test_pred = None
    for name, (X, y) in partitions.items():
        if not len(X):
            continue
        pred = split.inverse_target(model.predict(X, verbose=0))
        actual = split.inverse_target(y)
        metrics[f"{name}_rmse"] = _rmse(actual, pred)
        metrics[f"{name}_mae"] = float(mean_absolute_error(actual, pred))
        if name == "test":
            test_actual, test_pred = actual, pred

    return TrainResult(
        metrics=metrics,
        test_index=split.test_index,
        y_test_actual=test_actual if test_actual is not None else np.empty((0,)),
        y_test_pred=test_pred if test_pred is not None else np.empty((0,)),
        model=model,
    )


def walk_forward_eval(
    df: pd.DataFrame,
    feature_cols: tuple[str, ...] = DEFAULT_FIELDS,
    target_col: str = "Close",
    time_step: int = 60,
    n_splits: int = 5,
    epochs: int = 10,
    batch_size: int = 32,
    verbose: int = 0,
    seed: int = 42,
) -> dict:
    """Expanding-window walk-forward CV (sklearn ``TimeSeriesSplit``).

    Each fold trains only on its past and tests on the immediately following
    rows; scalers are re-fit on each fold's train slice (no leakage across folds).
    Fold construction lives in the TF-free ``windowing.walk_forward_splits`` so
    the leakage discipline is shared with ``prepare_splits`` and unit-testable.
    Returns per-fold test RMSE/MAE and their means.
    """
    wf_folds = walk_forward_splits(
        df, feature_cols=feature_cols, target_col=target_col, time_step=time_step, n_splits=n_splits
    )
    if not wf_folds:
        raise RuntimeError("no usable folds — reduce time_step or n_splits")

    folds: list[dict] = []
    for f in wf_folds:
        set_seeds(seed + f.fold)  # per-fold diversity
        model = build_model(time_step, f.X_train.shape[2])
        model.fit(f.X_train, f.y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

        pred = f.inverse_target(model.predict(f.X_test, verbose=0))
        actual = f.inverse_target(f.y_test)
        folds.append(
            {
                "fold": f.fold,
                "train_end": f.train_end,
                "n_test": int(len(f.y_test)),
                "rmse": _rmse(actual, pred),
                "mae": float(mean_absolute_error(actual, pred)),
            }
        )

    return {
        "folds": folds,
        "mean_rmse": float(np.mean([f["rmse"] for f in folds])),
        "mean_mae": float(np.mean([f["mae"] for f in folds])),
    }


def run(
    ticker: str,
    dataset: str | None = None,
    data_dir: str = "data",
    time_step: int = 60,
    epochs: int = 20,
    batch_size: int = 32,
    plot: bool = False,
    plot_path: str | None = None,
    walk_forward: int = 0,
    verbose: int = 0,
) -> dict:
    """Load a ticker, train a leakage-free LSTM, print and return metrics."""
    df = load_prices(ticker, data_dir=data_dir, dataset=dataset)
    split = prepare_splits(df, time_step=time_step)
    result = train_evaluate(split, epochs=epochs, batch_size=batch_size, verbose=verbose)

    print(f"\n=== {ticker} — single chronological split (leakage-free) ===")
    for k in ("train_rmse", "val_rmse", "test_rmse", "test_mae"):
        if k in result.metrics:
            print(f"  {k:>10}: {result.metrics[k]:.4f}")

    out: dict = {"ticker": ticker, "metrics": result.metrics}

    if walk_forward:
        wf = walk_forward_eval(df, time_step=time_step, n_splits=walk_forward, epochs=epochs)
        print(f"\n=== {ticker} — walk-forward ({len(wf['folds'])} folds) ===")
        for f in wf["folds"]:
            print(f"  fold {f['fold']}: test RMSE {f['rmse']:.4f}  (n_test={f['n_test']})")
        print(f"  mean test RMSE: {wf['mean_rmse']:.4f}")
        out["walk_forward"] = wf

    if plot:
        _plot(ticker, result, plot_path)

    return out


def _plot(ticker: str, result: TrainResult, plot_path: str | None) -> None:
    import matplotlib

    if plot_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(result.test_index, result.y_test_actual, label="Actual")
    plt.plot(result.test_index, result.y_test_pred, label="Predicted")
    plt.legend()
    plt.title(f"Stock Price Prediction for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    if plot_path:
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"\nSaved plot to {plot_path}")
    else:
        plt.show()
