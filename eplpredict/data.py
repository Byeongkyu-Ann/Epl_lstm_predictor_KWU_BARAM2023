"""Dataset helpers shared by the training scripts."""
import numpy as np
import pandas as pd


def chronological_split(df: pd.DataFrame, train_ratio: float):
    """Split without shuffling. The first rows train and the last rows test.

    The CSV must be sorted from the oldest to the newest match.
    """
    n_train = int(len(df) * train_ratio)
    return df.iloc[:n_train], df.iloc[n_train:]


def make_sequences(x: np.ndarray, y: np.ndarray, seq_length: int):
    """Build sliding windows.

    Window k covers rows k .. k+seq_length-1 and is paired with the label of its
    last row. For seq_length = 1 every row becomes one sample.
    Returns x with shape (samples, seq_length, features) and y with shape (samples, ...).
    """
    if len(x) < seq_length:
        raise ValueError(f"Need at least {seq_length} rows, got {len(x)}.")
    xs = [x[i:i + seq_length] for i in range(len(x) - seq_length + 1)]
    ys = [y[i + seq_length - 1] for i in range(len(x) - seq_length + 1)]
    return np.asarray(xs, dtype=np.float32), np.asarray(ys)
