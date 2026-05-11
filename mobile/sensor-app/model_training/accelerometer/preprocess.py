"""Windowing + normalization helpers. No file I/O at module level."""
from __future__ import annotations

import numpy as np

from config import WINDOW_SAMPLES, WINDOW_STRIDE_SAMPLES


def window_signal(
    signal: np.ndarray,
    window: int = WINDOW_SAMPLES,
    stride: int = WINDOW_STRIDE_SAMPLES,
) -> np.ndarray:
    """Return shape (num_windows, window, channels)."""
    if signal.shape[0] < window:
        return np.empty((0, window, signal.shape[1]), dtype=signal.dtype)
    num = 1 + (signal.shape[0] - window) // stride
    out = np.empty((num, window, signal.shape[1]), dtype=signal.dtype)
    for i in range(num):
        start = i * stride
        out[i] = signal[start : start + window]
    return out


def compute_norm_stats(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std computed over training windows only."""
    flat = windows.reshape(-1, windows.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    # Guard against zero-variance channels (shouldn't happen with real data).
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(windows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((windows - mean) / std).astype(np.float32)
