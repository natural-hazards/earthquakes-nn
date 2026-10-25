"""Resampling of waveform events between sampling rates."""

from math import gcd

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from tqdm import tqdm


__all__ = [
    'resample_events'
]


def resample_events(
    events: list[pd.DataFrame] | tuple[pd.DataFrame, ...],
    fs_in: float,
    fs_out: float
) -> list[pd.DataFrame]:
    """Resample every event from `fs_in` to `fs_out` Hz.

    Uses polyphase filtering (``scipy.signal.resample_poly``) with the
    built-in anti-aliasing low-pass, so down-sampling (e.g. 125 -> 100 Hz)
    does not alias energy above the new Nyquist frequency.

    Args:
        events: Event DataFrames, one column per channel
        fs_in: Sampling rate of the input events [Hz]
        fs_out: Target sampling rate [Hz]

    Returns:
        New list of float32 DataFrames with the same columns
    """
    if fs_in == fs_out:
        return list(events)

    # 125 -> 100 Hz becomes up=4, down=5
    scale = 1000
    up, down = int(fs_out * scale), int(fs_in * scale)
    g = gcd(up, down)
    up, down = up // g, down // g

    out = []
    for event in tqdm(events, desc=f'Resampling {fs_in:g} -> {fs_out:g} Hz'):
        data = resample_poly(event.to_numpy(dtype=np.float64), up, down, axis=0)
        out.append(pd.DataFrame(data.astype(np.float32), columns=event.columns))
    return out
