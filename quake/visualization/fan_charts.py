import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import IntFlag
from scipy import stats
from typing import List, Tuple, Optional


__all__ = [
    'Align',
    'plot_fan_chart',
    'plot_fan_chart_by_channel'
]


class Align(IntFlag):
    PAD = 1 << 24
    TRIM = 1 << 25
    FIXED = 1 << 26


# Default color palette
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


def _stack_events(
    events: List[pd.DataFrame],
    channels: List[str] = ['Z', 'N', 'E'],
    align: Align = Align.PAD
) -> np.ndarray:
    """
    Convert list of DataFrames to stacked numpy array.

    Args:
        events: List of DataFrames, each with channel columns
        channels: List of channel names to extract
        align: Align.PAD, Align.TRIM, or Align.FIXED | length

    Returns:
        Stacked array of shape [n_samples, seq_len, n_channels]
    """
    if len(events) == 0:
        raise ValueError("No events provided")

    if align & Align.TRIM:
        target_len = min(len(e) for e in events)
    elif align & Align.FIXED:
        target_len = int(align) & 0xFFFFFF
    else:
        target_len = max(len(e) for e in events)

    arrays = []
    for event_df in events:
        available_channels = [ch for ch in channels if ch in event_df.columns]
        if not available_channels:
            raise ValueError(f"None of channels {channels} found in DataFrame columns: {event_df.columns.tolist()}")

        arr = event_df[available_channels].values

        if len(arr) > target_len:
            arr = arr[:target_len]
        elif len(arr) < target_len:
            pad_len = target_len - len(arr)
            arr = np.pad(arr, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)

        arrays.append(arr)

    return np.stack(arrays)


def plot_fan_chart(
    events: List[pd.DataFrame],
    channels: List[str] = ['Z', 'N', 'E'],
    quantiles: List[float] = [0.1, 0.25, 0.75, 0.9],
    color: str = '#1f77b4',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    zscore: bool = False,
    log_scale: bool = False,
    align: Align = Align.PAD
) -> plt.Figure:
    """
    Plot fan chart (median with quantile bands) for a list of events.
    Each channel is plotted in a separate subplot.

    Args:
        events: List of DataFrames, each with channel columns
        channels: List of channel names (e.g., ['Z', 'N', 'E'])
        quantiles: Quantile values [q1_low, q2_low, q2_high, q1_high]
        color: Color for the plot
        title: Plot title
        figsize: Figure size
        zscore: If True, apply z-score normalization per event per channel
        log_scale: If True, use logarithmic scale for y-axis
        align: Align.PAD, Align.TRIM, or Align.FIXED | length

    Returns:
        Matplotlib figure object
    """
    stacked = _stack_events(events, channels, align=align)
    n_samples, seq_len, n_channels = stacked.shape

    if zscore:
        for ch in range(n_channels):
            stacked[:, :, ch] = stats.zscore(stacked[:, :, ch], axis=1)

    available_channels = [ch for ch in channels if ch in events[0].columns]

    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]

    median = np.median(stacked, axis=0)
    q_values = np.quantile(stacked, quantiles, axis=0)
    x = np.arange(seq_len)

    for ch_idx, (ax, ch_name) in enumerate(zip(axes, available_channels)):
        # Outer band
        ax.fill_between(x, q_values[0, :, ch_idx], q_values[3, :, ch_idx], alpha=0.2, color=color)
        # Inner band
        ax.fill_between(x, q_values[1, :, ch_idx], q_values[2, :, ch_idx], alpha=0.4, color=color)
        # Median line
        ax.plot(x, median[:, ch_idx], color=color, linewidth=1.5)
        ax.set_ylabel(ch_name)
        ax.grid(True, alpha=0.3)
        if log_scale:
            ax.set_yscale('symlog')

    axes[-1].set_xlabel('Time')

    if title:
        fig.suptitle(f'{title} (n={n_samples})')
    else:
        fig.suptitle(f'Fan Chart (n={n_samples})')

    plt.tight_layout()

    return fig


def plot_fan_chart_by_channel(
    events: List[pd.DataFrame],
    channels: List[str] = ['Z', 'N', 'E'],
    quantiles: List[float] = [0.1, 0.25, 0.75, 0.9],
    color: str = '#1f77b4',
    figsize: Tuple[int, int] = (14, 8),
    zscore: bool = False,
    log_scale: bool = False,
    align: Align = Align.PAD
) -> plt.Figure:
    """
    Plot fan chart with separate subplot for each channel.

    Args:
        events: List of DataFrames
        channels: List of channel names
        quantiles: Quantile values for bands
        color: Color for the plot
        figsize: Figure size
        zscore: If True, apply z-score normalization per event per channel
        log_scale: If True, use logarithmic scale for y-axis
        align: Align.PAD, Align.TRIM, or Align.FIXED | length

    Returns:
        Matplotlib figure object
    """
    stacked = _stack_events(events, channels, align=align)
    n_samples, seq_len, n_channels = stacked.shape

    if zscore:
        for ch in range(n_channels):
            stacked[:, :, ch] = stats.zscore(stacked[:, :, ch], axis=1)

    available_channels = [ch for ch in channels if ch in events[0].columns]

    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]

    median = np.median(stacked, axis=0)
    q_values = np.quantile(stacked, quantiles, axis=0)
    x = np.arange(seq_len)

    for ch_idx, (ax, ch_name) in enumerate(zip(axes, available_channels)):
        ax.fill_between(x, q_values[0, :, ch_idx], q_values[3, :, ch_idx], alpha=0.2, color=color)
        ax.fill_between(x, q_values[1, :, ch_idx], q_values[2, :, ch_idx], alpha=0.4, color=color)
        ax.plot(x, median[:, ch_idx], color=color, linewidth=1.5)
        ax.set_ylabel(ch_name)
        ax.grid(True, alpha=0.3)
        if log_scale:
            ax.set_yscale('symlog')

    axes[-1].set_xlabel('Time')
    fig.suptitle(f'Fan Chart by Channel (n={n_samples})')
    plt.tight_layout()

    return fig


