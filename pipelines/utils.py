"""
Shared Utilities for Training Pipelines
========================================

Common functions used across different training pipelines.
"""

import numpy as np
import matplotlib.pyplot as plt

from quake.data.loader import read_pickle
from quake.visualization import plot_fan_chart, Align


def load_events(paths: list[str]) -> tuple[list, np.ndarray]:
    """Load events and labels from multiple pickle files.

    Args:
        paths: List of paths to pickle files

    Returns:
        Tuple of (events list, labels array)
    """
    events, labels = [], []
    for path in paths:
        e, l = read_pickle(path)
        events.extend(e)
        labels.extend(l)
        print(f"Loaded {len(e)} events from {path}")
    return events, np.array(labels)


def show_fan_charts(
    events: list,
    labels: np.ndarray,
    channels: tuple[str, ...] = ('Z', 'N', 'E')
) -> None:
    """Show fan chart visualization for each class.

    Displays median waveform with quantile bands (10%, 25%, 75%, 90%)
    for each unique class label.

    Args:
        events: List of event DataFrames
        labels: Array of class labels
        channels: Channel names to plot
    """
    unique = np.unique(labels)
    for label in unique:
        class_events = [e for e, l in zip(events, labels) if l == label]
        plot_fan_chart(
            class_events,
            channels=list(channels),
            title=f'Class: {label}',
            align=Align.TRIM,
            zscore=True,
            log_scale=True
        )
    plt.show()


def plot_spectrogram(
    sample: np.ndarray,
    channels: tuple[str, ...] = ('Z', 'N', 'E'),
    title: str = 'Spectrogram'
) -> None:
    """Visualize spectrogram for each channel.

    Creates a side-by-side plot of spectrograms for each seismic channel,
    with frequency on y-axis and time on x-axis. Uses viridis colormap.

    Args:
        sample: Spectrogram array with shape [n_channels, freq_bins, time_frames]
        channels: Channel names for subplot titles
        title: Overall figure title
    """
    fig, axes = plt.subplots(1, len(channels), figsize=(4 * len(channels), 4))
    for i, (ax, ch) in enumerate(zip(axes, channels)):
        im = ax.imshow(sample[i], aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'Channel {ch}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        plt.colorbar(im, ax=ax)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
