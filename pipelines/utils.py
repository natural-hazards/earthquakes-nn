"""
Shared Utilities for Training Pipelines
========================================

Common functions used across different training pipelines.
"""

import numpy as np
import matplotlib.pyplot as plt

from quake.data.dataset import WaveformDataset
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


def filter_by_labels(
    events: list,
    labels: np.ndarray,
    keep: tuple[str, ...] = ('M', 'T')
) -> tuple[list, np.ndarray]:
    """Keep only events with labels in `keep`.

    Args:
        events: List of event DataFrames
        labels: Array of class labels
        keep: Label values to keep

    Returns:
        Filtered (events, labels)
    """
    mask = np.isin(labels, keep)
    return [e for e, m in zip(events, mask) if m], labels[mask]


def undersample(
    dataset: WaveformDataset,
    max_per_class: dict[int, int],
    seed: int | None = None
) -> WaveformDataset:
    """Randomly keep at most `max_per_class[c]` samples of class `c`.

    Intended for the training split only, so the test split keeps
    the natural class distribution.

    Args:
        dataset: Dataset to undersample
        max_per_class: Encoded class label -> maximum number of samples
        seed: Random seed

    Returns:
        New dataset with the same transform
    """
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(dataset.labels):
        idx = np.flatnonzero(dataset.labels == c)
        n = max_per_class.get(c, len(idx))
        keep.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    return dataset.subset(np.sort(np.concatenate(keep)))


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
