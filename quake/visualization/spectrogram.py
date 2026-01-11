"""Spectrogram visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt


__all__ = ['plot_spectrogram']


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
