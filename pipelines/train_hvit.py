"""
Training Pipeline for Hybrid CNN-ViT Model
===========================================

This pipeline trains the HybridViT model on seismic spectrogram data.

Data Flow:
    1. Load seismic events from pickle files
    2. Visualize class distribution and fan charts
    3. Apply STFT to convert waveforms to spectrograms
    4. Train HybridViT with pretrained ViT backbone

Usage:
    python pipelines/train_hvit.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchinfo import summary

from quake.data.adapter import WaveformDataAdapter, TransformOP
from quake.models import Backbone, HybridViT
from quake.procs.train import train_model
from quake.visualization import plot_spectrogram

from pipelines.utils import load_events, show_fan_charts


def train_hybrid_vit(
    events_train,
    events_test,
    img_size: tuple[int, int],
    channels: int = 3,
    device: str = 'cuda'
) -> None:
    """Train HybridViT model on spectrogram data.

    Args:
        events_train: Training dataset
        events_test: Test dataset
        img_size: Spectrogram dimensions (freq_bins, time_frames)
        channels: Number of input channels
        device: Device to train on ('cuda' or 'cpu')
    """
    model = HybridViT(
        backbone=Backbone.TINY,
        depth=4,  # Use first 4 pretrained blocks (of 12)
        in_channels=channels,
        num_classes=2,
        img_size=img_size,
        dropout=0.2
    ).to(device)

    summary(model, input_size=(1, channels, *img_size))

    train_model(
        events_train=events_train,
        events_test=events_test,
        batch_size=64,
        batch_size_test=64,
        model=model,
        epochs=20
    )


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    channels = ('Z', 'N', 'E')

    # Load data from multiple sources
    paths = [
        './resources/VRAC.pkl',
        './resources/MORC.pkl',
        './resources/hh_all.pkl',
    ]
    events, labels = load_events(paths)
    print(f'Total events: {len(events)}')

    unique, counts = np.unique(labels, return_counts=True)
    plt.bar(unique, counts)
    plt.title('Event Distribution')
    plt.xlabel('Event Type')
    plt.ylabel('Count')
    plt.show()

    # Fan charts show median waveform with quantile bands for each class
    show_fan_charts(events, labels, channels)

    # STFT parameters
    stft_nperseg = 64
    stft_hop = 32

    adapter = WaveformDataAdapter(
        events=events,
        labels=labels,
        channels=channels,
        stft_nperseg=stft_nperseg,
        stft_hop=stft_hop,
        transforms=TransformOP.DROP_NAN | TransformOP.TRIMMING | TransformOP.ZSCORE | TransformOP.SPECTROGRAM,
        test_ratio=0.3
    )
    events_train, events_test = adapter.get_datasets()

    # Get spectrogram info
    sample = events_train[0][0]
    freq_bins, time_frames = sample.shape[1], sample.shape[2]
    print(f"Spectrogram shape: channels={sample.shape[0]}, freq_bins={freq_bins}, time_frames={time_frames}")

    plot_spectrogram(sample, channels)

    train_hybrid_vit(
        events_train,
        events_test,
        img_size=(freq_bins, time_frames),
        channels=len(channels),
        device=device
    )


if __name__ == "__main__":
    main()
