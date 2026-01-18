"""
Training Pipeline for LSTM Models
==================================

This pipeline trains LSTM-based models on seismic data with data augmentation.
Time series augmentations are applied on-the-fly during training.

Data Flow:
    1. Load seismic events from pickle files
    2. Preprocess: DROP_NAN → TRIMMING → ZSCORE
    3. Training: augmentation → FFT → model
    4. Testing: FFT → model (no augmentation)

Usage:
    python pipelines/train_lstm.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchinfo import summary

from quake.data.adapter import WaveformDataAdapter, TransformOP
from quake.data.augmentation import (
    Compose, RandomApply, TimeShift, AddNoise,
    AmplitudeScale, TimeStretch
)
from quake.data.transforms import FFTTransform, ToTensor
from quake.models.lstm import LSTMModel
from quake.models.lstm_mhsa import LSTMAttentionModel
from quake.procs.train import train_model

from pipelines.utils import load_events, show_fan_charts


def train_lstm(
    events_train,
    events_test,
    seq_len: int,
    channels: int = 3,
    device: str = 'cuda'
) -> None:
    """Train standard LSTM model."""
    model = LSTMModel(
        channels=channels,
        classes=2,
        hidden=128,
        layers=3,
        dropout=0.3
    ).to(device)

    summary(model, input_size=(1, seq_len, channels))

    train_model(
        events_train=events_train,
        events_test=events_test,
        model=model,
        epochs=30
    )


def train_lstm_attention(
    events_train,
    events_test,
    seq_len: int,
    channels: int = 3,
    device: str = 'cuda'
) -> None:
    """Train LSTM model with multi-head self-attention."""
    model = LSTMAttentionModel(
        channels=channels,
        classes=2,
        hidden=64,
        layers=2,
        heads=2,
        dropout=0.3
    ).to(device)

    summary(model, input_size=(1, seq_len, channels))

    train_model(
        events_train=events_train,
        events_test=events_test,
        model=model
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

    unique, counts = np.unique(labels, return_counts=True)
    print(f'Number of events: {len(events)}')
    print(f'Event types: {unique}')

    plt.bar(unique, counts)
    plt.title('Event Distribution')
    plt.xlabel('Event Type')
    plt.ylabel('Count')
    plt.show()

    # Fan charts show median waveform with quantile bands for each class
    show_fan_charts(events, labels, channels)

    # Preprocessing: z-score only (FFT done dynamically)
    adapter = WaveformDataAdapter(
        events=events,
        labels=labels,
        channels=channels,
        transforms=TransformOP.DROP_NAN | TransformOP.TRIMMING | TransformOP.ZSCORE,
        test_ratio=0.3
    )

    # FFT parameters
    fft_size = 512

    # Training transform: augmentation → FFT → tensor
    train_transform = nn.Sequential(
        Compose([
            RandomApply(TimeShift(max_shift=0.1), p=0.5),
            RandomApply(AddNoise(snr_min=15, snr_max=40), p=0.5),
            RandomApply(AmplitudeScale(0.9, 1.1), p=0.3),
            RandomApply(TimeStretch(0.95, 1.05), p=0.3),
        ]),
        FFTTransform(fft_size=fft_size),
        ToTensor(),
    )

    # Test transform: FFT → tensor (no augmentation)
    test_transform = nn.Sequential(
        FFTTransform(fft_size=fft_size),
        ToTensor(),
    )

    events_train, events_test = adapter.get_datasets(
        transform_train=train_transform,
        transform_test=test_transform
    )

    # Get sequence length from data
    sample, _ = events_train[0]
    seq_len = sample.shape[0]
    print(f"Input shape: seq_len={seq_len}, channels={len(channels)}")

    train_lstm(events_train, events_test, seq_len=seq_len, channels=len(channels), device=device)
    # train_lstm_attention(events_train, events_test, seq_len=seq_len, channels=len(channels), device=device)


if __name__ == "__main__":
    main()
