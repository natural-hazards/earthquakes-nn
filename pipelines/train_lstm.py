"""
Training Pipeline for LSTM Models
==================================

This pipeline trains LSTM-based models on seismic data with data augmentation.
Time series augmentations are applied on-the-fly during training.

Data Flow:
    1. Load seismic events from pickle files
    2. Preprocess: DROP_NAN → crop to `duration` → ZSCORE → zero-pad (PADDING)
    3. Optionally undersample the majority class in the training split
    4. Training: augmentation → FFT → model
    5. Testing: FFT → model (no augmentation)

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
from quake.models.lstm_svm import LSTMAttentionSVMModel
from quake.procs.train import train_model, train_svm_model

from pipelines.utils import load_events, filter_by_labels, show_fan_charts, undersample


def train_lstm(
    events_train,
    events_test,
    seq_len: int,
    channels: int = 3,
    classes: int = 2,
    device: str = 'cuda'
) -> None:
    """Train standard LSTM model."""
    model = LSTMModel(
        channels=channels,
        classes=classes,
        hidden=128,
        layers=3,
        dropout=0.3
    ).to(device)

    summary(model, input_size=(1, seq_len, channels))

    train_model(
        events_train=events_train,
        events_test=events_test,
        model=model,
        epochs=30,
        device=device,
        weighting=None
    )


def train_lstm_attention(
    events_train,
    events_test,
    seq_len: int,
    channels: int = 3,
    classes: int = 2,
    device: str = 'cuda'
) -> None:
    """Train LSTM model with multi-head self-attention."""
    model = LSTMAttentionModel(
        channels=channels,
        classes=classes,
        hidden=64,
        layers=8,
        heads=4,
        dropout=0.3
    ).to(device)

    summary(model, input_size=(1, seq_len, channels))

    train_model(
        events_train=events_train,
        events_test=events_test,
        epochs=30,
        model=model,
        device=device,
        weighting=None
    )


def train_lstm_svm(
    events_train,
    events_test,
    seq_len: int,
    channels: int = 3,
    classes: int = 2,
    device: str = 'cuda'
) -> None:
    """Train LSTM + Attention model with SVM classifier."""
    model = LSTMAttentionSVMModel(
        channels=channels,
        classes=classes,
        hidden=128,
        layers=8,
        heads=4,
        dropout=0.3,
        svm_alpha=1e-6,
        svm_eta0=1e-3
    ).to(device)

    summary(model, input_size=(1, seq_len, channels))

    train_svm_model(
        events_train=events_train,
        events_test=events_test,
        model=model,
        epochs=30,
        compile_model=False
    )


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # channels = ('Z', 'N', 'E')
    channels = ('Z', 'N', 'E')

    # Event length: longer events are cropped, shorter ones zero-padded
    sampling_rate = 125  # Hz
    duration = 8  # s (1000 samples; longer records degraded the FFT-based LSTM)
    # Max training events per class (None = use all)
    max_train_per_class = None  # e.g. {'M': 2000}

    # Load data from multiple sources
    paths = [
        # './resources/VRAC.pkl',
        # './resources/MORC.pkl',
        # './resources/hh_all.pkl',
        # './resources/SPF.pkl',
        './resources/all_events.pkl'
    ]
    events, labels = load_events(paths)
    # events, labels = filter_by_labels(events, labels, keep=('T', 'M'))

    unique, counts = np.unique(labels, return_counts=True)
    classes = len(unique)
    print(f'Number of events: {len(events)}')
    print(f'Event types: {unique}')

    plt.bar(unique, counts)
    plt.title('Event Distribution')
    plt.xlabel('Event Type')
    plt.ylabel('Count')
    plt.show()

    # Fan charts show median waveform with quantile bands for each class
    # show_fan_charts(events, labels, channels)

    # Preprocessing: z-score only (FFT done dynamically)
    seq_len = int(duration * sampling_rate)
    adapter = WaveformDataAdapter(
        events=events,
        labels=labels,
        channels=channels,
        transforms=TransformOP.DROP_NAN | TransformOP.PADDING | TransformOP.ZSCORE,
        test_ratio=0.3,
        seq_len=seq_len,
        random_state=42
    )

    # FFT over the whole event (next power of two)
    fft_size = 1 << (seq_len - 1).bit_length()

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

    if max_train_per_class is not None:
        # Adapter encodes labels in sorted order (LabelEncoder)
        encoded = {int(np.searchsorted(unique, k)): n for k, n in max_train_per_class.items()}
        events_train = undersample(events_train, encoded, seed=42)
        print(f'Training events per class after undersampling: {np.bincount(events_train.labels)}')

    # Get model input length (FFT bins) from data
    sample, _ = events_train[0]
    seq_len = sample.shape[0]
    print(f"Input shape: seq_len={seq_len}, channels={len(channels)}")

    train_lstm(events_train, events_test, seq_len=seq_len, channels=len(channels), classes=classes, device=device)
    # train_lstm_attention(events_train, events_test, seq_len=seq_len, channels=len(channels), classes=classes, device=device)
    # train_lstm_svm(events_train, events_test, seq_len=seq_len, channels=len(channels), classes=classes, device=device)


if __name__ == "__main__":
    main()
