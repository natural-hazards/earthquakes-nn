"""
Training Pipeline for LSTM Models
==================================

This pipeline trains LSTM-based models on seismic data. The seismic
waveforms are converted to frequency domain using FFT transform before
being fed to the model.

Data Flow:
    1. Load seismic events from pickle files
    2. Visualize class distribution and fan charts
    3. Convert waveforms to frequency domain using FFT
    4. Train LSTM or LSTM with attention

Usage:
    python pipelines/train_lstm.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from quake.data.adapter import WaveformDataAdapter, TransformOP
from quake.models.lstm import LSTMModel
from quake.models.lstm_mhsa import LSTMAttentionModel
from quake.procs.train import train_model

from pipelines.utils import load_events, show_fan_charts


def train_lstm(
    events_train,
    events_test,
    channels: int = 3,
    device: str = 'cuda'
) -> None:
    """Train standard LSTM model.

    The model processes seismic waveforms that have been converted to
    frequency domain using FFT transform.

    Args:
        events_train: Training dataset
        events_test: Test dataset
        channels: Number of input channels
        device: Device to train on ('cuda' or 'cpu')
    """
    model = LSTMModel(
        channels=channels,
        classes=2,
        hidden=64,
        layers=2,
        dropout=0.3
    ).to(device)

    train_model(
        events_train=events_train,
        events_test=events_test,
        model=model
    )

def train_lstm_attention(
    events_train,
    events_test,
    channels: int = 3,
    device: str = 'cuda'
) -> None:
    """Train LSTM model with multi-head self-attention.

    The model processes seismic waveforms that have been converted to
    frequency domain using FFT transform. Adds attention mechanism on top
    of LSTM for better sequence modeling.

    Args:
        events_train: Training dataset
        events_test: Test dataset
        channels: Number of input channels
        device: Device to train on ('cuda' or 'cpu')
    """
    model = LSTMAttentionModel(
        channels=channels,
        classes=2,
        hidden=64,
        layers=2,
        heads=2,
        dropout=0.3
    ).to(device)

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

    adapter = WaveformDataAdapter(
        events=events,
        labels=labels,
        channels=channels,
        fft_size=512,
        transforms=TransformOP.DROP_NAN | TransformOP.TRIMMING | TransformOP.ZSCORE | TransformOP.FFT,
        test_ratio=0.3
    )
    events_train, events_test = adapter.get_datasets()

    train_lstm(events_train, events_test, channels=len(channels), device=device)
    # train_lstm_attention(events_train, events_test, channels=len(channels), device=device)

if __name__ == "__main__":
    main()