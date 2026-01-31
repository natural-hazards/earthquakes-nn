"""
Training Pipeline for EfficientNet Model
=========================================

This pipeline trains the EfficientNet model on seismic spectrogram data with augmentation.
Time series augmentations are applied before STFT, spectrogram augmentations after.

Data Flow:
    1. Load seismic events from pickle files
    2. Preprocess: DROP_NAN → TRIMMING → ZSCORE
    3. Training: time augmentation → STFT → tensor → spec augmentation → model
    4. Testing: STFT → tensor → model (no augmentation)

Usage:
    python pipelines/train_efficientnet.py
"""

import torch
import torch.nn as nn
from torchvision import transforms as T
from torchinfo import summary

from quake.data.adapter import WaveformDataAdapter, TransformOP
from quake.data.augmentation import (
    Compose, RandomApply, TimeShift, AddNoise, AmplitudeScale
)
from quake.data.transforms import STFTTransform, ToTensor
from quake.models import EfficientNetBackbone, EfficientNet
from quake.procs.train import train_model

from pipelines.utils import load_events


def train_efficientnet(
    events_train,
    events_test,
    channels: int = 3,
    device: str = 'cuda'
) -> None:
    """Train EfficientNet model on spectrogram data."""
    model = EfficientNet(
        backbone=EfficientNetBackbone.B0,
        in_channels=channels,
        num_classes=2,
        dropout=0.2
    ).to(device)

    summary(model, input_size=(1, channels, 64, 128))

    train_model(
        events_train=events_train,
        events_test=events_test,
        batch_size=64,
        batch_size_test=64,
        model=model,
        epochs=30
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

    # Preprocessing
    adapter = WaveformDataAdapter(
        events=events,
        labels=labels,
        channels=channels,
        transforms=TransformOP.DROP_NAN | TransformOP.TRIMMING | TransformOP.ZSCORE,
        test_ratio=0.3
    )

    # STFT parameters
    stft_nperseg = 64
    stft_hop = 32

    # Training transform: time augmentation → STFT → tensor → spec augmentation
    train_transform = nn.Sequential(
        Compose([
            RandomApply(TimeShift(max_shift=0.1), p=0.5),
            RandomApply(AddNoise(snr_min=15, snr_max=40), p=0.5),
            RandomApply(AmplitudeScale(0.9, 1.1), p=0.3),
        ]),
        STFTTransform(nperseg=stft_nperseg, hop=stft_hop),
        ToTensor(),
        T.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    )

    # Test transform: STFT → tensor (no augmentation)
    test_transform = nn.Sequential(
        STFTTransform(nperseg=stft_nperseg, hop=stft_hop),
        ToTensor(),
    )

    events_train, events_test = adapter.get_datasets(
        transform_train=train_transform,
        transform_test=test_transform
    )

    # Get spectrogram info
    sample, _ = events_train[0]
    freq_bins, time_frames = sample.shape[1], sample.shape[2]
    print(f"Spectrogram shape: channels={sample.shape[0]}, freq_bins={freq_bins}, time_frames={time_frames}")

    train_efficientnet(
        events_train,
        events_test,
        channels=len(channels),
        device=device
    )


if __name__ == "__main__":
    main()
