# Data Processing Module

This module provides data loading, preprocessing, augmentation, and transformation utilities for seismic waveform classification.

## Pipeline Architecture

```
PREPROCESSING (once, in WaveformDataAdapter)
┌─────────────────────────────────────────────────────────────────┐
│   Raw Events → DROP_NAN → TRIMMING → ZSCORE → [time series]    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
TRAINING (on-the-fly, in WaveformDataset via transform)
┌─────────────────────────────────────────────────────────────────┐
│   Time series                                                   │
│       ↓                                                         │
│   TIME AUGMENTATION (TimeShift, AddNoise, TimeStretch, etc.)   │
│       ↓                                                         │
│   TRANSFORM (FFTTransform or STFTTransform)                    │
│       ↓                                                         │
│   VISION AUGMENTATION (optional: torchvision transforms)       │
│       ↓                                                         │
│   Model                                                         │
└─────────────────────────────────────────────────────────────────┘
```

## Modules

### adapter.py

| Class | Description |
|-------|-------------|
| `WaveformDataAdapter` | Handles preprocessing and train/test splitting |
| `TransformOP` | Flags for preprocessing operations (DROP_NAN, TRIMMING, ZSCORE, PCA) |

### dataset.py

| Class | Description |
|-------|-------------|
| `WaveformDataset` | PyTorch-compatible dataset with optional on-the-fly transforms |

### augmentation.py

Time-series augmentations applied **before** FFT/STFT. All classes inherit from `nn.Module`.

| Class | Description | Key Parameters |
|-------|-------------|----------------|
| `TimeShift` | Circular shift in time domain | `max_shift`: fraction of sequence (0.0-1.0) |
| `AddNoise` | Gaussian noise with configurable SNR | `snr_min`, `snr_max`: SNR range in dB |
| `AmplitudeScale` | Random amplitude scaling | `scale_min`, `scale_max`: scale factor range |
| `TimeStretch` | Time stretching/compression | `rate_min`, `rate_max`: stretch rate range |
| `ChannelDropout` | Random channel zeroing | `p_drop`: probability, `max_channels`: max to drop |
| `Compose` | Chain multiple augmentations | `augmentations`: list of nn.Module |
| `RandomApply` | Apply augmentation with probability | `augmentation`: nn.Module, `p`: probability |

### transforms.py

Signal transforms for converting time series to frequency domain. All classes inherit from `nn.Module`.

| Class | Input Shape | Output Shape | Description |
|-------|-------------|--------------|-------------|
| `ToTensor` | `[seq_len, channels]` | `Tensor[seq_len, channels]` | Convert numpy to torch tensor |
| `FFTTransform` | `[seq_len, channels]` | `[fft_output_size, channels]` | FFT magnitude spectrum |
| `STFTTransform` | `[seq_len, channels]` | `[channels, freq_bins, time_frames]` | STFT spectrogram |

## Recommended Augmentation Parameters

### Time-Series Augmentations

| Augmentation | Parameter | Recommended | Rationale |
|--------------|-----------|-------------|-----------|
| TimeShift | max_shift | 0.1 (10%) | P-wave arrival time varies |
| AddNoise | snr_min/max | 15-40 dB | Typical seismic noise levels |
| AmplitudeScale | scale_min/max | 0.9-1.1 | Magnitude/distance variation |
| TimeStretch | rate_min/max | 0.95-1.05 | Path effects, small variations |
| ChannelDropout | p_drop | 0.3 | Simulate sensor failure |

### Vision Augmentations (for spectrograms)

| Transform | Parameter | Recommended | Rationale |
|-----------|-----------|-------------|-----------|
| `T.RandomErasing` | p, scale | 0.3, (0.02, 0.1) | SpecAugment-style masking |
| `T.GaussianBlur` | kernel_size, sigma | 3, (0.1, 0.5) | Mild frequency smoothing |
| `T.ColorJitter` | brightness, contrast | 0.2, 0.2 | Intensity variation |
| `T.RandomAffine` | translate | (0.05, 0.05) | Small time/freq shifts |

## Usage Examples

### LSTM Pipeline (FFT)

```python
from quake.data.adapter import WaveformDataAdapter, TransformOP
from quake.data.augmentation import Compose, RandomApply, TimeShift, AddNoise
from quake.data.transforms import FFTTransform, ToTensor
import torch.nn as nn

adapter = WaveformDataAdapter(
    events=events,
    labels=labels,
    transforms=TransformOP.DROP_NAN | TransformOP.TRIMMING | TransformOP.ZSCORE,
)

train_transform = nn.Sequential(
    Compose([
        RandomApply(TimeShift(max_shift=0.1), p=0.5),
        RandomApply(AddNoise(snr_min=15, snr_max=40), p=0.5),
    ]),
    FFTTransform(fft_size=512),
    ToTensor(),
)

test_transform = nn.Sequential(
    FFTTransform(fft_size=512),
    ToTensor(),
)

train_ds, test_ds = adapter.get_datasets(
    transform_train=train_transform,
    transform_test=test_transform
)
```

### HybridViT Pipeline (STFT + Vision Augmentation)

```python
from quake.data.adapter import WaveformDataAdapter, TransformOP
from quake.data.augmentation import Compose, RandomApply, TimeShift, AddNoise, AmplitudeScale
from quake.data.transforms import STFTTransform, ToTensor
from torchvision import transforms as T
import torch.nn as nn

adapter = WaveformDataAdapter(
    events=events,
    labels=labels,
    transforms=TransformOP.DROP_NAN | TransformOP.TRIMMING | TransformOP.ZSCORE,
)

train_transform = nn.Sequential(
    Compose([
        RandomApply(TimeShift(max_shift=0.1), p=0.5),
        RandomApply(AddNoise(snr_min=15, snr_max=40), p=0.5),
        RandomApply(AmplitudeScale(0.9, 1.1), p=0.3),
    ]),
    STFTTransform(nperseg=64, hop=32),
    ToTensor(),
    T.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
)

test_transform = nn.Sequential(
    STFTTransform(nperseg=64, hop=32),
    ToTensor(),
)

train_ds, test_ds = adapter.get_datasets(
    transform_train=train_transform,
    transform_test=test_transform
)
```
