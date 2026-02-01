# EfficientNet Training Pipeline

This pipeline trains an EfficientNet model on seismic spectrogram data with on-the-fly data augmentation. Time-series augmentations are applied before STFT, and vision augmentations are applied to the resulting spectrograms.

## Usage

```bash
python pipelines/train_efficientnet.py
```

## Data Flow

```
PREPROCESSING (once)
┌─────────────────────────────────────────────────────────────────┐
│   Load Events → DROP_NAN → TRIMMING → ZSCORE → [time series]   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
TRAINING (on-the-fly)
┌─────────────────────────────────────────────────────────────────┐
│   Time series → TIME AUG → STFT → VISION AUG → Model           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
TESTING (on-the-fly, no augmentation)
┌─────────────────────────────────────────────────────────────────┐
│   Time series → STFT → Model                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

### Data Sources

```python
paths = [
    './resources/VRAC.pkl',
    './resources/MORC.pkl',
    './resources/hh_all.pkl',
]
```

### Preprocessing

```python
adapter = WaveformDataAdapter(
    events=events,
    labels=labels,
    channels=channels,
    transforms=TransformOP.DROP_NAN | TransformOP.TRIMMING | TransformOP.ZSCORE,
    test_ratio=0.3
)
```

| Parameter | Description |
|-----------|-------------|
| `test_ratio` | Fraction of data for testing (default: 0.3) |

### Transform Pipelines

**Training Transform** (with augmentation):

```python
train_transform = nn.Sequential(
    # Time-series augmentations
    Compose([
        RandomApply(TimeShift(max_shift=0.1), p=0.5),
        RandomApply(AddNoise(snr_min=15, snr_max=40), p=0.5),
        RandomApply(AmplitudeScale(0.9, 1.1), p=0.3),
    ]),
    STFTTransform(nperseg=64, hop=32),
    ToTensor(),
    # Vision augmentation (SpecAugment-style)
    T.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
)
```

**Test Transform** (no augmentation):

```python
test_transform = nn.Sequential(
    STFTTransform(nperseg=64, hop=32),
    ToTensor(),
)
```

### Time-Series Augmentation Parameters

| Augmentation | Parameter | Value | Description |
|--------------|-----------|-------|-------------|
| `TimeShift` | max_shift | 0.1 | Circular shift up to 10% of sequence |
| `AddNoise` | snr_min/max | 15-40 dB | Gaussian noise with random SNR |
| `AmplitudeScale` | scale | 0.9-1.1 | Random amplitude scaling |

### Vision Augmentation Parameters

| Transform | Parameter | Value | Description |
|-----------|-----------|-------|-------------|
| `T.RandomErasing` | p | 0.3 | Probability of applying |
| `T.RandomErasing` | scale | (0.02, 0.1) | Area ratio of erased region |
| `T.RandomErasing` | ratio | (0.3, 3.3) | Aspect ratio of erased region |

### STFT Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `nperseg` | 64 | STFT window size |
| `hop` | 32 | STFT hop length (50% overlap) |
| `freq_bins` | 33 | Output frequency bins (`nperseg // 2 + 1`) |

### Model Parameters

```python
model = EfficientNet(
    backbone=EfficientNetBackbone.B0,  # Pretrained backbone
    in_channels=3,                      # Number of input channels (Z, N, E)
    num_classes=2,                      # Number of output classes
    dropout=0.2                         # Dropout probability
)
```

### Available Backbones

| Backbone | Parameters | Description |
|----------|------------|-------------|
| `EfficientNetBackbone.B0` | 5.3M | Smallest, fastest |
| `EfficientNetBackbone.B1` | 7.8M | |
| `EfficientNetBackbone.B2` | 9.2M | |
| `EfficientNetBackbone.B3` | 12M | |
| `EfficientNetBackbone.B4` | 19M | |
| `EfficientNetBackbone.V2_S` | 21M | EfficientNetV2 small |
| `EfficientNetBackbone.V2_M` | 54M | EfficientNetV2 medium |

### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 64 | Training batch size |
| `batch_size_test` | 64 | Test batch size |
| `epochs` | 30 | Number of training epochs |

## Output

The pipeline displays:
- Number of loaded events per file
- Total event count
- Spectrogram shape information (channels, freq_bins, time_frames)
- Model architecture summary (via torchinfo)
- Training progress with loss and F1 metrics
