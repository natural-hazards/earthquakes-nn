# LSTM Training Pipeline

This pipeline trains LSTM-based models on seismic data with on-the-fly data augmentation. Time-series augmentations are applied before FFT transform during training.

## Usage

```bash
python pipelines/train_lstm.py
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
│   Time series → AUGMENTATION → FFT → Model                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
TESTING (on-the-fly, no augmentation)
┌─────────────────────────────────────────────────────────────────┐
│   Time series → FFT → Model                                    │
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
    Compose([
        RandomApply(TimeShift(max_shift=0.1), p=0.5),
        RandomApply(AddNoise(snr_min=15, snr_max=40), p=0.5),
        RandomApply(AmplitudeScale(0.9, 1.1), p=0.3),
        RandomApply(TimeStretch(0.95, 1.05), p=0.3),
    ]),
    FFTTransform(fft_size=512),
    ToTensor(),
)
```

**Test Transform** (no augmentation):

```python
test_transform = nn.Sequential(
    FFTTransform(fft_size=512),
    ToTensor(),
)
```

### Augmentation Parameters

| Augmentation | Parameter | Value | Description |
|--------------|-----------|-------|-------------|
| `TimeShift` | max_shift | 0.1 | Circular shift up to 10% of sequence |
| `AddNoise` | snr_min/max | 15-40 dB | Gaussian noise with random SNR |
| `AmplitudeScale` | scale | 0.9-1.1 | Random amplitude scaling |
| `TimeStretch` | rate | 0.95-1.05 | Time stretching/compression |

### FFT Parameters

| Parameter | Description |
|-----------|-------------|
| `fft_size` | Number of FFT bins (default: 512) |
| Output size | `fft_size // 2 + 1` = 257 frequency bins |

### Model Selection

To switch between models, edit the `main()` function:

```python
# Standard LSTM
train_lstm(events_train, events_test, seq_len=seq_len, channels=len(channels), device=device)

# LSTM with attention (uncomment to use)
# train_lstm_attention(events_train, events_test, seq_len=seq_len, channels=len(channels), device=device)
```

### Model Parameters

**LSTMModel:**
```python
model = LSTMModel(
    channels=3,      # Number of input channels (Z, N, E)
    classes=2,       # Number of output classes
    hidden=64,       # LSTM hidden dimension
    layers=2,        # Number of LSTM layers
    dropout=0.3      # Dropout probability
)
```

**LSTMAttentionModel:**
```python
model = LSTMAttentionModel(
    channels=3,      # Number of input channels
    classes=2,       # Number of output classes
    hidden=64,       # LSTM hidden dimension
    layers=2,        # Number of LSTM layers
    heads=2,         # Number of attention heads
    dropout=0.3      # Dropout probability
)
```

## Output

The pipeline displays:
- Number of loaded events per file
- Total event count and class distribution
- Fan charts showing waveform variability per class
- Input shape information (seq_len, channels)
- Model architecture summary
- Training progress with loss and metrics
