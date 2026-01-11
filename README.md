# Seismic Event Classification with Neural Networks

A deep learning framework for classifying seismic events using LSTM and Vision Transformer models. The project processes seismic waveforms from multiple stations and channels (Z, N, E components) to distinguish between different event types.

## Features

- **Multiple model architectures**: LSTM, LSTM with attention, and Hybrid CNN-ViT
- **Flexible preprocessing**: FFT and STFT/Spectrogram transforms
- **Visualization tools**: Fan charts and spectrogram plotting
- **Data augmentation**: Time shift, noise injection, amplitude scaling, and more

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd earthquakes-nn

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
earthquakes-nn/
├── pipelines/              # Training pipelines
│   ├── train_lstm.py       # LSTM model training
│   ├── train_hvit.py       # Hybrid ViT training
│   └── utils.py            # Shared utilities
├── quake/
│   ├── data/
│   │   ├── adapter.py      # Data preprocessing (FFT, STFT)
│   │   ├── augmentation.py # Data augmentation transforms
│   │   ├── dataset.py      # PyTorch dataset
│   │   └── loader.py       # Pickle file loader
│   ├── models/
│   │   ├── lstm.py         # Standard LSTM model
│   │   ├── lstm_mhsa.py    # LSTM with multi-head attention
│   │   ├── hybrid_vit.py   # Hybrid CNN-ViT model
│   │   ├── LSTM.md         # LSTM documentation
│   │   └── HYBRID_VIT.md   # Hybrid ViT documentation
│   ├── procs/
│   │   └── train.py        # Training loop
│   └── visualization/
│       ├── fan_charts.py   # Fan chart visualization
│       └── spectrogram.py  # Spectrogram visualization
└── resources/              # Data files (pickle format)
```

## Data Format

Seismic events are stored as pickle files containing:
- **events**: List of pandas DataFrames with columns `Z`, `N`, `E` (seismic channels)
- **labels**: Array of class labels (e.g., earthquake vs noise)

## Preprocessing

The `WaveformDataAdapter` handles data preprocessing with configurable transforms:

```python
from quake.data.adapter import WaveformDataAdapter, TransformOP

adapter = WaveformDataAdapter(
    events=events,
    labels=labels,
    channels=('Z', 'N', 'E'),
    transforms=TransformOP.DROP_NAN | TransformOP.TRIMMING | TransformOP.ZSCORE | TransformOP.FFT,
    test_ratio=0.3
)
events_train, events_test = adapter.get_datasets()
```

### Available Transforms

| Transform | Description |
|-----------|-------------|
| `DROP_NAN` | Remove NaN values from waveforms |
| `TRIMMING` | Trim all events to minimum length |
| `ZSCORE` | Z-score normalization per channel |
| `FFT` | Convert to frequency domain using FFT |
| `SPECTROGRAM` | Convert to time-frequency representation using STFT |

**Note**: `FFT` and `SPECTROGRAM` are mutually exclusive.

## Models

### LSTM Models

Process frequency-domain data (after FFT) through recurrent layers.

- **LSTMModel**: Standard LSTM using final hidden state for classification
- **LSTMAttentionModel**: LSTM with multi-head self-attention for global context

See [quake/models/LSTM.md](quake/models/LSTM.md) for detailed architecture documentation.

```python
from quake.models.lstm import LSTMModel

model = LSTMModel(
    channels=3,      # Z, N, E components
    classes=2,       # Number of event classes
    hidden=64,       # Hidden dimension
    layers=2,        # LSTM layers
    dropout=0.3
)
```

```python
from quake.models.lstm_mhsa import LSTMAttentionModel

model = LSTMAttentionModel(
    channels=3,      # Z, N, E components
    classes=2,       # Number of event classes
    hidden=64,       # Hidden dimension
    layers=2,        # LSTM layers
    heads=2,         # Number of attention heads
    dropout=0.3
)
```

### Hybrid CNN-ViT

Combines CNN stem with pretrained Vision Transformer for spectrogram classification.

See [quake/models/HYBRID_VIT.md](quake/models/HYBRID_VIT.md) for detailed architecture documentation.

```python
from quake.models import Backbone, HybridViT

model = HybridViT(
    backbone=Backbone.TINY,  # Pretrained ViT backbone
    depth=4,                 # Use first 4 transformer blocks
    in_channels=3,           # Z, N, E channels
    num_classes=2,
    img_size=(33, 64),       # (freq_bins, time_frames)
    dropout=0.2
)
```

## Visualizations

### Fan Charts

Fan charts display the median waveform with quantile bands (10%, 25%, 75%, 90%) for each class, helping visualize the variability within event types.

```python
from quake.visualization import plot_fan_chart, Align

plot_fan_chart(
    events,
    channels=['Z', 'N', 'E'],
    title='Earthquake Events',
    align=Align.TRIM,
    zscore=True,
    log_scale=True
)
```

### Spectrograms

Visualize time-frequency representations of seismic signals.

```python
from quake.visualization import plot_spectrogram

# sample shape: [n_channels, freq_bins, time_frames]
plot_spectrogram(sample, channels=('Z', 'N', 'E'), title='Event Spectrogram')
```

## Training Pipelines

### LSTM Training

Trains LSTM models on FFT-transformed seismic data.

```bash
python pipelines/train_lstm.py
```

The pipeline:
1. Loads seismic events from pickle files
2. Displays class distribution and fan charts
3. Converts waveforms to frequency domain using FFT
4. Trains LSTM or LSTM with attention

### Hybrid ViT Training

Trains Hybrid CNN-ViT on spectrogram data.

```bash
python pipelines/train_hvit.py
```

The pipeline:
1. Loads seismic events from pickle files
2. Displays class distribution and fan charts
3. Converts waveforms to spectrograms using STFT
4. Trains HybridViT with pretrained ViT backbone

## Data Augmentation

The `quake.data.augmentation` module provides transforms for training:

```python
from quake.data.augmentation import (
    TimeShift,
    AddGaussianNoise,
    AmplitudeScale,
    ChannelDropout
)
```

## References

### LSTM

1. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
2. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.

### Hybrid CNN-ViT

1. Xiao, T., et al. (2021). *Early Convolutions Help Transformers See Better*. NeurIPS.
2. Dosovitskiy, A., et al. (2020). *An Image is Worth 16x16 Words*. ICLR.

## License

MIT License
