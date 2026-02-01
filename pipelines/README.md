# Training Pipelines

Training pipelines for seismic event classification models.

## Available Pipelines

| Pipeline | Model | Input Transform | Documentation |
|----------|-------|-----------------|---------------|
| `train_lstm.py` | LSTMModel, LSTMAttentionModel | FFT | [TRAIN_LSTM.md](TRAIN_LSTM.md) |
| `train_hvit.py` | HybridViT | STFT/Spectrogram | [TRAIN_HVIT.md](TRAIN_HVIT.md) |
| `train_efficientnet.py` | EfficientNet | STFT/Spectrogram | [TRAIN_EFFICIENTNET.md](TRAIN_EFFICIENTNET.md) |

## Quick Start

```bash
# Train LSTM model
python pipelines/train_lstm.py

# Train Hybrid ViT model
python pipelines/train_hvit.py

# Train EfficientNet model
python pipelines/train_efficientnet.py
```

## Shared Utilities

The `utils.py` module provides common functions:

| Function | Description |
|----------|-------------|
| `load_events(paths)` | Load and merge events from multiple pickle files |
| `show_fan_charts(events, labels, channels)` | Display fan charts for each class |

## Data Requirements

Pipelines expect pickle files in `./resources/` containing:
- **events**: List of pandas DataFrames with columns `Z`, `N`, `E`
- **labels**: Array of class labels
