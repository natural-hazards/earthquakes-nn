# LSTM Training Pipeline

This pipeline trains LSTM-based models on seismic data. The seismic waveforms are converted to frequency domain using FFT transform before being fed to the model.

## Usage

```bash
python pipelines/train_lstm.py
```

## Data Flow

1. Load seismic events from pickle files
2. Display class distribution bar chart
3. Show fan charts for each class (median waveform with quantile bands)
4. Apply preprocessing: DROP_NAN → TRIMMING → ZSCORE → FFT
5. Train LSTMModel or LSTMAttentionModel

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
    fft_size=512,
    transforms=TransformOP.DROP_NAN | TransformOP.TRIMMING | TransformOP.ZSCORE | TransformOP.FFT,
    test_ratio=0.3
)
```

| Parameter | Description |
|-----------|-------------|
| `fft_size` | Number of FFT bins (default: 512) |
| `test_ratio` | Fraction of data for testing (default: 0.3) |

### Model Selection

To switch between models, edit the `main()` function:

```python
# Standard LSTM
train_lstm(events_train, events_test, channels=len(channels), device=device)

# LSTM with attention (uncomment to use)
# train_lstm_attention(events_train, events_test, channels=len(channels), device=device)
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
- Training progress with loss and metrics
