# Hybrid ViT Training Pipeline

This pipeline trains a Hybrid CNN-ViT model on seismic data. The seismic waveforms are converted to spectrograms (time-frequency representation) using STFT before being fed to the model.

## Usage

```bash
python pipelines/train_hvit.py
```

## Data Flow

1. Load seismic events from pickle files
2. Display class distribution bar chart
3. Show fan charts for each class (median waveform with quantile bands)
4. Apply preprocessing: DROP_NAN → TRIMMING → ZSCORE → STFT/Spectrogram
5. Display sample spectrogram for each channel
6. Train HybridViT with pretrained ViT backbone

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
    stft_nperseg=64,
    stft_hop=32,
    transforms=TransformOP.DROP_NAN | TransformOP.TRIMMING | TransformOP.ZSCORE | TransformOP.SPECTROGRAM,
    test_ratio=0.3
)
```

| Parameter | Description |
|-----------|-------------|
| `stft_nperseg` | STFT window size (default: 64) |
| `stft_hop` | STFT hop length/stride (default: 32) |
| `test_ratio` | Fraction of data for testing (default: 0.3) |

### Model Parameters

```python
model = HybridViT(
    backbone=Backbone.TINY,  # Pretrained backbone
    depth=4,                 # Number of transformer blocks to use
    in_channels=3,           # Number of input channels (Z, N, E)
    num_classes=2,           # Number of output classes
    img_size=(freq_bins, time_frames),  # Spectrogram dimensions
    dropout=0.2              # Dropout probability
)
```

### Available Backbones

| Backbone | Model | Embed Dim | Heads | Parameters |
|----------|-------|-----------|-------|------------|
| `Backbone.TINY` | vit_tiny_patch16_224 | 192 | 3 | ~1.5M (depth=4) |
| `Backbone.SMALL` | vit_small_patch16_224 | 384 | 6 | ~6M (depth=4) |
| `Backbone.BASE` | vit_base_patch16_224 | 768 | 12 | ~24M (depth=4) |
| `Backbone.DEIT_TINY` | deit_tiny_patch16_224 | 192 | 3 | ~1.5M (depth=4) |
| `Backbone.DEIT_SMALL` | deit_small_patch16_224 | 384 | 6 | ~6M (depth=4) |

### Depth Selection

The `depth` parameter controls how many pretrained transformer blocks to use:

- `depth=4`: Use first 4 blocks (recommended for small datasets)
- `depth=None`: Use all 12 blocks

Early transformer layers learn general features that transfer well across domains, while later layers are more task-specific to ImageNet.

## Output

The pipeline displays:
- Number of loaded events per file
- Total event count and class distribution
- Fan charts showing waveform variability per class
- Spectrogram visualization for sample event
- Model architecture summary (via torchinfo)
- Training progress with loss and metrics
