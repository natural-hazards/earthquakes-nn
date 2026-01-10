# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python/PyTorch project for seismic event classification using neural networks. Classifies earthquake waveform data into event types using LSTM, LSTM+Attention, and Transformer models.

## Commands

```bash
# Run tests
pytest                          # All tests
pytest tests/data/loader_test.py  # Single test file

# Train models
python pipelines/train_lstm.py        # LSTM or LSTM+Attention
python pipelines/train_transformer.py # Transformer

# Run streaming app
streamlit run app/app.py

# Download dataset
bash resources/download.bash
```

## Architecture

### Data Pipeline
`quake/data/` - Data loading and transformation
- `loader.py`: Reads pickle files containing seismic events
- `adapter.py`: `WaveformDataAdapter` applies transformations (DROP_NAN, TRIMMING, ZSCORE, FFT) and creates train/test splits
- `dataset.py`: `WaveformDataset` wraps data for PyTorch DataLoader

### Models
`quake/models/` - Neural network architectures
- `lstm.py`: Basic LSTM classifier (3 layers, 256 hidden)
- `lstm_mhsa.py`: LSTM with multi-head self-attention (4 heads)
- `transformer.py`: Transformer encoder with sinusoidal positional encoding

All models: 3 input channels (Z, N, E seismic components), 2 output classes

### Training
`quake/procs/train.py` - PyTorch Ignite-based training with AdamW optimizer, exponential LR scheduler (gamma=0.85), F1-score evaluation

### Application
`app/` - Streamlit app for real-time seismic data visualization
- `client.py`: EIDA federation client for fetching live waveforms from CZ network

## Key Conventions

- Test files use `*_test.py` naming pattern
- Dataset expected at `resources/hh_selected.pkl` (4433 events)
- Models auto-detect CUDA, fall back to CPU
- Default batch sizes: 32-96 depending on model
