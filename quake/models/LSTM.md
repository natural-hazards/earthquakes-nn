# LSTM Models for Seismic Classification

## Overview

This module provides two LSTM-based models for classifying seismic events:

1. **LSTMModel** - Standard LSTM with final hidden state classification
2. **LSTMAttentionModel** - LSTM with multi-head self-attention for better sequence modeling

Both models process seismic waveforms that have been converted to frequency domain using FFT transform.

## Input Data

The models expect input tensors with shape `[batch, seq_len, channels]` where:
- `batch` - Number of samples in the batch
- `seq_len` - Sequence length (number of frequency bins from FFT)
- `channels` - Number of seismic channels (typically 3 for Z, N, E components)

## LSTMModel Architecture

The standard LSTM model uses the final hidden state for classification.

```
Input: [B, seq_len, channels]
        |
        v
+------------------+
|      LSTM        |  Multi-layer LSTM with dropout
|                  |  Processes sequence step by step
+------------------+
        |
        v
Final Hidden State: [B, hidden]
        |
        v
+------------------+
|   BatchNorm1d    |  Normalizes hidden state
+------------------+
        |
        v
+------------------+
|     Linear       |  Classification head
+------------------+
        |
        v
Output: [B, num_classes]
```

### LSTM Cell Equations

At each time step t, the LSTM computes:

```
Forget gate:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input gate:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Cell update:  c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
Cell state:   c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
Output gate:  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden state: h_t = o_t ⊙ tanh(c_t)
```

Where:
- σ is the sigmoid function
- ⊙ is element-wise multiplication
- h_t is the hidden state at time t
- c_t is the cell state at time t

### Why Final Hidden State?

The LSTM processes the sequence from left to right. The final hidden state `h_T` (where T is the last time step) contains a summary of the entire sequence, making it suitable for classification.

## LSTMAttentionModel Architecture

The attention model adds multi-head self-attention on top of LSTM outputs for better context aggregation.

```
Input: [B, seq_len, channels]
        |
        v
+------------------+
|      LSTM        |  Multi-layer LSTM with dropout
+------------------+
        |
        v
LSTM Output: [B, seq_len, hidden]
        |
        v
+------------------+
|   BatchNorm1d    |  Normalizes across hidden dimension
+------------------+
        |
        v
+------------------+
| Multi-Head Self  |  Attention over all time steps
|    Attention     |  Q = K = V = lstm_output
+------------------+
        |
    +---+---+
    |       |
    v       v
Residual Add + LayerNorm
        |
        v
+------------------+
|   Mean Pooling   |  Average over sequence dimension
+------------------+
        |
        v
Context: [B, hidden]
        |
        v
+------------------+
|     Linear       |  Classification head
+------------------+
        |
        v
Output: [B, num_classes]
```

### Multi-Head Self-Attention

Self-attention allows each position to attend to all other positions:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

With multi-head attention, the model can jointly attend to information from different representation subspaces:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Why Attention Helps

- **Global context**: LSTM hidden states are local; attention provides global context
- **Weighted aggregation**: Instead of just mean pooling, attention learns which time steps are important
- **Parallel computation**: Attention can process all positions simultaneously

## Usage

### LSTMModel

```python
from quake.models.lstm import LSTMModel

model = LSTMModel(
    channels=3,      # Z, N, E seismic channels
    classes=2,       # Number of event classes
    hidden=64,       # LSTM hidden dimension
    layers=2,        # Number of LSTM layers
    dropout=0.3      # Dropout between LSTM layers
)

# Input: [batch, seq_len, channels]
x = torch.randn(32, 257, 3)  # 257 = fft_size // 2 + 1
output = model(x)  # [32, 2]
```

### LSTMAttentionModel

```python
from quake.models.lstm_mhsa import LSTMAttentionModel

model = LSTMAttentionModel(
    channels=3,      # Z, N, E seismic channels
    classes=2,       # Number of event classes
    hidden=64,       # LSTM hidden dimension
    layers=2,        # Number of LSTM layers
    heads=2,         # Number of attention heads
    dropout=0.3      # Dropout rate
)

# Input: [batch, seq_len, channels]
x = torch.randn(32, 257, 3)
output = model(x)  # [32, 2]
```

## Comparison

| Feature | LSTMModel | LSTMAttentionModel |
|---------|-----------|-------------------|
| Sequence processing | Sequential (LSTM) | Sequential + Global (LSTM + Attention) |
| Output aggregation | Final hidden state | Attention-weighted mean pooling |
| Parameters | Fewer | More (attention weights) |
| Training speed | Faster | Slower |
| Context modeling | Local | Local + Global |

## Hyperparameter Guidelines

- **hidden**: 64-256 depending on dataset size. Larger values for more complex patterns.
- **layers**: 2-3 layers typically sufficient. More layers need more data.
- **dropout**: 0.2-0.5 for regularization. Higher for smaller datasets.
- **heads** (attention only): 2-8 heads. Must divide hidden size evenly.

## References

1. **Hochreiter, S., & Schmidhuber, J.** (1997).
   *Long Short-Term Memory*.
   Neural Computation, 9(8), 1735-1780.

2. **Vaswani, A., et al.** (2017).
   *Attention Is All You Need*.
   Advances in Neural Information Processing Systems (NeurIPS).
