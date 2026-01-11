"""
LSTM with Multi-Head Self-Attention for Seismic Classification
===============================================================

This module implements an LSTM model enhanced with multi-head self-attention
for classifying seismic events. The attention mechanism allows the model to
capture global dependencies across the entire sequence.

Architecture:
    Input: [B, seq_len, channels]
        |
    LSTM layers
        |
    LSTM output: [B, seq_len, hidden]
        |
    BatchNorm1d
        |
    Multi-Head Self-Attention (Q=K=V=lstm_output)
        |
    Residual connection + LayerNorm
        |
    Mean pooling over sequence
        |
    Linear classifier
        |
    Output: [B, num_classes]

The attention mechanism provides global context that complements
the local sequential processing of the LSTM.
"""

import torch as tch

from torch import nn


__all__ = [
    'LSTMAttentionModel'
]


class LSTMAttentionModel(nn.Module):
    """LSTM with multi-head self-attention for seismic classification.

    Combines LSTM's sequential processing with self-attention's global
    context modeling. Uses attention-weighted mean pooling instead of
    just the final hidden state.

    Args:
        channels: Number of input channels (e.g., 3 for Z, N, E components)
        classes: Number of output classes
        hidden: LSTM hidden dimension (must be divisible by heads)
        layers: Number of stacked LSTM layers
        heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        channels: int = 3,
        classes: int = 2,
        hidden: int = 256,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.75
    ) -> None:
        super(LSTMAttentionModel, self).__init__()

        # Multi-layer LSTM for sequential processing
        # Output: [batch, seq_len, hidden] - hidden state at each time step
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout
        )

        # Multi-head self-attention allows each position to attend to all others
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        # LayerNorm for post-attention normalization
        self.norm = nn.LayerNorm(hidden)

        # BatchNorm applied to LSTM output before attention
        self.batch_norm = nn.BatchNorm1d(hidden)

        # Project pooled context to class logits
        self.classifier = nn.Linear(hidden, classes)

    def forward(
        self,
        x: tch.Tensor
    ) -> tch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, channels]

        Returns:
            Class logits [batch, num_classes]
        """
        # Optimize LSTM weights for faster computation on GPU
        self.lstm.flatten_parameters()

        # Process sequence through LSTM
        # lstm_out: [batch, seq_len, hidden] - hidden state at each time step
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.contiguous()

        # Apply batch normalization across hidden dimension
        # Transpose: [batch, seq_len, hidden] -> [batch, hidden, seq_len]
        lstm_out_bn = self.batch_norm(lstm_out.transpose(1, 2)).transpose(1, 2)

        # Self-attention: each position attends to all positions
        # Q = K = V = lstm_out_bn (self-attention)
        attn_out, _ = self.self_attn(lstm_out_bn, lstm_out_bn, lstm_out_bn)

        # Residual connection + layer normalization for training stability
        attn_out = self.norm(attn_out + lstm_out_bn)

        # Mean pooling over sequence dimension to get fixed-size context
        # [batch, seq_len, hidden] -> [batch, hidden]
        context = attn_out.mean(dim=1)

        return self.classifier(context)
