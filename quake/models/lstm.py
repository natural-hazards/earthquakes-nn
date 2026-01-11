"""
LSTM Model for Seismic Classification
======================================

This module implements a standard LSTM model for classifying seismic events.
The model processes seismic waveforms converted to frequency domain using FFT.

Architecture:
    Input: [B, seq_len, channels]
        |
    LSTM layers (with dropout between layers)
        |
    Final hidden state: [B, hidden]
        |
    BatchNorm1d
        |
    Linear classifier
        |
    Output: [B, num_classes]

The final hidden state of the LSTM summarizes the entire input sequence,
making it suitable for sequence classification tasks.
"""

import torch as tch

from torch import nn


__all__ = [
    'LSTMModel'
]


class LSTMModel(nn.Module):
    """LSTM model for seismic event classification.

    Processes frequency-domain seismic data (after FFT) through stacked LSTM
    layers and uses the final hidden state for classification.

    Args:
        channels: Number of input channels (e.g., 3 for Z, N, E components)
        classes: Number of output classes
        hidden: LSTM hidden dimension
        layers: Number of stacked LSTM layers
        dropout: Dropout probability between LSTM layers
    """

    def __init__(
        self,
        channels: int = 3,
        classes: int = 2,
        hidden: int = 256,
        layers: int = 3,
        dropout: float = 0.75
    ) -> None:
        super(LSTMModel, self).__init__()

        # Multi-layer LSTM processes the sequence step by step
        # Input: [batch, seq_len, channels] -> Output: [batch, seq_len, hidden]
        # Also returns final hidden state: [layers, batch, hidden]
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout
        )

        # Normalize the hidden state before classification
        self.batch_norm = nn.BatchNorm1d(hidden)

        # Project hidden state to class logits
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
        # ht shape: [num_layers, batch, hidden] - hidden state at final time step
        # ct shape: [num_layers, batch, hidden] - cell state at final time step
        _, (ht, _) = self.lstm(x)

        # Use hidden state from the last LSTM layer
        # ht[-1] shape: [batch, hidden]
        out = ht[-1]

        # Normalize before classification for training stability
        out = self.batch_norm(out)

        return self.classifier(out)
