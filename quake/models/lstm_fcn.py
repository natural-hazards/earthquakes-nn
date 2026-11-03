"""
LSTM-FCN: LSTM branch + fully convolutional branch (Karim et al., 2018)
========================================================================

Two parallel branches read the same input sequence:

    Input: [B, seq_len, channels]
        |-------------------------------------------|
    LSTM (hidden)                       Conv1d(channels -> c1, k1) + BN + ReLU
        |                               Conv1d(c1 -> c2, k2) + BN + ReLU
    final hidden state [B, hidden]      Conv1d(c2 -> c3, k3) + BN + ReLU
        |                               global average pooling [B, c3]
    Dropout                                         |
        |-------------------- concat -------------------|
                                |
                         Linear -> [B, classes]

The convolutional branch is a 1-D convolution over the sequence with the
components as input channels, which is equivalent to the 2-D convolution over
the [seq_len, channels] matrix used in the original multivariate adaptation.

Reference:
    Karim, Majumdar, Darabi, Chen (2018): LSTM Fully Convolutional Networks for
    Time Series Classification. IEEE Access 6, 1662-1669.
"""

import torch as tch
import torch.nn as nn


__all__ = [
    'LSTMFCNModel'
]


class LSTMFCNModel(nn.Module):
    """LSTM-FCN classifier for multivariate sequences.

    Args:
        channels: Number of input channels (e.g., 3 for Z, N, E components)
        classes: Number of output classes
        hidden: LSTM hidden dimension
        layers: Number of stacked LSTM layers
        conv_channels: Output channels of the three convolutional blocks
        kernels: Kernel sizes of the three convolutional blocks
        dropout: Dropout probability applied to the LSTM output
    """

    def __init__(
        self,
        channels: int = 3,
        classes: int = 2,
        hidden: int = 128,
        layers: int = 1,
        conv_channels: tuple[int, int, int] = (128, 256, 128),
        kernels: tuple[int, int, int] = (8, 5, 3),
        dropout: float = 0.8
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)

        blocks = []
        in_ch = channels
        for out_ch, k in zip(conv_channels, kernels):
            blocks += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding='same'),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True)
            ]
            in_ch = out_ch
        self.fcn = nn.Sequential(*blocks)

        self.classifier = nn.Linear(hidden + conv_channels[-1], classes)

    def forward(
        self,
        x: tch.Tensor
    ) -> tch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, channels]

        Returns:
            Class logits [batch, classes]
        """
        self.lstm.flatten_parameters()

        # Recurrent branch: final hidden state of the uppermost layer
        _, (ht, _) = self.lstm(x)
        rec = self.dropout(ht[-1])

        # Convolutional branch over the sequence, components as channels
        conv = self.fcn(x.transpose(1, 2)).mean(dim=2)

        return self.classifier(tch.cat([rec, conv], dim=1))
