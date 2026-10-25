"""
LSTM with Attention and Linear SVM Classifier
==============================================

LSTM + multi-head self-attention for feature extraction, with a linear SVM
classifier head trained via sklearn's SGDClassifier. Gradients flow back
into the LSTM through a custom autograd function.

Architecture:
    Input: [B, seq_len, channels]
        |
    LSTM layers
        |
    BatchNorm1d
        |
    Multi-Head Self-Attention
        |
    Residual connection + LayerNorm
        |
    Mean pooling over sequence
        |
    SVMClassifier (sklearn SGDClassifier backend)
        |
    Output: [B] (binary decision values) or [B, classes] (one-vs-rest)
"""

import torch as tch

from torch import nn

from quake.models.svm import SVMClassifier


__all__ = [
    'LSTMAttentionSVMModel'
]


class LSTMAttentionSVMModel(nn.Module):
    """LSTM with multi-head self-attention and linear SVM classifier.

    Args:
        channels: Number of input channels (e.g., 3 for Z, N, E components)
        classes: Number of output classes
        hidden: LSTM hidden dimension (must be divisible by heads)
        layers: Number of stacked LSTM layers
        heads: Number of attention heads
        dropout: Dropout probability
        svm_alpha: SVM L2 regularization strength
        svm_eta0: SVM learning rate
    """

    def __init__(
        self,
        channels: int = 3,
        classes: int = 2,
        hidden: int = 256,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.75,
        svm_alpha: float = 1e-4,
        svm_eta0: float = 0.01
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout
        )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden)
        self.batch_norm = nn.BatchNorm1d(hidden)

        self.classifier = SVMClassifier(
            in_features=hidden,
            classes=classes,
            alpha=svm_alpha,
            eta0=svm_eta0
        )

    def extract_features(self, x: tch.Tensor) -> tch.Tensor:
        """Extract features before the SVM classifier.

        Args:
            x: Input tensor [batch, seq_len, channels]

        Returns:
            Feature tensor [batch, hidden]
        """
        self.lstm.flatten_parameters()

        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.contiguous()

        lstm_out_bn = self.batch_norm(lstm_out.transpose(1, 2)).transpose(1, 2)

        attn_out, _ = self.self_attn(lstm_out_bn, lstm_out_bn, lstm_out_bn)
        attn_out = self.norm(attn_out + lstm_out_bn)

        return attn_out.mean(dim=1)

    def forward(self, x: tch.Tensor) -> tch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, channels]

        Returns:
            SVM decision values [batch] (binary) or [batch, classes]
        """
        features = self.extract_features(x)
        return self.classifier(features)
