import torch as tch

from torch import nn


__all__ = [
    'LSTMModel'
]


class LSTMModel(nn.Module):

    def __init__(
        self,
        channels: int = 3,
        classes: int = 2,
        hidden: int = 256,
        layers: int = 3,
        dropout: float = 0.75
    ) -> None:
        super(LSTMModel, self).__init__()

        # recurrent layer
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout
        )

        # batch normalization after LSTM
        self.batch_norm = nn.BatchNorm1d(hidden)

        # classification layer
        self.classifier = nn.Linear(hidden, classes)

    def forward(
        self,
        x: tch.Tensor
    ) -> tch.Tensor:
        self.lstm.flatten_parameters()
        _, (ht, _) = self.lstm(x)
        out = ht[-1]  # [batch, hidden]

        # apply batch normalization
        out = self.batch_norm(out)

        return self.classifier(out)
