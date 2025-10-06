import torch as tch

from torch import nn

class LSTMAttentionModel(nn.Module):

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

        # recurrent layer
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout
        )

        # self-attention block
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        # layer normalization for stability
        self.norm = nn.LayerNorm(hidden)
        self.batch_norm = nn.BatchNorm1d(hidden)

        # classification layer
        self.classifier = nn.Linear(hidden, classes)

    def forward(
        self,
        x: tch.Tensor
    ) -> tch.Tensor:
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden]
        lstm_out = lstm_out.contiguous()

        # batch normalization
        lstm_out_bn = self.batch_norm(lstm_out.transpose(1, 2)).transpose(1, 2)

        # self-attention
        attn_out, _ = self.self_attn(lstm_out_bn, lstm_out_bn, lstm_out_bn)
        attn_out = self.norm(attn_out + lstm_out_bn)  # residual + normalization

        # mean pooling
        context = attn_out.mean(dim=1)  # [batch, hidden]

        return self.classifier(context)
