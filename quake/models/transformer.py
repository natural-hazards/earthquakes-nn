import torch as tch
import torch.nn as nn
import torch.fft as fft

class TransformerModel(nn.Module):

    def __init__(
        self,
        channels: int = 3,
        classes: int = 2,
        hidden: int = 256,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 4000
    ) -> None:
        super(TransformerModel, self).__init__()

        # input normalization
        self.input_norm = nn.LayerNorm(channels)

        # input projection
        self.input_proj = nn.Linear(channels, hidden)

        # sinusoidal positional encoding
        self.register_buffer("pos_encoding", self._generate_sinusoidal_encoding(max_seq_len, hidden))

        # transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        # multi-head attention pooling
        self.attn_pool = nn.MultiheadAttention(embed_dim=hidden, num_heads=heads, batch_first=True)

        # classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, classes)
        )

        # initialize weights
        self._init_weights()

    def _generate_sinusoidal_encoding(self, max_seq_len, hidden):
        position = tch.arange(0, max_seq_len).unsqueeze(1)
        div_term = tch.exp(tch.arange(0, hidden, 2) * -(tch.log(tch.tensor(10000.0)) / hidden))
        encoding = tch.zeros(max_seq_len, hidden)
        encoding[:, 0::2] = tch.sin(position * div_term)
        encoding[:, 1::2] = tch.cos(position * div_term)
        return encoding.unsqueeze(0)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0.)

    def forward(
        self,
        x: tch.Tensor
    ) -> tch.Tensor:
        # project input
        x = self.input_proj(x)  # [batch, seq_len, hidden]

        # add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]

        # transformer encoder
        x = self.transformer(x)  # [batch, seq_len, hidden]

        # multi-head attention pooling
        attn_output, _ = self.attn_pool(x, x, x)  # [batch, seq_len, hidden]
        context = attn_output.mean(dim=1)  # [batch, hidden]

        # classification
        return self.classifier(context)
