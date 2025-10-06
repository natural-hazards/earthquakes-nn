import torch as tch
import torch.nn as nn
import torch.nn.functional as F

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

        # input projection (like embedding)
        self.input_proj = nn.Linear(channels, hidden)

        # positional encoding (learnable)
        self.pos_embedding = nn.Parameter(tch.zeros(1, max_seq_len, hidden)) 

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

        # learnable attention pooling
        self.attn_pool = nn.Linear(hidden, 1)

        # classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, classes)
        )

        # initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0.)
        nn.init.xavier_uniform_(self.attn_pool.weight)
        nn.init.constant_(self.attn_pool.bias, 0.)

    def forward(
        self,
        x: tch.Tensor
    ) -> tch.Tensor:
        # project input
        x = self.input_proj(x)  # [batch, seq_len, hidden]

        # add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]

        # transformer encoder
        x = self.transformer(x)  # [batch, seq_len, hidden]

        # attention pooling over sequence
        attn_scores = self.attn_pool(x).squeeze(-1)        # [batch, seq_len]
        attn_weights = F.softmax(attn_scores, dim=1)        # [batch, seq_len]
        context = tch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # [batch, hidden]

        # classification
        return self.classifier(context)
