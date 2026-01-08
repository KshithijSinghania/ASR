import math
import torch
import torch.nn as nn


# -------------------------
# Positional Encoding
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, T, D)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1)]


# -------------------------
# Transformer ASR Model
# -------------------------
class TransformerASR(nn.Module):
    def __init__(
        self,
        input_dim=80,
        model_dim=256,
        num_heads=4,
        num_layers=4,
        ff_dim=1024,
        dropout=0.1,
        vocab_size=29,  # 28 chars + blank
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Linear(model_dim, vocab_size)

    def forward(self, x, lengths):
        """
        x: (B, T, 80)
        lengths: (B,)
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        # Padding mask: True where padding exists
        max_len = x.size(1)
        lengths = lengths.to(x.device)

        padding_mask = (
            torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]
        )

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        logits = self.classifier(x)  # (B, T, vocab)

        return logits
