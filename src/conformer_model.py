import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer

class ConformerBlock(nn.Module):
    def __init__(self, dim, heads, ff_dim, dropout):
        super().__init__()
        self.attn = TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=31, padding=15, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask):
        x = self.attn(x, src_key_padding_mask=mask)

        # Depthwise convolution
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x + y)

        return x


class ConformerASR(nn.Module):
    def __init__(self, input_dim=80, dim=256, layers=4, heads=4, ff_dim=1024, vocab=29):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, dim)

        self.blocks = nn.ModuleList([
            ConformerBlock(dim, heads, ff_dim, 0.1)
            for _ in range(layers)
        ])

        self.classifier = nn.Linear(dim, vocab)

    def forward(self, x, lengths):
        x = self.in_proj(x)
        lengths = lengths.to(x.device)

        mask = torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None]

        for block in self.blocks:
            x = block(x, mask)

        return self.classifier(x)
