import torch
import torch.nn as nn


class ASRModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # CNN layers
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # BiGRU layers
        self.rnn = nn.GRU(
            input_size=64 * 80,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Output layer
        self.fc = nn.Linear(256 * 2, num_classes)

    def _reshape(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3)
        return x.contiguous().view(b, t, c * f)

    def forward(self, x):
        # x: (B, T, 80)
        x = x.unsqueeze(1)        # (B, 1, T, 80)
        x = self.conv(x)
        x = self._reshape(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x
