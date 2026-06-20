"""CRNN recognizer: CNN backbone -> map-to-sequence -> BiLSTM -> CTC linear head.

Architecture follows the classic Shi et al. CRNN (CNN + BiLSTM + CTC), adapted to a
fixed input height of 32. ``forward`` returns log-probabilities shaped
``(T, B, num_classes)`` ready for :class:`torch.nn.CTCLoss`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    """A BiLSTM followed by a linear projection (per time-step)."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                           bidirectional=True, batch_first=False,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, input_size)
        recurrent, _ = self.rnn(x)
        return self.linear(recurrent)


class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network for line recognition.

    Args:
        num_classes: output classes including the CTC blank.
        in_channels: 1 (grayscale) or 3 (RGB).
        cnn_out_channels: channels produced by the final conv block.
        rnn_hidden: hidden units per LSTM direction.
        rnn_layers: number of stacked BiLSTM blocks (>=1).
        dropout: dropout inside the recurrent stack.
    """

    def __init__(self, num_classes: int, in_channels: int = 1,
                 cnn_out_channels: int = 512, rnn_hidden: int = 256,
                 rnn_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes

        def conv_bn(cin, cout, k=3, s=1, p=1, bn=False):
            layers = [nn.Conv2d(cin, cout, k, s, p)]
            if bn:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.ReLU(inplace=True))
            return layers

        c = cnn_out_channels
        self.cnn = nn.Sequential(
            *conv_bn(in_channels, 64),
            nn.MaxPool2d(2, 2),                              # H/2,  W/2
            *conv_bn(64, 128),
            nn.MaxPool2d(2, 2),                              # H/4,  W/4
            *conv_bn(128, 256),
            *conv_bn(256, 256),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),           # H/8,  W/4
            *conv_bn(256, 512, bn=True),
            *conv_bn(512, c, bn=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),           # H/16, W/4
            *conv_bn(c, c, k=2, s=1, p=0, bn=True),         # H/16 - 1 -> 1 for H=32
        )

        # Two stacked BiLSTM blocks (map-to-sequence -> recurrent -> classes).
        self.rnn = nn.Sequential(
            BidirectionalLSTM(c, rnn_hidden, rnn_hidden, num_layers=rnn_layers,
                              dropout=dropout),
            BidirectionalLSTM(rnn_hidden, rnn_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        conv = self.cnn(x)                  # (B, C', 1, W')
        b, ch, h, w = conv.size()
        assert h == 1, f"expected feature height 1, got {h} (input height must be 32)"
        conv = conv.squeeze(2)              # (B, C', W')
        conv = conv.permute(2, 0, 1)        # (W'=T, B, C')  -- map to sequence
        out = self.rnn(conv)               # (T, B, num_classes)
        return out.log_softmax(2)


def build_crnn(num_classes: int, cfg: dict | None = None,
               in_channels: int = 1) -> CRNN:
    """Construct a CRNN from a model-config dict (``configs/default.yaml`` 'model')."""
    cfg = cfg or {}
    return CRNN(
        num_classes=num_classes,
        in_channels=in_channels,
        cnn_out_channels=int(cfg.get("cnn_out_channels", 512)),
        rnn_hidden=int(cfg.get("rnn_hidden", 256)),
        rnn_layers=int(cfg.get("rnn_layers", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
    )