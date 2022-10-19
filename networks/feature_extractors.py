import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, n_channels, kernel_size):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=n_channels,
                      out_channels=n_channels,
                      kernel_size=(kernel_size, kernel_size),
                      padding=(kernel_size // 2, kernel_size // 2)),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=n_channels,
                      out_channels=n_channels,
                      kernel_size=(kernel_size, kernel_size),
                      padding=(kernel_size // 2, kernel_size // 2)),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f):
        return f + self.body(f)


class Extractor(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.small = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.avg_pooling = nn.AdaptiveAvgPool2d(64)
        self.large = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        # x: (B, input_channels, 320, 320)
        h = self.small(x)  # (B, 256, 64, 64)
        f_small = self.avg_pooling(h)
        h = self.large(h)
        f_large = h.flatten(1)
        return f_small, f_large
