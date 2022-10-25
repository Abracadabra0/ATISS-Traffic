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
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.mid = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(160)
        )
        self.large = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(160)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=512, kernel_size=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B, input_channels, 320, 320)
        h_small = self.small(x)  # (B, 128, 160, 160)
        h_mid = self.mid(x)  # (B, 128, 160, 160)
        h_large = self.large(x)
        h = torch.cat([h_small, h_mid, h_large], dim=1)
        h = self.refine(h)
        return h
