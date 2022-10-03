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
        self.large = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            ResBlock(128, 5),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            ResBlock(128, 3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            ResBlock(128, 3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1)),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.AdaptiveAvgPool2d((320, 320))
        )

        self.small = nn.Sequential(
            nn.Conv2d(in_channels=input_channels + 128, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            ResBlock(128, 3),
            ResBlock(128, 1)
        )

    def forward(self, x):
        # x: (B, input_channels, 320, 320)
        large_f = self.large(x)  # (B, 64, 320, 320)
        x = torch.cat([x, large_f], dim=1)  # (B, input_channels + 128, 320, 320)
        f = self.small(x)  # (B, 128, 320, 320)
        return f
