import torch
from torch import nn
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, input_channels, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self._feature_extractor = models.resnet18(pretrained=False)
        self._feature_extractor.conv1 = torch.nn.Conv2d(
            input_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False
        )
        self._feature_extractor.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.feature_size)
        )

    def forward(self, feature_map):
        return self._feature_extractor(feature_map)
