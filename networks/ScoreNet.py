import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class SwishActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class EncodeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, t_dim):
        super().__init__()
        self.dense = Dense(t_dim, out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.gnorm = nn.GroupNorm(4, num_channels=out_channels)
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, h, t):
        h = self.conv(h)
        h += self.dense(t)
        h = self.act(self.gnorm(h))
        return h


class DecodeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, t_dim):
        super().__init__()
        self.dense = Dense(t_dim, out_channels)
        self.tconv = nn.ConvTranspose2d(in_channels * 2, out_channels, 3, stride=2, padding=1, output_padding=1)
        self.gnorm = nn.GroupNorm(4, num_channels=out_channels)
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, h, t, skip):
        h = torch.cat([h, skip], dim=1)
        h = self.tconv(h)
        h += self.dense(t)
        h = self.act(self.gnorm(h))
        return h


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, t_dim=256):
        """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
    """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=t_dim),
            nn.Linear(t_dim, t_dim)
        )
        # Encoding layers
        self.h_head = nn.Sequential(
            nn.Conv2d(1, 32, 1),
            nn.GroupNorm(4, num_channels=32),
            SwishActivation()
        )
        self.encode1 = EncodeLayer(32, 64, t_dim)
        self.encode2 = EncodeLayer(64, 128, t_dim)
        self.encode3 = EncodeLayer(128, 256, t_dim)
        self.mid = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(4, num_channels=128),
            SwishActivation()
        )
        self.decode2 = DecodeLayer(128, 64, t_dim)
        self.decode1 = DecodeLayer(64, 32, t_dim)
        self.tail = nn.Conv2d(32, 1, 1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t
        # x: (B, 4, 80, 80)
        h = self.h_head(x)
        t_embed = self.act(self.embed(t))
        # Encoding path
        h1 = self.encode1(h, t_embed)
        h2 = self.encode2(h1, t_embed)
        h3 = self.encode3(h2, t_embed)
        # Decoding path
        h = self.mid(h3)
        h = self.decode2(h, t_embed, h2)
        h = self.decode1(h, t_embed, h1)
        h = self.tail(h)
        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h