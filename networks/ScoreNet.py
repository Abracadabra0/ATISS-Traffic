import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools


def marginal_prob_std(t, sigma):
    """Compute the standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

  Returns:
    The standard deviation.
  """
    t = torch.tensor(t)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

  Returns:
    The vector of diffusion coefficients.
  """
    return torch.tensor(sigma ** t)


sigma = 25.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)


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
        self.cond = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.GroupNorm(4, num_channels=out_channels),
            SwishActivation()
        )
        self.conv1 = nn.Conv2d(in_channels * 2, out_channels, 3, stride=1, padding=1)
        self.gnorm1 = nn.GroupNorm(4, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.gnorm2 = nn.GroupNorm(4, num_channels=out_channels)
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, h, t, fmap):
        h = torch.cat([h, fmap], dim=1)
        h = self.conv1(h)
        h = self.act(self.gnorm1(h))
        h += self.dense(t)
        h = self.conv2(h)
        h = self.act(self.gnorm2(h))
        fmap = self.cond(fmap)
        return h, fmap


class DecodeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, t_dim):
        super().__init__()
        self.dense = Dense(t_dim, in_channels)
        self.conv = nn.Conv2d(in_channels * 3, in_channels, 3, stride=1, padding=1)
        self.gnorm1 = nn.GroupNorm(4, num_channels=in_channels)
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
        self.gnorm2 = nn.GroupNorm(4, num_channels=out_channels)
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, h, t, skip, fmap):
        h = torch.cat([h, skip, fmap], dim=1)
        h = self.conv(h)
        h = self.act(self.gnorm1(h))
        h += self.dense(t)
        h = self.tconv(h)
        h = self.act(self.gnorm2(h))
        return h


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, t_dim=256, cond_dim=128):
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
            nn.Conv2d(4, 32, 1),
            nn.GroupNorm(4, num_channels=32),
            SwishActivation()
        )
        self.cond_head = nn.Sequential(
            nn.Conv2d(cond_dim, 32, 1),
            nn.GroupNorm(4, num_channels=32),
            SwishActivation()
        )
        self.encode1 = EncodeLayer(32, 64, t_dim)
        self.encode2 = EncodeLayer(64, 128, t_dim)
        self.encode3 = EncodeLayer(128, 256, t_dim)
        self.encode4 = EncodeLayer(256, 512, t_dim)
        self.mid = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(4, num_channels=256),
            SwishActivation()
        )
        self.decode3 = DecodeLayer(256, 128, t_dim)
        self.decode2 = DecodeLayer(128, 64, t_dim)
        self.decode1 = DecodeLayer(64, 32, t_dim)
        self.tail = nn.Conv2d(32, 4, 1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t, fmap):
        # Obtain the Gaussian random feature embedding for t
        # x: (B, 4, 80, 80)
        h = self.h_head(x)
        t_embed = self.act(self.embed(t))
        fmap = self.cond_head(fmap)
        # Encoding path
        h1, fmap1 = self.encode1(h, t_embed, fmap)
        h2, fmap2 = self.encode2(h1, t_embed, fmap1)
        h3, fmap3 = self.encode3(h2, t_embed, fmap2)
        h4, fmap4 = self.encode4(h3, t_embed, fmap3)
        # Decoding path
        h = self.mid(h4)
        h = self.decode3(h, t_embed, h3, fmap3)
        h = self.decode2(h, t_embed, h2, fmap2)
        h = self.decode1(h, t_embed, h1, fmap1)
        h = self.tail(h)
        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h
