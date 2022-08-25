from .ScoreNet import ScoreNet
from .feature_extractors import Extractor
import torch
from torch import nn
import numpy as np


def marginal_prob_std(t, sigma):
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma):
    return sigma ** t


class Generator(nn.Module):
    def __init__(self, sigma=25.):
        super().__init__()
        self.sigma = sigma
        self.marginal_prob_std_fn = lambda t: marginal_prob_std(t, self.sigma)
        self.diffusion_coeff_fn = lambda t: diffusion_coeff(t, self.sigma)
        self.score_net = ScoreNet(self.marginal_prob_std_fn)
        self.extractor = Extractor(26)
        self._eps = 1e-5

    def forward(self, x, map_layers):
        fmap = self.extractor(map_layers)
        t = torch.rand(x.shape[0], device=x.device) * (1 - self._eps) + self._eps
        z = torch.randn_like(x)
        std = self.marginal_prob_std_fn(t)
        perturbed_x = x + z * std[:, None, None, None]
        score = self.score_net(perturbed_x, t, fmap)
        loss = torch.mean(torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1, 2, 3)))
        return loss
