from .ScoreNet import ScoreNet
from .feature_extractors import Extractor
import torch
from torch import nn
import numpy as np
from tqdm import tqdm


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

    def forward(self, x, t=None):
        if t is None:
            t = torch.rand(x.shape[0], device=x.device) * (1 - self._eps) + self._eps
        z = torch.randn_like(x)
        std = self.marginal_prob_std_fn(t)
        perturbed_x = x + z * std[:, None, None, None]
        score = self.score_net(perturbed_x, t)
        loss = torch.mean(torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1, 2, 3)))
        return score, loss

    def generate(self, num_steps=1000, snr=0.16):
        self.eval()
        B = 1
        device = torch.device(0)
        t = torch.ones(B, device=device)
        init_x = torch.randn(B, 1, 28, 28, device=device) * self.marginal_prob_std_fn(t)[:, None, None, None]
        time_steps = np.linspace(1., self._eps, num_steps)
        step_size = time_steps[0] - time_steps[1]
        x = init_x
        with torch.no_grad():
            for time_step in tqdm(time_steps):      
                batch_time_step = torch.ones(B, device=device) * time_step
                # Corrector step (Langevin MCMC)
                grad = self.score_net(x, batch_time_step)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = np.sqrt(np.prod(x.shape[1:]))
                langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
                x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

                # Predictor step (Euler-Maruyama)
                g = self.diffusion_coeff_fn(batch_time_step)
                x_mean = x + (g**2)[:, None, None, None] * self.score_net(x, batch_time_step) * step_size
                x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      
            
        self.train()
        return x_mean
