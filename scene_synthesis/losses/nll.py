import torch
from torch import nn
import math


class WeightedNLL(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = dict(weights)
        total = sum(weights.values())
        for k in self.weights:
            self.weights[k] = self.weights[k] / total
        self._eps = 1e-6  # numerical stability for LogNorm

    def _loss_vonmises(self, x, prob):
        # x: (B, 1)
        mixture = prob.params['mixture']  # (B, n_mixture)
        cos = prob.params['cos'].squeeze()  # (B, n_mixture)
        sin = prob.params['sin'].squeeze()  # (B, n_mixture)
        kappa = prob.params['kappa'].squeeze()  # (B, n_mixture)
        B, n_mixture = mixture.shape
        x = x.expand(B, n_mixture)
        log_prob = kappa * (torch.cos(x) * cos + torch.sin(x) * sin) - _log_modified_bessel_fn(kappa) - math.log(2 * math.pi)  # (B, n_mixture)
        logits = torch.log_softmax(mixture, dim=-1)
        loss = -torch.logsumexp(log_prob + logits, dim=-1)
        return loss

    def forward(self, probs, gt):
        device = gt['category'].device

        loss_category = -probs['category'].log_prob(gt['category'])

        loss_location = -probs['location'].log_prob(gt['location'])
        loss_location = torch.where(gt['category'] == 0,
                                    torch.tensor(0., device=device),
                                    loss_location)

        loss_wl = -probs['wl'].log_prob(gt['bbox'][:, :2] + self._eps)
        loss_wl = torch.where(gt['category'] == 0,
                              torch.tensor(0., device=device),
                              loss_wl)
        loss_theta = self._loss_vonmises(gt['bbox'][:, 2:], probs['theta'])
        loss_theta = torch.where(gt['category'] == 0,
                                 torch.tensor(0., device=device),
                                 loss_theta)

        loss_s = -probs['s'].log_prob(gt['velocity'][:, :1] + self._eps)
        loss_s = torch.where(gt['velocity'][:, 0] == 0,
                             torch.tensor(0., device=device),
                             loss_s)
        loss_s = torch.where(gt['category'] == 0,
                             torch.tensor(0., device=device),
                             loss_s)
        loss_omega = self._loss_vonmises(gt['velocity'][:, 1:], probs['omega'])
        loss_omega = torch.where(gt['velocity'][:, 0] == 0,
                                 torch.tensor(0., device=device),
                                 loss_omega)
        loss_omega = torch.where(gt['category'] == 0,
                                 torch.tensor(0., device=device),
                                 loss_omega)
        loss_moving = torch.where(gt['velocity'][:, 0] == 0,
                                  -probs['moving'].log_prob(torch.tensor(0., device=device)).squeeze(),
                                  -probs['moving'].log_prob(torch.tensor(1., device=device)).squeeze())
        loss_moving = torch.where(gt['category'] == 0,
                                  torch.tensor(0., device=device),
                                  loss_moving)

        loss = loss_category * self.weights['category'] + \
               loss_location * self.weights['location'] + \
               loss_wl * self.weights['wl'] + \
               loss_theta * self.weights['theta'] + \
               loss_moving * self.weights['moving'] + \
               loss_s * self.weights['s'] + \
               loss_omega * self.weights['omega']

        components = {
            'all': loss,
            'category': loss_category * self.weights['category'],
            'location': loss_location * self.weights['location'],
            'wl': loss_wl * self.weights['wl'],
            'theta': loss_theta * self.weights['theta'],
            'moving': loss_moving * self.weights['moving'],
            's': loss_s * self.weights['s'],
            'omega': loss_omega * self.weights['omega']
        }
        return components


def lr_func(warmup):
    def inner(iters):
        return min((iters + 1) ** -0.5, (iters + 1) * warmup ** -1.5)

    return inner


def _eval_poly(y, coef):
    coef = list(coef)
    result = coef.pop()
    while coef:
        result = coef.pop() + y * result
    return result


def _log_modified_bessel_fn(x):
    # compute small solution
    y = (x / 3.75)
    y = y * y
    small = _eval_poly(y, [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2])
    small = small.log()
    # compute large solution
    y = 3.75 / x
    large = x - 0.5 * x.log() + _eval_poly(y, [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2,
                                              -0.2057706e-1, 0.2635537e-1, -0.1647633e-1, 0.392377e-2]).log()
    result = torch.where(x < 3.75, small, large)
    return result
