import torch
from torch import nn


class WeightedNLL(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = dict(weights)
        total = sum(weights.values())
        for k in self.weights:
            self.weights[k] = self.weights[k] / total
        self._eps = 1e-6  # numerical stability for LogNorm

    def forward(self, probs, gt):
        device = gt['category'].device

        loss_category = -probs['category'].log_prob(gt['category'])

        loc = ((gt['location'] + 40) / 4).long()
        loc = loc[..., 0] * 20 + loc[..., 1]
        loss_location = -probs['location'].log_prob(loc)
        loss_location = torch.where(gt['category'] == 0,
                                    torch.tensor(0., device=device),
                                    loss_location)

        loss_wl = -probs['bbox']['wl'].log_prob(gt['bbox'][:, :2] + self._eps)
        loss_wl = torch.where(gt['category'] == 0,
                              torch.tensor(0., device=device),
                              loss_wl)
        loss_theta = -probs['bbox']['theta'].log_prob(gt['bbox'][:, 2:])
        loss_theta = torch.where(gt['category'] == 0,
                                 torch.tensor(0., device=device),
                                 loss_theta)

        loss_s = -probs['velocity']['s'].log_prob(gt['velocity'][:, :1] + self._eps)
        loss_s = torch.where(gt['velocity'][:, 0] == 0,
                             torch.tensor(0., device=device),
                             loss_s)
        loss_s = torch.where(gt['category'] == 0,
                             torch.tensor(0., device=device),
                             loss_s)
        loss_omega = -probs['velocity']['omega'].log_prob(gt['velocity'][:, 1:])
        loss_omega = torch.where(gt['velocity'][:, 0] == 0,
                                 torch.tensor(0., device=device),
                                 loss_omega)
        loss_omega = torch.where(gt['category'] == 0,
                                 torch.tensor(0., device=device),
                                 loss_omega)
        loss_moving = torch.where(gt['velocity'][:, 0] == 0,
                                  -probs['velocity']['moving'].log_prob(torch.tensor(0., device=device)).squeeze(),
                                  -probs['velocity']['moving'].log_prob(torch.tensor(1., device=device)).squeeze())
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
            'bbox': {
                'all': loss_wl * self.weights['wl'] + loss_theta * self.weights['theta'],
                'wl': loss_wl * self.weights['wl'],
                'theta': loss_theta * self.weights['theta']
            },
            'velocity': {
                'all': loss_moving * self.weights['moving'] + loss_s * self.weights['s'] + loss_omega * self.weights['omega'],
                'moving': loss_moving * self.weights['moving'],
                's': loss_s * self.weights['s'],
                'omega': loss_omega * self.weights['omega']
            }
        }
        return components


def lr_func(warmup):
    def inner(iters):
        return min((iters + 1) ** -0.5, (iters + 1) * warmup ** -1.5)

    return inner
