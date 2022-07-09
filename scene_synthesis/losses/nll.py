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
        loss_wl = -probs['bbox']['wl'].log_prob(gt['bbox'][:, :2] + self._eps)
        loss_theta = -probs['bbox']['theta'].log_prob(gt['bbox'][:, 2:])
        loss_bbox = loss_wl + loss_theta
        loss_s = -probs['velocity']['s'].log_prob(gt['velocity'][:, :1] + self._eps)
        loss_omega = -probs['velocity']['omega'].log_prob(gt['velocity'][:, 1:])
        loss_moving = loss_s + loss_omega
        loss_velocity = torch.where(gt['velocity'][:, 0] == 0,
                                    -probs['velocity']['moving'].log_prob(torch.tensor(0., device=device)).squeeze(),
                                    -probs['velocity']['moving'].log_prob(torch.tensor(1., device=device)).squeeze()
                                    + loss_moving)

        loss = loss_category * self.weights['category'] + \
               torch.where(gt['category'] == 0,
                           torch.tensor(0., device=device),
                           loss_location * self.weights['location'] +
                           loss_bbox * self.weights['bbox'] +
                           loss_velocity * self.weights['velocity'])

        components = {
            'all': loss,
            'category': loss_category,
            'location': torch.where(gt['category'] == 0,
                                    torch.tensor(0., device=device),
                                    loss_location),
            'bbox': torch.where(gt['category'] == 0,
                                torch.tensor(0., device=device),
                                loss_bbox),
            'velocity': torch.where(gt['category'] == 0,
                                    torch.tensor(0., device=device),
                                    loss_velocity)
        }
        return components
