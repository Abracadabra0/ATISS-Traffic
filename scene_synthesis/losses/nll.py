import torch
from torch import nn


class WeightedNLL(nn.Module):
    def __init__(self, weights, with_components=False):
        super().__init__()
        self.weights = weights
        self._eps = 1e-6  # numerical stability for LogNorm
        self.with_components = with_components

    def forward(self, probs, gt):
        device = gt['category'].device
        loss_category = probs['category'].log_prob(gt['category']) * self.weights['category']

        loss_location = probs['location'].log_prob(gt['location']) * self.weights['location']

        loss_wl = probs['bbox'][0].log_prob(gt['bbox'][:, :2] + self._eps)
        loss_theta = probs['bbox'][1].log_prob(gt['bbox'][:, 2])
        loss_bbox = (loss_wl + loss_theta) * self.weights['bbox']

        loss_s = probs['velocity'][0].log_prob(gt['velocity'][:, 0] + self._eps)
        loss_omega = probs['velocity'][1].log_prob(gt['velocity'][:, 1])
        loss_moving = loss_s + loss_omega
        loss_velocity = torch.where(gt['velocity'][:, 0] == 0,
                                    probs['velocity'][2].log_prob(torch.tensor(0., device=device)),
                                    probs['velocity'][2].log_prob(torch.tensor(1., device=device)) + loss_moving) * self.weights['velocity']

        loss = loss_category + torch.where(gt['category'] == 0,
                                           torch.tensor(0., device=device),
                                           loss_location + loss_bbox + loss_velocity)
        loss = -loss.mean()

        if self.with_components:
            components = {
                'category': -loss_category.mean(),
                'location': -torch.where(gt['category'] == 0,
                                           torch.tensor(0., device=device),
                                           loss_location).mean(),
                'bbox': -torch.where(gt['category'] == 0,
                                           torch.tensor(0., device=device),
                                           loss_bbox).mean(),
                'velocity': -torch.where(gt['category'] == 0,
                                           torch.tensor(0., device=device),
                                           loss_velocity).mean()
            }
            return loss, components
        else:
            return loss
