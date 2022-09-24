import torch
from torch import nn


def get_length_mask(lengths):
    N = lengths.shape[0]
    L = lengths.max()
    idx_range = torch.arange(L, device=lengths.device).expand(N, -1)
    lengths = lengths.reshape(-1, 1).expand(-1, L)
    return idx_range < lengths


class DiffusionLoss(nn.Module):
    def __init__(self, weights_entry, weights_category):
        super().__init__()
        self.weights_entry = dict(weights_entry)
        total = sum(weights_entry.values())
        for k in self.weights_entry:
            self.weights_entry[k] = self.weights_entry[k] / total
        self.weights_category = dict(weights_category)
        total = sum(weights_category.values())
        for k in self.weights_category:
            self.weights_category[k] = self.weights_category[k] / total

    def forward(self, pred, target, sigmas):
        loss_dict = {
            'pedestrian': {},
            'bicyclist': {},
            'vehicle': {}
        }
        for name in ['pedestrian', 'bicyclist', 'vehicle']:
            loss = -pred[name]['length'].log_prob(target[name]['length']).mean()
            loss_dict[name]['length'] = loss
        for name in ['pedestrian', 'bicyclist', 'vehicle']:
            B = len(target[name]['length'])
            mask = get_length_mask(target[name]['length'])  # (B, L)
            tgt = -1 / sigmas[:, None, None] ** 2 * target[name]['noise']
            loss = ((pred[name]['score'] - tgt) ** 2).sum(dim=-1)  # (B, L)
            loss = (loss * sigmas[:, None] ** 2 * mask).mean(dim=-1)  # (B, )
            loss_dict[name]['noise_all'] = torch.where(target[name]['length'] > 0, loss, torch.tensor(0., device=loss.device))
            loss = loss[target[name]['length'] > 0].sum() / B
            loss_dict[name]['noise'] = loss
        # aggregate loss
        loss = torch.tensor(0, device=target['pedestrian']['length'].device)
        for category in ['pedestrian', 'bicyclist', 'vehicle']:
            for entry in ['length', 'noise']:
                loss = loss + loss_dict[category][entry] * self.weights_category[category] * self.weights_entry[entry]
        loss_dict['all'] = loss
        return loss_dict
