import torch
import torch.nn as nn


def get_mlp(hidden_size, output_size):
    mlp_layers = [
        nn.Linear(hidden_size, 2 * hidden_size),
        nn.BatchNorm1d(2 * hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(2 * hidden_size, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, output_size)
    ]
    return nn.Sequential(*mlp_layers)


def get_length_mask(lengths):
    N = lengths.shape[0]
    L = lengths.max()
    idx_range = torch.arange(L, device=lengths.device).expand(N, -1)
    lengths = lengths.reshape(-1, 1).expand(-1, L)
    return idx_range >= lengths
