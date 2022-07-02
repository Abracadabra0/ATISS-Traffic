import torch
import torch.nn as nn
import numpy as np
import math


class FixedPositionalEncoding(nn.Module):
    def __init__(self, proj_dims, t_min=1, t_max=64):
        super().__init__()
        ll = proj_dims // 2
        exp = torch.linspace(np.log2(t_max), np.log2(t_min), ll)
        exp = 2 ** exp
        self.sigma = 2 * np.pi / exp

    def forward(self, x):
        return torch.cat([
            torch.sin(x[..., None] * self.sigma.to(x.device)),
            torch.cos(x[..., None] * self.sigma.to(x.device))
        ], dim=-1).flatten(start_dim=-2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def get_mlp(hidden_size, output_size):
    mlp_layers = [
        nn.Linear(hidden_size, 2 * hidden_size),
        nn.ReLU(),
        nn.Linear(2 * hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    ]
    return nn.Sequential(*mlp_layers)


def get_length_mask(lengths):
    N = lengths.shape[0]
    L = lengths.max()
    idx_range = torch.arange(L, device=lengths.device).expand(N, -1)
    lengths = lengths.reshape(-1, 1).expand(-1, L)
    return idx_range >= lengths


def sample_from_dmll(pred, num_classes=256):
    """Sample from mixture of logistics.

    Arguments
    ---------
        pred: NxC where C is 3*number of logistics
    """
    assert len(pred.shape) == 2

    N = pred.size(0)
    nr_mix = pred.size(1) // 3

    probs = torch.softmax(pred[:, :nr_mix], dim=-1)
    means = pred[:, nr_mix:2 * nr_mix]
    scales = torch.nn.functional.elu(pred[:, 2*nr_mix:3*nr_mix]) + 1.0001

    indices = torch.multinomial(probs, 1).squeeze()
    batch_indices = torch.arange(N, device=probs.device)
    mu = means[batch_indices, indices]
    s = scales[batch_indices, indices]
    u = torch.rand(N, device=probs.device)
    preds = mu + s*(torch.log(u) - torch.log(1-u))

    return torch.clamp(preds, min=-1, max=1)[:, None]


def optimizer_factory(config, parameters):
    """Based on the input arguments create a suitable optimizer object."""
    optimizer = config.get("optimizer", "Adam")
    lr = config.get("lr", 1e-3)
    momentum = config.get("momentum", 0.9)

    if optimizer == "SGD":
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum)
    elif optimizer == "Adam":
        return torch.optim.Adam(parameters, lr=lr)
    else:
        raise NotImplementedError()
