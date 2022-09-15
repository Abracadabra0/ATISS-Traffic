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


class MapIndexLayer(nn.Module):
    def __init__(self, axes_limit=40, resolution=0.25, n_feature=128):
        super().__init__()
        self.axes_limit = axes_limit
        self.resolution = resolution
        self.wl = int(self.axes_limit * 2 / self.resolution)
        self.empty = nn.Parameter(torch.randn(n_feature))

    def forward(self, fmap, loc):
        # fmap: (B, C, wl, wl)
        # loc: (B, L, 2)
        C = fmap.size(1)
        mask = (loc[..., 0] > -1) & (loc[..., 0] < 1) & (loc[..., 1] > -1) & (loc[..., 1] < 1)
        loc = loc.clamp(min=-0.999, max=0.999)
        x = loc[..., 0] * self.axes_limit
        y = loc[..., 1] * self.axes_limit
        row = ((self.axes_limit - y) / self.resolution).long()
        col = ((self.axes_limit + x) / self.resolution).long()
        idx = row * self.wl + col  # (B, L)
        idx = idx[..., None].repeat(1, 1, C)  # (B, L, C)
        fmap = fmap.flatten(2, 3).permute(0, 2, 1)  # (B, wl * wl, C)
        indexed = fmap.gather(dim=1, index=idx)  # (B, L, C)
        indexed = torch.where(mask[..., None], indexed, self.empty[None, None, :])
        return indexed
