import torch
import torch.nn as nn


def get_mlp(hidden_size, output_size):
    mlp_layers = [
        nn.Linear(hidden_size, 2 * hidden_size),
        nn.LayerNorm(2 * hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(2 * hidden_size, hidden_size),
        nn.LayerNorm(hidden_size),
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

    def forward(self, fmap, loc):
        # fmap: (B, C, wl, wl)
        # loc: (B, L, 2)
        C = fmap.size(1)
        loc = loc.clamp(min=-0.999, max=0.999)
        x = loc[..., 0] * self.axes_limit
        y = loc[..., 1] * self.axes_limit
        row = ((self.axes_limit - y) / self.resolution).long()
        col = ((self.axes_limit + x) / self.resolution).long()
        idx = row * self.wl + col  # (B, L)
        idx = idx[..., None].repeat(1, 1, C)  # (B, L, C)
        fmap = fmap.flatten(2, 3).permute(0, 2, 1)  # (B, wl * wl, C)
        indexed = fmap.gather(dim=1, index=idx)  # (B, L, C)
        return indexed


class TrainableIndexLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=2),
            nn.ReLU(),
            nn.ConvTranspose2d(384, 192, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2),
            nn.LayerNorm([96, 8, 8]),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2),
            nn.LayerNorm([48, 16, 16]),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2),
            nn.LayerNorm([24, 32, 32]),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, kernel_size=2, stride=2),
            nn.LayerNorm([12, 64, 64]),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 1, kernel_size=5, stride=5),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, f):
        # f: (B, L, 768)
        B = f.shape[0]
        L = f.shape[1]
        f = f[..., None, None]  # (B, L, 768, 1, 1)
        f = f.flatten(0, 1)  # (B * L, 768, 1, 1)
        weight = self.body(f)  # (B * L, 1, 320, 320)
        weight = weight.reshape(B * L, 320 * 320)
        weight = self.softmax(weight)
        weight = weight.reshape(B, L, 1, 320, 320)
        return weight
