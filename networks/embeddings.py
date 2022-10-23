import torch
import torch.nn as nn
import numpy as np
import math


class FixedPositionalEncoding(nn.Module):
    def __init__(self, proj_dims):
        super().__init__()
        proj_dims = proj_dims // 2
        exp = torch.linspace(0, 1, proj_dims)
        exp = 0.1 ** exp
        self.sigma = 2 * np.pi / exp
        self.sigma_angle = torch.round(torch.linspace(1, 8, 32))

    def forward(self, x):
        non_angle = x[..., :-1]
        angle = x[..., -1:]
        pe_non_angle = torch.cat([
            torch.sin(non_angle[..., None] * self.sigma.to(x.device)),
            torch.cos(non_angle[..., None] * self.sigma.to(x.device))
        ], dim=-1)
        pe_angle = torch.cat([
            torch.sin(angle[..., None] * self.sigma_angle.to(x.device)),
            torch.cos(angle[..., None] * self.sigma_angle.to(x.device))
        ], dim=-1)
        return torch.cat([pe_non_angle, pe_angle], dim=-2).flatten(start_dim=-2)


class TrainablePE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(TrainablePE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


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
        return x


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding2D, self).__init__()
        d_pe = d_model // 2
        pe = torch.zeros(max_len, d_pe)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_pe, 2).float() * (-math.log(10000.0) / d_pe))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # (1, max_len, d_pe)

    @staticmethod
    def get_rank_from_pts(pts):
        # pts: (B, L, 2)
        x = pts[..., 0]
        y = pts[..., 1]
        _, x_indices = x.sort()
        _, x_rank = x_indices.sort()  # (B, L)
        _, y_indices = y.sort()
        _, y_rank = y_indices.sort()
        rank = torch.stack([x_rank, y_rank], dim=-1)
        return rank

    def forward(self, f, rank):
        # f: (B, L, d_model)
        # rank: (B, L, 2)
        B = f.shape[0]
        L = f.shape[1]
        d_pe = f.shape[-1] // 2
        x_rank = rank[..., 0]
        x_rank = x_rank.unsqueeze(-1).repeat(1, 1, d_pe)  # (B, L, d_pe)
        x_pe = self.pe.repeat(B, 1, 1).gather(dim=1, index=x_rank)  # (B, L, d_pe)
        y_rank = rank[..., 1]
        y_rank = y_rank.unsqueeze(-1).repeat(1, 1, d_pe)  # (B, L, d_pe)
        y_pe = self.pe.repeat(B, 1, 1).gather(dim=1, index=y_rank)  # (B, L, d_pe)
        pe = torch.cat([x_pe, y_pe], dim=-1)
        f = f + pe
        return f


class SinusoidalEmb(nn.Module):
    def __init__(self, dim, input_dim, T_min, T_max):
        super().__init__()
        self.dim = dim
        T = torch.linspace(math.log(T_min), math.log(T_max), self.dim // input_dim // 2)
        T = torch.exp(T)
        self.weights = 2 * math.pi / T

    def forward(self, x):
        # x: (B, ) or (B, L, input_dim)
        device = x.device
        emb = x[..., None] * self.weights.to(device)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if len(x.shape) == 1:
            return emb.flatten(1)  # (B, dim)
        else:
            return emb.flatten(2)  # (B, L, dim)

