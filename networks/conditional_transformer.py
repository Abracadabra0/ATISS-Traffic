import torch
from torch import nn
from torch.nn.modules.activation import MultiheadAttention
from .embeddings import SinusoidalEmb, TrainablePE


class MapIndexLayer(nn.Module):
    def __init__(self, axes_limit=40, resolution=0.25):
        super().__init__()
        self.axes_limit = axes_limit
        self.resolution = resolution
        self.wl = int(self.axes_limit * 2 / self.resolution)

    def forward(self, fmap, loc):
        # fmap: (B, C, wl, wl)
        # loc: (B, L, 2)
        C = fmap.size(1)
        loc = loc.clamp(min=-1, max=0.999)
        x = loc[..., 0] * self.axes_limit
        y = loc[..., 1] * self.axes_limit
        row = ((self.axes_limit - y) / self.resolution).long()
        col = ((self.axes_limit + x) / self.resolution).long()
        idx = row * self.wl + col  # (B, L)
        idx = idx[..., None].repeat(1, 1, C)  # (B, L, C)
        fmap = fmap.flatten(2, 3).permute(0, 2, 1)  # (B, wl * wl, C)
        indexed = fmap.gather(dim=1, index=idx)  # (B, L, C)
        return indexed


class ConditionalEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dim_t_embed=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.time_mlp = nn.Sequential(
            SinusoidalEmb(dim_t_embed),
            nn.Linear(dim_t_embed, dim_feedforward * 2),
            nn.GELU(),
            nn.Linear(dim_feedforward * 2, dim_feedforward * 2)
        )

        self.act = nn.GELU()

    def forward(self, src, t, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)  # (B, L, d_model)

        t_embed = self.time_mlp(t)  # (B, dim_feedforward * 2)
        weight = t_embed[:, :self.dim_feedforward].unsqueeze(1)
        bias = t_embed[:, self.dim_feedforward:].unsqueeze(1)
        src2 = self.linear1(src)
        src2 = weight * src2 + bias
        src2 = self.linear2(self.dropout(self.act(src2)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class ConditionalEncoder(nn.Module):
    def __init__(self, d_model, n_layers, nhead=12, dim_feedforward=2048, dim_t_embed=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.layers = [ConditionalEncoderLayer(d_model, nhead, dim_feedforward, dim_t_embed, dropout)
                       for _ in range(n_layers)]
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, t, src_mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, t, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        output = self.norm(output)
        return output


class TransformerBackbone(nn.Module):
    def __init__(self,
                 d_model,
                 n_layers,
                 dim_pos_embed=64,
                 dim_map_embed=128,
                 nhead=12, dim_feedforward=2048, dim_t_embed=256, dropout=0.1):
        super().__init__()
        self.pos_embedding = SinusoidalEmb(dim_pos_embed)
        self.indexing = MapIndexLayer(axes_limit=40, resolution=0.25)
        self.head = nn.Sequential(
            nn.Linear(dim_pos_embed * 2 + dim_map_embed, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.pe = TrainablePE(d_model)
        self.body = ConditionalEncoder(d_model=d_model,
                                       n_layers=n_layers,
                                       nhead=nhead,
                                       dim_feedforward=dim_feedforward,
                                       dim_t_embed=dim_t_embed,
                                       dropout=dropout)
        self.tail = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, pos, fmap, t, mask=None):
        # pos: (B, L, 2)
        map_info = self.indexing(fmap, pos)  # (B, L, dim_map_embed)
        pos_embed = self.pos_embedding(pos)  # (B, L, dim_pos_embed)
        x = torch.cat([pos_embed, map_info], dim=2)
        x = self.pe(self.head(x))  # (B, L, d_model)
        x = self.body(x, t, src_key_padding_mask=mask)
        x = self.tail(x)
        return x

