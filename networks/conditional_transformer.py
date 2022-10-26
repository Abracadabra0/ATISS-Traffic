import torch
from torch import nn
from torch.nn.modules.activation import MultiheadAttention
from .embeddings import SinusoidalEmb, PositionalEncoding2D
from .utils import MapIndexLayer
from torch.nn import Transformer


class ConditionalEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_map_embed=512, dim_feedforward=2048, dim_t_embed=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.in_layers = nn.ModuleList([
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        ])
        self.time_mlp = nn.Sequential(
            SinusoidalEmb(dim_t_embed, input_dim=1, T_min=1e-3, T_max=10),
            nn.Linear(dim_t_embed, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model * 2)
        )
        self.out_layers = nn.ModuleList([
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        ])
        self.skip_conn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.map_mlp = nn.Sequential(
            nn.Linear(dim_map_embed, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.indexing = MapIndexLayer()

    def forward(self, src, t, fmap, loc, src_mask=None, src_key_padding_mask=None):
        # incorporate map info
        map_info = self.indexing(fmap, loc)
        map_embed = self.map_mlp(map_info)  # (B, L, d_model)
        src[:, 1:] = src[:, 1:] + map_embed
        h = self.in_layers[0](src)
        h = self.in_layers[1](h)
        h = self.in_layers[2](h, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        h = self.out_layers[0](h)
        t_embed = self.time_mlp(t).unsqueeze(1)  # (B, 1, d_model * 2)
        scale = t_embed[..., :self.d_model]
        shift = t_embed[..., self.d_model:]
        h = (1 + scale) * h + shift
        h = self.out_layers[1](h)
        h = self.out_layers[2](h, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        src = self.skip_conn(src) + h
        return src


class NoisePredictor(nn.Module):
    def __init__(self, d_model, dim_t_embed):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalEmb(dim_t_embed, input_dim=1, T_min=1e-3, T_max=10),
            nn.Linear(dim_t_embed, dim_t_embed),
            nn.GELU(),
            nn.Linear(dim_t_embed, dim_t_embed)
        )
        self.body = nn.Sequential(
            nn.Linear(d_model + dim_t_embed, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, f, t):
        L = f.shape[1]
        t_embed = self.time_mlp(t).unsqueeze(1).repeat(1, L, 1)
        f = torch.cat([f, t_embed], dim=-1)
        return self.body(f)


class ConditionalEncoder(nn.Module):
    def __init__(self, d_model, n_layers, nhead=12, dim_map_embed=512, dim_feedforward=2048, dim_t_embed=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            ConditionalEncoderLayer(d_model, nhead, dim_map_embed, dim_feedforward, dim_t_embed, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.noise_predictor = NoisePredictor(d_model, dim_t_embed=dim_t_embed)

    def forward(self, src, t, fmap, loc, src_mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, t, fmap, loc, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            noise = self.noise_predictor(output[:, 1:], t)
            loc = (loc - noise).clamp(min=-1, max=1)
        return noise


class TransformerBackbone(nn.Module):
    def __init__(self,
                 d_model=768,
                 n_layers=6,
                 dim_pos_embed=256,
                 dim_map_embed=512,
                 dim_category_embed=128,
                 nhead=12,
                 dim_feedforward=2048,
                 dim_t_embed=256,
                 dropout=0.1):
        super().__init__()
        self.pos_embedding = SinusoidalEmb(dim_pos_embed, input_dim=2, T_min=1e-3, T_max=1e3)
        self.head = nn.Sequential(
            nn.Linear(dim_pos_embed + dim_category_embed, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.pe = PositionalEncoding2D(d_model, max_len=100)
        self.body = ConditionalEncoder(d_model=d_model,
                                       n_layers=n_layers,
                                       nhead=nhead,
                                       dim_map_embed=dim_map_embed,
                                       dim_feedforward=dim_feedforward,
                                       dim_t_embed=dim_t_embed,
                                       dropout=dropout)

        self.category_embedding = nn.Embedding(3, dim_category_embed)
        self.empty_token = nn.Parameter(torch.randn(d_model))

    def forward(self, pos, original, fmap, t, mask=None):
        # pos: (B, L, 2)
        B = fmap.shape[0]
        x = [self.empty_token.reshape(1, 1, -1).repeat(B, 1, 1)]
        length = []
        for i, field in enumerate(['pedestrian', 'bicyclist', 'vehicle']):
            B, L, *_ = pos[field].shape
            length.append(L)
            pos_embed = self.pos_embedding(pos[field])
            category = torch.ones((B, L), dtype=torch.long).to(fmap.device) * i
            fcategory = self.category_embedding(category)  # (B, L, dim_category_embed)
            feature = torch.cat([fcategory, pos_embed], dim=-1)
            feature = self.head(feature)
            rank = self.pe.get_rank_from_pts(original[field])
            feature = self.pe(feature, rank)
            x.append(feature)
        x = torch.cat(x, dim=1)  # (B, L + 1, d_model)
        pad = torch.zeros(B, 1).to(x.device)
        mask = torch.cat([pad, mask], dim=1)
        loc = torch.cat(list(pos.values()), dim=1)
        x = self.body(x, t, fmap, loc, src_key_padding_mask=mask)

        return {
            'pedestrian': x[:, :length[0]],
            'bicyclist': x[:, length[0]:length[0] + length[1]],
            'vehicle': x[:, length[0] + length[1]:]
        }

