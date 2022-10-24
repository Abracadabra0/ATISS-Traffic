import torch
from torch import nn
from torch.nn.modules.activation import MultiheadAttention
from .embeddings import SinusoidalEmb, PositionalEncoding2D
from .utils import MapIndexLayer


class ConditionalEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_map_embed=512, size_fmap=160, dim_feedforward=2048, dim_t_embed=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.time_mlp = nn.Sequential(
            SinusoidalEmb(dim_t_embed, input_dim=1, T_min=1e-3, T_max=10),
            nn.Linear(dim_t_embed, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, dim_feedforward)
        )

        self.act = nn.GELU()
        self.indexing = MapIndexLayer(size_fmap)
        self.map_mlp = nn.Sequential(
            nn.Linear(dim_map_embed, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, src, t, fmap, loc, src_mask=None, src_key_padding_mask=None):
        # incorporate map info
        map_info = self.indexing(fmap, loc)
        map_embed = self.map_mlp(map_info)  # (B, L + 1, C)
        src[:, 1:] = src[:, 1:] + map_embed[:, 1:]
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)  # (B, L, d_model)
        t_embed = self.time_mlp(t)  # (B, dim_feedforward)
        src2 = self.linear1(src)
        src2 = self.dropout(self.act(src2))
        src2 = src2 + t_embed[:, None, :]
        src = src + self.dropout(self.linear2(src2))
        src = self.norm2(src)
        return src


class PositionPredictor(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
            nn.Sigmoid()
        )

    def forward(self, f):
        return self.body(f) * 2 - 1


class ConditionalEncoder(nn.Module):
    def __init__(self, d_model, n_layers, nhead=12, dim_map_embed=512, size_fmap=160, dim_feedforward=2048, dim_t_embed=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            ConditionalEncoderLayer(d_model, nhead, dim_map_embed, size_fmap, dim_feedforward, dim_t_embed, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pos_predictor = PositionPredictor(d_model)

    def forward(self, src, t, fmap, src_mask=None, src_key_padding_mask=None):
        output = src
        loc = self.pos_predictor(output)
        for mod in self.layers:
            output = mod(output, t, fmap, loc, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            loc = self.pos_predictor(output)

        return loc[:, 1:]


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
        x = self.body(x, t, fmap, src_key_padding_mask=mask)

        return {
            'pedestrian': x[:, :length[0]],
            'bicyclist': x[:, length[0]:length[0] + length[1]],
            'vehicle': x[:, length[0] + length[1]:]
        }

