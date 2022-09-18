import torch
from torch import nn
from torch.nn.modules.activation import MultiheadAttention
from .embeddings import SinusoidalEmb, TrainablePE, TrainablePE2D
from .utils import MapIndexLayer


class ConditionalEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dim_t_embed=256, dropout=0.1):
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
            SinusoidalEmb(dim_t_embed, input_dim=1),
            nn.Linear(dim_t_embed, dim_feedforward * 2),
            nn.GELU(),
            nn.Linear(dim_feedforward * 2, dim_feedforward * 2)
        )

        self.act = nn.GELU()

    def forward(self, src, t, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)  # (B, L, d_model)
        t_embed = self.time_mlp(t)  # (B, dim_feedforward * 2)
        weight = t_embed[:, :self.dim_feedforward].unsqueeze(1)
        bias = t_embed[:, self.dim_feedforward:].unsqueeze(1)
        src2 = self.linear1(src)
        src2 = self.dropout(self.act(src2))
        src2 = weight * src2 + bias
        src = src + self.dropout(self.linear2(src2))
        src = self.norm2(src)
        return src


class ConditionalEncoder(nn.Module):
    def __init__(self, d_model, n_layers, nhead=12, dim_feedforward=1024, dim_t_embed=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            ConditionalEncoderLayer(d_model, nhead, dim_feedforward, dim_t_embed, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, t, src_mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, t, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        output = self.norm(output)
        return output


class TransformerBackbone(nn.Module):
    def __init__(self,
                 d_model=768,
                 n_layers=6,
                 dim_pos_embed=256,
                 dim_map_embed=128,
                 dim_category_embed=128,
                 nhead=12,
                 dim_feedforward=1024,
                 dim_t_embed=256,
                 dropout=0.1):
        super().__init__()
        self.pos_embedding = SinusoidalEmb(dim_pos_embed, input_dim=2)
        self.indexing = MapIndexLayer(axes_limit=40, resolution=0.25)
        self.head = nn.Linear(dim_pos_embed + dim_map_embed + dim_category_embed, d_model)
        self.pe = nn.ModuleDict({
            'pedestrian': TrainablePE(d_model),
            'bicyclist': TrainablePE(d_model),
            'vehicle': TrainablePE(d_model)
        })
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
        self.category_embedding = nn.Embedding(3, dim_category_embed)

    def forward(self, pos, fmap, t, sigmas, mask=None):
        # pos: (B, L, 2)
        x = []
        length = []
        for i, field in enumerate(['pedestrian', 'bicyclist', 'vehicle']):
            B, L, *_ = pos[field].shape
            length.append(L)
            map_info = self.indexing(fmap[field], pos[field])
            pos_embed = self.pos_embedding(pos[field])
            category = torch.ones((B, L), dtype=torch.long).to(map_info.device) * i
            fcategory = self.category_embedding(category)  # (B, L, dim_category_embed)
            feature = torch.cat([fcategory, pos_embed, map_info], dim=-1)
            feature = self.pe[field](self.head(feature))
            x.append(feature)
        x = torch.cat(x, dim=1)
        x = self.body(x, t, src_key_padding_mask=mask)
        x = self.tail(x) / sigmas[:, None, None]
        return {
            'pedestrian': x[:, :length[0]],
            'bicyclist': x[:, length[0]:length[0] + length[1]],
            'vehicle': x[:, length[0] + length[1]:]
        }

