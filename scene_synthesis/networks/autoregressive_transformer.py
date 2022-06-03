# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli, Normal, LogNormal, VonMises, Independent, MixtureSameFamily
from .utils import FixedPositionalEncoding, get_mlp, get_length_mask


class AutoregressiveTransformer(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        # Build a transformer encoder
        self.transformer_encoder = nn.Transformer(
            d_model=768,
            nhead=12,
            num_encoder_layers=6,
            dim_feedforward=2048,
            activation='gelu',
            batch_first=True
        ).encoder
        self.d_model = 768

        # extract features from maps
        self.feature_extractor = feature_extractor

        # Embedding matix for each category
        self.category_embedding = nn.Embedding(4, 64)

        # Positional encoding for other attributes
        self.pe_location = FixedPositionalEncoding(proj_dims=64, t_min=1, t_max=64)
        self.pe_bbox = FixedPositionalEncoding(proj_dims=64, t_min=1/8, t_max=8)
        self.pe_velocity = FixedPositionalEncoding(proj_dims=64, t_min=1/8, t_max=8)

        # map from object feature to transformer input
        self.fc_map = get_mlp(512, self.d_model)
        self.fc_object = get_mlp(512, self.d_model)

        # embed attribute extractor
        self.q = nn.Parameter(torch.randn(self.d_model))

        # used for autoregressive decoding
        self.n_mixture = 10
        self.decoder_rnn = [get_mlp(self.d_model, 512),
                            get_mlp(512 + 64, 512),
                            get_mlp(512 + 64 * 2, 512),
                            get_mlp(512 + 64 * 3, 512)]
        self.decoder_rnn = nn.Sequential(*self.decoder_rnn)
        self.prob_category = get_mlp(512, 4)  # categorical distribution
        self.prob_location = get_mlp(512, self.n_mixture + (2 + 2) * self.n_mixture)  # 2D normal distribution
        self.prob_wl = get_mlp(512, self.n_mixture + (2 + 2) * self.n_mixture)  # 2D LogNorm distribution
        self.prob_theta = get_mlp(512, self.n_mixture + 2 * self.n_mixture)  # VonMises distribution
        self.prob_moving = get_mlp(512, 1)
        self.prob_s = get_mlp(512, self.n_mixture + 2 * self.n_mixture)  # LogNorm distribution
        self.prob_omega = get_mlp(512, self.n_mixture + 2 * self.n_mixture)  # VonMises distribution

    def forward(self, samples, lengths, gt):
        # Unpack the samples
        category = samples["category"]  # (B, L)
        location = samples["location"]
        bbox = samples["bbox"]
        velocity = samples["velocity"]
        maps = samples["map"]
        B, L, *_ = category.shape

        # extract features from map
        map_f = self.feature_extractor(maps)
        map_f = self.fc_map(map_f)  # (B, d_model)

        # embed category
        category_f = self.category_embedding(category)
        # positional encoding for location
        location_f = self.pe_location(location)
        # positional encoding for bounding box
        bbox_f = self.pe_bbox(bbox)
        # positional encoding for velocity
        velocity_f = self.pe_velocity(velocity)
        object_f = torch.cat([category_f, location_f, bbox_f, velocity_f], dim=-1)  # (B, L, 512)
        object_f = self.fc_object(torch.flatten(object_f, start_dim=0, end_dim=1)).reshape(B, L, self.d_model)  # (B, L, d_model)

        input_f = torch.cat([map_f[:, None, :],
                             self.q.expand(B, 1, self.d_model),
                             object_f],
                            dim=1)  # (B, L + 2, d_model)

        # Compute the features using causal masking
        length_mask = get_length_mask(lengths + 2)
        output_f = self.transformer_encoder(input_f, src_key_padding_mask=length_mask)
        # take only the encoded q token
        output_f = output_f[:, 1, :]  # (B, d_model)

        # predict category
        f = self.decoder_rnn[0](output_f)  # (B, 512)
        prob_category = Categorical(logits=self.prob_category(f))

        # predict location
        gt_category = gt['category']  # (B,)
        gt_category_f = self.category_embedding(gt_category)
        f = self.decoder_rnn[1](torch.cat([f, gt_category_f], dim=-1))
        prob_location = self.prob_location(f)
        mixture_distribution = Categorical(logits=prob_location[:, :self.n_mixture])
        prob_location = prob_location[:, self.n_mixture:].reshape(B, self.n_mixture, 4)  # (B, n_mixture, 4)
        prob_location = Normal(prob_location[..., :2], torch.exp(prob_location[..., 2:]))  # batch_shape = (B, n_mixture, 2)
        prob_location = Independent(prob_location, reinterpreted_batch_ndims=1)  # batch_shape = (B, n_mixture), event_shape = 2
        prob_location = MixtureSameFamily(mixture_distribution, prob_location)  # batch_shape = B, event_shape = 2

        # predict wl and theta
        gt_location = gt['location']  # (B, 2)
        gt_location_f = self.pe_location(gt_location)  # (B, 64 * 2)
        f = self.decoder_rnn[2](torch.cat([f, gt_location_f], dim=-1))
        prob_wl = self.prob_wl(f)
        mixture_distribution = Categorical(logits=prob_wl[:, :self.n_mixture])
        prob_wl = prob_wl[:, self.n_mixture:].reshape(B, self.n_mixture, 4)
        prob_wl = LogNormal(prob_wl[..., :2], torch.exp(prob_wl[..., 2:]))
        prob_wl = Independent(prob_wl, reinterpreted_batch_ndims=1)
        prob_wl = MixtureSameFamily(mixture_distribution, prob_wl)

        prob_theta = self.prob_theta(f)  # (B, n_mixture + 2 * n_mixture)
        mixture_distribution = Categorical(logits=prob_theta[:, :self.n_mixture])
        prob_theta = prob_theta[:, self.n_mixture:].reshape(B, self.n_mixture, 2)  # (B, n_mixture, 2)
        prob_theta = VonMises(prob_theta[..., 0], torch.exp(prob_theta[..., 1]))  # batch_shape = (B, n_mixture), event_shape = 1
        prob_theta = MixtureSameFamily(mixture_distribution, prob_theta)  # batch_shape = B, event_shape = 1

        # predict s and omega
        gt_bbox = gt['bbox']
        gt_bbox_f = self.pe_bbox(gt_bbox)  # (B, 64 * 3)
        f = self.decoder_rnn[3](torch.cat([f, gt_bbox_f], dim=-1))
        # predict the probability of s != 0
        prob_moving = self.prob_moving(f).flatten()
        prob_moving = Bernoulli(logits=prob_moving)

        prob_s = self.prob_s(f)
        mixture_distribution = Categorical(logits=prob_s[:, :self.n_mixture])
        prob_s = prob_s[:, self.n_mixture:].reshape(B, self.n_mixture, 2)
        prob_s = LogNormal(prob_s[..., 0], torch.exp(prob_s[..., 1]))
        prob_s = MixtureSameFamily(mixture_distribution, prob_s)

        prob_omega = self.prob_omega(f)
        mixture_distribution = Categorical(logits=prob_omega[:, :self.n_mixture])
        prob_omega = prob_omega[:, self.n_mixture:].reshape(B, self.n_mixture, 2)
        prob_omega = VonMises(prob_omega[..., 0], torch.exp(prob_omega[..., 1]))
        prob_omega = MixtureSameFamily(mixture_distribution, prob_omega)

        return {'category': prob_category,
                'location': prob_location,
                'bbox': (prob_wl, prob_theta),
                'velocity': (prob_s, prob_omega, prob_moving)}



def train_on_batch(model, optimizer, sample_params, config):
    # Make sure that everything has the correct size
    optimizer.zero_grad()
    X_pred = model(sample_params)
    # Compute the loss
    loss = X_pred.reconstruction_loss(sample_params, sample_params["lengths"])
    # Do the backpropagation
    loss.backward()
    # Do the update
    optimizer.step()

    return loss.item()


@torch.no_grad()
def validate_on_batch(model, sample_params, config):
    X_pred = model(sample_params)
    # Compute the loss
    loss = X_pred.reconstruction_loss(sample_params, sample_params["lengths"])
    return loss.item()
