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
from .utils import FixedPositionalEncoding, PositionalEncoding, get_mlp, get_length_mask


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
        # self.pe_location = FixedPositionalEncoding(proj_dims=64, t_min=1 / 8, t_max=8)
        # self.pe_bbox = FixedPositionalEncoding(proj_dims=64, t_min=1 / 8, t_max=8)
        # self.pe_velocity = FixedPositionalEncoding(proj_dims=64, t_min=1 / 8, t_max=8)
        self.pe_location = get_mlp(2, 64 * 2)
        self.pe_bbox = get_mlp(3, 64 * 3)
        self.pe_velocity = get_mlp(2, 64 * 2)

        # map from object feature to transformer input
        self.fc_map = get_mlp(512, self.d_model)
        self.fc_object = get_mlp(512, self.d_model)

        # embed attribute extractor
        self.q = nn.Parameter(torch.randn(self.d_model))

        # positional encoding for transformer input
        self.pe = PositionalEncoding(self.d_model)

        # used for autoregressive decoding
        self.n_mixture = 10
        self.prob_category = get_mlp(self.d_model, 4)  # categorical distribution

        self.decoder_pedestrian = nn.Sequential(get_mlp(self.d_model,
                                                        (1 + 2 * 2) * self.n_mixture),  # location
                                                get_mlp(self.d_model + 64 * 2,
                                                        (1 + 2 * 2) * self.n_mixture +
                                                        (1 + 1 * 2) * self.n_mixture),  # bbox
                                                get_mlp(self.d_model + 64 * 5,
                                                        1 + (1 + 1 * 2) * self.n_mixture +
                                                        (1 + 1 * 2) * self.n_mixture))  # velocity
        self.decoder_bicyclist = nn.Sequential(get_mlp(self.d_model,
                                                       (1 + 2 * 2) * self.n_mixture),  # location
                                               get_mlp(self.d_model + 64 * 2,
                                                       (1 + 2 * 2) * self.n_mixture +
                                                       (1 + 1 * 2) * self.n_mixture),  # bbox
                                               get_mlp(self.d_model + 64 * 5,
                                                       1 + (1 + 1 * 2) * self.n_mixture +
                                                       (1 + 1 * 2) * self.n_mixture))  # velocity

        self.decoder_vehicle = nn.Sequential(get_mlp(self.d_model,
                                                     (1 + 2 * 2) * self.n_mixture),  # location
                                             get_mlp(self.d_model + 64 * 2,
                                                     (1 + 2 * 2) * self.n_mixture +
                                                     (1 + 1 * 2) * self.n_mixture),  # bbox
                                             get_mlp(self.d_model + 64 * 5,
                                                     1 + (1 + 1 * 2) * self.n_mixture +
                                                     (1 + 1 * 2) * self.n_mixture))  # velocity
        self.iters = 0

    def mix_distribution(self, f, distribution, event_shape):
        # f: (B, (1 + event_shape * 2) * self.n_mixture)
        B = f.shape[0]
        mixture = Categorical(logits=f[:, :self.n_mixture])
        prob = f[:, self.n_mixture:].reshape(B, self.n_mixture, 2 * event_shape)
        prob = distribution(prob[..., :event_shape], torch.sigmoid(prob[..., event_shape:]) * 0.5)  # batch_shape = (B, n_mixture, event_shape)
        prob = Independent(prob, reinterpreted_batch_ndims=1)  # batch_shape = (B, n_mixture)
        prob = MixtureSameFamily(mixture, prob)  # batch_shape = B, event_shape
        return prob

    def forward(self, samples, lengths, gt, loss_fn):
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
        object_f = self.fc_object(torch.flatten(object_f, start_dim=0, end_dim=1)).reshape(B, L,
                                                                                           self.d_model)  # (B, L, d_model)

        input_f = torch.cat([map_f[:, None, :],
                             self.q.expand(B, 1, self.d_model),
                             self.pe(object_f)],
                            dim=1)  # (B, L + 2, d_model)

        # Compute the features using causal masking
        length_mask = get_length_mask(lengths + 2)
        output_f = self.transformer_encoder(input_f, src_key_padding_mask=length_mask)
        # take only the encoded q token
        output_f = output_f[:, 1, :]  # (B, d_model)

        # predict category
        prob_category = self.prob_category(output_f)  # (B, 4)
        prob_category = Categorical(logits=prob_category)
        if self.iters % 100 == 99:
            print(f'logits: {prob_category.logits}')

        loss_select = []
        for decoder in [self.decoder_pedestrian, self.decoder_bicyclist, self.decoder_vehicle]:
            location_f = decoder[0](output_f)
            prob_location = self.mix_distribution(location_f, Normal, 2)
            pred_location = prob_location.sample()

            bbox_f = decoder[1](torch.cat([output_f, self.pe_location(pred_location)], dim=-1))
            prob_wl = self.mix_distribution(bbox_f[:, :(1 + 2 * 2) * self.n_mixture], LogNormal, 2)
            prob_theta = self.mix_distribution(bbox_f[:, (1 + 2 * 2) * self.n_mixture:], VonMises, 1)
            pred_bbox = torch.cat([prob_wl.sample(), prob_theta.sample()], dim=-1)

            velocity_f = decoder[2](torch.cat([output_f, self.pe_location(pred_location), self.pe_bbox(pred_bbox)], dim=-1))
            prob_moving = Bernoulli(logits=velocity_f[:, :1])
            prob_s = self.mix_distribution(velocity_f[:, 1:1 + (1 + 1 * 2) * self.n_mixture], LogNormal, 1)
            prob_omega = self.mix_distribution(velocity_f[:, 1 + (1 + 1 * 2) * self.n_mixture:], VonMises, 1)

            probs = {
                "category": prob_category,
                "location": prob_location,
                "bbox": {"wl": prob_wl, "theta": prob_theta},
                "velocity": {"moving": prob_moving, "s": prob_s, "omega": prob_omega}
            }
            loss_components = loss_fn(probs, gt)
            loss_select.append(loss_components)
            if self.iters % 100 == 99:
                print(f'loc: {pred_location.squeeze() * 40}')

        loss = {}
        for k in ['all', 'category', 'location', 'bbox', 'velocity']:
            loss[k] = loss_select[0][k]
            loss[k] = torch.where(gt['category'] == 2, loss_select[1][k], loss[k])
            loss[k] = torch.where(gt['category'] == 3, loss_select[2][k], loss[k])
            loss[k] = loss[k].mean()

        self.iters += 1
        return loss

    def generate(self, samples, lengths, condition):
        # Unpack the samples
        # B = 1
        self.eval()
        with torch.no_grad():
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
            object_f = self.fc_object(torch.flatten(object_f, start_dim=0, end_dim=1)).reshape(B, L,
                                                                                               self.d_model)  # (B, L, d_model)

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
            if condition['category'] is not None:
                prob_category = None
                pred_category = condition['category']
            else:
                prob_category = self.prob_category(output_f)  # (B, 4)
                prob_category = Categorical(logits=prob_category)
                pred_category = prob_category.sample().item()

            if pred_category == 0:
                probs = {
                    "category": prob_category,
                    "location": None,
                    "bbox": {"wl": None, "theta": None},
                    "velocity": {"moving": None, "s": None, "omega": None}
                }
                preds = {
                    "category": pred_category,
                    "location": None,
                    "bbox": {"wl": None, "theta": None},
                    "velocity": {"moving": None, "s": None, "omega": None}
                }
                return preds, probs
            elif pred_category == 1:
                decoder = self.decoder_pedestrian
            elif pred_category == 2:
                decoder = self.decoder_bicyclist
            else:
                decoder = self.decoder_vehicle

            if condition['location'] is not None:
                prob_location = None
                pred_location = condition['location']
            else:
                location_f = decoder[0](output_f)
                prob_location = self.mix_distribution(location_f, Normal, 2)
                pred_location = prob_location.sample()

            if condition['bbox'] is not None:
                prob_wl = None
                prob_theta = None
                pred_wl = condition['bbox']['wl']
                pred_theta = condition['bbox']['theta']
            else:
                bbox_f = decoder[1](torch.cat([output_f, self.pe_location(pred_location)], dim=-1))
                prob_wl = self.mix_distribution(bbox_f[:, :(1 + 2 * 2) * self.n_mixture], LogNormal, 2)
                prob_theta = self.mix_distribution(bbox_f[:, (1 + 2 * 2) * self.n_mixture:], VonMises, 1)
                pred_wl = prob_wl.sample()
                pred_theta = prob_theta.sample()
            pred_bbox = torch.cat([pred_wl, pred_theta], dim=-1)

            if condition['velocity'] is not None:
                prob_moving = None
                prob_s = None
                prob_omega = None
                pred_moving = condition['velocity']['moving']
                pred_s = condition['velocity']['s']
                pred_omega = condition['velocity']['omega']
            else:
                velocity_f = decoder[2](
                    torch.cat([output_f, self.pe_location(pred_location), self.pe_bbox(pred_bbox)], dim=-1))
                prob_moving = Bernoulli(logits=velocity_f[:, :1])
                prob_s = self.mix_distribution(velocity_f[:, 1:1 + (1 + 1 * 2) * self.n_mixture], LogNormal, 1)
                prob_omega = self.mix_distribution(velocity_f[:, 1 + (1 + 1 * 2) * self.n_mixture:], VonMises, 1)
                pred_moving = prob_moving.sample()
                pred_s = prob_s.sample()
                pred_omega = prob_omega.sample()

            probs = {
                "category": prob_category,
                "location": prob_location,
                "bbox": {"wl": prob_wl, "theta": prob_theta},
                "velocity": {"moving": prob_moving, "s": prob_s, "omega": prob_omega}
            }
            preds = {
                "category": pred_category,
                "location": pred_location.squeeze(0) * 40,
                "bbox": {"wl": pred_wl.squeeze(0), "theta": pred_theta.squeeze(0)},
                "velocity": {"moving": pred_moving.squeeze(0), "s": pred_s.squeeze(0), "omega": pred_omega.squeeze(0)}
            }
            return preds, probs


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
