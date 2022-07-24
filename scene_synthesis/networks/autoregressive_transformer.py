# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli, LogNormal, VonMises, Independent, MixtureSameFamily
from .utils import FixedPositionalEncoding, TrainablePE, get_mlp, get_length_mask
from .feature_extractors import Extractor


class Decoder(nn.Module):
    def __init__(self, d_model=768, n_mixture=10):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.n_mixture = n_mixture
        self.location = nn.Sequential(
            nn.Conv2d(in_channels=self.d_model + 128, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1)),
        )
        self.wl = get_mlp(self.d_model + 128, (1 + 2 * 2) * self.n_mixture)
        self.theta = get_mlp(self.d_model + 128, (1 + 1 * 2) * self.n_mixture)
        self.moving = get_mlp(self.d_model + 128 + 192, 1)
        self.s = get_mlp(self.d_model + 128 + 192, (1 + 1 * 2) * self.n_mixture)
        self.omega = get_mlp(self.d_model + 128 + 192, (1 + 1 * 2) * self.n_mixture)

    def forward(self, f, field):
        if field == 'location':
            # f: (B, 10000, d_model + 128)
            B, _, n_feature = f.shape
            f = f.permute([0, 2, 1]).reshape(B, n_feature, 100, 100).contiguous()  # (B, n_feature, 100, 100)
            out = self.location(f)  # (B, 1, 100, 100)
            out = out.flatten(1)  # (B, 10000)
            return out
        if field == 'wl':
            # f: (B, d_model + 128)
            out = self.wl(f)
            return out
        if field == 'theta':
            out = self.theta(f)
            return out
        if field == 'moving':
            out = self.moving(f)
            return out
        if field == 's':
            out = self.s(f)
            return out
        if field == 'omega':
            out = self.omega(f)
            return out


class AutoregressiveTransformer(nn.Module):
    def __init__(self):
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
        self.q = nn.Parameter(torch.randn(self.d_model))

        # extract features from maps
        self.feature_extractor = Extractor(4)

        # Embedding matix for each category
        self.category_embedding = nn.Embedding(4, 64)

        # Positional encoding for other attributes
        self.pe_bbox = FixedPositionalEncoding(proj_dims=64)
        self.pe_velocity = FixedPositionalEncoding(proj_dims=64)

        # map from object feature to transformer input
        self.fc_object = nn.Linear(512, self.d_model)

        # positional encoding for transformer input
        self.pe = TrainablePE(self.d_model)

        # used for autoregressive decoding
        self.n_mixture = 10
        self.prob_category = get_mlp(self.d_model + 128, 4)  # categorical distribution

        self.decoder_pedestrian = Decoder()
        self.decoder_bicyclist = Decoder()
        self.decoder_vehicle = Decoder()

        self.register_buffer('iters', torch.tensor(0))

    def _discrete_loc(self, loc):
        row = ((40 - loc[..., 1]) / 0.8).long()
        col = ((loc[..., 0] + 40) / 0.8).long()
        return row * 100 + col

    def _smooth_loc(self, loc):
        row = torch.div(loc, 100, rounding_mode='trunc')
        col = loc - row * 100
        x = col * 0.8 - 40
        x = x + torch.rand(*x.shape) * 0.8
        y = 40 - row * 0.8
        y = y + torch.rand(*y.shape) * 0.8
        return torch.stack([x, y], dim=-1)

    def _mix_distribution(self, f, distribution, event_shape):
        # f: (B, (1 + event_shape * 2) * self.n_mixture)
        B = f.shape[0]
        mixture = Categorical(logits=f[..., :self.n_mixture])
        prob = f[..., self.n_mixture:].reshape(B, self.n_mixture, 2 * event_shape)
        assert distribution in ['LogNormal', 'VonMises']
        if distribution == 'LogNormal':
            deviation = torch.sigmoid(prob[..., event_shape:]) * 0.5
            deviation = torch.clamp(deviation, min=0.1, max=10)
            prob = LogNormal(prob[..., :event_shape], deviation)  # batch_shape = (B, n_mixture, event_shape)
        elif distribution == 'VonMises':
            if self.iters < 5000:
                deviation = 8
            else:
                deviation = 7 + torch.exp(prob[..., event_shape:])
                deviation = torch.clamp(deviation, min=0.1, max=10)
            prob = VonMises(prob[..., :event_shape], deviation)  # batch_shape = (B, n_mixture, event_shape)
        prob = Independent(prob, reinterpreted_batch_ndims=1)  # batch_shape = (B, n_mixture)
        prob = MixtureSameFamily(mixture, prob)  # batch_shape = B, event_shape
        return prob

    def _decode(self, decoder, output_f, map_f):
        B = output_f.size(0)
        f_in = torch.cat([output_f[:, None, :].expand(B, 10000, self.d_model), map_f], dim=-1)
        f_out = decoder(f_in, 'location')  # (B, 10000)
        prob_location = Categorical(logits=f_out)
        pred_location = prob_location.sample()  # (B, )
        location_f = []
        for location_one, map_f_one in zip(pred_location, map_f):
            location_f.append(map_f_one[location_one])
        location_f = torch.stack(location_f, dim=0)  # (B, 128)

        f_in = torch.cat([
            output_f,
            location_f
        ], dim=-1)  # (B, d_model + 128)
        f_out = decoder(f_in, 'wl')
        prob_wl = self._mix_distribution(f=f_out,
                                         distribution='LogNormal',
                                         event_shape=2)
        f_out = decoder(f_in, 'theta')
        prob_theta = self._mix_distribution(f=f_out,
                                            distribution='VonMises',
                                            event_shape=1)
        pred_wl = prob_wl.sample()
        pred_theta = prob_theta.sample()
        pred_bbox = torch.cat([pred_wl, pred_theta], dim=-1)
        bbox_f = self.pe_bbox(pred_bbox)

        f_in = torch.cat([
            output_f,
            location_f,
            bbox_f
        ], dim=-1)  # (B, d_model + 128 + 192)
        f_out = decoder(f_in, 'moving')
        prob_moving = Bernoulli(logits=f_out)
        f_out = decoder(f_in, 's')
        prob_s = self._mix_distribution(f=f_out,
                                        distribution='LogNormal',
                                        event_shape=1)
        f_out = decoder(f_in, 'omega')
        prob_omega = self._mix_distribution(f=f_out,
                                            distribution='VonMises',
                                            event_shape=1)
        pred_moving = prob_moving.sample()
        pred_s = prob_s.sample()
        pred_omega = prob_omega.sample()

        probs = {
            "location": prob_location,
            "wl": prob_wl,
            "theta": prob_theta,
            "moving": prob_moving,
            "s": prob_s,
            "omega": prob_omega
        }
        preds = {
            "location": pred_location,
            "wl": pred_wl,
            "theta": pred_theta,
            "moving": pred_moving,
            "s": pred_s,
            "omega": pred_omega
        }
        return probs, preds

    def forward(self, samples, lengths, gt, loss_fn):
        # Unpack the samples
        category = samples["category"]  # (B, L)
        location = self._discrete_loc(samples['location'])  # (B, L)
        gt['location'] = self._discrete_loc(gt['location'])
        bbox = samples["bbox"]
        velocity = samples["velocity"]
        maps = samples["map"]
        B, L, *_ = category.shape

        # extract features from map
        map_f = self.feature_extractor(maps)  # (B, 128, 100, 100)
        map_f = map_f.flatten(2, 3).permute([0, 2, 1]).contiguous()  # (B, 10000, 128)

        # embed category
        category_f = self.category_embedding(category)
        # positional encoding for location
        location_f = []
        for location_one, map_f_one in zip(location, map_f):
            location_f.append(map_f_one[location_one])
        location_f = torch.stack(location_f, dim=0)  # (B, L, 128)
        # positional encoding for bounding box
        bbox_f = self.pe_bbox(bbox)
        # positional encoding for velocity
        velocity_f = self.pe_velocity(velocity)
        object_f = torch.cat([category_f, location_f, bbox_f, velocity_f], dim=-1)  # (B, L, 512)
        object_f = self.fc_object(object_f)  # (B, L, d_model)
        input_f = torch.cat([self.q.expand(B, 1, self.d_model),
                            self.pe(object_f)],
                            dim=1)  # (B, L + 1, d_model)

        # Compute the features using causal masking
        length_mask = get_length_mask(lengths + 1)
        output_f = self.transformer_encoder(input_f, src_key_padding_mask=length_mask)  # (B, L + 1, d_model)

        # max pooling
        output_f = output_f.max(dim=1)[0]  # (B, d_model)

        # predict category
        prob_category = self.prob_category(torch.cat([output_f, map_f.mean(dim=1)], dim=-1))  # (B, 4)
        prob_category = Categorical(logits=prob_category)

        loss_select = []
        for decoder in [self.decoder_pedestrian, self.decoder_bicyclist, self.decoder_vehicle]:
            probs, _ = self._decode(decoder, output_f, map_f)
            probs["category"] = prob_category
            loss_components = loss_fn(probs, gt)
            loss_select.append(loss_components)

        loss = {}
        for k in ['all', 'category', 'location', 'wl', 'theta', 'moving', 's', 'omega']:
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
            location = self._discrete_loc(samples['location'])  # (B, L)
            bbox = samples["bbox"]
            velocity = samples["velocity"]
            maps = samples["map"]
            B, L, *_ = category.shape

            # extract features from map
            map_f = self.feature_extractor(maps)  # (B, 128, 100, 100)
            map_f = map_f.flatten(2, 3).permute([0, 2, 1]).contiguous()  # (B, 10000, 128)

            # embed category
            category_f = self.category_embedding(category)
            # positional encoding for location
            location_f = []
            for location_one, map_f_one in zip(location, map_f):
                location_f.append(map_f_one[location_one])
            location_f = torch.stack(location_f, dim=0)  # (B, L, 128)
            # positional encoding for bounding box
            bbox_f = self.pe_bbox(bbox)
            # positional encoding for velocity
            velocity_f = self.pe_velocity(velocity)
            object_f = torch.cat([category_f, location_f, bbox_f, velocity_f], dim=-1)  # (B, L, 512)
            object_f = self.fc_object(object_f)  # (B, L, d_model)
            input_f = torch.cat([self.q.expand(B, 1, self.d_model),
                               self.pe(object_f)],
                               dim=1)  # (B, L + 1, d_model)

            # Compute the features using causal masking
            length_mask = get_length_mask(lengths + 1)
            output_f = self.transformer_encoder(input_f, src_key_padding_mask=length_mask)  # (B, L + 1, d_model)
            # max pooling
            output_f = output_f.max(dim=1)[0]  # (B, d_model)

            # predict category
            if condition['category'] is not None:
                prob_category = None
                pred_category = condition['category']
            else:
                prob_category = self.prob_category(torch.cat([output_f, map_f.mean(dim=1)], dim=-1))  # (B, 4)
                prob_category = Categorical(logits=prob_category)
                pred_category = prob_category.sample().item()

            if pred_category == 0:
                probs = {
                    "category": prob_category,
                    "location": None,
                    "wl": None,
                    "theta": None,
                    "moving": None,
                    "s": None,
                    "omega": None
                }
                preds = {
                    "category": pred_category,
                    "location": None,
                    "wl": None,
                    "theta": None,
                    "moving": None,
                    "s": None,
                    "omega": None
                }
                return preds, probs
            elif pred_category == 1:
                decoder = self.decoder_pedestrian
            elif pred_category == 2:
                decoder = self.decoder_bicyclist
            else:
                decoder = self.decoder_vehicle
            probs, preds = self._decode(decoder, output_f, map_f)
            probs['category'] = prob_category
            for k in preds:
                preds[k] = preds[k].squeeze(0)
            preds['category'] = pred_category
            return preds, probs
