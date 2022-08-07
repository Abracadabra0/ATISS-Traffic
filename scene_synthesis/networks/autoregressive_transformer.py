# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import math
import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli, LogNormal, VonMises, Independent, MixtureSameFamily
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from .utils import FixedPositionalEncoding, TrainablePE, get_mlp, get_length_mask
from .feature_extractors import Extractor
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import cv2


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
        self.wl = get_mlp(128, (1 + 2 + 2) * self.n_mixture)
        self.theta = get_mlp(128, (1 + 2 + 1) * self.n_mixture)
        self.moving = get_mlp(self.d_model + 128 + 192, 1)
        self.s = get_mlp(self.d_model + 128 + 192, (1 + 1 + 1) * self.n_mixture)
        self.omega = get_mlp(self.d_model + 128 + 192, (1 + 2 + 1) * self.n_mixture)

    def forward(self, f, field):
        if field == 'location':
            # f: (B, 6400, d_model + 128)
            B, _, n_feature = f.shape
            f = f.permute([0, 2, 1]).reshape(B, n_feature, 80, 80).contiguous()  # (B, n_feature, 80, 80)
            out = self.location(f)  # (B, 1, 80, 80)
            out = out.flatten(1)  # (B, 6400)
            return out
        if field == 'wl':
            # f: (B, 128)
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
    def __init__(self, loss_fn=None, lr=None, scheduler=None, logger=None):
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
        self.feature_extractor = Extractor(9)

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
        self.loss_fn = loss_fn
        if lr is not None:
            self.optimizer = Adam(self.parameters(), lr=lr)
        if scheduler is not None:
            self.scheduler = LambdaLR(self.optimizer, scheduler)
        self.logger = logger

    def _discrete_loc(self, loc):
        row = (40 - loc[..., 1]).long()
        col = (loc[..., 0] + 40).long()
        return row * 80 + col

    def _smooth_loc(self, loc):
        row = torch.div(loc, 80, rounding_mode='trunc')
        col = loc - row * 80
        x = col - 40
        x = x + torch.rand(*x.shape).to(x.device)
        y = 40 - row
        y = y + torch.rand(*y.shape).to(y.device)
        return torch.stack([x, y], dim=-1)

    def _mix_lognormal(self, f, event_shape):
        # f: (B, (1 + event_shape * 2) * self.n_mixture)
        B = f.shape[0]
        mixture = Categorical(logits=f[..., :self.n_mixture])
        f = f[..., self.n_mixture:].reshape(B, self.n_mixture, 2 * event_shape)
        mu = f[..., :event_shape]
        sigma = torch.sigmoid(torch.clamp(f[..., event_shape:], min=-5)) * 0.5
        prob = LogNormal(mu, sigma)  # batch_shape = (B, n_mixture, event_shape)
        prob = Independent(prob, reinterpreted_batch_ndims=1)  # batch_shape = (B, n_mixture)
        prob = MixtureSameFamily(mixture, prob)  # batch_shape = B, event_shape
        return prob

    def _mix_vonmises(self, f):
        # f: (B, (1 + 2 + 1) * self.n_mixture)
        B = f.shape[0]
        mixture = Categorical(logits=f[..., :self.n_mixture])
        f = f[..., self.n_mixture:].reshape(B, self.n_mixture, 3)
        cos = f[..., 0:1]
        sin = f[..., 1:2]
        kappa = 7 + torch.exp(torch.clamp(f[..., 2:3], max=5))
        norm = torch.sqrt(cos ** 2 + sin ** 2) + 1e-3
        cos = cos / norm
        sin = sin / norm
        mu = torch.arcsin(sin)
        mu = torch.where(cos >= 0, mu, torch.sign(sin) * math.pi - mu)
        prob = VonMises(mu, kappa)
        prob = Independent(prob, reinterpreted_batch_ndims=1)
        prob = MixtureSameFamily(mixture, prob)
        prob.params = {'mixture': mixture.logits,
                       'cos': cos,
                       'sin': sin,
                       'kappa': kappa}
        return prob

    def _max_prob_sample(self, prob, n_sample):
        assert prob.batch_shape[0] == 1
        sample = prob.sample(torch.tensor([n_sample]))
        log_prob = prob.log_prob(sample)
        max_prob = log_prob.argmax(dim=0).squeeze()
        pred = sample[max_prob]
        return pred

    def _decode(self, decoder, output_f, map_f, n_sample=1):
        B = output_f.size(0)
        f_in = torch.cat([output_f[:, None, :].expand(B, 6400, self.d_model), map_f], dim=-1)
        f_out = decoder(f_in, 'location')  # (B, 6400)
        prob_location = Categorical(logits=f_out)
        if n_sample == 1:
            pred_location = prob_location.sample()  # (B, )
        else:
            pred_location = self._max_prob_sample(prob_location, n_sample)
        location_f = []
        for location_one, map_f_one in zip(pred_location, map_f):
            location_f.append(map_f_one[location_one])
        location_f = torch.stack(location_f, dim=0)  # (B, 128)

        f_in = torch.cat([
            location_f
        ], dim=-1)  # (B, 128)
        f_out = decoder(f_in, 'wl')
        prob_wl = self._mix_lognormal(f=f_out,
                                      event_shape=2)
        f_out = decoder(f_in, 'theta')
        prob_theta = self._mix_vonmises(f=f_out)
        if n_sample == 1:
            pred_wl = prob_wl.sample()
            pred_theta = prob_theta.sample()
        else:
            pred_wl = self._max_prob_sample(prob_wl, n_sample)
            pred_theta = self._max_prob_sample(prob_theta, n_sample)
        pred_bbox = torch.cat([pred_wl, pred_theta], dim=-1)
        bbox_f = self.pe_bbox(pred_bbox)

        f_in = torch.cat([
            self.d_model,
            location_f,
            bbox_f
        ], dim=-1)  # (B, 128 + 192)
        f_out = decoder(f_in, 'moving')
        prob_moving = Bernoulli(logits=f_out)
        f_out = decoder(f_in, 's')
        prob_s = self._mix_lognormal(f=f_out, event_shape=1)
        f_out = decoder(f_in, 'omega')
        prob_omega = self._mix_vonmises(f=f_out)
        if n_sample == 1:
            pred_moving = prob_moving.sample()
            pred_s = prob_s.sample()
            pred_omega = prob_omega.sample()
        else:
            pred_moving = self._max_prob_sample(prob_moving, n_sample)
            pred_s = self._max_prob_sample(prob_s, n_sample)
            pred_omega = self._max_prob_sample(prob_omega, n_sample)

        probs = {
            "location": prob_location,
            "wl": prob_wl,
            "theta": prob_theta,
            "moving": prob_moving,
            "s": prob_s,
            "omega": prob_omega
        }
        preds = {
            "location": self._smooth_loc(pred_location),
            "wl": pred_wl,
            "theta": pred_theta,
            "moving": pred_moving,
            "s": pred_s,
            "omega": pred_omega
        }
        return probs, preds

    def _rasterize(self, occupancy, category, location, bbox):
        B = occupancy.size(0)
        device = occupancy.device
        category = category.to('cpu').numpy()
        location = location.to('cpu').numpy()
        bbox = bbox.to('cpu').numpy()
        occupancy = occupancy.to('cpu').numpy()
        for i in range(B):
            if category[i] == 0:
                continue
            layer = category[i] - 1
            w, l, theta = bbox[i]
            corners = np.array([[l / 2, w / 2],
                                [-l / 2, w / 2],
                                [-l / 2, -w / 2],
                                [l / 2, -w / 2]])
            rotation = np.array([[np.cos(theta), np.sin(theta)],
                                 [-np.sin(theta), np.cos(theta)]])
            corners = np.dot(corners, rotation) + location[i]
            corners[:, 0] = corners[:, 0] + 40
            corners[:, 1] = 40 - corners[:, 1]
            corners = np.floor(corners / 0.25).astype(int)
            cv2.fillConvexPoly(occupancy[i, layer], corners, 1)
        return torch.tensor(occupancy, dtype=torch.float32, device=device)

    def _forward_step(self, samples, lengths, gt):
        B, L, *_ = samples["category"].shape
        # Unpack the samples
        category = samples["category"]  # (B, L)
        location = samples['location']  # (B, L)
        bbox = samples["bbox"]
        velocity = samples["velocity"]
        maps = samples["map"]

        # extract features from map
        map_f = self.feature_extractor(maps)  # (B, 128, 80, 80)
        map_f = map_f.flatten(2, 3).permute([0, 2, 1]).contiguous()  # (B, 6400, 128)

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

        # masking
        mask = get_length_mask(lengths + 1)
        output_f = self.transformer_encoder(input_f, src_key_padding_mask=mask)  # (B, L + 1, d_model)

        # max pooling
        output_f = output_f.max(dim=1)[0]  # (B, d_model)

        # predict category
        prob_category = self.prob_category(torch.cat([output_f, map_f.mean(dim=1)], dim=-1))  # (B, 4)
        prob_category = Categorical(logits=prob_category)

        loss_select = []
        pred_select = []
        for decoder in [self.decoder_pedestrian, self.decoder_bicyclist, self.decoder_vehicle]:
            probs, preds = self._decode(decoder, output_f, map_f)
            probs["category"] = prob_category
            loss_components = self.loss_fn(probs, gt)
            loss_select.append(loss_components)
            pred_select.append(preds)

        loss = {}
        for k in ['all', 'category', 'location', 'wl', 'theta', 'moving', 's', 'omega']:
            loss[k] = loss_select[0][k]
            loss[k] = torch.where(gt['category'] == 2, loss_select[1][k], loss[k])
            loss[k] = torch.where(gt['category'] == 3, loss_select[2][k], loss[k])

        pred = {k: [] for k in ['location', 'wl', 'theta', 'moving', 's', 'omega']}
        for k in ['location', 'wl', 'theta', 'moving', 's', 'omega']:
            for i in range(B):
                if gt['category'][i] == 0:
                    pred[k].append(torch.zeros_like(pred_select[0][k][0]))
                else:
                    choice = gt['category'][i].item() - 1
                    pred[k].append(pred_select[choice][k][i])
            pred[k] = torch.stack(pred[k], dim=0)
        pred['category'] = gt['category']

        return loss, pred

    def forward(self, samples, lengths, gt):
        samples['location'] = self._discrete_loc(samples['location'])
        gt['location'] = self._discrete_loc(gt['location'])
        B, L, *_ = samples["category"].shape

        fields = ['all', 'category', 'location', 'wl', 'theta', 'moving', 's', 'omega']
        all_loss = {k: torch.zeros(B, device=lengths.device) for k in fields}
        window_size = gt['category'].size(1)
        for step in range(window_size):
            # keep the first step objects
            self.optimizer.zero_grad()
            gt_step = {}
            for field in ['category', 'location', 'bbox', 'velocity']:
                gt_step[field] = gt[field][:, step]
            loss, pred = self._forward_step(samples, lengths, gt_step)  # (B, )
            for k in fields:
                all_loss[k] += loss[k].detach()
            loss['all'].mean().backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
            self.optimizer.step()
            self.scheduler.step()

            new_samples = {field: [] for field in ['category', 'location', 'bbox', 'velocity']}
            pred['bbox'] = torch.cat([pred['wl'], pred['theta']], dim=-1)
            occupancy = self._rasterize(samples['map'][:, 3:], pred['category'], pred['location'], pred['bbox'])
            new_samples['map'] = torch.cat([samples['map'][:, :3], occupancy], dim=1)
            pred['location'] = self._discrete_loc(pred['location'])
            pred['velocity'] = torch.cat([pred['s'], pred['omega']], dim=-1) * pred['moving']
            for i in range(B):
                length = lengths[i].item()
                new_samples_one = {}
                for field in ['category', 'location', 'bbox', 'velocity']:
                    new_samples_one[field] = torch.cat([samples[field][i, :length], pred[field][i].unsqueeze(0)], dim=0)
                idx = list(range(length + 1))
                idx.sort(key=lambda x: new_samples_one['location'][x])
                for field in ['category', 'location', 'bbox', 'velocity']:
                    new_samples_one[field] = new_samples_one[field][idx]
                    new_samples[field].append(new_samples_one[field])
            for field in ['category', 'location', 'bbox', 'velocity']:
                new_samples[field] = pad_sequence(new_samples[field], batch_first=True)
            lengths = lengths + 1
            samples = new_samples

            self.iters += 1

        for k in fields:
            all_loss[k] = all_loss[k].mean() / window_size
        return all_loss

    def generate(self, samples, lengths, condition, n_sample):
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

            mask = get_length_mask(lengths + 1)
            # Compute the features using causal masking
            output_f = self.transformer_encoder(input_f, src_key_padding_mask=mask)  # (B, L + 1, d_model)
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
            probs, preds = self._decode(decoder, output_f, map_f, n_sample)
            probs['category'] = prob_category
            for k in preds:
                preds[k] = preds[k].squeeze(0)
            preds['category'] = pred_category
            return preds, probs
