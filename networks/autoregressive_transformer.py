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


def _eval_poly(y, coef):
    coef = list(coef)
    result = coef.pop()
    while coef:
        result = coef.pop() + y * result
    return result


def _log_modified_bessel_fn(x):
    # compute small solution
    y = (x / 3.75)
    y = y * y
    small = _eval_poly(y, [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2])
    small = small.log()
    # compute large solution
    y = 3.75 / x
    large = x - 0.5 * x.log() + _eval_poly(y, [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2,
                                               -0.2057706e-1, 0.2635537e-1, -0.1647633e-1, 0.392377e-2]).log()
    result = torch.where(x < 3.75, small, large)
    return result


class WeightedNLL(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = dict(weights)
        total = sum(weights.values())
        for k in self.weights:
            self.weights[k] = self.weights[k] / total
        self._eps = 1e-6  # numerical stability for LogNorm

    def _loss_vonmises(self, x, prob):
        # x: (B, 1)
        mixture = prob.params['mixture']  # (B, n_mixture)
        cos = prob.params['cos'].squeeze()  # (B, n_mixture)
        sin = prob.params['sin'].squeeze()  # (B, n_mixture)
        kappa = prob.params['kappa'].squeeze()  # (B, n_mixture)
        B, n_mixture = mixture.shape
        x = x.expand(B, n_mixture)
        log_prob = kappa * (torch.cos(x) * cos + torch.sin(x) * sin) - _log_modified_bessel_fn(kappa) - math.log(
            2 * math.pi)  # (B, n_mixture)
        logits = torch.log_softmax(mixture, dim=-1)
        loss = -torch.logsumexp(log_prob + logits, dim=-1)
        return loss

    def forward(self, probs, gt):
        device = gt['category'].device

        loss_category = -probs['category'].log_prob(gt['category'])

        loss_location = -probs['location'].log_prob(gt['location'])
        loss_location = torch.where(gt['category'] == 0,
                                    torch.tensor(0., device=device),
                                    loss_location)

        loss_wl = -probs['wl'].log_prob(gt['bbox'][:, :2] + self._eps)
        loss_wl = torch.where(gt['category'] == 0,
                              torch.tensor(0., device=device),
                              loss_wl)
        loss_theta = self._loss_vonmises(gt['bbox'][:, 2:], probs['theta'])
        loss_theta = torch.where(gt['category'] == 0,
                                 torch.tensor(0., device=device),
                                 loss_theta)

        loss_s = -probs['s'].log_prob(gt['velocity'][:, :1] + self._eps)
        loss_s = torch.where(gt['velocity'][:, 0] == 0,
                             torch.tensor(0., device=device),
                             loss_s)
        loss_s = torch.where(gt['category'] == 0,
                             torch.tensor(0., device=device),
                             loss_s)
        loss_omega = self._loss_vonmises(gt['velocity'][:, 1:], probs['omega'])
        loss_omega = torch.where(gt['velocity'][:, 0] == 0,
                                 torch.tensor(0., device=device),
                                 loss_omega)
        loss_omega = torch.where(gt['category'] == 0,
                                 torch.tensor(0., device=device),
                                 loss_omega)
        loss_moving = torch.where(gt['velocity'][:, 0] == 0,
                                  -probs['moving'].log_prob(torch.tensor(0., device=device)).squeeze(),
                                  -probs['moving'].log_prob(torch.tensor(1., device=device)).squeeze())
        loss_moving = torch.where(gt['category'] == 0,
                                  torch.tensor(0., device=device),
                                  loss_moving)

        loss = loss_category * self.weights['category'] + \
               loss_location * self.weights['location'] + \
               loss_wl * self.weights['wl'] + \
               loss_theta * self.weights['theta'] + \
               loss_moving * self.weights['moving'] + \
               loss_s * self.weights['s'] + \
               loss_omega * self.weights['omega']

        components = {
            'all': loss,
            'category': loss_category * self.weights['category'],
            'location': loss_location * self.weights['location'],
            'wl': loss_wl * self.weights['wl'],
            'theta': loss_theta * self.weights['theta'],
            'moving': loss_moving * self.weights['moving'],
            's': loss_s * self.weights['s'],
            'omega': loss_omega * self.weights['omega']
        }
        return components


class Decoder(nn.Module):
    def __init__(self, d_model=768, n_mixture=8):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.n_mixture = n_mixture
        self.location = nn.Sequential(
            nn.Conv2d(in_channels=self.d_model + 128, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1))
        )
        self.wl = get_mlp(128, (1 + 2 + 2) * self.n_mixture)
        self.theta = get_mlp(128, (1 + 2 + 1) * self.n_mixture)
        self.moving = get_mlp(128 + 192, 1)
        self.s = get_mlp(128 + 192, (1 + 1 + 1) * self.n_mixture)
        self.omega = get_mlp(128 + 192, (1 + 2 + 1) * self.n_mixture)

    def forward(self, f, field):
        if field == 'location':
            # f: (B, 320 * 320, d_model + 128)
            B, _, n_feature = f.shape
            f = f.permute([0, 2, 1]).reshape(B, n_feature, 320, 320).contiguous()  # (B, n_feature, 320, 320)
            out = self.location(f)  # (B, 1, 320, 320)
            out = out.flatten(1)  # (B, 320 * 320)
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
        self.feature_extractor = Extractor(26)

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
        self.n_mixture = 8
        self.prob_category = get_mlp(self.d_model + 128, 4)  # categorical distribution

        self.decoder_pedestrian = Decoder()
        self.decoder_bicyclist = Decoder()
        self.decoder_vehicle = Decoder()

        self.loss_fn = WeightedNLL(weights={
            'category': 0.1,
            'location': 1.,
            'wl': 0.4,
            'theta': 0.4,
            'moving': 0.2,
            's': 0.2,
            'omega': 0.2
        })

    def _discrete_loc(self, loc):
        row = ((40 - loc[..., 1]) / 0.25).long()
        col = ((loc[..., 0] + 40) / 0.25).long()
        return row * 320 + col

    def _smooth_loc(self, loc):
        row = torch.div(loc, 320, rounding_mode='trunc')
        col = loc - row * 320
        x = col * 0.25 - 40
        x = x + torch.rand(*x.shape).to(x.device) * 0.25
        y = 40 - row * 0.25
        y = y + torch.rand(*y.shape).to(y.device) * 0.25
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
        kappa = 15 + torch.exp(torch.clamp(f[..., 2:3], max=5))
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

    def _mix_vonmises_delta(self, f, sin_theta, cos_theta):
        # f: (B, (1 + 2 + 1) * self.n_mixture)
        B = f.shape[0]
        mixture = Categorical(logits=f[..., :self.n_mixture])
        f = f[..., self.n_mixture:].reshape(B, self.n_mixture, 3)
        cos_delta = f[..., 0:1]
        sin_delta = f[..., 1:2]
        kappa = 15 + torch.exp(torch.clamp(f[..., 2:3], max=5))
        norm = torch.sqrt(cos_delta ** 2 + sin_delta ** 2) + 1e-3
        cos_delta = cos_delta / norm
        sin_delta = sin_delta / norm
        sin = sin_theta * cos_delta + cos_theta * sin_delta
        cos = cos_theta * cos_delta - sin_theta * sin_delta
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

    def _decode(self, decoder, output_f, map_f,
                gt=None,
                n_sample=1,
                prev_occupancy=None):
        B = output_f.size(0)
        f_in = torch.cat([output_f[:, None, :].expand(B, 320 * 320, self.d_model), map_f], dim=-1)
        f_out = decoder(f_in, 'location')  # (B, 320 * 320)
        location_f = []
        if n_sample == 1:
            prob_location = Categorical(logits=f_out)
            pred_location = prob_location.sample()  # (B, )
            pred_location_smoothed = self._smooth_loc(pred_location)
            # teacher forcing
            for location_one, map_f_one in zip(gt['location'], map_f):
                location_f.append(map_f_one[location_one])
        else:
            prob_location = Categorical(logits=f_out)
            for _ in range(50):
                pred_location = self._max_prob_sample(prob_location, n_sample)
                pred_location_smoothed = self._smooth_loc(pred_location)
                # reject overlapping location
                row = int((40 - pred_location_smoothed.squeeze().numpy()[1]) / 0.25)
                col = int((pred_location_smoothed.squeeze().numpy()[0] + 40) / 0.25)
                if prev_occupancy[row, col]:
                    continue
                break
            for location_one, map_f_one in zip(pred_location, map_f):
                location_f.append(map_f_one[location_one])
        location_f = torch.stack(location_f, dim=0)  # (B, 128)

        f_in = location_f  # (B, 128)
        f_out = decoder(f_in, 'wl')
        prob_wl = self._mix_lognormal(f=f_out, event_shape=2)
        f_out = decoder(f_in, 'theta')
        prob_theta = self._mix_vonmises(f=f_out)
        if n_sample == 1:
            pred_wl = prob_wl.sample()
            pred_theta = prob_theta.sample()
        else:
            for _ in range(50):
                pred_wl = self._max_prob_sample(prob_wl, n_sample)
                pred_theta = self._max_prob_sample(prob_theta, n_sample * 2)
                # reject overlapping bounding box
                location = pred_location_smoothed.squeeze().numpy()
                w, l = pred_wl.squeeze().numpy()
                theta = pred_theta.item()
                corners = np.array([[l / 2, w / 2],
                                    [-l / 2, w / 2],
                                    [-l / 2, -w / 2],
                                    [l / 2, -w / 2]])
                rotation = np.array([[np.cos(theta), np.sin(theta)],
                                     [-np.sin(theta), np.cos(theta)]])
                corners = np.dot(corners, rotation) + location
                corners[:, 0] = corners[:, 0] + 40
                corners[:, 1] = 40 - corners[:, 1]
                corners = np.floor(corners / 0.25).astype(int)
                new_occupancy = np.zeros((320, 320), dtype=np.uint8)
                cv2.fillConvexPoly(new_occupancy, corners, 1)
                if (new_occupancy.astype(bool) & prev_occupancy.astype(bool)).sum() > 0:
                    continue
                break

        pred_bbox = torch.cat([pred_wl, pred_theta], dim=-1)
        bbox_f = self.pe_bbox(pred_bbox)

        f_in = torch.cat([
            location_f,
            bbox_f
        ], dim=-1)  # (B, 128 + 192)
        f_out = decoder(f_in, 'moving')
        prob_moving = Bernoulli(logits=f_out)
        f_out = decoder(f_in, 's')
        prob_s = self._mix_lognormal(f=f_out, event_shape=1)
        f_out = decoder(f_in, 'omega')
        prob_omega = self._mix_vonmises_delta(f=f_out, sin_theta=prob_theta.params['sin'],
                                              cos_theta=prob_theta.params['cos'])
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
            "location": pred_location_smoothed,
            "wl": pred_wl,
            "theta": pred_theta,
            "moving": pred_moving,
            "s": pred_s,
            "omega": pred_omega
        }
        return probs, preds

    def _rasterize(self, object_layers, category, location, bbox, velocity):
        B = object_layers.size(0)
        device = object_layers.device
        location = location.to('cpu').numpy()
        bbox = bbox.to('cpu').numpy()
        velocity = velocity.to('cpu').numpy()
        object_layers = object_layers.to('cpu').numpy()
        for i in range(B):
            if category[i] == 0:
                continue
            working_layers = object_layers[i, (category[i] - 1) * 6:category[i] * 6]
            w, l, theta = bbox[i]
            speed, heading = velocity[i]
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
            occupancy = np.zeros((320, 320), dtype=np.uint8)
            cv2.fillConvexPoly(occupancy, corners, 255)
            working_layers[0] = np.where(occupancy > 0, 1., working_layers[0])

            row = int((40 - location[i, 1]) / 0.25)
            col = int((location[i, 0] + 40) / 0.25)
            working_layers[1, row, col] = np.sin(theta)
            working_layers[2, row, col] = np.cos(theta)
            working_layers[3, row, col] = speed
            working_layers[4, row, col] = np.sin(heading)
            working_layers[5, row, col] = np.cos(heading)

        return torch.tensor(object_layers, dtype=torch.float32, device=device)

    def _forward_step(self, samples, lengths, gt):
        B, L, *_ = samples["category"].shape
        # Unpack the samples
        category = samples["category"]  # (B, L)
        location = samples['location']  # (B, L)
        bbox = samples["bbox"]
        velocity = samples["velocity"]
        maps = samples["map"]

        # extract features from map
        map_f = self.feature_extractor(maps)  # (B, 128, 320, 320)
        map_f = map_f.flatten(2, 3).permute([0, 2, 1]).contiguous()  # (B, 320 * 320, 128)

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

        # mean pooling
        output_f = output_f.mean(dim=1)  # (B, d_model)

        # predict category
        prob_category = self.prob_category(torch.cat([output_f, map_f.mean(dim=1)], dim=-1))  # (B, 4)
        prob_category = Categorical(logits=prob_category)

        loss_select = []
        pred_select = []
        for decoder in [self.decoder_pedestrian, self.decoder_bicyclist, self.decoder_vehicle]:
            probs, preds = self._decode(decoder, output_f, map_f, gt=gt)
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
        L = lengths.max().item()
        for field in ['category', 'location', 'bbox', 'velocity']:
            samples[field] = samples[field][:, :L]
        samples['location'] = self._discrete_loc(samples['location'])
        gt['location'] = self._discrete_loc(gt['location'])

        gt_step = {}
        for field in ['category', 'location', 'bbox', 'velocity']:
            gt_step[field] = gt[field][:, 0]
        loss, pred = self._forward_step(samples, lengths, gt_step)  # (B, )
        return loss

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
            map_f = self.feature_extractor(maps)  # (B, 128, 320, 320)
            map_f = map_f.flatten(2, 3).permute([0, 2, 1]).contiguous()  # (B, 320 * 320, 128)

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
                return preds, probs, samples, lengths
            elif pred_category == 1:
                decoder = self.decoder_pedestrian
            elif pred_category == 2:
                decoder = self.decoder_bicyclist
            else:
                decoder = self.decoder_vehicle
            prev_occupancy = maps[0, 8].numpy()

            probs, preds = self._decode(decoder, output_f, map_f,
                                        n_sample=n_sample,
                                        prev_occupancy=prev_occupancy)
            probs['category'] = prob_category
            preds['category'] = torch.tensor([pred_category])

            new_samples = {field: [] for field in ['category', 'location', 'bbox', 'velocity']}
            preds['bbox'] = torch.cat([preds['wl'], preds['theta']], dim=-1)
            preds['velocity'] = torch.cat([preds['s'], preds['omega']], dim=-1) * preds['moving']
            object_layers = self._rasterize(samples['map'][:, 8:],
                                            preds['category'],
                                            preds['location'],
                                            preds['bbox'],
                                            preds['velocity'])
            new_samples['map'] = torch.cat([samples['map'][:, :8], object_layers], dim=1)
            for i in range(B):
                length = lengths[i].item()
                new_samples_one = {}
                for field in ['category', 'location', 'bbox', 'velocity']:
                    new_samples_one[field] = torch.cat([samples[field][i, :length], preds[field][i].unsqueeze(0)],
                                                       dim=0)
                idx = list(range(length + 1))
                idx.sort(key=lambda x: (-new_samples_one['location'][x, 1], new_samples_one['location'][x, 0]))
                for field in ['category', 'location', 'bbox', 'velocity']:
                    new_samples_one[field] = new_samples_one[field][idx]
                    new_samples[field].append(new_samples_one[field])
            for field in ['category', 'location', 'bbox', 'velocity']:
                new_samples[field] = pad_sequence(new_samples[field], batch_first=True)
            lengths = lengths + 1
            samples = new_samples

        self.train()
        return preds, probs, samples, lengths
