import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from .conditional_transformer import TransformerBackbone
from .feature_extractors import Extractor
from .utils import get_length_mask
from .losses import DiffusionLoss
import math


def sigmas_schedule(timesteps, start=1e-3, end=20):
    sigmas = torch.linspace(math.log(start), math.log(end), timesteps)
    return torch.exp(sigmas)


class ObjectNumberPredictor(nn.Module):
    def __init__(self, dim_feature, dim_hidden=128, max=63):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_feature, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, max + 1)
        )

    def forward(self, fmap):
        logits = self.model(fmap)
        return Categorical(logits=logits)


class DiffusionBasedModel(nn.Module):
    def __init__(self, time_steps, axes_limit=40):
        super().__init__()
        self.time_steps = time_steps
        sigmas = sigmas_schedule(time_steps)
        self.register_buffer('sigmas', sigmas)
        self.feature_extractor = nn.ModuleDict({
            'pedestrian': Extractor(8),
            'bicyclist': Extractor(8),
            'vehicle': Extractor(8)
        })

        self.n_pedestrian = ObjectNumberPredictor(128)
        self.n_bicyclist = ObjectNumberPredictor(128)
        self.n_vehicle = ObjectNumberPredictor(128)
        self.backbone = TransformerBackbone()

        self.axes_limit = axes_limit
        self.loss_fn = DiffusionLoss(
            weights_entry={
                'length': 1,
                'location': 1
            },
            weights_category={
                'pedestrian': 1,
                'bicyclist': 1,
                'vehicle': 1
            }
        )

    def perturb(self, x, sigmas):
        noise = torch.randn_like(x)
        perturbed = x + noise * sigmas[:, None, None]
        return perturbed

    def forward(self, pedestrians, bicyclists, vehicles, maps):
        B = maps.size(0)
        device = maps.device
        for category in [pedestrians, bicyclists, vehicles]:
            L = category['length'].max().item()
            for field in ['location', 'bbox', 'velocity']:
                category[field] = category[field][:, :L]
        fmap = {}  # (B, 128, 320, 320)
        for field in ['pedestrian', 'bicyclist', 'vehicle']:
            fmap[field] = self.feature_extractor[field](maps)
        pred = {
            'pedestrian': {},
            'bicyclist': {},
            'vehicle': {}
        }
        target = {
            'pedestrian': {},
            'bicyclist': {},
            'vehicle': {}
        }
        # predict number of objects
        pred['pedestrian']['length'] = self.n_pedestrian(fmap['pedestrian'].mean(dim=(2, 3)))
        pred['bicyclist']['length'] = self.n_bicyclist(fmap['bicyclist'].mean(dim=(2, 3)))
        pred['vehicle']['length'] = self.n_vehicle(fmap['vehicle'].mean(dim=(2, 3)))
        target['pedestrian']['length'] = pedestrians['length']
        target['bicyclist']['length'] = bicyclists['length']
        target['vehicle']['length'] = vehicles['length']

        # predict location noise
        t = torch.randint(0, self.time_steps, (B, ), device=device)
        sigmas = self.sigmas[t].to(device)
        # perturb data
        inputs = {'pedestrian': pedestrians,
                  'bicyclist': bicyclists,
                  'vehicle': vehicles}
        pos = {}
        for field in ['pedestrian', 'bicyclist', 'vehicle']:
            perturbed, noise = self.perturb(inputs[field]['location'], sigmas)
            target[field]['noise'] = noise
            pos[field] = perturbed
        mask = torch.cat([
            get_length_mask(pedestrians['length']),
            get_length_mask(bicyclists['length']),
            get_length_mask(vehicles['length'])
        ], dim=1)
        result = self.backbone(pos, fmap, t, sigmas, mask)
        for field in ['pedestrian', 'bicyclist', 'vehicle']:
            pred[field]['score'] = result[field]

        loss_dict = self.loss_fn(pred, target)
        return loss_dict

    def sample_step(self, x, t, noise):
        x = x - (1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_bar[t]) * noise
        x = 1 / torch.sqrt(self.alpha[t]) * x
        if t != 0:
            x = x + self.posterior_std[t] * torch.randn_like(x)
        return x

    @torch.no_grad()
    def generate(self, maps):
        B = maps.size(0)
        assert B == 1
        device = maps.device
        fmap = self.feature_extractor(maps)  # (B, 128, 320, 320)
        pred = {
            'pedestrian': {},
            'bicyclist': {},
            'vehicle': {}
        }
        # predict number of objects
        fmap_avg = fmap.mean(dim=(2, 3))
        pred['pedestrian']['length'] = self.n_pedestrian(fmap_avg).sample().item()
        pred['bicyclist']['length'] = self.n_bicyclist(fmap_avg).sample().item()
        pred['vehicle']['length'] = self.n_vehicle(fmap_avg).sample().item()
        print(pred['pedestrian']['length'], pred['bicyclist']['length'], pred['vehicle']['length'])

        # pedestrian
        pred['pedestrian']['location'] = []
        location = torch.randn((pred['pedestrian']['length'], 2), device=device)
        idx = list(range(pred['pedestrian']['length']))
        idx.sort(key=lambda x: (-location[x, 1], location[x, 0]))
        location = location[idx].unsqueeze(0)
        mask = get_length_mask(torch.tensor([pred['pedestrian']['length']], device=device))
        for t in reversed(range(self.time_steps)):
            t = torch.tensor([t], device=device)
            noise = self.pedestrian_backbone(location, fmap, t, mask=mask)
            location = self.sample_step(location, t, noise)
            pred['pedestrian']['location'].append(location)
        # bicyclist
        pred['bicyclist']['location'] = []
        location = torch.randn((pred['bicyclist']['length'], 2), device=device)
        idx = list(range(pred['bicyclist']['length']))
        idx.sort(key=lambda x: (-location[x, 1], location[x, 0]))
        location = location[idx].unsqueeze(0)
        mask = get_length_mask(torch.tensor([pred['bicyclist']['length']], device=device))
        for t in reversed(range(self.time_steps)):
            t = torch.tensor([t], device=device)
            noise = self.bicyclist_backbone(location, fmap, t, mask=mask)
            location = self.sample_step(location, t, noise)
            pred['bicyclist']['location'].append(location)
        # vehicle
        pred['vehicle']['location'] = []
        location = torch.randn((pred['vehicle']['length'], 2), device=device)
        idx = list(range(pred['vehicle']['length']))
        idx.sort(key=lambda x: (-location[x, 1], location[x, 0]))
        location = location[idx].unsqueeze(0)
        mask = get_length_mask(torch.tensor([pred['vehicle']['length']], device=device))
        for t in reversed(range(self.time_steps)):
            t = torch.tensor([t], device=device)
            noise = self.vehicle_backbone(location, fmap, t, mask=mask)
            location = self.sample_step(location, t, noise)
            pred['vehicle']['location'].append(location)
        return pred
