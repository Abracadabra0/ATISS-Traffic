import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from .conditional_transformer import TransformerBackbone
from .feature_extractors import Extractor
from .utils import get_length_mask
from .losses import DiffusionLoss


def linear_beta_schedule(timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, timesteps, dtype=torch.float32)


class ObjectNumberPredictor(nn.Module):
    def __init__(self, dim_feature, dim_hidden=128, max=99):
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
    def __init__(self, time_steps, d_model=768, n_layers=12, axes_limit=40):
        super().__init__()
        self.time_steps = time_steps
        beta = linear_beta_schedule(time_steps)
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar_rshift = F.pad(alpha_bar[:-1], (1, 0), value=1.)
        posterior_variance = beta * (1. - alpha_bar_rshift) / (1. - alpha_bar)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('prior_std', torch.sqrt(1 - alpha_bar))
        self.register_buffer('posterior_std', torch.sqrt(posterior_variance))
        self.feature_extractor = Extractor(8)

        self.n_pedestrian = ObjectNumberPredictor(128)
        self.n_bicyclist = ObjectNumberPredictor(128)
        self.n_vehicle = ObjectNumberPredictor(128)
        self.pedestrian_backbone = TransformerBackbone(d_model, n_layers)
        self.bicyclist_backbone = TransformerBackbone(d_model, n_layers)
        self.vehicle_backbone = TransformerBackbone(d_model, n_layers)

        self.axes_limit = axes_limit
        self.loss_fn = DiffusionLoss(
            weights_entry={
                'length': 0.4,
                'noise': 1
            },
            weights_category={
                'pedestrian': 1,
                'bicyclist': 1,
                'vehicle': 1
            }
        )

    def forward(self, pedestrians, bicyclists, vehicles, maps):
        B = maps.size(0)
        device = maps.device
        for category in [pedestrians, bicyclists, vehicles]:
            L = category['length'].max().item()
            for field in ['location', 'bbox', 'velocity']:
                category[field] = category[field][:, :L]
        fmap = self.feature_extractor(maps)  # (B, 128, 320, 320)
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
        fmap_avg = fmap.mean(dim=(2, 3))
        pred['pedestrian']['length'] = self.n_pedestrian(fmap_avg)
        pred['bicyclist']['length'] = self.n_bicyclist(fmap_avg)
        pred['vehicle']['length'] = self.n_vehicle(fmap_avg)
        target['pedestrian']['length'] = pedestrians['length']
        target['bicyclist']['length'] = bicyclists['length']
        target['vehicle']['length'] = vehicles['length']

        # predict location noise
        t = torch.randint(0, self.time_steps, (B, ), device=device)
        # perturb data
        scale = torch.sqrt(self.alpha_bar[t]).to(device)  # (B, )
        std = self.prior_std[t].to(device)  # (B, )
        # pedestrian
        noise = torch.randn_like(pedestrians['location'])
        target['pedestrian']['noise'] = noise
        perturbed = pedestrians['location'] * scale[:, None, None] + noise * std[:, None, None]
        mask = get_length_mask(pedestrians['length'])
        pred['pedestrian']['noise'] = self.pedestrian_backbone(perturbed, fmap, t, mask=mask)
        # bicyclist
        noise = torch.randn_like(bicyclists['location'])
        target['bicyclist']['noise'] = noise
        perturbed = bicyclists['location'] * scale[:, None, None] + noise * std[:, None, None]
        mask = get_length_mask(bicyclists['length'])
        pred['bicyclist']['noise'] = self.bicyclist_backbone(perturbed, fmap, t, mask=mask)
        # vehicle
        noise = torch.randn_like(vehicles['location'])
        target['vehicle']['noise'] = noise
        perturbed = vehicles['location'] * scale[:, None, None] + noise * std[:, None, None]
        mask = get_length_mask(vehicles['length'])
        pred['vehicle']['noise'] = self.vehicle_backbone(perturbed, fmap, t, mask=mask)

        loss_dict = self.loss_fn(pred, target)
        return loss_dict

    def sample_step(self, x, t, noise):
        x = x - (1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_bar[t]) * noise
        x = 1 / torch.sqrt(self.alpha[t]) * x
        if t != 0:
            x = x + self.posterior_std[t] * torch.randn_like(x)
        x = x.clamp(min=-0.999, max=0.999)
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
        location = torch.rand((pred['pedestrian']['length'], 2), device=device) * 2 - 1
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
        location = torch.rand((pred['bicyclist']['length'], 2), device=device) * 2 - 1
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
        location = torch.randn((pred['vehicle']['length'], 2), device=device) * 2 - 1
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
