import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from .conditional_transformer import TransformerBackbone
from .feature_extractors import Extractor
from .utils import get_length_mask
from .losses import DiffusionLoss
import math


def sigmas_schedule(timesteps, start=1e-2, end=10):
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
                'noise': 1
            },
            weights_category={
                'pedestrian': 1,
                'bicyclist': 1,
                'vehicle': 1
            }
        )

    def perturb(self, x, sigmas):
        noise = torch.randn_like(x) * sigmas[:, None, None]
        perturbed = x + noise
        return perturbed, noise

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
        result = self.backbone(pos, fmap, t / self.time_steps, sigmas, mask)
        for field in ['pedestrian', 'bicyclist', 'vehicle']:
            pred[field]['score'] = result[field]

        loss_dict = self.loss_fn(pred, target, sigmas)
        loss_dict['t'] = t.item()
        return loss_dict

    def sample_score_model(self, pred, fmap, step_lr=2e-7, n_steps_each=50):
        fields = ['pedestrian', 'bicyclist', 'vehicle']
        device = fmap['pedestrian'].device
        mask = torch.cat([
            get_length_mask(torch.tensor([pred[field]['length']], device=device)) for field in fields
        ], dim=1)
        B = mask.shape[0]
        L = sum([pred[field]['length'] for field in fields])
        x = torch.rand((B, L, 2), device=device) * 2 - 1
        for t in reversed(range(self.time_steps)):
            t = torch.tensor([t], device=device)
            sigma = self.sigmas[t].to(device)
            step_size = step_lr * (sigma / self.sigmas[0]) ** 2
            pos = {
                'pedestrian': x[:, :pred['pedestrian']['length']],
                'bicyclist': x[:, pred['pedestrian']['length']:pred['pedestrian']['length'] + pred['bicyclist']['length']],
                'vehicle': x[:, pred['pedestrian']['length'] + pred['bicyclist']['length']:]
            }
            for s in range(n_steps_each):
                grad = self.backbone(pos, fmap, t, sigma, mask)
                grad = torch.cat(list(grad.values()), dim=1)
                noise = torch.randn_like(x)
                grad_norm = torch.norm(grad.view(B, -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(B, -1), dim=-1).mean()
                x = x + step_size * grad + noise * math.sqrt(step_size * 2)
                pos = {
                    'pedestrian': x[:, :pred['pedestrian']['length']],
                    'bicyclist': x[:, pred['pedestrian']['length']:pred['pedestrian']['length'] + pred['bicyclist']['length']],
                    'vehicle': x[:, pred['pedestrian']['length'] + pred['bicyclist']['length']:]
                }

                x_norm = torch.norm(x.view(B, -1), dim=-1).mean()
                snr = math.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2
                print(x[0, 0])
                print("step: {}, step_size: {}, grad_norm: {}, x_norm: {}, snr: {}, grad_mean_norm: {}".format(
                    t, step_size, grad_norm.item(), x_norm.item(), snr.item(), grad_mean_norm.item()))
            for field in fields:
                pred[field]['location'].append(pos[field])
        return pred

    @torch.no_grad()
    def generate(self, maps):
        B = maps.size(0)
        assert B == 1
        fmap = {}  # (B, 128, 320, 320)
        for field in ['pedestrian', 'bicyclist', 'vehicle']:
            fmap[field] = self.feature_extractor[field](maps)
        pred = {
            'pedestrian': {},
            'bicyclist': {},
            'vehicle': {}
        }
        # predict number of objects
        pred['pedestrian']['length'] = self.n_pedestrian(fmap['pedestrian'].mean(dim=(2, 3))).sample().item()
        pred['bicyclist']['length'] = self.n_bicyclist(fmap['bicyclist'].mean(dim=(2, 3))).sample().item()
        pred['vehicle']['length'] = self.n_vehicle(fmap['vehicle'].mean(dim=(2, 3))).sample().item()
        print(pred['pedestrian']['length'], pred['bicyclist']['length'], pred['vehicle']['length'])

        pred['pedestrian']['location'] = []
        pred['bicyclist']['location'] = []
        pred['vehicle']['location'] = []
        pred = self.sample_score_model(pred, fmap)
        return pred
