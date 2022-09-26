import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
import torchvision.transforms.functional as functional
from .conditional_transformer import TransformerBackbone
from .feature_extractors import Extractor
from .utils import get_length_mask
from .losses import DiffusionLoss
import math


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
    @staticmethod
    def blur_factor_schedule(timesteps, start=1, end=64):
        factors = torch.linspace(math.log(start), math.log(end), timesteps)
        return torch.exp(factors)

    @staticmethod
    def diffuse_factor_schedule(timesteps, start=1e-2, end=10):
        factors = torch.linspace(math.log(start), math.log(end), timesteps)
        return torch.exp(factors)

    @staticmethod
    def blur(img, factor):
        size = img.shape[-1]
        new_size = int(size / factor)
        blurred = functional.resize(img, [new_size, new_size])
        blurred = functional.gaussian_blur(blurred, [3, 3], [0.5, 0.5])
        blurred = functional.resize(blurred, [size, size])
        blurred = functional.gaussian_blur(blurred, [3, 3], [0.5, 0.5])
        return blurred

    @staticmethod
    def diffuse(pts, size, factor):
        y, x = torch.meshgrid(torch.linspace(1, -1, size), torch.linspace(-1, 1, size))
        x = x[None, None]
        y = y[None, None]
        pts_x = pts[..., 0][..., None, None]
        pts_y = pts[..., 1][..., None, None]
        prob_x = 1 / factor * torch.exp(-(x - pts_x)**2 / (2 * factor**2))
        prob_y = 1 / factor * torch.exp(-(y - pts_y)**2 / (2 * factor**2))
        prob = prob_x * prob_y
        prob = prob / prob.sum(dim=(2, 3))  # (B, L, size, size)
        return prob

    def __init__(self, time_steps, axes_limit=40):
        super().__init__()
        self.time_steps = time_steps
        blur_factors = self.blur_factor_schedule(time_steps)
        self.register_buffer('blur_factors', blur_factors)
        diffuse_factors = self.diffuse_factor_schedule(time_steps)
        self.register_buffer('diffuse_factors', diffuse_factors)
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

    def perturb(self, pts, t, area):
        # pts: (B, L, 2)
        # area: (B, 320, 320)
        B = pts.shape[0]
        L = pts.shape[1]
        size = area.shape[1]
        blur_factor = self.blur_factors[t].item()
        blurred = self.blur(area, blur_factor)  # (B, 320, 320)
        diffuse_factor = self.diffuse_factor[t].item()
        diffused = self.diffuse(pts, size, diffuse_factor)  # (B, L, 320, 320)
        prob = blurred.unsqueeze(1) * diffused
        prob = prob / prob.sum(dim=(2, 3))  # (B, L, 320, 320)
        # sample from prob
        prob = prob.flatten(0, 1).flatten(1, 2)  # (B * L, 320 * 320)
        prob = Categorical(probs=prob)
        sample = prob.sample()  # (B * L, )
        row = torch.div(sample, size, rounding_mode='trunc')
        col = sample - row * size
        x = col * 0.25 - 40
        y = 40 - row * 0.25
        perturbed = torch.stack([x, y], dim=-1).reshape(B, L, -1)
        noise = perturbed - pts
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
        t = torch.randint(0, self.time_steps, (1, )).item()
        # perturb data
        inputs = {'pedestrian': pedestrians,
                  'bicyclist': bicyclists,
                  'vehicle': vehicles}
        pos = {}
        drivable_area = maps[:, 0]
        walkable_area = (maps[:, 1] + maps[:, 2]).clamp(max=1)
        for field in ['pedestrian', 'bicyclist', 'vehicle']:
            if field == 'vehicle':
                area = drivable_area
            else:
                area = walkable_area
            perturbed, noise = self.perturb(inputs[field]['location'], t, area)
            target[field]['noise'] = noise
            pos[field] = perturbed
        mask = torch.cat([
            get_length_mask(pedestrians['length']),
            get_length_mask(bicyclists['length']),
            get_length_mask(vehicles['length'])
        ], dim=1)
        t_normed = torch.ones(B, dtype=torch.float, device=device) * t / self.time_steps
        result = self.backbone(pos, fmap, t_normed, self.diffuse_factor[t], mask)
        for field in ['pedestrian', 'bicyclist', 'vehicle']:
            pred[field]['score'] = result[field]

        loss_dict = self.loss_fn(pred, target, self.diffuse_factor[t])
        loss_dict['t'] = t
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
