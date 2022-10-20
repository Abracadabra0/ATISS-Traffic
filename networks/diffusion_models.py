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
            nn.Linear(dim_feature, dim_hidden // 2),
            nn.LayerNorm(dim_hidden // 2),
            nn.ReLU(),
            nn.Linear(dim_hidden // 2, dim_hidden // 4),
            nn.LayerNorm(dim_hidden // 4),
            nn.ReLU(),
            nn.Linear(dim_hidden // 4, max + 1)
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
    def diffuse_factor_schedule(timesteps, start=1e-2, end=20):
        factors = torch.linspace(math.log(start), math.log(end), timesteps)
        return torch.exp(factors)

    @staticmethod
    def blur(img, factor):
        size = img.shape[-1]
        new_size = int(size / factor)
        blurred = nn.AdaptiveAvgPool2d(new_size)(img.unsqueeze(1)).squeeze(1)
        blurred = functional.gaussian_blur(blurred, [3, 3], [0.5, 0.5])
        blurred = functional.resize(blurred, [size, size])
        blurred = functional.gaussian_blur(blurred, [3, 3], [0.5, 0.5])
        return blurred

    @staticmethod
    def diffuse(pts, size, factor):
        y, x = torch.meshgrid(torch.linspace(1, -1, size), torch.linspace(-1, 1, size))
        x = x[None, None].to(pts.device)
        y = y[None, None].to(pts.device)
        pts_x = pts[..., 0][..., None, None]
        pts_y = pts[..., 1][..., None, None]
        prob_x = 1 / factor * torch.exp(-(x - pts_x)**2 / (2 * factor**2))
        prob_y = 1 / factor * torch.exp(-(y - pts_y)**2 / (2 * factor**2))
        prob = prob_x * prob_y
        prob = prob / prob.sum(dim=(2, 3), keepdim=True)  # (B, L, size, size)
        return prob

    def __init__(self, time_steps, axes_limit=40):
        super().__init__()
        self.time_steps = time_steps
        blur_factors = self.blur_factor_schedule(time_steps)
        self.register_buffer('blur_factors', blur_factors)
        diffuse_factors = self.diffuse_factor_schedule(time_steps)
        self.register_buffer('diffuse_factors', diffuse_factors)
        self.feature_extractor = Extractor(8)

        self.n_pedestrian = ObjectNumberPredictor(256)
        self.n_bicyclist = ObjectNumberPredictor(256)
        self.n_vehicle = ObjectNumberPredictor(256)
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
        axes_limit = size // 2
        blur_factor = self.blur_factors[t].item()
        blurred = self.blur(area, blur_factor) + 1e-3  # (B, 320, 320)
        diffuse_factor = self.diffuse_factors[t].item()
        diffused = self.diffuse(pts, size, diffuse_factor)  # (B, L, 320, 320)
        prob = blurred.unsqueeze(1) * diffused
        prob = prob / prob.sum(dim=(2, 3), keepdim=True)  # (B, L, 320, 320)
        # sample from prob
        prob = prob.flatten(0, 1).flatten(1, 2)  # (B * L, 320 * 320)
        prob = Categorical(probs=prob)
        sample = prob.sample()  # (B * L, )
        row = torch.div(sample, size, rounding_mode='trunc')
        col = sample - row * size
        x = col / axes_limit - 1
        y = 1 - row / axes_limit
        perturbed = torch.stack([x, y], dim=-1).reshape(B, L, 2)
        gt = pts
        return perturbed, pts

    def forward(self, batch):
        maps = batch['map']
        pedestrians = batch['pedestrian']
        bicyclists = batch['bicyclist']
        vehicles = batch['vehicle']
        B = maps.size(0)
        device = maps.device
        fmap, avg = self.feature_extractor(maps)  # (B, 512, 320, 320)
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
        pred['pedestrian']['length'] = self.n_pedestrian(avg)
        pred['bicyclist']['length'] = self.n_bicyclist(avg)
        pred['vehicle']['length'] = self.n_vehicle(avg)
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
            perturbed, pts = self.perturb(inputs[field]['location'], t, area)
            target[field]['gt'] = pts
            pos[field] = perturbed
        mask = torch.cat([
            get_length_mask(pedestrians['length']),
            get_length_mask(bicyclists['length']),
            get_length_mask(vehicles['length'])
        ], dim=1)
        t_normed = torch.ones(B, dtype=torch.float, device=device) * t / self.time_steps
        result = self.backbone(pos, fmap, t_normed, mask)
        for field in ['pedestrian', 'bicyclist', 'vehicle']:
            pred[field]['score'] = result[field]

        loss_dict = self.loss_fn(pred, target, self.diffuse_factors[t])
        loss_dict['t'] = t
        return loss_dict

    def init_pts(self, lengths, areas):
        B = areas['pedestrian'].shape[0]
        L = sum(lengths.values())
        device = areas['pedestrian'].device
        return (torch.rand(B, L, 2) * 2 - 1).to(device)

    def sample_score_model(self, pred, maps, fmap):
        from tqdm import tqdm
        import warnings
        warnings.filterwarnings("ignore")
        fields = ['pedestrian', 'bicyclist', 'vehicle']
        device = fmap.device
        mask = torch.cat([
            get_length_mask(torch.tensor([pred[field]['length']], device=device)) for field in fields
        ], dim=1)
        B = mask.shape[0]
        areas = {'pedestrian': (maps[:, 1] + maps[:, 2]).clamp(max=1),
                 'bicyclist': (maps[:, 1] + maps[:, 2]).clamp(max=1),
                 'vehicle': maps[:, 0]}
        lengths = {field: pred[field]['length'] for field in fields}
        x = self.init_pts(lengths, areas)
        pos = {
            'pedestrian': x[:, :pred['pedestrian']['length']],
            'bicyclist': x[:, pred['pedestrian']['length']:pred['pedestrian']['length'] + pred['bicyclist']['length']],
            'vehicle': x[:, pred['pedestrian']['length'] + pred['bicyclist']['length']:]
        }
        for t in tqdm(reversed(range(self.time_steps))):
            sigma = self.diffuse_factors[t]
            t_normed = torch.ones(B, dtype=torch.float, device=device) * t / self.time_steps
            grad = self.backbone(pos, fmap, t_normed, sigma, mask)
            grad = torch.cat(list(grad.values()), dim=1)
            x = x + sigma**2 * grad
            pos = {
                'pedestrian': x[:, :pred['pedestrian']['length']],
                'bicyclist': x[:, pred['pedestrian']['length']:pred['pedestrian']['length'] + pred['bicyclist']['length']],
                'vehicle': x[:, pred['pedestrian']['length'] + pred['bicyclist']['length']:]
            }
            if t > 0:
                for field in fields:
                    pos[field], _ = self.perturb(pos[field], t - 1, areas[field])
                x = torch.cat([
                    pos['pedestrian'],
                    pos['bicyclist'],
                    pos['vehicle']
                ], dim=1)
            for field in fields:
                pred[field]['location'].append(pos[field])
        return pred

    @torch.no_grad()
    def generate(self, maps, lengths=None):
        B = maps.size(0)
        assert B == 1
        fmap, avg = self.feature_extractor(maps)  # (B, 512, 320, 320)
        pred = {
            'pedestrian': {},
            'bicyclist': {},
            'vehicle': {}
        }
        # predict number of objects
        if lengths is None:
            pred['pedestrian']['length'] = self.n_pedestrian(avg).sample().item()
            pred['bicyclist']['length'] = self.n_bicyclist(avg).sample().item()
            pred['vehicle']['length'] = self.n_vehicle(avg).sample().item()
        else:
            pred['pedestrian']['length'] = lengths[0]
            pred['bicyclist']['length'] = lengths[1]
            pred['vehicle']['length'] = lengths[2]
        print(pred['pedestrian']['length'], pred['bicyclist']['length'], pred['vehicle']['length'])

        pred['pedestrian']['location'] = []
        pred['bicyclist']['location'] = []
        pred['vehicle']['location'] = []
        pred = self.sample_score_model(pred, maps, fmap)
        return pred
