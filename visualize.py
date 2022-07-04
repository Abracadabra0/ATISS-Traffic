import numpy as np
from matplotlib import pyplot as plt
import torch
from scene_synthesis.datasets.nuScenes import NuScenesDataset
from scene_synthesis.datasets.utils import collate_train
from scene_synthesis.networks.feature_extractors import ResNet18
from scene_synthesis.networks.autoregressive_transformer import AutoregressiveTransformer


def to_numpy(data: dict):
    for k in data:
        if isinstance(data[k], torch.Tensor):
            data[k] = data[k].squeeze()
            if not data[k].shape:
                data[k] = data[k].item()
            else:
                data[k] = data[k].numpy()
        elif isinstance(data[k], dict):
            to_numpy(data[k])

plt.ion()
np.random.seed(0)
torch.manual_seed(0)
dataset = NuScenesDataset("/media/yifanlin/My Passport/data/nuScene-processed", train=True)
axes_limit = 40
_, ax = plt.subplots()
cat2color = {1: 'red', 2: 'blue', 3: 'green'}
feature_extractor = ResNet18(4, 512)
model = AutoregressiveTransformer(feature_extractor)
model.load_state_dict(torch.load('./ckpts/pe+mlp+0.5'))

data = dataset[11]
input_data, length, _ = collate_train([data])
for k in data:
    data[k] = input_data[k].squeeze(0).numpy()

map_layers = data['map'].sum(axis=0)
ax.imshow(map_layers, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit])
for i in range(length.item()):
    if data['category'][i] != 0:
        color = cat2color[data['category'][i]]
        loc = data['location'][i]
        ax.plot(loc[0], loc[1], 'x', color=color)
        w, l, theta = data['bbox'][i]
        corners = np.array([[l / 2, w / 2],
                            [-l / 2, w / 2],
                            [-l / 2, -w / 2],
                            [l / 2, -w / 2],
                            [l / 2, w / 2]])
        rotation = np.array([[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]])
        corners = np.dot(corners, rotation) + loc
        ax.plot(corners[:, 0], corners[:, 1], color=color, linewidth=2)
        speed, omega = data['velocity'][i]
        rotation = np.array([[np.cos(omega), np.sin(omega)],
                             [-np.sin(omega), np.cos(omega)]])
        velocity = np.dot(np.array([speed, 0]), rotation)
        ax.arrow(loc[0], loc[1], velocity[0] * 5, velocity[1] * 5, color=color, width=0.05)

condition = {
    "category": 2,  # int
    "location": None,  # (1, 2)
    "bbox": None,  # (1, 2), (1, 1)
    "velocity": None  # (1, 1), (1, 1), (1, 1)
}

preds, probs = model.generate(input_data, length, condition)
to_numpy(preds)
category = preds['category']
if category != 0:
    color = cat2color[category]
    loc = preds['location']
    ax.plot(loc[0], loc[1], 'x', color=color)
    w, l = preds['bbox']['wl']
    theta = preds['bbox']['theta']
    corners = np.array([[l / 2, w / 2],
                        [-l / 2, w / 2],
                        [-l / 2, -w / 2],
                        [l / 2, -w / 2],
                        [l / 2, w / 2]])
    rotation = np.array([[np.cos(theta), np.sin(theta)],
                         [-np.sin(theta), np.cos(theta)]])
    corners = np.dot(corners, rotation) + loc
    ax.plot(corners[:, 0], corners[:, 1], color=color, linewidth=2)
    speed = preds['velocity']['s'] * preds['velocity']['moving']
    omega = preds['velocity']['omega']
    rotation = np.array([[np.cos(omega), np.sin(omega)],
                         [-np.sin(omega), np.cos(omega)]])
    velocity = np.dot(np.array([speed, 0]), rotation)
    ax.arrow(loc[0], loc[1], velocity[0] * 5, velocity[1] * 5, color=color, width=0.05)
    data['category'] = np.concatenate([data['category'], np.array([preds['category']])], axis=0)
    data['location'] = np.concatenate([data['location'], preds['location'].reshape(1, -1)], axis=0)
    data['bbox'] = np.concatenate([data['bbox'], np.concatenate([preds['bbox']['wl'], np.array([preds['bbox']['theta']])], axis=0).reshape(1, -1)], axis=0)
    data['velocity'] = np.concatenate([data['velocity'], np.array([[preds['velocity']['s'] * preds['velocity']['moving'], preds['velocity']['omega']]])], axis=0)
    input_data = {}
    input_data['category'] = torch.tensor(data['category'])
    for k in ['location', 'bbox', 'velocity', 'map']:
        input_data[k] = torch.tensor(data[k], dtype=torch.float)
    input_data, length, _ = collate_train([input_data])

_, ax_prob = plt.subplots()
grid = np.meshgrid(np.linspace(-axes_limit, axes_limit, 400), np.linspace(axes_limit, -axes_limit, 400))
grid = np.stack(grid, axis=-1)
prob_map = np.exp(probs['location'].log_prob(torch.tensor(grid / 40)))
ax_prob.imshow(prob_map, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit])
mean = probs['location'].component_distribution.mean.squeeze().numpy() * 40
ax_prob.scatter(mean[:, 0], mean[:, 1])
