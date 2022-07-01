import numpy as np
from matplotlib import pyplot as plt
import torch
from scene_synthesis.datasets.nuScenes import NuScenesDataset
from scene_synthesis.datasets.utils import collate_train
from scene_synthesis.networks.feature_extractors import ResNet18
from scene_synthesis.networks.autoregressive_transformer import AutoregressiveTransformer


plt.ion()
np.random.seed(0)
torch.manual_seed(0)
dataset = NuScenesDataset("/media/yifanlin/My Passport/data/nuScene-processed", train=True)
axes_limit = 40
cat2color = {1: 'red', 2: 'blue', 3: 'green'}
feature_extractor = ResNet18(4, 512)
model = AutoregressiveTransformer(feature_extractor)
model.load_state_dict(torch.load('./ckpts/07-01-04:15:14'))

data = dataset[11]
input_data, length, _ = collate_train([data])
for k in data:
    data[k] = input_data[k].squeeze(0).numpy()

map_layers = data['map'].sum(axis=0)
plt.imshow(map_layers, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit])
for i in range(length.item()):
    if data['category'][i] != 0:
        color = cat2color[data['category'][i]]
        loc = data['location'][i]
        plt.plot(loc[0], loc[1], 'x', color=color)
        w, l, theta = data['bbox'][i]
        corners = np.array([[l / 2, w / 2],
                            [-l / 2, w / 2],
                            [-l / 2, -w / 2],
                            [l / 2, -w / 2],
                            [l / 2, w / 2]])
        rotation = np.array([[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]])
        corners = np.dot(corners, rotation) + loc
        plt.plot(corners[:, 0], corners[:, 1], color=color, linewidth=2)
        speed, omega = data['velocity'][i]
        rotation = np.array([[np.cos(omega), np.sin(omega)],
                             [-np.sin(omega), np.cos(omega)]])
        velocity = np.dot(np.array([speed, 0]), rotation)
        plt.arrow(loc[0], loc[1], velocity[0] * 5, velocity[1] * 5, color=color, width=0.05)

preds, probs = model.generate(input_data, length)
for k in preds:
    if isinstance(preds[k], torch.Tensor):
        if len(preds[k]) == 1:
            preds[k] = preds[k].item()
        else:
            preds[k] = preds[k].numpy()
    elif isinstance(preds[k], dict):
        for j in preds[k]:
            if isinstance(preds[k][j], torch.Tensor):
                if len(preds[k][j]) == 1:
                    preds[k][j] = preds[k][j].item()
                else:
                    preds[k][j] = preds[k][j].numpy()
category = preds['category']
if category != 0:
    color = cat2color[category]
    loc = preds['location']
    plt.plot(loc[0], loc[1], 'x', color=color)
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
    plt.plot(corners[:, 0], corners[:, 1], color=color, linewidth=2)
    speed = preds['velocity']['s'] * preds['velocity']['moving']
    omega = preds['velocity']['omega']
    rotation = np.array([[np.cos(omega), np.sin(omega)],
                         [-np.sin(omega), np.cos(omega)]])
    velocity = np.dot(np.array([speed, 0]), rotation)
    plt.arrow(loc[0], loc[1], velocity[0] * 5, velocity[1] * 5, color=color, width=0.05)
data['category'] = np.concatenate([data['category'], np.array([preds['category']])], axis=0)
data['location'] = np.concatenate([data['location'], preds['location'].reshape(1, -1)], axis=0)
data['bbox'] = np.concatenate([data['bbox'], np.concatenate([preds['bbox']['wl'], np.array([preds['bbox']['theta']])], axis=0).reshape(1, -1)], axis=0)
data['velocity'] = np.concatenate([data['velocity'], np.array([[preds['velocity']['s'] * preds['velocity']['moving'], preds['velocity']['omega']]])], axis=0)
input_data = {}
input_data['category'] = torch.tensor(data['category'])
for k in ['location', 'bbox', 'velocity', 'map']:
    input_data[k] = torch.tensor(data[k], dtype=torch.float)
input_data, length, _ = collate_train([input_data])

grid = np.meshgrid(np.linspace(-axes_limit, axes_limit, 400), np.linspace(-axes_limit, axes_limit, 400))
