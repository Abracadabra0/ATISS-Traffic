import numpy as np
from matplotlib import pyplot as plt
import torch
from scene_synthesis.datasets.nuScenes import NuScenesDataset
from scene_synthesis.datasets.utils import collate_train


plt.ion()
np.random.seed(0)
torch.manual_seed(0)
dataset = NuScenesDataset("/media/yifanlin/My Passport/data/nuScene-processed", train=True)
axes_limit = 40
cat2color = {1: 'red', 2: 'blue', 3: 'green'}
model = torch.load('./ckpts/06-16-07:30:15')

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

generated, probs = model.generate(input_data, length)
for k in generated:
    generated[k] = generated[k].squeeze(0).numpy()
category = generated['category'].item()
if category != 0:
    color = cat2color[category]
    loc = generated['location']
    plt.plot(loc[0], loc[1], 'x', color=color)
    w, l, theta = generated['bbox']
    corners = np.array([[l / 2, w / 2],
                        [-l / 2, w / 2],
                        [-l / 2, -w / 2],
                        [l / 2, -w / 2],
                        [l / 2, w / 2]])
    rotation = np.array([[np.cos(theta), np.sin(theta)],
                         [-np.sin(theta), np.cos(theta)]])
    corners = np.dot(corners, rotation) + loc
    plt.plot(corners[:, 0], corners[:, 1], color=color, linewidth=2)
    speed, omega = generated['velocity']
    rotation = np.array([[np.cos(omega), np.sin(omega)],
                         [-np.sin(omega), np.cos(omega)]])
    velocity = np.dot(np.array([speed, 0]), rotation)
    plt.arrow(loc[0], loc[1], velocity[0] * 5, velocity[1] * 5, color=color, width=0.05)
data['category'] = np.concatenate([data['category'], generated['category'].reshape(1)], axis=0)
data['location'] = np.concatenate([data['location'], generated['location'].reshape(1, -1)], axis=0)
data['bbox'] = np.concatenate([data['bbox'], generated['bbox'].reshape(1, -1)], axis=0)
data['velocity'] = np.concatenate([data['velocity'], generated['velocity'].reshape(1, -1)], axis=0)
input_data = {}
for k in data:
    input_data[k] = torch.tensor(data[k])
input_data, length, _ = collate_train([input_data])

grid = np.meshgrid(np.linspace(-axes_limit, axes_limit, 400), np.linspace(-axes_limit, axes_limit, 400))
