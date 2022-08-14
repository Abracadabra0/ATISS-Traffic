import numpy as np
from matplotlib import pyplot as plt
import torch
from scene_synthesis.datasets.nuScenes import NuScenesDataset
from scene_synthesis.datasets.utils import collate_test
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

np.random.seed(0)
torch.manual_seed(0)
plt.ion()
dataset = NuScenesDataset("../../data/nuScene-processed/train")
axes_limit = 40
cat2color = {1: 'red', 2: 'blue', 3: 'green'}
model = AutoregressiveTransformer()
model.load_state_dict(torch.load('./ckpts/08-11-10:02:02'))

data = dataset[10]
input_data, length = collate_test([data], keep=10)
for k in data:
    data[k] = input_data[k].squeeze(0).numpy()

condition = {
    "category": None,  # int
}

cnt = 0
while True:
    preds, probs, input_data, length = model.generate(input_data, length, condition, n_sample=5)
    cnt += 1
    print(cnt)
    to_numpy(preds)
    category = preds['category']
    if category == 0 or cnt > 100:
        break

fig, ax = plt.subplots(figsize=(10, 10))
map_layers = np.stack(input_data['map'][0, :3], axis=-1) * 0.2
ax.imshow(map_layers, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit])
for i in range(length.item()):
    if input_data['category'][0, i] != 0:
        color = cat2color[input_data['category'][0, i].item()]
        loc = input_data['location'][0, i].numpy()
        ax.plot(loc[0], loc[1], 'x', color=color)
        w, l, theta = input_data['bbox'][0, i].numpy()
        corners = np.array([[l / 2, w / 2],
                            [-l / 2, w / 2],
                            [-l / 2, -w / 2],
                            [l / 2, -w / 2],
                            [l / 2, w / 2]])
        rotation = np.array([[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]])
        corners = np.dot(corners, rotation) + loc
        ax.plot(corners[:, 0], corners[:, 1], color=color, linewidth=2)
        speed, omega = input_data['velocity'][0, i].numpy()
        rotation = np.array([[np.cos(omega), np.sin(omega)],
                             [-np.sin(omega), np.cos(omega)]])
        velocity = np.dot(np.array([speed, 0]), rotation)
        ax.arrow(loc[0], loc[1], velocity[0] * 5, velocity[1] * 5, color=color, width=0.05)
ax.set_xlim([-axes_limit, axes_limit])
ax.set_ylim([-axes_limit, axes_limit])

fig.savefig("./train-10-10.png")