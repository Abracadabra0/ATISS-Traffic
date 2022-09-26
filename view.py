import numpy as np
from matplotlib import pyplot as plt
import torch
from datasets import NuScenesDataset, collate_fn, AutoregressivePreprocessor
from torch.utils.data import DataLoader
from torchvision.transforms import GaussianBlur


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


dataset = NuScenesDataset("/media/yifanlin/My Passport/data/nuSceneProcessed/train")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
processor = AutoregressivePreprocessor('cpu').test()
axes_limit = 40
cat2color = {1: 'red', 2: 'blue', 3: 'green'}

for idx, batch in enumerate(dataloader):
    print(idx)
    batch, length, _ = processor(batch, n_keep=-1)

    fig, ax = plt.subplots(figsize=(10, 10))
    map_layers = batch['map'][0, :3].permute(1, 2, 0) * 0.2
    ax.imshow(map_layers, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit])
    for i in range(length.item()):
        if batch['category'][0, i] != 0:
            color = cat2color[batch['category'][0, i].item()]
            loc = batch['location'][0, i].numpy()
            ax.plot(loc[0], loc[1], 'x', color=color)
            w, l, theta = batch['bbox'][0, i].numpy()
            corners = np.array([[0, 0],
                                [l / 2, 0],
                                [l / 2, w / 2],
                                [-l / 2, w / 2],
                                [-l / 2, -w / 2],
                                [l / 2, -w / 2],
                                [l / 2, 0]])
            rotation = np.array([[np.cos(theta), np.sin(theta)],
                                 [-np.sin(theta), np.cos(theta)]])
            corners = np.dot(corners, rotation) + loc
            ax.plot(corners[:, 0], corners[:, 1], color=color, linewidth=2)
            speed, omega = batch['velocity'][0, i].numpy()
            rotation = np.array([[np.cos(omega), np.sin(omega)],
                                 [-np.sin(omega), np.cos(omega)]])
            velocity = np.dot(np.array([speed, 0]), rotation)
            ax.arrow(loc[0], loc[1], velocity[0] * 5, velocity[1] * 5, color=color, width=0.05)
    ax.set_xlim([-axes_limit, axes_limit])
    ax.set_ylim([-axes_limit, axes_limit])
    fig.savefig(f'./view/{idx}.png')
    plt.close(fig)
