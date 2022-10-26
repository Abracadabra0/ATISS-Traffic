from networks.diffusion_models import DiffusionBasedModel
from matplotlib import pyplot as plt
from datasets import NuScenesDataset, collate_fn, DiffusionModelPreprocessor
from torch.utils.data import DataLoader
import numpy as np


dataset = NuScenesDataset("/media/yifanlin/My Passport/data/nuSceneProcessed/train")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
batch = next(iter(dataloader))

preprocessor = DiffusionModelPreprocessor('cpu').test()
batch = preprocessor(batch)
maps = batch['map']
model = DiffusionBasedModel(time_steps=1000)
fields = ['pedestrian', 'bicyclist', 'vehicle']
areas = {
    'pedestrian': (maps[:, 1] + maps[:, 2]).clamp(max=1),
    'bicyclist': (maps[:, 1] + maps[:, 2]).clamp(max=1),
    'vehicle': maps[:, 0]
}
for step in range(20):
    perturbed = {}
    for field in fields:
        perturbed[field], _ = model.perturb(batch[field]['location'], 999, areas[field])
    axes_limit = 40
    name2color = {'pedestrian': 'red', 'bicyclist': 'blue', 'vehicle': 'green'}
    fig, ax = plt.subplots(figsize=(10, 10))
    drivable_area = maps[0, 0]
    ped_crossing = maps[0, 1]
    walkway = maps[0, 2]
    lane_divider = maps[0, 4]
    map_layers = np.stack([
        drivable_area + lane_divider,
        ped_crossing,
        walkway
    ], axis=-1) * 0.2
    ax.imshow(map_layers, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit])
    for field in fields:
        color = name2color[field]
        for i in range(perturbed[field].shape[1]):
            loc = perturbed[field][0, i] * axes_limit
            ax.plot(loc[0], loc[1], 'x', color=color)
            ax.annotate(str(i), loc)
    ax.set_xlim(-1.5 * axes_limit, 1.5 * axes_limit)
    ax.set_ylim(-1.5 * axes_limit, 1.5 * axes_limit)
    fig.savefig("./view/test_%02d.png" % step)
    plt.close(fig)
