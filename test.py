import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import NuScenesDataset, DiffusionModelPreprocessor, collate_fn
from networks import DiffusionBasedModel
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    device = torch.device(0)
    dataset = NuScenesDataset("/projects/perception/personals/yefanlin/data/nuSceneProcessed/train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)
    preprocessor = DiffusionModelPreprocessor(device).test()
    B = 1
    model = DiffusionBasedModel(time_steps=1000)
    model.load_state_dict(torch.load('./ckpts/09-07-10:25:52'))
    model.to(device)
    model.eval()

    axes_limit = 40
    name2color = {'pedestrian': 'red', 'bicyclist': 'blue', 'vehicle': 'green'}

    for batch in dataloader:
        pedestrians, bicyclists, vehicles, maps = preprocessor(batch)
        pred = model.generate(maps)

        maps = maps.cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 10))
        drivable_area = maps[0, 0]
        ped_crossing = maps[0, 1]
        walkway = maps[0, 2]
        lane_divider = maps[0, 5]
        orientation = maps[0, 6:8]
        map_layers = np.stack([
            drivable_area + lane_divider,
            ped_crossing,
            walkway
        ], axis=-1) * 0.2
        ax.imshow(map_layers, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit])
        for name in ['pedestrian', 'bicyclist', 'vehicle']:
            color = name2color[name]
            for i in range(pred[name]['length']):
                loc = pred[name]['location'][0, i].cpu().numpy() * axes_limit
                ax.plot(loc[0], loc[1], 'x', color=color)
        fig.savefig(f"./result/test.png")
        plt.close(fig)
