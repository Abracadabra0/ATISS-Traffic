import torch
from torch.utils.data import DataLoader
from datasets import NuScenesDataset, ScoreModelProcessor, collate_fn
from networks.ScoreBasedGenerator import Generator
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    device = torch.device(0)
    np.random.seed(0)
    torch.manual_seed(0)
    dataset = NuScenesDataset("../../data/nuSceneProcessed/train")
    B = 32
    dataloader = DataLoader(dataset, batch_size=B, shuffle=False, num_workers=4, collate_fn=collate_fn)
    processor = ScoreModelProcessor(device).test()
    model = Generator()
    model.load_state_dict(torch.load('./ckpts/08-25-09:42:07'))
    model.to(device)
    cat2color = {1: 'red', 2: 'blue', 3: 'green'}
    for batch in dataloader:
        _, map_layers = processor(batch)
        x = model.generate(map_layers)
        for i in range(B):
            generated = x[i]
            for channel in range(4):
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(generated[channel], extent=[-80, 80, -80, 80])
                fig.savefig(f"./result/test-{i}({channel}).png")
                plt.close(fig)
            generated = generated.max(dim=0)[1]
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(map_layers[i, :3].permute(1, 2, 0) * 0.2, extent=[-80, 80, -80, 80])
            for row in range(80):
                for col in range(80):
                    if generated[row, col] != 0:
                        color = cat2color[generated[row, col].item()]
                        x = (col - 40) + 0.5
                        y = (40 - row) + 0.5
                        ax.scatter(x, y, c=color, marker='x')
            fig.savefig(f"./result/test-{i}.png")
            plt.close(fig)
