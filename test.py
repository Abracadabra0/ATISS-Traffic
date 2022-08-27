import torch
from torch.utils.data import DataLoader
from datasets import NuScenesDataset, ScoreModelProcessor, collate_fn
from networks.ScoreBasedGenerator import Generator
import numpy as np
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


if __name__ == '__main__':
    device = torch.device(0)
    np.random.seed(0)
    torch.manual_seed(0)
    dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
    dataset = torch.utils.data.Subset(dataset, range(1))
    B = 1
    dataloader = DataLoader(dataset, batch_size=B, shuffle=False, num_workers=4)
    model = Generator()
    model.load_state_dict(torch.load('./ckpts/08-27-02:53:47'))
    model.to(device)
    for x, y in dataloader:
        x = model.generate().to('cpu')
        for i in range(B):
            generated = x[i]
            generated = generated.clamp(min=0, max=1)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(generated[0], extent=[-40, 40, -40, 40])
            fig.savefig(f"./result/test-{i}.png")
            plt.close(fig)
