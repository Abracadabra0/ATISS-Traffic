import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import NuScenesDataset, ScoreModelProcessor, collate_fn
from networks.ScoreBasedGenerator import Generator
import numpy as np
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


if __name__ == '__main__':
    device = torch.device(0)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(800)
    ])
    dataset = MNIST('.', train=True, transform=transform, download=True)
    dataset = torch.utils.data.Subset(dataset, range(1))
    B = 1
    dataloader = DataLoader(dataset, batch_size=B, shuffle=False, num_workers=4)
    model = Generator()
    model.load_state_dict(torch.load('./ckpts/08-27-08:14:22'))
    model.to(device)
    for x, y in dataloader:
        x = model.generate().to('cpu')
        for i in range(B):
            generated = x[i]
            generated = generated.clamp(min=0, max=1)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(generated[0])
            fig.savefig(f"./result/test-{i}.png")
            plt.close(fig)
