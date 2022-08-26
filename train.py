import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from datasets import NuScenesDataset, ScoreModelProcessor, collate_fn
from networks.ScoreBasedGenerator import Generator
import numpy as np
import time

if __name__ == '__main__':
    device = torch.device(0)
    np.random.seed(0)
    torch.manual_seed(0)
    timestamp = time.strftime('%m-%d-%H:%M:%S')
    writer = SummaryWriter(log_dir=f'./log/{timestamp}')
    os.makedirs('./ckpts', exist_ok=True)
    dataset = NuScenesDataset("../../data/nuSceneProcessed/train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)
    processor = ScoreModelProcessor(device).train()
    model = Generator()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    n_epochs = 4000
    iters = 0

    for epoch in range(n_epochs):
        print(f'----------------Epoch {epoch}----------------')
        for batch in dataloader:
            x, map_layers = processor(batch)
            _, loss = model(x, map_layers)
            print(iters, loss.item())
            writer.add_scalar('loss', loss, iters)
            loss.backward()
            optimizer.step()
            iters += 1

        # if (epoch + 1) % 100 == 0:
        #     model.cpu()
        #     torch.save(model.state_dict(), os.path.join('./ckpts', timestamp + f'-{epoch}'))
        #     model.to(device)

    model.cpu()
    torch.save(model.state_dict(), os.path.join('./ckpts', timestamp))
