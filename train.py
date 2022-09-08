import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from datasets import NuScenesDataset, DiffusionModelPreprocessor, collate_fn
from networks import DiffusionBasedModel
import numpy as np
import time


if __name__ == '__main__':
    device = torch.device(0)
    np.random.seed(0)
    torch.manual_seed(0)
    timestamp = time.strftime('%m-%d-%H:%M:%S')
    writer = SummaryWriter(log_dir=f'./log/{timestamp}')
    os.makedirs('./ckpts', exist_ok=True)
    dataset = NuScenesDataset("/projects/perception/personals/yefanlin/data/nuSceneProcessed/train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)
    preprocessor = DiffusionModelPreprocessor(device).test()
    model = DiffusionBasedModel(time_steps=1000)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=2e-5)
    n_epochs = 4000

    iters = 0
    for epoch in range(n_epochs):
        for batch in dataloader:
            pedestrians, bicyclists, vehicles, maps = preprocessor(batch)
            loss_dict = model(pedestrians, bicyclists, vehicles, maps)
            for name in ['pedestrian', 'bicyclist', 'vehicle']:
                for entry in ['length', 'noise']:
                    writer.add_scalar(f'{name}+{entry}', loss_dict[name][entry], iters)
            writer.add_scalar('all', loss_dict['all'], iters)
            print(iters, loss_dict['all'].item())
            optimizer.zero_grad()
            loss_dict['all'].backward()
            optimizer.step()
            iters += 1
            
    model.cpu()
    torch.save(model.state_dict(), os.path.join('./ckpts', timestamp))
