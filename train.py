import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from datasets import NuScenesDataset, DiffusionModelPreprocessor, collate_fn
from networks import DiffusionBasedModel
import numpy as np
import time


def lr_func(warmup):
    def inner(iters):
        return min((iters + 1) ** -0.5, (iters + 1) * warmup ** -1.5)
    return inner


if __name__ == '__main__':
    device = torch.device(0)
    np.random.seed(0)
    torch.manual_seed(0)
    timestamp = time.strftime('%m-%d-%H:%M:%S')
    writer = SummaryWriter(log_dir=f'./log/{timestamp}')
    os.makedirs('./ckpts', exist_ok=True)
    B = 4
    dataset = NuScenesDataset("/projects/perception/personals/yefanlin/data/nuSceneProcessed/train")
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True, num_workers=8, collate_fn=collate_fn)
    preprocessor = DiffusionModelPreprocessor(device).train()
    model = DiffusionBasedModel(time_steps=1000)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lr_func(4000))
    n_epochs = 400

    iters = 0
    hist = np.zeros(1000)
    cnt = np.ones(1000)
    for epoch in range(n_epochs):
        print(f"------------ Epoch {epoch} ------------")
        for batch in dataloader:
            pedestrians, bicyclists, vehicles, maps = preprocessor(batch)
            loss_dict = model(pedestrians, bicyclists, vehicles, maps)
            for name in ['pedestrian', 'bicyclist', 'vehicle']:
                for entry in ['length', 'noise']:
                    loss_dict[name][entry] = loss_dict[name][entry].mean()
                    writer.add_scalar(f'loss/{name}+{entry}', loss_dict[name][entry], iters)
            loss_dict['all'] = loss_dict['all'].mean()
            writer.add_scalar('all', loss_dict['all'], iters)
            print(iters, loss_dict['all'].item())
            t = loss_dict['t']
            hist[t] += loss_dict['pedestrian']['noise'].item()
            cnt[t] += B
            optimizer.zero_grad()
            loss_dict['all'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            scheduler.step()
            iters += 1
            
    hist = hist / cnt
    for step, y in enumerate(hist):
        writer.add_scalar('sigma', y, step)
    model.cpu()
    torch.save(model.state_dict(), f'./ckpts/{timestamp}')
