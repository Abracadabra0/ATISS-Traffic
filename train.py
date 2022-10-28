import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from datasets import NuScenesDataset, DiffusionModelPreprocessor, collate_fn
from networks import DiffusionBasedModel
import numpy as np
import time


def lr_func(warmup):
    def inner(iters):
        return min((iters + 1) ** -0.5, (iters + 1) * warmup ** -1.5)
    return inner

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
n_gpus = torch.cuda.device_count()
n_epochs = 20
batch_size = 16

def main(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if rank == 0:
        timestamp = time.strftime('%m-%d-%H:%M:%S')
        writer = SummaryWriter(log_dir=f'./log/{timestamp}')
        os.makedirs(f'./ckpts/{timestamp}', exist_ok=True)
    device = torch.device(rank)
    dataset = NuScenesDataset('/projects/perception/datasets/nuScenesProcessed/train')
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size // world_size, shuffle=False, sampler=sampler, collate_fn=collate_fn)
    preprocessor = DiffusionModelPreprocessor(device).train()
    model = DiffusionBasedModel(time_steps=1000)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lr_func(8000))

    iters = 0
    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            batch = preprocessor(batch)
            loss_dict = model(batch)
            loss_dict['all'] = loss_dict['all'].mean()
            print(iters, loss_dict['all'].item())
            if rank == 0:
                for name in ['pedestrian', 'bicyclist', 'vehicle']:
                    for entry in ['length', 'noise']:
                        loss_dict[name][entry] = loss_dict[name][entry].mean()
                        writer.add_scalar(f'loss/{name}+{entry}', loss_dict[name][entry], iters)
                writer.add_scalar('all', loss_dict['all'], iters)
            optimizer.zero_grad()
            loss_dict['all'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            scheduler.step()
            iters += 1
        if rank == 0:
            torch.save(model.module.state_dict(), os.path.join(f'./ckpts/{timestamp}', f'model-{epoch}'))
            torch.save(optimizer.state_dict(), os.path.join(f'./ckpts/{timestamp}', f'optimizer-{epoch}'))
            torch.save(scheduler.state_dict(), os.path.join(f'./ckpts/{timestamp}', f'scheduler-{epoch}'))
    if rank == 0:
        torch.save(model.module.state_dict(), f'./ckpts/{timestamp}/final')

if __name__ == '__main__':
    mp.spawn(main, args=(n_gpus,), nprocs=n_gpus, join=True)