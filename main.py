import os
import torch
from torch import nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from datasets import NuScenesDataset, AutoregressivePreprocessor, collate_fn
from networks.autoregressive_transformer import AutoregressiveTransformer
import numpy as np
from networks.losses.nll import lr_func
import time


os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
timestamp = time.strftime('%m-%d-%H:%M:%S')
writer = SummaryWriter(log_dir=f'./log/{timestamp}')
os.makedirs(f'./ckpts/{timestamp}', exist_ok=True)
n_gpus = torch.cuda.device_count()
n_epochs = 30
batch_size = 12


def main(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(rank)
    dataset = NuScenesDataset("/shared/perception/datasets/nuScenesProcessed/train")
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size // world_size, shuffle=False, collate_fn=collate_fn, sampler=sampler)
    preprocessor = AutoregressivePreprocessor('cpu').train()

    model = AutoregressiveTransformer()
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lr_func(4000))

    iters = 0
    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            batch, lengths, gt = preprocessor(batch, window_size=1)
            loss = model(batch, lengths, gt)
            if rank == 0:
                for k, v in loss.items():
                    writer.add_scalar(f'loss/{k}', v.mean(), iters)
            optimizer.zero_grad()
            loss['all'].mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()
            iters += 1
        if rank == 0:
            torch.save(model.module.state_dict(), os.path.join(f'./ckpts/{timestamp}', f'model-{epoch}'))
            torch.save(optimizer.state_dict(), os.path.join(f'./ckpts/{timestamp}', f'optimizer-{epoch}'))
            torch.save(scheduler.state_dict(), os.path.join(f'./ckpts/{timestamp}', f'scheduler-{epoch}'))
    if rank == 0:
        torch.save(model.module.state_dict(), os.path.join(f'./ckpts/{timestamp}', 'final'))


if __name__ == '__main__':
    mp.spawn(main, args=(n_gpus,), nprocs=n_gpus, join=True)
