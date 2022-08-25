import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from datasets import NuScenesDataset, AutoregressiveProcessor, collate_fn
from networks.autoregressive_transformer import AutoregressiveTransformer
import numpy as np
from losses.nll import WeightedNLL, lr_func
import time

if __name__ == '__main__':
    device = torch.device(0)
    np.random.seed(0)
    torch.manual_seed(0)
    timestamp = time.strftime('%m-%d-%H:%M:%S')
    writer = SummaryWriter(log_dir=f'./log/{timestamp}')
    os.makedirs('./ckpts', exist_ok=True)
    dataset = NuScenesDataset("../../data/nuScene-processed/train")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn)
    processor = AutoregressiveProcessor(device).train()
    loss_fn = WeightedNLL(weights={
        'category': 0.1,
        'location': 1.,
        'wl': 0.4,
        'theta': 0.4,
        'moving': 0.2,
        's': 0.2,
        'omega': 0.2
    })
    loss_fn.to(device)
    model = AutoregressiveTransformer(loss_fn=loss_fn, lr=1e-3, scheduler=lr_func(1000), logger=writer)
    model.to(device)
    n_epochs = 400
    iters = 0

    for epoch in range(n_epochs):
        print(f'----------------Epoch {epoch}----------------')
        for batch in dataloader:
            batch, lengths, gt = processor(batch, window_size=1)
            loss = model(batch, lengths, gt)
            print(iters, loss['all'].item())
            for k, v in loss.items():
                writer.add_scalar(f'loss/{k}', loss[k], iters)
            iters += 1

        if (epoch + 1) % 100 == 0:
            model.cpu()
            torch.save(model.state_dict(), os.path.join('./ckpts', timestamp + f'-{epoch}'))
            model.to(device)

    model.cpu()
    torch.save(model.state_dict(), os.path.join('./ckpts', timestamp))
