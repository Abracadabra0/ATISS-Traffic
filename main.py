import os
import torch
from torch import nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from datasets import NuScenesDataset, AutoregressivePreprocessor, collate_fn
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
    os.makedirs(f'./ckpts/{timestamp}', exist_ok=True)
    dataset = NuScenesDataset("/shared/perception/datasets/nuScenesProcessed/train")
    dataloader = DataLoader(dataset, batch_size=72, shuffle=True, num_workers=6, collate_fn=collate_fn)
    processor = AutoregressivePreprocessor('cpu').train()

    model = AutoregressiveTransformer()
    model = DataParallel(model).to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lr_func(4000))

    n_epochs = 20
    iters = 0
    print("Running on %d GPUs " % torch.cuda.device_count())

    for epoch in range(n_epochs):
        for batch in dataloader:
            batch, lengths, gt = processor(batch, window_size=1)
            loss = model(batch, lengths, gt)
            for k, v in loss.items():
                writer.add_scalar(f'loss/{k}', loss[k].mean(), iters)
            print(f"{iters}: {loss['all'].mean().item()}")
            optimizer.zero_grad()
            loss['all'].mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()
            iters += 1

        if (epoch + 1) % 2 == 0:
            torch.save(model.module.state_dict(), os.path.join(f'./ckpts/{timestamp}', f'model-{epoch}'))
            torch.save(optimizer.state_dict(), os.path.join(f'./ckpts/{timestamp}', f'optimizer-{epoch}'))
            torch.save(scheduler.state_dict(), os.path.join(f'./ckpts/{timestamp}', f'scheduler-{epoch}'))

    model.cpu()
    torch.save(model.module.state_dict(), os.path.join(f'./ckpts/{timestamp}', 'final'))
