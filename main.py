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
from networks.losses.nll import lr_func
import time

if __name__ == '__main__':
    device = torch.device(0)
    np.random.seed(0)
    torch.manual_seed(0)
    dataset = NuScenesDataset("/projects/perception/datasets/nuScenesProcessed/train")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=6, collate_fn=collate_fn)
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
            print(f"{iters}: {loss['all'].mean().item()}")
            optimizer.zero_grad()
            loss['all'].mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            print('loss ok')
            optimizer.step()
            print('step ok')
            scheduler.step()
            iters += 1
