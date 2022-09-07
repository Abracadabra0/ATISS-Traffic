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
from tqdm import tqdm


if __name__ == '__main__':
    device = torch.device(0)
    np.random.seed(0)
    torch.manual_seed(0)
    timestamp = time.strftime('%m-%d-%H:%M:%S')
    writer = SummaryWriter(log_dir=f'./log/{timestamp}')
    os.makedirs('./ckpts', exist_ok=True)
    dataset = NuScenesDataset("/media/yifanlin/My Passport/data/nuSceneProcessed/train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)
    model = DiffusionBasedModel(time_steps=1000)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    n_epochs = 5000

    iters = 0
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        total_loss = 0.
        num_items = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            x = x.to(device)
            score, loss = model(x)
            total_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            writer.add_scalar('loss', loss, iters)
            loss.backward()
            optimizer.step()
            iters += 1
        pbar.set_description(f'Average loss: {total_loss / num_items}')

    model.cpu()
    torch.save(model.state_dict(), os.path.join('./ckpts', timestamp))
