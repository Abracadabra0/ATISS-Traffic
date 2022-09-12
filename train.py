import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import DataParallel
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
    dataset = NuScenesDataset("/shared/perception/personals/yefanlin/data/nuSceneProcessed/train")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8, collate_fn=collate_fn)
    preprocessor = DiffusionModelPreprocessor('cpu').train()
    model = DiffusionBasedModel(time_steps=1000)
    model = DataParallel(model).to(device)
    optimizer = Adam(model.parameters(), lr=2e-5)
    n_epochs = 250

    iters = 0
    print("Running on %d GPUs " % torch.cuda.device_count())
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
            optimizer.zero_grad()
            loss_dict['all'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            iters += 1
        if (epoch + 1) % 50 == 0:
            torch.save(model.module.state_dict(), os.path.join(f'./ckpts/{timestamp}', f'model-{epoch}'))
            torch.save(optimizer.state_dict(), os.path.join(f'./ckpts/{timestamp}', f'optimizer-{epoch}'))
            
    model.cpu()
    torch.save(model.module.state_dict(), os.path.join(f'./ckpts/{timestamp}', 'final'))
