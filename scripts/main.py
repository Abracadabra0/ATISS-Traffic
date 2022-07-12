import sys

sys.path.append('/home/yefanlin/project/ATISS-Traffic')

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from scene_synthesis.datasets.nuScenes import NuScenesDataset
from scene_synthesis.datasets.utils import collate_train
from scene_synthesis.networks.autoregressive_transformer import AutoregressiveTransformer
from scene_synthesis.networks.feature_extractors import ResNet18
from torch.optim import Adam
import numpy as np
from scene_synthesis.losses.nll import WeightedNLL, lr_func
import time

if __name__ == '__main__':
    device = torch.device(0)
    np.random.seed(0)
    torch.manual_seed(0)
    timestamp = time.strftime('%m-%d-%H:%M:%S')
    writer = SummaryWriter(log_dir=f'/home/yefanlin/scratch/project/ATISS-Traffic/log/{timestamp}')
    os.makedirs('/home/yefanlin/scratch/project/ATISS-Traffic/ckpts', exist_ok=True)
    dataset = NuScenesDataset("/home/yefanlin/scratch/data/nuScene-processed", train=True)
    # dataset = NuScenesDataset("/media/yifanlin/My Passport/data/nuScene-processed", train=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_train)
    feature_extractor = ResNet18(4, 512)
    feature_extractor.to(device)
    model = AutoregressiveTransformer(feature_extractor, writer)
    model.to(device)
    loss_fn = WeightedNLL(weights={
        'category': 0.2,
        'location': 1.,
        'wl': 0.6,
        'theta': 0.3,
        'moving': 0.4,
        's': 0.4,
        'omega': 0.2
    })
    loss_fn.to(device)
    optimizer = Adam(model.parameters(), lr=768**-0.5 * 0.1)
    scheduler = LambdaLR(optimizer, lr_func(1000))
    n_epochs = 6000
    iters = 0

    for epoch in range(n_epochs):
        print(f'----------------Epoch {epoch}----------------')
        for samples, lengths, gt in dataloader:
            for k in samples:
                samples[k] = samples[k].to(device)
            lengths = lengths.to(device)
            for k in gt:
                gt[k] = gt[k].to(device)

            optimizer.zero_grad()
            loss = model(samples, lengths, gt, loss_fn)

            for k, v in loss.items():
                print(f"{k}: {v.item()}")
                writer.add_scalar(f'loss/{k}', v, iters)
            loss['all'].backward()
            optimizer.step()
            scheduler.step()
            iters += 1

    model.cpu()
    torch.save(model.state_dict(), os.path.join('/home/yefanlin/scratch/project/ATISS-Traffic/ckpts', timestamp))
