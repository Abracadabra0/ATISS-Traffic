import torch
from torch import nn
from torch.utils.data import DataLoader
from scene_synthesis.datasets.nuScenes import NuScenesDataset
from scene_synthesis.datasets.utils import collate_train
from scene_synthesis.networks.autoregressive_transformer import AutoregressiveTransformer
from scene_synthesis.networks.feature_extractors import ResNet18
from torch.optim import Adam
import numpy as np
from scene_synthesis.losses.nll import WeightedNLL


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    dataset = NuScenesDataset("/media/yifanlin/My Passport/data/nuScene-processed", train=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_train)
    feature_extractor = ResNet18(10, 512)
    model = AutoregressiveTransformer(feature_extractor)
    loss_fn = WeightedNLL(weights={
        'category': 1.,
        'location': 0.01,
        'bbox': 1.,
        'velocity': 1
    }, with_components=True)
    optimizer = Adam(model.parameters(), lr=1e-5)
    n_epoch = 100

    for _ in range(n_epoch):
        for i, (samples, lengths, gt) in enumerate(dataloader):
            optimizer.zero_grad()
            probs = model(samples, lengths, gt)
            loss, components = loss_fn(probs, gt)
            print(i, loss.item())
            loss.backward()
            optimizer.step()
