# %%
import sys
sys.path.append('/home/yefanlin/project/ATISS-Traffic')

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
# %%
device = torch.device(0)
np.random.seed(0)
torch.manual_seed(0)
dataset = NuScenesDataset("/home/yefanlin/scratch/data/nuScene-processed", train=True)
# %%
dataset[0]
# %%
