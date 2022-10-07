from networks.diffusion_models import DiffusionBasedModel
from matplotlib import pyplot as plt
from datasets import NuScenesDataset, collate_fn, DiffusionModelPreprocessor
from torch.utils.data import DataLoader


dataset = NuScenesDataset("/media/yifanlin/My Passport/data/nuSceneProcessed/train")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
batch = next(iter(dataloader))
preprocessor = DiffusionModelPreprocessor('cpu').test()
pedestrians, bicyclists, vehicles, maps = preprocessor(batch)
drivable_area = maps[:, 0]
blur_factors = DiffusionBasedModel.blur_factor_schedule(1000)
diffuse_factors = DiffusionBasedModel.diffuse_factor_schedule(1000)
for t in range(1000):
    blur_factor = blur_factors[t].item()
    blurred = DiffusionBasedModel.blur(drivable_area, blur_factor) + 1e-3
    diffuse_factor = diffuse_factors[t].item()
    diffused = DiffusionBasedModel.diffuse(vehicles['location'], 320, diffuse_factor)
    prob = blurred.unsqueeze(1) * diffused
    plt.imsave(f'./view/{str(t).zfill(3)}.png', prob[0, 1])
