from scene_synthesis.datasets.nuScenes import NuScenesDataset


if __name__ == '__main__':
    dataroot = '/media/yifanlin/My Passport/data/nuScene'
    version = 'v1.0-mini'
    output_path = '/media/yifanlin/My Passport/data/nuScene-processed'
    NuScenesDataset.preprocess(dataroot, version, output_path)
