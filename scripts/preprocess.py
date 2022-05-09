from scene_synthesis.datasets.nuScenes import NuScenesDataset


if __name__ == '__main__':
    NuScenesDataset.preprocess(dataroot="/media/yifanlin/My Passport/data/nuScene",
                               version='v1.0-mini',
                               output_path="/media/yifanlin/My Passport/data/nuScene-processed",
                               axes_limit=40)
