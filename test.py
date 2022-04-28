from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from matplotlib import pyplot as plt


nusc = NuScenes(version='v1.0-mini', dataroot='/media/yifanlin/My Passport/data/nuScene', verbose=True)
my_sample = nusc.sample[2]
nusc.render_sample_data(my_sample['data']['LIDAR_TOP'], axes_limit=40)
plt.show()
