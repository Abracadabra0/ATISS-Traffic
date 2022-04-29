import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes

dataroot = '/media/yifanlin/My Passport/data/nuScene'
nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)
sample = nusc.sample[20]
axes_limit = 80

def crop_image(image: np.array,
               x_px: int,
               y_px: int,
               axes_limit_px: int) -> np.array:
    x_min = int(x_px - axes_limit_px)
    x_max = int(x_px + axes_limit_px)
    y_min = int(y_px - axes_limit_px)
    y_max = int(y_px + axes_limit_px)
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image

# Get data.
sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
scene = nusc.get('scene', sample['scene_token'])
log = nusc.get('log', scene['log_token'])
map_name = log['location']
map_ = nusc.get('map', log['map_token'])
map_mask = map_['mask']
pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
# Retrieve and crop mask.
pixel_coords = map_mask.to_pixel_coords(pose['translation'][0], pose['translation'][1])
scaled_limit_px = int(axes_limit * (1.0 / map_mask.resolution))
mask_raster = map_mask.mask()
cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], scaled_limit_px)
plt.imshow(cropped)
plt.show()

nusc.render_sample_data(sample['data']['LIDAR_TOP'], axes_limit=40)
plt.show()

nusc_map = NuScenesMap(dataroot=dataroot, map_name=map_name)
patch_box = (pose['translation'][0], pose['translation'][1], axes_limit * 2, axes_limit * 2)
patch_angle = 0  # Default orientation where North is up
layer_names = None
canvas_size = None
map_mask = nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
plt.imshow(map_mask[0])
plt.show()
