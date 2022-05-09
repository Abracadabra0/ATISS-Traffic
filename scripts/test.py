import numpy as np
from matplotlib import pyplot as plt
from pyquaternion import Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility


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


def get_homogeneous_matrix(translation: np.array,
                           rotation: np.array) -> np.array:
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation
    homogeneous_matrix[:3, 3] = translation
    return homogeneous_matrix


dataroot = '/media/yifanlin/My Passport/data/nuScene'
nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)
sample = nusc.sample[8]
axes_limit = 40
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
# patch_box = (pose['translation'][0], pose['translation'][1], axes_limit * 2, axes_limit * 2)
# patch_angle = 0  # Default orientation where North is up
# layer_names = nusc_map.non_geometric_layers
# map_mask = nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size=None)
# map_mask = np.flip(map_mask, 1)
# for layer, mask in zip(layer_names, map_mask):
#     plt.imshow(mask)
#     plt.title(layer)
#     plt.show()

ego_to_world = get_homogeneous_matrix(np.zeros(3), Quaternion(pose['rotation']).rotation_matrix)
_, boxes, _ = nusc.get_sample_data(sample['data']['LIDAR_TOP'], box_vis_level=BoxVisibility.ANY,
                                   use_flat_vehicle_coordinates=True)
_, ax = plt.subplots(1, 1, figsize=(9, 9))
ax.imshow(cropped, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit])
for box in boxes:
    box_to_ego = get_homogeneous_matrix(box.center, box.rotation_matrix)
    box_to_world = np.dot(ego_to_world, box_to_ego)
    x, y = box_to_world[:2, 3]
    ax.plot(x, y, 'x', color='red')
    dx, dy, _ = nusc.box_velocity(box.token)
    ax.arrow(x, y, dx * 10, dy * 10, color='blue', width=0.05)
ax.set_xlim(-axes_limit, axes_limit)
ax.set_ylim(-axes_limit, axes_limit)
plt.show()
