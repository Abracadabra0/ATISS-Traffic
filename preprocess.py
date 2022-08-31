import math
import torch
import numpy as np
import os
import cv2
from pyquaternion import Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
from datasets.utils import get_homogeneous_matrix, cartesian_to_polar
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")


layer_names = ['drivable_area',
                'ped_crossing',
                'walkway',
                'carpark_area',
                'lane',
                'lane_divider',
                'road_segment']
Q = 0
PEDESTRIAN = 1
BICYCLIST = 2
VEHICLE = 3
END = 4
category_mapping = {'human.pedestrian.adult': PEDESTRIAN,
                    'human.pedestrian.child': PEDESTRIAN,
                    'human.pedestrian.wheelchair': PEDESTRIAN,
                    'human.pedestrian.stroller': PEDESTRIAN,
                    'human.pedestrian.personal_mobility': PEDESTRIAN,
                    'human.pedestrian.police_officer': PEDESTRIAN,
                    'human.pedestrian.construction_worker': PEDESTRIAN,
                    'vehicle.car': VEHICLE,
                    'vehicle.motorcycle': BICYCLIST,
                    'vehicle.bicycle': BICYCLIST,
                    'vehicle.bus.bendy': VEHICLE,
                    'vehicle.bus.rigid': VEHICLE,
                    'vehicle.truck': VEHICLE,
                    'vehicle.construction': VEHICLE,
                    'vehicle.emergency.ambulance': VEHICLE,
                    'vehicle.emergency.police': VEHICLE,
                    'vehicle.trailer': VEHICLE}

resolution = 0.25
axes_limit = 40
wl = int(axes_limit * 2 / resolution)

dataroot = '/shared/perception/datasets/nuScenesMetadata'
version = 'v1.0-test'
output_path = '/shared/perception/datasets/nuScenesProcessed'
split = 'test'

n_process = None

nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
os.makedirs(output_path, exist_ok=True)
os.chdir(output_path)
if split == 'train':
    os.makedirs('train', exist_ok=True)
    os.chdir('train')
else:
    os.makedirs('test', exist_ok=True)
    os.chdir('test')

def preprocess_scene(i_scene):
    scene = nusc.scene[i_scene]
    sample_token = scene['first_sample_token']
    while sample_token:
        sample = nusc.get('sample', sample_token)
        # get data from that sample
        sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        scene = nusc.get('scene', sample['scene_token'])
        log = nusc.get('log', scene['log_token'])
        map_name = log['location']
        pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        ego_to_world = get_homogeneous_matrix(np.zeros(3), Quaternion(pose['rotation']).rotation_matrix)

        # create directory
        os.makedirs(sample_data['token'], exist_ok=True)

        # get annotated map
        nusc_map = NuScenesMap(dataroot=dataroot, map_name=map_name)
        patch_box = (pose['translation'][0], pose['translation'][1], axes_limit * 2, axes_limit * 2)
        patch_angle = math.degrees(Quaternion(pose['rotation']).yaw_pitch_roll[0])
        rad = patch_angle / 180 * np.pi
        rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        rot_inv = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])

        # parse center lines
        patch = (pose['translation'][0] - axes_limit * np.sqrt(2),
                    pose['translation'][1] - axes_limit * np.sqrt(2),
                    pose['translation'][0] + axes_limit * np.sqrt(2),
                    pose['translation'][1] + axes_limit * np.sqrt(2))
        lane_tokens = nusc_map.get_records_in_patch(patch, ['lane'], mode='intersect')['lane']
        center_lines = {}
        for lane_token in lane_tokens:
            center_lines[lane_token] = {}
            lane_record = nusc_map.get_arcline_path(lane_token)
            arcs = arcline_path_utils.discretize_lane(lane_record, resolution_meters=1)
            arcs = np.array(arcs)
            arcs[:, :2] = np.dot(arcs[:, :2] - pose['translation'][:2], rot)
            arcs[:, 2:] = arcs[:, 2:] - patch_angle / 180 * np.pi
            arcs[:, 2:] = np.where(arcs[:, 2:] > np.pi, arcs[:, 2:] - 2 * np.pi, arcs[:, 2:])
            arcs[:, 2:] = np.where(arcs[:, 2:] < -np.pi, arcs[:, 2:] + 2 * np.pi, arcs[:, 2:])
            arcs = list(filter(lambda x: -axes_limit * 2 < x[0] < axes_limit * 2
                                            and -axes_limit * 2 < x[1] < axes_limit * 2,
                                list(arcs)))
            center_lines[lane_token]['arcs'] = np.array(arcs)
            node_tokens = nusc_map.get('lane', lane_token)['exterior_node_tokens']
            nodes = []
            for node_token in node_tokens:
                node = nusc_map.get('node', node_token)
                nodes.append(np.array([node['x'], node['y']]))
            nodes = np.stack(nodes, axis=0)
            center_lines[lane_token]['nodes'] = nodes

        # get map masks
        # scaled: drivable_area, ped_crossing, walkway, carpark_area,
        # lane, lane_divider, road_segment
        map_mask = nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size=None)
        map_mask = np.flip(map_mask, 1)
        drivable_area = cv2.resize(map_mask[0], (wl, wl))
        ped_crossing = cv2.resize(map_mask[1], (wl, wl))
        walkway = cv2.resize(map_mask[2], (wl, wl))
        carpark_area = cv2.resize(map_mask[3], (wl, wl))
        lane = cv2.resize(map_mask[4], (wl, wl))
        lane_divider = cv2.resize(map_mask[5], (wl, wl))
        road_segment = cv2.resize(map_mask[6], (wl, wl))

        # get lane orientation
        lane_pts = []
        orientation = np.zeros_like(lane, dtype=float)
        hit = lane_tokens[0]
        for row in range(orientation.shape[0]):
            for col in range(orientation.shape[1]):
                if lane[row, col]:
                    coord = np.array([(col - 160) * 0.25, (160 - row) * 0.25])
                    coord_global = np.dot(coord, rot_inv) + pose['translation'][:2]
                    result = -1.0
                    for lane_token in [hit] + lane_tokens:
                        nodes = center_lines[lane_token]['nodes']
                        result = cv2.pointPolygonTest(nodes.reshape((-1, 1, 2)).astype(np.float32),
                                                        tuple(coord_global.astype(np.float32)),
                                                        False)
                        if result > 0:
                            hit = lane_token
                            break
                    if result < 0:
                        lane[row, col] = 0
                        continue
                    arcs = center_lines[hit]['arcs']
                    dist = np.linalg.norm(coord - arcs[:, :2], axis=1)
                    argmin = np.argmin(dist)
                    orientation[row, col] = arcs[argmin, 2]
                    lane_pts.append(np.array([row, col]))
        lane_pts = lane_pts[0:-1:7]
        lane_pts = np.array(lane_pts)

        # get road_segment orientation
        road_segment = np.where(lane == 0, road_segment, 0)
        for row in range(road_segment.shape[0]):
            for col in range(road_segment.shape[1]):
                if road_segment[row, col] > 0:
                    dst = np.linalg.norm(lane_pts - np.array([row, col]), axis=1)
                    closest = lane_pts[dst.argmin()]
                    orientation[row, col] = orientation[closest[0], closest[1]]
        lane = lane + road_segment

        map_layers = np.stack([
            drivable_area,
            ped_crossing,
            walkway,
            carpark_area,
            lane,
            lane_divider,
            orientation
        ], axis=0)

        # convert to torch.tensor and save it
        map_layers = torch.tensor(map_layers.copy(), dtype=torch.float32)
        torch.save(map_layers, os.path.join(sample_data['token'], 'map'))

        # retrieve all objects that fall inside the boundaries
        _, boxes, _ = nusc.get_sample_data(sample['data']['LIDAR_TOP'], box_vis_level=BoxVisibility.ALL,
                                            use_flat_vehicle_coordinates=True)
        boxes = filter(
            lambda x: -axes_limit < x.center[0] < axes_limit and -axes_limit < x.center[1] < axes_limit,
            boxes)
        # filter out relevant categories
        boxes = filter(lambda x: x.name in category_mapping, boxes)
        boxes = list(boxes)
        boxes.sort(key=lambda x: (-x.center[1], x.center[0]))
        # parse data
        category = []
        location = []
        bbox = []
        velocity = []
        for box in boxes:
            # filter out vehicles outside roads
            if category_mapping[box.name] == VEHICLE:
                x, y = box.center[0], box.center[1]
                row = int((axes_limit - y) / resolution)
                col = int((x + axes_limit) / resolution)
                if lane[row, col] == 0:
                    continue
            box_to_ego = get_homogeneous_matrix(box.center, box.rotation_matrix)
            # calculates vehicle heading direction
            _, heading = cartesian_to_polar(box_to_ego[:2, 0])
            # calculates velocity by differentiate
            v = nusc.box_velocity(box.token)
            # velocity could be nan. If so, drop it
            if True in np.isnan(v):
                continue
            # convert to ego coordinate
            v = np.dot(np.linalg.inv(ego_to_world[:3, :3]), v[..., None]).flatten()[:2]
            category.append(category_mapping[box.name])
            location.append(box.center[:2])
            bbox.append((box.wlh[0], box.wlh[1], heading))
            velocity.append(cartesian_to_polar(v))
        # append end token
        category.append(0)
        location.append(np.zeros(2))
        bbox.append(np.zeros(3))
        velocity.append(np.zeros(2))
        # convert to tensor and save
        torch.save(torch.tensor(np.array(category), dtype=torch.int64), os.path.join(sample_data['token'], 'category'))
        torch.save(torch.tensor(np.array(location), dtype=torch.float32), os.path.join(sample_data['token'], 'location'))
        torch.save(torch.tensor(np.array(bbox), dtype=torch.float32), os.path.join(sample_data['token'], 'bbox'))
        torch.save(torch.tensor(np.array(velocity), dtype=torch.float32), os.path.join(sample_data['token'], 'velocity'))

        sample_token = sample['next']
    return i_scene


def callback(res):
    print(f'Scene {res} finished')


pool = Pool(processes=n_process)
for i_scene in range(len(nusc.scene)):
    pool.apply_async(preprocess_scene, args=(i_scene, ), callback=callback)
pool.close()
pool.join()
print('All done')
