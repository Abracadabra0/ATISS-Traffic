import cv2
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import functional as F
from torchvision.transforms import RandomRotation


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


def cartesian_to_polar(vector: np.array) -> np.array:
    rho = np.linalg.norm(vector)
    if rho == 0:
        return np.array([0., 0.])
    theta = np.arctan(vector[1] / vector[0]) + (vector[0] < 0) * np.pi
    theta = theta + (theta < 0) * np.pi * 2
    return np.array([rho, theta])


def collate_test(samples, keep='random'):
    fields = ['category', 'location', 'bbox', 'velocity']
    for sample in samples:
        # drivable_area, ped_crossing, walkway, lane, orientation
        base_layers = sample['map'][:4]
        lane = sample['map'][3:4]
        orientation = sample['map'][4:]
        sample['map'] = torch.cat([base_layers,
                                   torch.sin(orientation) * lane,
                                   torch.cos(orientation) * lane], dim=0)
        # reorder
        idx = list(range(len(sample['category']) - 1))
        idx.sort(key=lambda x: (-sample['location'][x, 1], sample['location'][x, 0]))
        idx.append(len(sample['category']) - 1)
        for field in ['category', 'location', 'bbox', 'velocity']:
            sample[field] = sample[field][idx]

    # random masking
    lengths = [len(sample['category']) for sample in samples]
    if keep == 'random':
        keep_lengths = [np.random.randint(0, length + 1) for length in lengths]
    elif keep == -1:
        keep_lengths = lengths
    else:
        keep_lengths = [keep for _ in lengths]
    collated = {field: [] for field in fields}
    maps = []
    for sample, keep_length in zip(samples, keep_lengths):
        for field in fields:
            collated[field].append(sample[field][:keep_length])
        object_layers = rasterize_objects(sample['category'][:keep_length],
                                          sample['location'][:keep_length],
                                          sample['bbox'][:keep_length],
                                          sample['velocity'][:keep_length])
        # drivable_area, ped_crossing, walkway, lane, orientation(sin), orientation(cos),
        # occupancy, orientation(sin), orientation(cos), speed,
        # heading(sin), heading(cos)
        # 12 layers in total
        all_layers = torch.cat([
            sample['map'],
            object_layers['occupancy'],
            torch.sin(object_layers['orientation']) * object_layers['occupancy'],
            torch.cos(object_layers['orientation']) * object_layers['occupancy'],
            object_layers['speed'],
            torch.sin(object_layers['heading']) * object_layers['occupancy'],
            torch.cos(object_layers['heading']) * object_layers['occupancy']
        ], dim=0)
        maps.append(all_layers)
    for field in fields:
        collated[field] = pad_sequence(collated[field], batch_first=True)
    collated['map'] = torch.stack(maps)
    return collated, torch.tensor(keep_lengths)


def collate_train(samples, window_size=1):
    fields = ['category', 'location', 'bbox', 'velocity']
    # random rotation
    angles = np.random.rand(len(samples)) * 360
    for sample, angle in zip(samples, angles):
        sample['map'] = F.rotate(sample['map'], angle)
        # drivable_area, ped_crossing, walkway, lane, lane_divider, orientation
        base_layers = sample['map'][0:5]
        lane = sample['map'][3:4]
        orientation = sample['map'][5:6]
        rad = angle / 180 * np.pi
        rotation_mat = torch.tensor([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]], dtype=torch.float32)
        orientation += rad
        sample['map'] = torch.cat([base_layers,
                                   torch.sin(orientation) * lane,
                                   torch.cos(orientation) * lane], dim=0)
        sample['location'] = sample['location'] @ rotation_mat
        sample['bbox'][:, -1] += rad
        sample['velocity'][:, -1] += rad
        # filter out objects fallen outside the image
        r = sample['location'].norm(dim=1)
        mask = (r < 40)
        for field in fields:
            sample[field] = sample[field][mask]
        # reorder
        idx = list(range(len(sample['category']) - 1))
        idx.sort(key=lambda x: (-sample['location'][x, 1], sample['location'][x, 0]))
        idx.append(len(sample['category']) - 1)
        for field in ['category', 'location', 'bbox', 'velocity']:
            sample[field] = sample[field][idx]
    # random masking
    lengths = [len(sample['category']) for sample in samples]
    window_size = min(window_size, min(lengths))
    keep_lengths = [np.random.randint(0, length - window_size + 1) for length in lengths]
    collated = {field: [] for field in fields}
    gt = {field: [] for field in fields}
    maps = []
    for sample, keep_length in zip(samples, keep_lengths):
        for field in fields:
            collated[field].append(sample[field][:keep_length])
            gt[field].append(sample[field][keep_length:keep_length + window_size])
        object_layers = rasterize_objects(sample['category'][:keep_length],
                                          sample['location'][:keep_length],
                                          sample['bbox'][:keep_length],
                                          sample['velocity'][:keep_length])
        # drivable_area, ped_crossing, walkway, lane, lane_divider, orientation(sin), orientation(cos),
        # (occupancy, orientation(sin), orientation(cos), speed, heading(sin), heading(cos)) * 3
        # 25 layers in total
        for name in range(1, 4):
            layers = object_layers[name]
            object_layers[name] = torch.cat([
                layers['occupancy'],
                torch.sin(layers['orientation']) * layers['occupancy'],
                torch.cos(layers['orientation']) * layers['occupancy'],
                layers['speed'],
                torch.sin(layers['heading']) * layers['occupancy'],
                torch.cos(layers['heading']) * layers['occupancy']
            ], dim=0)
        all_layers = torch.cat([
            sample['map'],
            object_layers[1],
            object_layers[2],
            object_layers[3]
        ], dim=0)
        maps.append(all_layers)
    for field in fields:
        collated[field] = pad_sequence(collated[field], batch_first=True)
        gt[field] = pad_sequence(gt[field], batch_first=True)
    collated['map'] = torch.stack(maps)
    return collated, torch.tensor(keep_lengths), gt


def rasterize_objects(category, location, bbox, velocity):
    category = category.numpy()
    location = location.numpy()
    bbox = bbox.numpy()
    velocity = velocity.numpy()
    L = category.shape[0]
    object_layers = {k: None for k in range(1, 4)}
    for name in object_layers:
        object_layers[name] = {k: np.zeros((320, 320), dtype=np.float32)
                               for k in ['occupancy', 'orientation', 'speed', 'heading']}
    for i in range(L):
        layers = object_layers[category[i]]
        w, l, theta = bbox[i]
        speed, heading = velocity[i]
        corners = np.array([[l / 2, w / 2],
                            [-l / 2, w / 2],
                            [-l / 2, -w / 2],
                            [l / 2, -w / 2]])
        rotation = np.array([[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]])
        corners = np.dot(corners, rotation) + location[i]
        corners[:, 0] = corners[:, 0] + 40
        corners[:, 1] = 40 - corners[:, 1]
        corners = np.floor(corners / 0.25).astype(int)
        occupancy = np.zeros((320, 320), dtype=np.uint8)
        cv2.fillConvexPoly(occupancy, corners, 255)
        layers['occupancy'] = np.where(occupancy > 0, 1., layers['occupancy'])
        row = int((40 - location[i, 1]) / 0.25)
        col = int((location[i, 0] + 40) / 0.25)
        layers['orientation'][row, col] = theta
        layers['speed'][row, col] = speed
        layers['heading'][row, col] = heading
    for name in range(1, 4):
        for k in ['occupancy', 'orientation', 'speed', 'heading']:
            object_layers[name][k] = torch.tensor(object_layers[name][k], dtype=torch.float32).unsqueeze(0)
    return object_layers
