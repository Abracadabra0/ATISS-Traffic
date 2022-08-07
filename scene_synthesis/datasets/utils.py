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
    # random masking
    lengths = [len(sample['category']) for sample in samples]
    if keep == 'all':
        keep_lengths = lengths
    elif keep == 'random':
        keep_lengths = [np.random.randint(0, length + 1) for length in lengths]
    else:
        keep_lengths = [0 for length in lengths]
    collated = {field: [] for field in fields}
    maps = []
    for sample, keep_length in zip(samples, keep_lengths):
        for field in fields:
            collated[field].append(sample[field][:keep_length])
        occupancy = rasterize_objects(sample['category'][:keep_length],
                                      sample['location'][:keep_length],
                                      sample['bbox'][:keep_length])
        maps.append(torch.cat([sample['map'][:3], occupancy], dim=0))
    for field in fields:
        collated[field] = pad_sequence(collated[field], batch_first=True)
    collated['map'] = torch.stack(maps)
    return collated, torch.tensor(keep_lengths)


def collate_train(samples, window_size=3):
    fields = ['category', 'location', 'bbox', 'velocity']
    # random rotation
    angles = np.random.rand(len(samples)) * 360
    for sample, angle in zip(samples, angles):
        sample['map'] = F.rotate(sample['map'], angle)
        orientation = sample['map'][-1:]
        rad = angle / 180 * np.pi
        rotation_mat = torch.tensor([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]], dtype=torch.float32)
        orientation += rad
        sample['map'] = torch.cat([sample['map'][:-1], torch.sin(orientation), torch.cos(orientation)], dim=0)
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
        occupancy = rasterize_objects(sample['category'][:keep_length],
                                      sample['location'][:keep_length],
                                      sample['bbox'][:keep_length])
        maps.append(torch.cat([sample['map'], occupancy], dim=0))
    for field in fields:
        collated[field] = pad_sequence(collated[field], batch_first=True)
        gt[field] = pad_sequence(gt[field], batch_first=True)
    collated['map'] = torch.stack(maps)
    return collated, torch.tensor(keep_lengths), gt


def rasterize_objects(category, location, bbox):
    category = category.numpy()
    location = location.numpy()
    bbox = bbox.numpy()
    maps = np.zeros((3, 320, 320), dtype=np.int8)
    for category_one, location_one, bbox_one in zip(category, location, bbox):
        if category_one == 0:
            continue
        layer = category_one - 1
        w, l, theta = bbox_one
        corners = np.array([[l / 2, w / 2],
                            [-l / 2, w / 2],
                            [-l / 2, -w / 2],
                            [l / 2, -w / 2]])
        rotation = np.array([[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]])
        corners = np.dot(corners, rotation) + location_one
        corners[:, 0] = corners[:, 0] + 40
        corners[:, 1] = 40 - corners[:, 1]
        corners = np.floor(corners / 0.25).astype(int)
        cv2.fillConvexPoly(maps[layer], corners, 255)
    maps = maps / 255
    return torch.tensor(maps, dtype=torch.float32)
