import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

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


def collate_train(samples, keep_all=False):
    lengths = [len(sample['category']) for sample in samples]
    if not keep_all:
        keep_lengths = [np.random.randint(0, length + 1) for length in lengths]
    else:
        keep_lengths = lengths.copy()
    collated = {}
    gt = {}
    for k in ['category', 'location', 'bbox', 'velocity']:
        collated[k] = pad_sequence([sample[k][:l] for sample, l in zip(samples, keep_lengths)], batch_first=True)
        gt_list = []
        for sample, l in zip(samples, keep_lengths):
            try:
                gt_list.append(sample[k][l])
            except IndexError:
                if k == 'category':
                    gt_list.append(torch.tensor(0))
                elif k == 'bbox':
                    gt_list.append(torch.tensor([0., 0., 0.]))
                else:
                    gt_list.append(torch.tensor([0., 0.]))
        gt[k] = torch.stack(gt_list)
    collated['map'] = torch.stack([sample['map'] for sample in samples])

    return collated, torch.tensor(keep_lengths), gt
