import cv2
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import functional as F


class AutoregressivePreprocessor:
    """
    input_layers:
        drivable_area, ped_crossing, walkway, carpark_area, lane, lane_divider, orientation
        7 layers in total

    output_layers:
        drivable_area, ped_crossing, walkway, carpark_area, lane, lane_divider, orientation(sin), orientation(cos),
        (occupancy, orientation(sin), orientation(cos), speed, heading(sin), heading(cos)) * 3
        26 layers in total
    """

    def __init__(self, device, window_scheduler=None):
        self.device = device
        self.axes_limit = 40
        self.wl = 320
        self.resolution = 0.25
        self.train_iters = 0
        self.scheduler = window_scheduler
        self.state = 'train'

    def train(self):
        self.state = 'train'
        return self

    def test(self):
        self.state = 'test'
        return self

    def __call__(self, batch, *args, **kwargs):
        B = len(batch['length'])
        batch['map'] = batch['map'].to(self.device)
        for field in batch:
            batch[field] = list(batch[field])
        batch['length'] = [each.item() for each in batch['length']]
        for field in ['category', 'location', 'bbox', 'velocity']:
            for i in range(B):
                batch[field][i] = batch[field][i][:batch['length'][i]]

        if self.state == 'train':
            batch, gt = self.train_process(batch, *args, **kwargs)
        else:
            batch, gt = self.test_process(batch, *args, **kwargs)

        for field in ['category', 'location', 'bbox', 'velocity']:
            batch[field] = pad_sequence(batch[field], batch_first=True)
            batch[field] = batch[field].to(self.device)
        lengths = torch.tensor(batch['length']).to(self.device)
        del batch['length']
        batch['map'] = torch.stack(batch['map'], dim=0)
        for field in ['category', 'location', 'bbox', 'velocity']:
            gt[field] = torch.stack(gt[field], dim=0)
            gt[field] = gt[field].to(self.device)
        return batch, lengths, gt
    
    def train_process(self, batch, window_size=1):
        if self.scheduler is not None:
            window_size = self.scheduler(self.train_iters)
        batch = self._random_rotate(batch)
        batch = self._sort_obj(batch)
        batch, gt = self._random_masking(batch, window_size=window_size)
        batch = self._rasterize_object(batch)
        self.train_iters += 1
        return batch, gt

    def test_process(self, batch, n_keep=0):
        batch = self._random_rotate(batch, rotate=False)
        batch = self._sort_obj(batch)
        batch, gt = self._random_masking(batch, n_keep=n_keep)
        batch = self._rasterize_object(batch)
        return batch, gt

    def _random_rotate(self, batch, rotate=True):
        B = len(batch['length'])
        for i in range(B):
            deg = rad = rotation_mat = None
            if rotate:
                deg = np.random.rand() * 360
                rad = deg / 180 * np.pi
                rotation_mat = torch.tensor([[np.cos(rad), np.sin(rad)],
                                             [-np.sin(rad), np.cos(rad)]],
                                            dtype=torch.float32)
                batch['map'][i] = F.rotate(batch['map'][i], deg)
                batch['map'][i][6] += rad
            # drivable_area, ped_crossing, walkway, carpark_area, lane, lane_divider, orientation
            drivable_area = batch['map'][i][0]
            ped_crossing = batch['map'][i][1]
            walkway = batch['map'][i][2]
            carpark_area = batch['map'][i][3]
            lane = batch['map'][i][4]
            lane_divider = batch['map'][i][5]
            orientation = batch['map'][i][6]
            batch['map'][i] = torch.stack([
                drivable_area,
                ped_crossing,
                walkway,
                carpark_area,
                lane,
                lane_divider,
                torch.sin(orientation) * lane,
                torch.cos(orientation) * lane
            ], dim=0)
            if rotate:
                batch['location'][i] = batch['location'][i] @ rotation_mat
                batch['bbox'][i][:, -1] += rad
                batch['velocity'][i][:, -1] += rad
                # filter out objects fallen outside the image
                mask = (batch['location'][i][:, 0] > -self.axes_limit) & \
                       (batch['location'][i][:, 0] < self.axes_limit) & \
                       (batch['location'][i][:, 1] > -self.axes_limit) & \
                       (batch['location'][i][:, 1] < self.axes_limit)
                for field in ['category', 'location', 'bbox', 'velocity']:
                    batch[field][i] = batch[field][i][mask]
                batch['length'][i] = len(batch['category'][i])
        return batch

    def _sort_obj(self, batch):
        B = len(batch['length'])
        for i in range(B):
            L = len(batch['category'][i])
            idx = list(range(L - 1))
            idx.sort(key=lambda x: (-batch['location'][i][x, 1], batch['location'][i][x, 0]))
            idx.append(L - 1)  # append index for end token
            for field in ['category', 'location', 'bbox', 'velocity']:
                batch[field][i] = batch[field][i][idx]
        return batch

    def _random_masking(self, batch, window_size=1, n_keep='random'):
        B = len(batch['length'])
        window_size = min(window_size, min(batch['length']))
        if n_keep == 'random':
            keep_lengths = [np.random.randint(0, length - window_size + 1) for length in batch['length']]
        elif n_keep == -1:
            keep_lengths = [length - 1 for length in batch['length']]
        else:
            keep_lengths = [min(n_keep, length) for length in batch['length']]
        fields = ['category', 'location', 'bbox', 'velocity']
        gt = {field: [] for field in fields}
        for i in range(B):
            length = keep_lengths[i]
            batch['length'][i] = length
            for field in fields:
                all = batch[field][i]
                batch[field][i] = all[:length]
                gt[field].append(all[length:length + window_size])
        return batch, gt

    def _rasterize_object(self, batch):
        B = len(batch['length'])
        for i in range(B):
            category = batch['category'][i].numpy()
            location = batch['location'][i].numpy()
            bbox = batch['bbox'][i].numpy()
            velocity = batch['velocity'][i].numpy()
            L = category.shape[0]
            object_layers = {k: None for k in [1, 2, 3]}
            for type_id in [1, 2, 3]:
                object_layers[type_id] = {k: np.zeros((self.wl, self.wl), dtype=np.float32)
                                          for k in ['occupancy', 'orientation', 'speed', 'heading']}
            for j in range(L):
                working_layers = object_layers[category[j]]
                w, l, theta = bbox[j]
                speed, heading = velocity[j]
                corners = np.array([[l / 2, w / 2],
                                    [-l / 2, w / 2],
                                    [-l / 2, -w / 2],
                                    [l / 2, -w / 2]])
                rotation = np.array([[np.cos(theta), np.sin(theta)],
                                     [-np.sin(theta), np.cos(theta)]])
                corners = np.dot(corners, rotation) + location[j]
                corners[:, 0] = corners[:, 0] + self.axes_limit
                corners[:, 1] = self.axes_limit - corners[:, 1]
                corners = np.floor(corners / self.resolution).astype(int)
                cv2.fillConvexPoly(working_layers['occupancy'], corners, 1)
                row = int((self.axes_limit - location[j, 1]) / self.resolution)
                col = int((location[j, 0] + self.axes_limit) / self.resolution)
                working_layers['orientation'][row, col] = theta
                working_layers['speed'][row, col] = speed
                working_layers['heading'][row, col] = heading
            for type_id in [1, 2, 3]:
                tmp = object_layers[type_id]
                object_layers[type_id] = np.stack([
                    tmp['occupancy'],
                    np.sin(tmp['orientation']) * tmp['occupancy'],
                    np.cos(tmp['orientation']) * tmp['occupancy'],
                    tmp['speed'],
                    np.sin(tmp['heading']) * tmp['occupancy'],
                    np.cos(tmp['heading']) * tmp['occupancy']
                ], axis=0)
            object_layers = np.concatenate([object_layers[1], object_layers[2], object_layers[3]], axis=0)
            object_layers = torch.tensor(object_layers, dtype=torch.float32).to(self.device)
            batch['map'][i] = torch.cat([batch['map'][i], object_layers], dim=0)
        return batch


class DiffusionModelPreprocessor:
    """
        input_layers:
            drivable_area, ped_crossing, walkway, carpark_area, lane, lane_divider, orientation
            7 layers in total

        output_layers:
            drivable_area, ped_crossing, walkway, carpark_area, lane, lane_divider, orientation(sin), orientation(cos),
            8 layers in total
    """

    def __init__(self, device):
        self.device = device
        self.axes_limit = 40
        self.wl = 320
        self.resolution = 0.25
        self.train_iters = 0
        self.state = 'train'

    def train(self):
        self.state = 'train'
        return self

    def test(self):
        self.state = 'test'
        return self

    def __call__(self, batch):
        B = len(batch['length'])
        batch['map'] = batch['map'].to(self.device)
        for field in batch:
            batch[field] = list(batch[field])
        # remove padding
        batch['length'] = [each.item() for each in batch['length']]
        for field in ['category', 'location', 'bbox', 'velocity']:
            for i in range(B):
                batch[field][i] = batch[field][i][:batch['length'][i]]
        for name in ['pedestrian', 'bicyclist', 'vehicle']:
            batch[name] = {
                'length': [],
                'location': [],
                'bbox': [],
                'velocity': []
            }

        if self.state == 'train':
            batch = self.train_process(batch)
        else:
            batch = self.test_process(batch)

        for name in ['pedestrian', 'bicyclist', 'vehicle']:
            for field in ['location', 'bbox', 'velocity']:
                batch[name][field] = pad_sequence(batch[name][field], batch_first=True)
                batch[name][field] = batch[name][field].to(self.device)
            batch[name]['length'] = torch.tensor(batch[name]['length']).to(self.device)
            batch[name]['location'] = batch[name]['location'] / self.axes_limit
        batch['map'] = torch.stack(batch['map'], dim=0)
        return batch['pedestrian'], batch['bicyclist'], batch['vehicle'], batch['map']

    def train_process(self, batch):
        batch = self._random_rotate(batch)
        batch = self._sort_obj(batch)
        batch = self._split_obj(batch)
        self.train_iters += 1
        return batch

    def test_process(self, batch):
        batch = self._random_rotate(batch, rotate=False)
        batch = self._sort_obj(batch)
        batch = self._split_obj(batch)
        return batch

    def _random_rotate(self, batch, rotate=True):
        B = len(batch['length'])
        for i in range(B):
            deg = rad = rotation_mat = None
            if rotate:
                deg = np.random.rand() * 360
                rad = deg / 180 * np.pi
                rotation_mat = torch.tensor([[np.cos(rad), np.sin(rad)],
                                             [-np.sin(rad), np.cos(rad)]],
                                            dtype=torch.float32)
                batch['map'][i] = F.rotate(batch['map'][i], deg)
                batch['map'][i][6] += rad
            # drivable_area, ped_crossing, walkway, carpark_area, lane, lane_divider, orientation
            drivable_area = batch['map'][i][0]
            ped_crossing = batch['map'][i][1]
            walkway = batch['map'][i][2]
            carpark_area = batch['map'][i][3]
            lane = batch['map'][i][4]
            lane_divider = batch['map'][i][5]
            orientation = batch['map'][i][6]
            batch['map'][i] = torch.stack([
                drivable_area,
                ped_crossing,
                walkway,
                carpark_area,
                lane,
                lane_divider,
                torch.sin(orientation) * lane,
                torch.cos(orientation) * lane
            ], dim=0)
            if rotate:
                batch['location'][i] = batch['location'][i] @ rotation_mat
                batch['bbox'][i][:, -1] += rad
                batch['velocity'][i][:, -1] += rad
                # filter out objects fallen outside the image
                mask = (batch['location'][i][:, 0] > -self.axes_limit) & \
                       (batch['location'][i][:, 0] < self.axes_limit) & \
                       (batch['location'][i][:, 1] > -self.axes_limit) & \
                       (batch['location'][i][:, 1] < self.axes_limit)
                for field in ['category', 'location', 'bbox', 'velocity']:
                    batch[field][i] = batch[field][i][mask]
                batch['length'][i] = len(batch['category'][i])
        return batch

    def _sort_obj(self, batch):
        B = len(batch['length'])
        for i in range(B):
            L = len(batch['category'][i])
            idx = list(range(L - 1))
            idx.sort(key=lambda x: (-batch['location'][i][x, 1], batch['location'][i][x, 0]))
            idx.append(L - 1)  # append index for end token
            for field in ['category', 'location', 'bbox', 'velocity']:
                batch[field][i] = batch[field][i][idx]
        return batch

    def _split_obj(self, batch):
        B = len(batch['length'])
        mapping = {
            1: 'pedestrian',
            2: 'bicyclist',
            3: 'vehicle'
        }
        for i in range(B):
            for category in [1, 2, 3]:
                mask = (batch['category'][i] == category)
                name = mapping[category]
                batch[name]['length'].append(sum(mask).item())
                batch[name]['location'].append(batch['location'][i][mask])
                batch[name]['bbox'].append(batch['bbox'][i][mask])
                batch[name]['velocity'].append(batch['velocity'][i][mask])
        del batch['length']
        del batch['category']
        del batch['location']
        del batch['bbox']
        del batch['velocity']
        return batch
