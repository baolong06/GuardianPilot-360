import numpy as np
import torch
from .base import BaseDataset

class SequenceDataset(BaseDataset):
    pass

class AugmentedSequenceDataset(BaseDataset):
    def __init__(self, json_file, feature_key='features', label_key='label',
                 noise_std=0.0, scaling_range=(1.0, 1.0), rotation_deg=0.0):
        super().__init__(json_file, feature_key, label_key)
        self.noise_std = noise_std
        self.scaling_range = scaling_range
        self.rotation_deg = rotation_deg

    def __getitem__(self, idx):
        features, label = super().__getitem__(idx)
        features = features.numpy().copy()
        seq_len, dim = features.shape

        if self.noise_std > 0:
            features += np.random.normal(0, self.noise_std, features.shape)

        if self.scaling_range[0] != 1.0 or self.scaling_range[1] != 1.0:
            scale = np.random.uniform(*self.scaling_range)
            if dim >= 2:
                features[:, :2] *= scale

        if self.rotation_deg != 0:
            angle = np.radians(np.random.uniform(-self.rotation_deg, self.rotation_deg))
            cos, sin = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([[cos, -sin], [sin, cos]])
            if dim >= 2:
                features[:, :2] = np.dot(features[:, :2], rot_matrix.T)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)