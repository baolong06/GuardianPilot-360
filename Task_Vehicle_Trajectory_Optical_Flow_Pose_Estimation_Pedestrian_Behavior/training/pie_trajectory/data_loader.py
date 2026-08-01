"""
Data loader for PIE trajectory dataset (using JSON files)
"""
import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np

class PieTrajectoryDataset(Dataset):
    def __init__(self, json_file, obs_len=10, pred_len=12, norm_params=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.norm_params = norm_params

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        obs = np.array(item['obs'], dtype=np.float32)   # (obs_len, 2)
        pred = np.array(item['pred'], dtype=np.float32) # (pred_len, 2)

        if self.norm_params is not None:
            x_mean, x_std = self.norm_params['x_mean'], self.norm_params['x_std']
            y_mean, y_std = self.norm_params['y_mean'], self.norm_params['y_std']
            obs[:, 0] = (obs[:, 0] - x_mean) / (x_std + 1e-8)
            obs[:, 1] = (obs[:, 1] - y_mean) / (y_std + 1e-8)
            pred[:, 0] = (pred[:, 0] - x_mean) / (x_std + 1e-8)
            pred[:, 1] = (pred[:, 1] - y_mean) / (y_std + 1e-8)

        # Thêm flow = 0 để thành 4 chiều (phù hợp với input của LSTM)
        obs_4d = np.concatenate([obs, np.zeros((obs.shape[0], 2))], axis=1)
        return torch.tensor(obs_4d, dtype=torch.float32), torch.tensor(pred, dtype=torch.float32)

def compute_normalization_params(json_file):
    """Tính mean và std từ tất cả các điểm trong dataset (từ obs)"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    all_coords = []
    for item in data:
        obs = np.array(item['obs'])  # (obs_len, 2)
        all_coords.append(obs)
    all_coords = np.concatenate(all_coords, axis=0)
    x_mean = np.mean(all_coords[:, 0])
    x_std = np.std(all_coords[:, 0])
    y_mean = np.mean(all_coords[:, 1])
    y_std = np.std(all_coords[:, 1])
    return {'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}

def get_dataloader(json_file, batch_size=64, obs_len=10, pred_len=12, shuffle=True, norm_params=None):
    dataset = PieTrajectoryDataset(json_file, obs_len, pred_len, norm_params)
    if len(dataset) == 0:
        print(f"⚠️ Dataset empty for {json_file}")
        return None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)