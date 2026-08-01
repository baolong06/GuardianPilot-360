"""
Data loader cho pedestrian crossing intention dataset (PIE).

JSON schema expected (mỗi record):
    {
        "ped_id": "1_1_7",
        "obs": [[cx_norm, cy_norm, speed, heading, looking_flag, crossing_flag], ...],
        "label": 0 or 1,
        "intention_prob": 0.867,
        "time_to_event": 45,       # số frame đến crossing_point
        "set_id": "set01",
        "video_id": "video_0001"
    }

input_dim:
    0: cx_norm          (center_x / frame_width)
    1: cy_norm          (center_y / frame_height)
    2: speed            (normalized displacement magnitude)
    3: heading          (arctan2 in radians)
    4: looking_flag     (1.0 = looking, 0.0 = not-looking)
    5: crossing_flag    (1.0 = crossing, 0.0 = not-crossing)
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class IntentionDataset(Dataset):
    """PyTorch Dataset cho pedestrian intention prediction."""

    def __init__(
        self,
        json_file: str,
        obs_len: int = 16,
        input_dim: int = 6,
        noise_std: float = 0.0,
    ):
        with open(json_file, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        self.obs_len = obs_len
        self.input_dim = input_dim
        self.noise_std = noise_std
        self.data = self._validate(raw)

    def _validate(self, raw):
        valid = []
        for item in raw:
            if 'obs' not in item or 'label' not in item:
                continue
            obs = np.array(item['obs'])
            if obs.ndim != 2:
                continue
            if item['label'] not in (0, 1):
                continue
            valid.append(item)
        return valid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        obs = np.array(item['obs'], dtype=np.float32)  # (seq_len, >=4)
        label = int(item['label'])

        # Cắt hoặc pad đến obs_len
        if obs.shape[0] > self.obs_len:
            obs = obs[-self.obs_len:]            # lấy obs_len frame cuối
        elif obs.shape[0] < self.obs_len:
            pad = np.zeros((self.obs_len - obs.shape[0], obs.shape[1]), dtype=np.float32)
            obs = np.concatenate([pad, obs], axis=0)

        # Đảm bảo input_dim = 6
        if obs.shape[1] < self.input_dim:
            pad = np.zeros((obs.shape[0], self.input_dim - obs.shape[1]), dtype=np.float32)
            obs = np.concatenate([obs, pad], axis=1)
        elif obs.shape[1] > self.input_dim:
            obs = obs[:, :self.input_dim]

        if self.noise_std > 0:
            obs[:, :4] += np.random.normal(0, self.noise_std, (obs.shape[0], 4)).astype(np.float32)

        return torch.tensor(obs, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def compute_class_weights(dataset: IntentionDataset):
    """Tính class weight để handle class imbalance."""
    labels = [item['label'] for item in dataset.data]
    n_total = len(labels)
    n_pos = sum(labels)
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    w_neg = n_pos / n_total
    w_pos = n_neg / n_total
    return torch.tensor([w_neg, w_pos], dtype=torch.float32)


def get_dataloader(
    json_file: str,
    batch_size: int = 64,
    obs_len: int = 16,
    input_dim: int = 6,
    shuffle: bool = True,
    noise_std: float = 0.0,
    num_workers: int = 0,  # Windows: luôn dùng 0 tránh MemoryError khi pickle dataset
) -> DataLoader:
    """Factory function cho IntentionDataset → DataLoader."""
    dataset = IntentionDataset(json_file, obs_len=obs_len, input_dim=input_dim, noise_std=noise_std)
    if len(dataset) == 0:
        print(f"WARNING: Dataset empty for {json_file}")
        return None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
