import torch
from torch.utils.data import Dataset
import json
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, json_file, feature_key='features', label_key='label'):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.feature_key = feature_key
        self.label_key = label_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = np.array(item[self.feature_key], dtype=np.float32)
        label = item[self.label_key]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)