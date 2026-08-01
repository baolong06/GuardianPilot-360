"""
Training script for trajectory prediction (vehicle & pedestrian) using PIE dataset
"""
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import json
import yaml

# ========== CONFIG ==========
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'training', 'configs', 'config.yaml')

def to_float(val):
    if isinstance(val, str):
        try:
            return float(val)
        except:
            return val
    return val

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def get_config_for_task(task_name):
    config = load_config()
    return config.get(task_name, {})

def get_paths():
    config = load_config()
    return config.get('paths', {})

# ========== LOGGER ==========
class Logger:
    def __init__(self, log_root, task_name):
        self.log_dir = os.path.join(log_root, task_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, 'metrics.csv')
        self._init_csv()
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
        except:
            self.writer = None
            print("TensorBoard not available, logging only to CSV")

    def _init_csv(self):
        import csv
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_metric', 'lr'])

    def log_epoch(self, epoch, train_loss, val_metric, lr, metric_name='ade'):
        import csv
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_metric, lr])
        if self.writer:
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar(f'Metric/val_{metric_name}', val_metric, epoch)
            self.writer.add_scalar('LR', lr, epoch)

    def close(self):
        if self.writer:
            self.writer.close()

# ========== DATASET ==========
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
        obs = np.array(item['obs'], dtype=np.float32)
        pred = np.array(item['pred'], dtype=np.float32)

        if self.norm_params is not None:
            x_mean = to_float(self.norm_params['x_mean'])
            x_std = to_float(self.norm_params['x_std'])
            y_mean = to_float(self.norm_params['y_mean'])
            y_std = to_float(self.norm_params['y_std'])
            obs[:, 0] = (obs[:, 0] - x_mean) / (x_std + 1e-8)
            obs[:, 1] = (obs[:, 1] - y_mean) / (y_std + 1e-8)
            pred[:, 0] = (pred[:, 0] - x_mean) / (x_std + 1e-8)
            pred[:, 1] = (pred[:, 1] - y_mean) / (y_std + 1e-8)

        obs_4d = np.concatenate([obs, np.zeros((obs.shape[0], 2))], axis=1)
        return torch.tensor(obs_4d, dtype=torch.float32), torch.tensor(pred, dtype=torch.float32)

# ========== MODEL ==========
class TrajectoryLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, pred_len=12, obs_len=10):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 2)

    def forward(self, obs_traj):
        batch_size = obs_traj.size(0)
        _, (h_n, c_n) = self.encoder(obs_traj)
        decoder_input = obs_traj[:, -1:, :]
        preds = []
        for _ in range(self.pred_len):
            decoder_output, (h_n, c_n) = self.decoder(decoder_input, (h_n, c_n))
            pred = self.fc_out(decoder_output)
            preds.append(pred)
            pred_with_flow = torch.cat([pred, torch.zeros_like(pred)], dim=-1)
            decoder_input = pred_with_flow
        preds = torch.cat(preds, dim=1)
        return preds

def compute_ade_fde(pred, target):
    batch_ade = []
    batch_fde = []
    for b in range(pred.size(0)):
        p = pred[b].cpu().numpy()
        t = target[b].cpu().numpy()
        batch_ade.append(np.mean(np.linalg.norm(p - t, axis=1)))
        batch_fde.append(np.linalg.norm(p[-1] - t[-1]))
    return np.mean(batch_ade), np.mean(batch_fde)

def compute_normalization_params(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    all_coords = []
    for item in data:
        obs = np.array(item['obs'])
        all_coords.append(obs)
    all_coords = np.concatenate(all_coords, axis=0)
    return {
        'x_mean': float(np.mean(all_coords[:, 0])),
        'x_std': float(np.std(all_coords[:, 0])),
        'y_mean': float(np.mean(all_coords[:, 1])),
        'y_std': float(np.std(all_coords[:, 1]))
    }

def get_dataloader(json_file, batch_size=64, obs_len=10, pred_len=12, shuffle=True, norm_params=None):
    dataset = PieTrajectoryDataset(json_file, obs_len, pred_len, norm_params)
    if len(dataset) == 0:
        print(f"⚠️ Dataset empty for {json_file}")
        return None
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=0,       # Windows: num_workers>0 gây MemoryError khi pickle dataset lớn
        pin_memory=False,
    )

# ========== TRAINING ==========
def train_for_type(obj_type):
    config = get_config_for_task('pie_trajectory')
    paths = get_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚗 Training {obj_type} trajectory model...")

    data_cfg = config[obj_type]['data']
    model_cfg = config[obj_type]['model']
    train_cfg = config[obj_type]['training']

    norm_params = compute_normalization_params(data_cfg['train_json'])
    print(f"📊 Normalization: x_mean={norm_params['x_mean']:.2f}, x_std={norm_params['x_std']:.2f}, "
          f"y_mean={norm_params['y_mean']:.2f}, y_std={norm_params['y_std']:.2f}")

    train_loader = get_dataloader(
        data_cfg['train_json'], batch_size=int(to_float(train_cfg['batch_size'])),
        obs_len=int(to_float(model_cfg['obs_len'])), pred_len=int(to_float(model_cfg['pred_len'])),
        norm_params=norm_params
    )
    val_loader = get_dataloader(
        data_cfg['val_json'], batch_size=int(to_float(train_cfg['batch_size'])),
        obs_len=int(to_float(model_cfg['obs_len'])), pred_len=int(to_float(model_cfg['pred_len'])),
        shuffle=False, norm_params=norm_params
    )

    if train_loader is None or val_loader is None:
        print(f"⚠️ No data for {obj_type}, skipping.")
        return

    model = TrajectoryLSTM(
        input_dim=int(to_float(model_cfg['input_dim'])),
        hidden_dim=int(to_float(model_cfg['hidden_dim'])),
        num_layers=int(to_float(model_cfg['num_layers'])),
        pred_len=int(to_float(model_cfg['pred_len'])),
        obs_len=int(to_float(model_cfg['obs_len']))
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=to_float(train_cfg['lr']),
        weight_decay=to_float(train_cfg['weight_decay'])
    )
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=train_cfg['scheduler']['mode'],
        factor=to_float(train_cfg['scheduler']['factor']),
        patience=int(to_float(train_cfg['scheduler']['patience']))
    )

    logger = Logger(paths['log_root'], f'pie_trajectory_{obj_type}')
    save_dir = os.path.join(paths['model_root'], 'pie_trajectory', obj_type, 'weights')
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'norm_params.json'), 'w') as f:
        json.dump(norm_params, f)

    best_val_ade = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(int(to_float(train_cfg['epochs']))):
        model.train()
        total_loss = 0.0
        for obs, pred in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            obs, pred = obs.to(device), pred.to(device)
            optimizer.zero_grad()
            output = model(obs)
            loss = criterion(output, pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), to_float(train_cfg['gradient_clip']))
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        all_ade, all_fde = [], []
        with torch.no_grad():
            for obs, pred in val_loader:
                obs, pred = obs.to(device), pred.to(device)
                output = model(obs)
                ade, fde = compute_ade_fde(output, pred)
                all_ade.append(ade)
                all_fde.append(fde)

        val_ade = np.mean(all_ade)
        val_fde = np.mean(all_fde)
        lr = optimizer.param_groups[0]['lr']

        logger.log_epoch(epoch, avg_loss, val_ade, lr, metric_name='ADE')
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val ADE={val_ade:.4f}, Val FDE={val_fde:.4f}, LR={lr:.6f}")

        scheduler.step(val_ade)

        if val_ade < best_val_ade:
            best_val_ade = val_ade
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
            print(f"✅ Best model saved (ADE={val_ade:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= int(to_float(train_cfg['patience'])):
            print(f"Early stopping at epoch {epoch+1}")
            break

    logger.close()
    print(f"Training complete. Best ADE: {best_val_ade:.4f} at epoch {best_epoch+1}")

def train():
    for obj_type in ['vehicle', 'pedestrian']:
        train_for_type(obj_type)

if __name__ == "__main__":
    train()