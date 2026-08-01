"""
Training script for pedestrian behavior classification (PIE)
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
from sklearn.metrics import accuracy_score, f1_score
import json
import numpy as np
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

    def log_epoch(self, epoch, train_loss, val_metric, lr, metric_name='accuracy'):
        import csv
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_metric, lr])
        if self.writer:
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar(f'Metric/val', val_metric, epoch)
            self.writer.add_scalar('LR', lr, epoch)

    def close(self):
        if self.writer:
            self.writer.close()

# ========== DATASET ==========
class AugmentedSequenceDataset(Dataset):
    def __init__(self, json_file, noise_std=0.0, scaling_range=(1.0, 1.0), rotation_deg=0.0):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.noise_std = noise_std
        self.scaling_range = scaling_range
        self.rotation_deg = rotation_deg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = np.array(item['features'], dtype=np.float32)
        label = item['label']

        if self.noise_std > 0:
            features += np.random.normal(0, self.noise_std, features.shape)

        if self.scaling_range[0] != 1.0 or self.scaling_range[1] != 1.0:
            scale = np.random.uniform(*self.scaling_range)
            if features.shape[1] >= 2:
                features[:, :2] *= scale

        if self.rotation_deg != 0:
            angle = np.radians(np.random.uniform(-self.rotation_deg, self.rotation_deg))
            cos, sin = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([[cos, -sin], [sin, cos]])
            if features.shape[1] >= 2:
                features[:, :2] = np.dot(features[:, :2], rot_matrix.T)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ========== MODEL ==========
class PedestrianBehaviorGRU(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

# ========== TRAINING ==========
def train():
    config = get_config_for_task('pedestrian_behavior')
    paths = get_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = AugmentedSequenceDataset(
        config['data']['train_json'],
        noise_std=to_float(config.get('augmentation', {}).get('noise_std', 0.005)),
        scaling_range=[to_float(x) for x in config.get('augmentation', {}).get('scaling_range', (0.95, 1.05))],
        rotation_deg=to_float(config.get('augmentation', {}).get('rotation_range', 2))
    )
    val_ds = AugmentedSequenceDataset(
        config['data']['val_json'],
        noise_std=0.0, scaling_range=(1.0, 1.0), rotation_deg=0.0
    )

    train_loader = DataLoader(train_ds, batch_size=int(to_float(config['training']['batch_size'])), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(to_float(config['training']['batch_size'])), shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    model = PedestrianBehaviorGRU(
        input_dim=int(to_float(config['model']['input_dim'])),
        hidden_dim=int(to_float(config['model']['hidden_dim'])),
        num_layers=int(to_float(config['model']['num_layers'])),
        num_classes=int(to_float(config['model']['num_classes'])),
        dropout=to_float(config['model']['dropout'])
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=to_float(config['training']['lr']),
        weight_decay=to_float(config['training']['weight_decay'])
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config['training']['scheduler']['mode'],
        factor=to_float(config['training']['scheduler']['factor']),
        patience=int(to_float(config['training']['scheduler']['patience']))
    )

    logger = Logger(paths['log_root'], 'pedestrian_behavior')
    save_dir = os.path.join(paths['model_root'], 'pedestrian_behavior', 'weights')
    os.makedirs(save_dir, exist_ok=True)

    best_acc = 0.0
    patience_counter = 0
    best_epoch = 0

    for epoch in range(int(to_float(config['training']['epochs']))):
        model.train()
        total_loss = 0.0
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), to_float(config['training']['gradient_clip']))
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, pred = torch.max(outputs, 1)
                preds.extend(pred.cpu().numpy())
                truths.extend(labels.cpu().numpy())

        acc = accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average='weighted')
        lr = optimizer.param_groups[0]['lr']

        logger.log_epoch(epoch, avg_loss, acc, lr)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={acc:.4f}, F1={f1:.4f}, LR={lr:.6f}")

        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
            print(f"✅ Best model saved (Acc={acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= int(to_float(config['training']['patience'])):
            print(f"Early stopping at epoch {epoch+1}")
            break

    logger.close()
    print(f"Training complete. Best accuracy: {best_acc:.4f} at epoch {best_epoch+1}")

if __name__ == "__main__":
    train()