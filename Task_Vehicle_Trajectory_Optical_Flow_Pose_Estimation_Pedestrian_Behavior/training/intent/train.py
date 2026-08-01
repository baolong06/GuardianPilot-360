"""
Training script for pedestrian crossing intention prediction (PIE dataset).

Reads config from: training/configs/config.yaml → section 'intent'
Data expected at: data/processed/pie_intention/train.json, val.json

Sample JSON record:
    {
        "ped_id": "1_1_7",
        "obs": [[cx, cy, speed, heading, looking_flag, crossing_flag], ...],  # (obs_len, 6)
        "label": 0 or 1,   # 0=not crossing, 1=will cross
        "intention_prob": 0.867,
        "time_to_event": 45
    }

CRITICAL:
    - obs window phải kết thúc TRƯỚC critical_point của pedestrian.
    - Không để frame sau critical_point lọt vào obs (data leakage).
    - Label = 1 nếu intention_prob >= threshold (mặc định 0.5).
    - Class imbalance thường xảy ra: dùng class_weight trong CrossEntropyLoss.
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import json
import numpy as np
import yaml

# ========== CONFIG ==========
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'training', 'configs', 'config.yaml')


def to_float(val):
    if isinstance(val, str):
        try:
            return float(val)
        except Exception:
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
        except Exception:
            self.writer = None
            print("TensorBoard not available, logging only to CSV")

    def _init_csv(self):
        import csv
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_acc', 'val_f1', 'val_auc', 'lr'])

    def log_epoch(self, epoch, train_loss, val_acc, val_f1, val_auc, lr):
        import csv
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_acc, val_f1, val_auc, lr])
        if self.writer:
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Metric/val_acc', val_acc, epoch)
            self.writer.add_scalar('Metric/val_f1', val_f1, epoch)
            self.writer.add_scalar('Metric/val_auc', val_auc, epoch)
            self.writer.add_scalar('LR', lr, epoch)

    def close(self):
        if self.writer:
            self.writer.close()


# ========== DATASET ==========
class IntentionDataset(Dataset):
    """Dataset cho intention prediction.

    Mỗi sample là dict với:
        obs   : list of list, shape (obs_len, input_dim=6)
        label : int, 0 hoặc 1
    """

    def __init__(self, json_file: str, noise_std: float = 0.0):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.noise_std = noise_std
        self._validate()

    def _validate(self):
        valid = []
        for item in self.data:
            if 'obs' not in item or 'label' not in item:
                continue
            obs = np.array(item['obs'])
            if obs.ndim != 2 or obs.shape[1] < 4:
                continue
            if item['label'] not in (0, 1):
                continue
            valid.append(item)
        n_removed = len(self.data) - len(valid)
        if n_removed > 0:
            print(f"[IntentionDataset] Removed {n_removed} invalid samples")
        self.data = valid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        obs = np.array(item['obs'], dtype=np.float32)  # (obs_len, input_dim)
        label = int(item['label'])

        # Pad input_dim đến 6 nếu thiếu (backward compat)
        if obs.shape[1] < 6:
            pad = np.zeros((obs.shape[0], 6 - obs.shape[1]), dtype=np.float32)
            obs = np.concatenate([obs, pad], axis=1)

        if self.noise_std > 0:
            obs += np.random.normal(0, self.noise_std, obs.shape).astype(np.float32)

        return torch.tensor(obs, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ========== MODEL (import từ models/intent/model.py) ==========
from models.intent.model import IntentionGRU


# ========== TRAINING ==========
def train():
    # Try to read from config, fall back to defaults nếu chưa có section 'intent'
    config = get_config_for_task('intent')
    paths = get_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Data paths ===
    data_cfg = config.get('data', {})
    train_json = data_cfg.get('train_json', 'data/processed/pie_intention/train.json')
    val_json = data_cfg.get('val_json', 'data/processed/pie_intention/val.json')

    if not os.path.exists(train_json):
        print(f"ERROR: train_json not found: {train_json}")
        print("Hãy chạy scripts/parse_pie.py trước để tạo dữ liệu intention.")
        return
    if not os.path.exists(val_json):
        print(f"ERROR: val_json not found: {val_json}")
        return

    # === Dataset ===
    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})

    train_ds = IntentionDataset(train_json, noise_std=to_float(config.get('augmentation', {}).get('noise_std', 0.005)))
    val_ds = IntentionDataset(val_json, noise_std=0.0)
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("ERROR: Dataset rỗng. Kiểm tra file JSON.")
        return

    # === Class weights để handle imbalance ===
    labels = [item['label'] for item in train_ds.data]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos > 0 and n_neg > 0:
        weight_neg = n_pos / len(labels)
        weight_pos = n_neg / len(labels)
        class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float32).to(device)
        print(f"Class weights: neg={weight_neg:.3f}, pos={weight_pos:.3f} (pos={n_pos}, neg={n_neg})")
    else:
        class_weights = None

    batch_size = int(to_float(train_cfg.get('batch_size', 64)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # === Model ===
    model = IntentionGRU(
        input_dim=int(to_float(model_cfg.get('input_dim', 6))),
        hidden_dim=int(to_float(model_cfg.get('hidden_dim', 128))),
        num_layers=int(to_float(model_cfg.get('num_layers', 2))),
        num_classes=2,
        dropout=to_float(model_cfg.get('dropout', 0.3)),
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=to_float(train_cfg.get('lr', 0.001)),
        weight_decay=to_float(train_cfg.get('weight_decay', 1e-5)),
    )
    sched_cfg = train_cfg.get('scheduler', {})
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=sched_cfg.get('mode', 'max'),
        factor=to_float(sched_cfg.get('factor', 0.5)),
        patience=int(to_float(sched_cfg.get('patience', 5))),
    )

    log_root = paths.get('log_root', 'runs')
    model_root = paths.get('model_root', 'models')
    logger_obj = Logger(log_root, 'intent')
    save_dir = os.path.join(model_root, 'intent', 'weights')
    os.makedirs(save_dir, exist_ok=True)

    best_f1 = 0.0
    patience_counter = 0
    max_patience = int(to_float(train_cfg.get('patience', 10)))
    epochs = int(to_float(train_cfg.get('epochs', 50)))

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for obs, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            obs, labels_batch = obs.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(obs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            grad_clip = to_float(train_cfg.get('gradient_clip', 1.0))
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        preds_list, truths_list, probs_list = [], [], []
        with torch.no_grad():
            for obs, labels_batch in val_loader:
                obs = obs.to(device)
                outputs = model(obs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                preds_list.extend(preds.tolist())
                truths_list.extend(labels_batch.numpy().tolist())
                probs_list.extend(probs[:, 1].tolist())  # prob của class 1 (will cross)

        acc = accuracy_score(truths_list, preds_list)
        f1 = f1_score(truths_list, preds_list, average='weighted', zero_division=0)
        try:
            auc = roc_auc_score(truths_list, probs_list)
        except Exception:
            auc = 0.0

        lr = optimizer.param_groups[0]['lr']
        logger_obj.log_epoch(epoch, avg_loss, acc, f1, auc, lr)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, LR={lr:.6f}")

        scheduler.step(f1)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
            print(f"[+] Best model saved (F1={f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    logger_obj.close()
    print(f"Training complete. Best weighted F1: {best_f1:.4f}")


if __name__ == "__main__":
    train()
