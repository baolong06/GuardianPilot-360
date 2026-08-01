"""
Training script for traffic light classification using PIE dataset
"""
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from glob import glob
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
class ImageClassificationDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ========== MODEL ==========
class TrafficLightClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# ========== TRAINING ==========
def train():
    config = get_config_for_task('traffic_light')
    paths = get_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_dir = config['data']['img_dir']
    img_size = int(to_float(config['data']['img_size']))
    test_split = to_float(config['data']['test_split'])
    augment = config['augmentation']['use']

    img_files = glob(os.path.join(img_dir, "*.jpg"))
    if not img_files:
        print(f"❌ No images found in {img_dir}")
        return

    labels = []
    for f in img_files:
        basename = os.path.basename(f)
        state = basename.split('_')[-1].replace('.jpg', '')
        labels.append(state)

    label_map = {'red': 0, 'yellow': 1, 'green': 2}
    y = [label_map.get(l, 0) for l in labels]

    X_train, X_temp, y_train, y_temp = train_test_split(
        img_files, y, test_size=test_split, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    if augment:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_ds = ImageClassificationDataset(X_train, y_train, train_transform)
    val_ds = ImageClassificationDataset(X_val, y_val, val_transform)
    test_ds = ImageClassificationDataset(X_test, y_test, val_transform)

    train_loader = DataLoader(train_ds, batch_size=int(to_float(config['training']['batch_size'])), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(to_float(config['training']['batch_size'])), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=int(to_float(config['training']['batch_size'])), shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")

    model = TrafficLightClassifier(
        num_classes=int(to_float(config['model']['num_classes'])),
        pretrained=config['model']['pretrained']
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

    logger = Logger(paths['log_root'], 'traffic_light')
    save_dir = os.path.join(paths['model_root'], 'traffic_light', 'weights')
    os.makedirs(save_dir, exist_ok=True)

    best_acc = 0.0
    patience_counter = 0
    best_epoch = 0

    for epoch in range(int(to_float(config['training']['epochs']))):
        model.train()
        total_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), to_float(config['training']['gradient_clip']))
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
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

    # Test best model
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best.pth')))
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, pred = torch.max(outputs, 1)
            preds.extend(pred.cpu().numpy())
            truths.extend(labels.cpu().numpy())
    test_acc = accuracy_score(truths, preds)
    test_f1 = f1_score(truths, preds, average='weighted')
    print(f"\nTest results: Acc={test_acc:.4f}, F1={test_f1:.4f}")

    logger.close()
    print(f"Training complete. Best validation accuracy: {best_acc:.4f} at epoch {best_epoch+1}")

if __name__ == "__main__":
    train()