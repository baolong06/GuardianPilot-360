import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

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

def load_traffic_light_data(img_dir, test_split=0.2, img_size=128, augment=True):
    img_files = glob(os.path.join(img_dir, "*.jpg"))
    if not img_files:
        raise FileNotFoundError(f"No images found in {img_dir}")

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

    return train_ds, val_ds, test_ds