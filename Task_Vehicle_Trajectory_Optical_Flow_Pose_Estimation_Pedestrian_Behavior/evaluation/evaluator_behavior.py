import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from models.behavior_clf.model import BehaviorGRU
from training.behavior_clf.data_loader import get_dataloader

def evaluate_behavior():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model_path = "models/behavior_clf/weights/best.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    state_dict = torch.load(model_path, map_location=device)
    num_layers = 1
    for k in state_dict.keys():
        if k.startswith("gru.weight_ih_l"):
            try:
                layer_idx = int(k.split("l")[1]) + 1
                num_layers = max(num_layers, layer_idx)
            except ValueError:
                pass

    print(f"Detected BehaviorGRU num_layers={num_layers} from checkpoint.")
    model = BehaviorGRU(input_dim=4, hidden_dim=128, num_layers=num_layers, num_classes=4)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load validation data
    val_loader = get_dataloader("data/processed/behavior/val.json", batch_size=64, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print("\n" + "="*50)
    print("📊 BEHAVIOR EVALUATION RESULTS")
    print("="*50)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['stop', 'straight', 'turn_left', 'turn_right']))
    print("\n🔢 Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("="*50)

if __name__ == "__main__":
    evaluate_behavior()