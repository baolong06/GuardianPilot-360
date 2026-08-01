import sys
import os
# Thêm thư mục gốc vào sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import numpy as np
from models.trajectory.model import TrajectoryLSTM
from training.trajectory.data_loader import get_dataloader
from evaluation.metrics.trajectory import compute_ade_fde, compute_miss_rate

def evaluate():
    # Load config
    config_path = os.path.join("models", "trajectory", "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = TrajectoryLSTM(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        pred_len=config["model"]["pred_len"],
        obs_len=config["model"]["obs_len"]
    )
    model_path = os.path.join(config["training"]["save_dir"], "best.pth")
    if not os.path.exists(model_path):
        print(f"❌ Model weights not found at {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load validation data
    data_dir = config["data"]["data_dir"]
    val_csv = os.path.join(data_dir, "val_trajectory.csv")
    if not os.path.exists(val_csv):
        print(f"❌ Validation CSV not found at {val_csv}")
        return

    val_loader = get_dataloader(
        val_csv,
        batch_size=config["training"]["batch_size"],
        obs_len=config["model"]["obs_len"],
        pred_len=config["model"]["pred_len"],
        shuffle=False
    )

    if len(val_loader.dataset) == 0:
        print("❌ Validation dataset is empty.")
        return

    all_ade, all_fde, all_mr = [], [], []
    with torch.no_grad():
        for obs, gt in val_loader:
            obs, gt = obs.to(device), gt.to(device)
            pred = model(obs)
            ade, fde = compute_ade_fde(pred, gt)
            mr = compute_miss_rate(pred, gt, threshold=1.0)
            all_ade.append(ade)
            all_fde.append(fde)
            all_mr.append(mr)

    final_ade = np.mean(all_ade)
    final_fde = np.mean(all_fde)
    final_mr = np.mean(all_mr)

    print("\n" + "="*50)
    print("📊 FINAL EVALUATION RESULTS")
    print("="*50)
    print(f"  ADE (Average Displacement Error):   {final_ade:.4f}")
    print(f"  FDE (Final Displacement Error):     {final_fde:.4f}")
    print(f"  Miss Rate (threshold=1.0m):         {final_mr:.4f}")
    print("="*50)

    # Save report
    report_dir = os.path.join("evaluation", "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "trajectory_eval.txt")
    with open(report_path, "w") as f:
        f.write(f"ADE: {final_ade:.4f}\n")
        f.write(f"FDE: {final_fde:.4f}\n")
        f.write(f"Miss Rate (1.0m): {final_mr:.4f}\n")
    print(f"✅ Report saved to {report_path}")

if __name__ == "__main__":
    evaluate()