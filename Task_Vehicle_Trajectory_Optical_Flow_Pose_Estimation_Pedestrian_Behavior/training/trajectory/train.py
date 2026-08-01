import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
import numpy as np
from models.trajectory.model import TrajectoryLSTM
from training.trajectory.data_loader import get_dataloader
from evaluation.metrics.trajectory import compute_ade_fde

def train():
    # Load config
    config_path = os.path.join("models", "trajectory", "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    data_dir = config["data"]["data_dir"]
    train_csv = os.path.join(data_dir, "train_trajectory.csv")
    val_csv = os.path.join(data_dir, "val_trajectory.csv")

    train_loader = get_dataloader(train_csv, config["training"]["batch_size"],
                                  config["model"]["obs_len"], config["model"]["pred_len"], shuffle=True)
    val_loader = get_dataloader(val_csv, config["training"]["batch_size"],
                                config["model"]["obs_len"], config["model"]["pred_len"], shuffle=False)

    # Model
    model = TrajectoryLSTM(input_dim=config["model"]["input_dim"],
                           hidden_dim=config["model"]["hidden_dim"],
                           num_layers=config["model"]["num_layers"],
                           pred_len=config["model"]["pred_len"],
                           obs_len=config["model"]["obs_len"])
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = torch.nn.MSELoss()

    writer = SummaryWriter("runs/trajectory")
    best_val_ade = float("inf")
    save_dir = config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        total_loss = 0
        for obs, pred in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}"):
            obs, pred = obs.to(device), pred.to(device)
            optimizer.zero_grad()
            output = model(obs)
            loss = criterion(output, pred)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)

        # Validation
        model.eval()
        all_ade = []
        all_fde = []
        with torch.no_grad():
            for obs, pred in val_loader:
                obs, pred = obs.to(device), pred.to(device)
                output = model(obs)
                ade, fde = compute_ade_fde(output, pred)
                all_ade.append(ade)
                all_fde.append(fde)
        val_ade = np.mean(all_ade)
        val_fde = np.mean(all_fde)
        writer.add_scalar("ADE/val", val_ade, epoch)
        writer.add_scalar("FDE/val", val_fde, epoch)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val ADE={val_ade:.4f}, Val FDE={val_fde:.4f}")

        if val_ade < best_val_ade:
            best_val_ade = val_ade
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
            print(f"   Best model saved (ADE={val_ade:.4f})")

    writer.close()
    print("Training complete.")

if __name__ == "__main__":
    train() 