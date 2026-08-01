import torch
import numpy as np

def compute_ade_fde(pred, gt):
    dist = torch.norm(pred - gt, dim=-1)
    ade = dist.mean(dim=1)
    fde = dist[:, -1]
    return ade.mean().item(), fde.mean().item()

def compute_miss_rate(pred, gt, threshold=1.0):
    fde = torch.norm(pred[:, -1, :] - gt[:, -1, :], dim=-1)
    miss = (fde > threshold).float().mean().item()
    return miss