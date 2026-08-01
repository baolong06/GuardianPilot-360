import numpy as np

def compute_ade_fde(pred, target):
    batch_ade = []
    batch_fde = []
    for b in range(pred.size(0)):
        p = pred[b].cpu().numpy()
        t = target[b].cpu().numpy()
        batch_ade.append(np.mean(np.linalg.norm(p - t, axis=1)))
        batch_fde.append(np.linalg.norm(p[-1] - t[-1]))
    return np.mean(batch_ade), np.mean(batch_fde)