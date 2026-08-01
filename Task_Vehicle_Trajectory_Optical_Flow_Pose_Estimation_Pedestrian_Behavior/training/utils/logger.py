import os
import csv
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_root, task_name):
        self.log_dir = os.path.join(log_root, task_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        self.csv_path = os.path.join(self.log_dir, 'metrics.csv')
        self._init_csv()

    def _init_csv(self):
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_metric', 'lr'])

    def log_epoch(self, epoch, train_loss, val_metric, lr, metric_name='accuracy'):
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar(f'Metric/val', val_metric, epoch)
        self.writer.add_scalar('LR', lr, epoch)
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_metric, lr])

    def close(self):
        self.writer.close()