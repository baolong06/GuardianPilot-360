import torch
import torch.nn as nn

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, pred_len=3, obs_len=3):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Chỉ output (x, y) – 2 chiều
        self.fc_out = nn.Linear(hidden_dim, 2)

    def forward(self, obs_traj):
        """
        obs_traj: (batch, obs_len, input_dim) = (batch, obs_len, 4)
        """
        batch_size = obs_traj.size(0)
        # Encode
        _, (h_n, c_n) = self.encoder(obs_traj)

        # Decoder: bắt đầu từ frame cuối cùng của obs
        decoder_input = obs_traj[:, -1:, :]  # (batch, 1, 4)
        preds = []
        for _ in range(self.pred_len):
            decoder_output, (h_n, c_n) = self.decoder(decoder_input, (h_n, c_n))
            pred = self.fc_out(decoder_output)  # (batch, 1, 2)
            preds.append(pred)
            # Tạo input tiếp theo: ghép pred với flow = 0 (vì chỉ dự đoán vị trí)
            pred_with_flow = torch.cat([pred, torch.zeros_like(pred)], dim=-1)  # (batch, 1, 4)
            decoder_input = pred_with_flow

        preds = torch.cat(preds, dim=1)  # (batch, pred_len, 2)
        return preds

class SocialLSTM(TrajectoryLSTM):
    """Phiên bản có tương tác xã hội (đơn giản)"""
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, pred_len=3, obs_len=3, neighbor_dim=4):
        super().__init__(input_dim, hidden_dim, num_layers, pred_len, obs_len)
        self.neighbor_dim = neighbor_dim
        self.encoder = nn.LSTM(input_dim + neighbor_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_dim + neighbor_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 2)

    def forward(self, obs_traj, neighbor_features=None):
        if neighbor_features is None:
            neighbor_features = torch.zeros(obs_traj.size(0), obs_traj.size(1), self.neighbor_dim).to(obs_traj.device)
        encoder_input = torch.cat([obs_traj, neighbor_features], dim=-1)
        # Sau đó gọi forward của lớp cha, nhưng cần override để dùng encoder/decoder đã thay
        # Đơn giản: gọi trực tiếp logic của forward
        batch_size = obs_traj.size(0)
        _, (h_n, c_n) = self.encoder(encoder_input)
        decoder_input = torch.cat([obs_traj[:, -1:, :], neighbor_features[:, -1:, :]], dim=-1)
        preds = []
        for _ in range(self.pred_len):
            decoder_output, (h_n, c_n) = self.decoder(decoder_input, (h_n, c_n))
            pred = self.fc_out(decoder_output)
            preds.append(pred)
            # Tạo input tiếp theo: concat pred với zeros cho flow và neighbor
            pred_with_flow = torch.cat([pred, torch.zeros_like(pred)], dim=-1)  # (batch,1,4)
            # Giả sử neighbor_features không đổi (hoặc bằng 0)
            zero_neighbor = torch.zeros(obs_traj.size(0), 1, self.neighbor_dim).to(obs_traj.device)
            decoder_input = torch.cat([pred_with_flow, zero_neighbor], dim=-1)
        preds = torch.cat(preds, dim=1)
        return preds