# model.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]


class TrajectoryTransformer(nn.Module):
    """
    Input: sequence of past deltas (B, n_in, 2)
    We project to d_model, add pos enc, pass through TransformerEncoder,
    then use a simple MLP decoder (autoregressive or direct) to predict m_pred displacements.
    We'll use direct multi-step decoder (non-autoregressive) for simplicity.
    """
    def __init__(self, n_in, m_pred, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.n_in = n_in
        self.m_pred = m_pred
        self.d_model = d_model
        self.input_proj = nn.Linear(2, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max(n_in, m_pred)+10)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, m_pred * 2)
        )

    def forward(self, x):
        # x: (B, n_in, 2)
        B = x.size(0)
        z = self.input_proj(x)  # (B, n_in, d_model)
        z = self.pos_enc(z)
        z = self.encoder(z)  # (B, n_in, d_model)
        z_pooled = z.permute(0,2,1)  # (B, d_model, n_in)
        z_p = self.pool(z_pooled).squeeze(-1)  # (B, d_model)
        out = self.decoder(z_p)  # (B, m_pred*2)
        out = out.view(B, self.m_pred, 2)
        return out

class TrajectoryLSTM(nn.Module):
    def __init__(self, n_in, m_pred, hidden=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, m_pred*2))

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        out = self.fc(last).view(x.size(0), -1, 2)
        return out
