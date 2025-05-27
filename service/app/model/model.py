from performer_pytorch import Performer
import torch.nn as nn
import torch


class HybridPerformerBiLSTM(nn.Module):
    def __init__(self, input_dim, horizon=6, lstm_hidden=64, lstm_layers=2,
                 performer_dim=128, performer_depth=3, performer_heads=4, dim_head=32, dropout=0.1):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, lstm_hidden, num_layers=lstm_layers,
                              batch_first=True, bidirectional=True, dropout=dropout)
        self.bilstm_fc = nn.Linear(lstm_hidden * 2, horizon)

        self.project = nn.Linear(input_dim, performer_dim)
        self.performer = Performer(
            dim=performer_dim,
            depth=performer_depth,
            heads=performer_heads,
            dim_head=dim_head,
            causal=False,
            ff_dropout=0.1,
            attn_dropout=0.1
        )
        self.performer_fc = nn.Linear(performer_dim, horizon)

        self.weight = nn.Parameter(torch.tensor(0.5))  # обучаемый вес объединения

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        lstm_last = lstm_out[:, -1, :]
        lstm_pred = self.bilstm_fc(lstm_last)  # (batch, horizon)

        proj = self.project(x)
        performer_out = self.performer(proj)
        performer_last = performer_out[:, -1, :]
        performer_pred = self.performer_fc(performer_last)  # (batch, horizon)

        w = torch.sigmoid(self.weight)
        out = w * lstm_pred + (1 - w) * performer_pred
        return out
