from torch import nn
import torch


class HybridLSTMGRU(nn.Module):
    
    def __init__(self, input_size=1, hidden_size_lstm=128, hidden_size_gru=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size_lstm, batch_first=True)
        self.gru = nn.GRU(input_size=hidden_size_lstm, hidden_size=hidden_size_gru, batch_first=True)
        self.linear = nn.Linear(hidden_size_gru, 1)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        lstm_out, _ = self.lstm(x)  # output shape: [batch_size, seq_len, hidden_size_lstm]
        gru_out, _ = self.gru(lstm_out)  # output shape: [batch_size, seq_len, hidden_size_gru]
        out = self.linear(gru_out[:, -1, :])  # take output at last time step
        return out
 