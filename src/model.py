import torch
import torch.nn as nn
from src.config import N_WINDOW, HIDDEN_SIZE, NUM_LSTM_LAYERS, DROPOUT


class CandleHLPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LSTM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LSTM_LAYERS > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE + 1, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE // 2, 2),
        )

    def forward(self, hist_seq, cur_close):
        _, (hn, _) = self.lstm(hist_seq)
        lstm_out = hn[-1]
        combined = torch.cat([lstm_out, cur_close], dim=1)
        return self.head(combined)
