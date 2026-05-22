import os
import numpy as np
import torch
from src.config import N_WINDOW, MODELS_DIR, PIP_MULTIPLIER
from src.data_loader import normalize_pips, load_raw_data, unnormalize
from src.model import CandleHLPredictor


class CandleDecorator:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CandleHLPredictor().to(self.device)

        if model_path is None:
            model_path = os.path.join(MODELS_DIR, "best_model.pt")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def _prepare_input(self, historical_candles, current_close):
        ohlc = np.array(historical_candles, dtype=np.float64)
        if ohlc.ndim != 2 or ohlc.shape[1] < 4 or len(ohlc) != N_WINDOW:
            raise ValueError(
                f"historical_candles must have shape ({N_WINDOW}, 4), got {ohlc.shape}"
            )

        norm, prev_close = normalize_pips(np.vstack([ohlc, [[0, 0, 0, current_close]]]))

        hist_seq = norm[:N_WINDOW, 1:4]
        cur_close_norm = norm[N_WINDOW, 3:4]

        hist_tensor = torch.from_numpy(hist_seq.reshape(1, N_WINDOW, 3)).float().to(self.device)
        cur_tensor = torch.from_numpy(cur_close_norm.reshape(1, 1)).float().to(self.device)

        return hist_tensor, cur_tensor, prev_close[N_WINDOW]

    @torch.no_grad()
    def predict(self, historical_candles, current_close):
        hist_tensor, cur_tensor, prev_close_val = self._prepare_input(
            historical_candles, current_close
        )
        pred = self.model(hist_tensor, cur_tensor).cpu().numpy()[0]
        high_pips, low_pips = pred[0], pred[1]

        pred_high = unnormalize(high_pips, prev_close_val)
        pred_low = unnormalize(low_pips, prev_close_val)

        open_val = prev_close_val
        close_val = current_close

        return {
            "open": round(float(open_val), 5),
            "high": round(float(max(open_val, close_val, pred_high)), 5),
            "low": round(float(min(open_val, close_val, pred_low)), 5),
            "close": round(float(close_val), 5),
        }


def load_historical_window(csv_path, start_idx):
    ohlc = load_raw_data(csv_path)
    window = ohlc[start_idx - N_WINDOW:start_idx]
    current_close = ohlc[start_idx, 3]
    actual_ohlc = {
        "open": round(float(ohlc[start_idx, 0]), 5),
        "high": round(float(ohlc[start_idx, 1]), 5),
        "low": round(float(ohlc[start_idx, 2]), 5),
        "close": round(float(ohlc[start_idx, 3]), 5),
    }
    return window, current_close, actual_ohlc


def evaluate_on_data(csv_path, num_samples=50, model_path=None):
    decorator = CandleDecorator(model_path)
    ohlc = load_raw_data(csv_path)

    errors_high = []
    errors_low = []

    for i in range(N_WINDOW, min(N_WINDOW + num_samples, len(ohlc))):
        historical = ohlc[i - N_WINDOW:i]
        current_close = ohlc[i, 3]
        actual = {
            "open": ohlc[i, 0],
            "high": ohlc[i, 1],
            "low": ohlc[i, 2],
            "close": ohlc[i, 3],
        }
        pred = decorator.predict(historical, current_close)

        errors_high.append(abs(pred["high"] - actual["high"]))
        errors_low.append(abs(pred["low"] - actual["low"]))

    print(f"MAE High: {np.mean(errors_high):.6f}  ({np.mean(errors_high) * PIP_MULTIPLIER:.2f} pips)")
    print(f"MAE Low:  {np.mean(errors_low):.6f}   ({np.mean(errors_low) * PIP_MULTIPLIER:.2f} pips)")
    print()
    print("Sample predictions vs actual (first 5):")
    print(f"{'Open':>8}  {'PredH':>8}  {'TrueH':>8}  {'PredL':>8}  {'TrueL':>8}  {'Close':>8}")
    for i in range(N_WINDOW, N_WINDOW + 5):
        historical = ohlc[i - N_WINDOW:i]
        current_close = ohlc[i, 3]
        pred = decorator.predict(historical, current_close)
        print(
            f"{pred['open']:8.5f}  {pred['high']:8.5f}  {ohlc[i,1]:8.5f}  "
            f"{pred['low']:8.5f}  {ohlc[i,2]:8.5f}  {pred['close']:8.5f}"
        )


if __name__ == "__main__":
    from src.config import DATA_PATH
    evaluate_on_data(DATA_PATH)
