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

    def predict_recursive(self, seed_candles, future_closes):
        seed = np.array(seed_candles, dtype=np.float64)
        if seed.ndim != 2 or seed.shape[1] < 4 or len(seed) != N_WINDOW:
            raise ValueError(
                f"seed_candles must have shape ({N_WINDOW}, 4), got {seed.shape}"
            )

        buffer = seed.copy()
        results = []

        for close_val in future_closes:
            candle = self.predict(buffer, close_val)
            results.append(candle)
            new_row = np.array(
                [[candle["open"], candle["high"], candle["low"], candle["close"]]],
                dtype=np.float64,
            )
            buffer = np.vstack([buffer[1:], new_row])

        return results


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


def draw_candles(ax, candles, offset, color_up, color_down, bar_width=0.6):
    import matplotlib.patches as mpatches

    for i, c in enumerate(candles):
        x = offset + i
        o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
        color = color_up if cl >= o else color_down

        ax.plot([x, x], [l, h], color=color, linewidth=0.8)

        body_bottom = min(o, cl)
        body_height = abs(cl - o) or 1e-8
        rect = mpatches.Rectangle(
            (x - bar_width / 2, body_bottom),
            bar_width,
            body_height,
            linewidth=0.5,
            edgecolor=color,
            facecolor=color,
        )
        ax.add_patch(rect)

    ax.set_xlim(offset - 1, offset + len(candles))


def visualize_recursive_prediction(
    csv_path, start_idx=None, num_steps=60, model_path=None, save_path=None
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ohlc = load_raw_data(csv_path)

    if start_idx is None:
        start_idx = N_WINDOW + 50

    end_idx = min(start_idx + num_steps, len(ohlc))
    num_steps = end_idx - start_idx

    seed = ohlc[start_idx - N_WINDOW:start_idx]
    future_closes = ohlc[start_idx:end_idx, 3]
    actual_candles = [
        {
            "open": round(float(ohlc[i, 0]), 5),
            "high": round(float(ohlc[i, 1]), 5),
            "low": round(float(ohlc[i, 2]), 5),
            "close": round(float(ohlc[i, 3]), 5),
        }
        for i in range(start_idx, end_idx)
    ]

    decorator = CandleDecorator(model_path)
    pred_candles = decorator.predict_recursive(seed, future_closes)

    cum_high_err = 0.0
    cum_low_err = 0.0
    for i in range(num_steps):
        cum_high_err += abs(pred_candles[i]["high"] - actual_candles[i]["high"])
        cum_low_err += abs(pred_candles[i]["low"] - actual_candles[i]["low"])

    mae_high_pips = cum_high_err / num_steps * PIP_MULTIPLIER
    mae_low_pips = cum_low_err / num_steps * PIP_MULTIPLIER

    print(f"Recursive prediction over {num_steps} steps:")
    print(f"  MAE High: {cum_high_err / num_steps:.6f}  ({mae_high_pips:.2f} pips)")
    print(f"  MAE Low:  {cum_low_err / num_steps:.6f}   ({mae_low_pips:.2f} pips)")

    green = "#26a69a"
    red = "#ef5350"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)
    fig.suptitle(
        f"Recursive OHLC prediction ({num_steps} steps)  |  "
        f"MAE High: {mae_high_pips:.2f} pips  |  MAE Low: {mae_low_pips:.2f} pips",
        fontsize=13,
        fontweight="bold",
    )

    draw_candles(ax1, actual_candles, 0, green, red)
    ax1.set_ylabel("Actual candles", fontsize=12)

    draw_candles(ax2, pred_candles, 0, green, red)
    ax2.set_ylabel("Predicted candles", fontsize=12)
    ax2.set_xlabel("Step", fontsize=12)

    xticks = range(0, num_steps, max(1, num_steps // 15))
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([str(t) for t in xticks])

    handles = [
        mpatches.Patch(facecolor=green, edgecolor=green, label="Bullish (Close >= Open)"),
        mpatches.Patch(facecolor=red, edgecolor=red, label="Bearish (Close < Open)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to: {save_path}")
    else:
        plt.show()


def visualize_recursive_pipeline(
    csv_path, start_idx=None, num_steps=30, model_path=None, save_path=None
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    ohlc = load_raw_data(csv_path)

    if start_idx is None:
        start_idx = N_WINDOW + 50

    end_idx = min(start_idx + num_steps, len(ohlc))
    num_steps = end_idx - start_idx

    seed_ohlc = ohlc[start_idx - N_WINDOW:start_idx]
    seed_candles = [
        {"open": round(float(seed_ohlc[i, 0]), 5),
         "high": round(float(seed_ohlc[i, 1]), 5),
         "low": round(float(seed_ohlc[i, 2]), 5),
         "close": round(float(seed_ohlc[i, 3]), 5)}
        for i in range(N_WINDOW)
    ]

    future_closes = ohlc[start_idx:end_idx, 3]
    actual_candles = [
        {"open": round(float(ohlc[i, 0]), 5),
         "high": round(float(ohlc[i, 1]), 5),
         "low": round(float(ohlc[i, 2]), 5),
         "close": round(float(ohlc[i, 3]), 5)}
        for i in range(start_idx, end_idx)
    ]

    decorator = CandleDecorator(model_path)
    pred_candles = decorator.predict_recursive(seed_ohlc, future_closes)

    cum_high_err = 0.0
    cum_low_err = 0.0
    for i in range(num_steps):
        cum_high_err += abs(pred_candles[i]["high"] - actual_candles[i]["high"])
        cum_low_err += abs(pred_candles[i]["low"] - actual_candles[i]["low"])

    mae_high_pips = cum_high_err / num_steps * PIP_MULTIPLIER
    mae_low_pips = cum_low_err / num_steps * PIP_MULTIPLIER

    print(f"Recursive pipeline over {num_steps} steps:")
    print(f"  MAE High: {cum_high_err / num_steps:.6f}  ({mae_high_pips:.2f} pips)")
    print(f"  MAE Low:  {cum_low_err / num_steps:.6f}   ({mae_low_pips:.2f} pips)")

    seed_highs = [c["high"] for c in seed_candles]
    seed_lows = [c["low"] for c in seed_candles]
    actual_highs = [c["high"] for c in actual_candles]
    actual_lows = [c["low"] for c in actual_candles]
    predicted_highs = [c["high"] for c in pred_candles]
    predicted_lows = [c["low"] for c in pred_candles]

    all_highs = seed_highs + predicted_highs + actual_highs
    all_lows = seed_lows + predicted_lows + actual_lows
    y_min = min(all_lows)
    y_max = max(all_highs)
    y_pad = (y_max - y_min) * 0.15

    green = "#26a69a"
    red = "#ef5350"
    blue = "#42a5f5"
    orange = "#ff9800"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    seed_positions = list(range(N_WINDOW))
    future_positions = list(range(N_WINDOW, N_WINDOW + num_steps))
    divider = N_WINDOW - 0.5

    # ---- Upper plot: Input data ----
    draw_candles(ax1, seed_candles, 0, green, red, bar_width=0.7)

    close_line = ax1.plot(
        [N_WINDOW - 0.5] + future_positions,
        [seed_candles[-1]["close"]] + list(future_closes),
        color=blue, linewidth=1.5, marker=".", markersize=4, zorder=3, label="Close (input)"
    )

    draw_candles_ghost(ax1, actual_candles, N_WINDOW, green, red, alpha=0.25, bar_width=0.7)

    ax1.axvline(x=divider, color="white", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
    ax1.set_ylabel("Input Data", fontsize=12, fontweight="bold")
    ax1.text(
        N_WINDOW / 2, y_max + y_pad * 0.6,
        f"Seed ({N_WINDOW} candles)", ha="center", va="bottom",
        fontsize=9, fontstyle="italic", color="gray"
    )
    ax1.text(
        N_WINDOW + num_steps / 2, y_max + y_pad * 0.6,
        f"Future closes ({num_steps}) + Actual (faded)", ha="center", va="bottom",
        fontsize=9, fontstyle="italic", color="gray"
    )

    handles1 = [
        Line2D([0], [0], color=blue, linewidth=1.5, marker=".", markersize=6, label="Close (input)"),
        mpatches.Patch(facecolor=green, edgecolor=green, alpha=0.25, label="Actual candle (faded)"),
    ]
    ax1.legend(handles=handles1, loc="upper right", fontsize=8)

    # ---- Lower plot: Recursive output ----
    draw_candles(ax2, seed_candles, 0, green, red, bar_width=0.7)
    draw_candles(ax2, pred_candles, N_WINDOW, green, red, bar_width=0.7)

    ax2.axvline(x=divider, color="white", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
    ax2.set_ylabel("Predicted Output", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Candle index", fontsize=12)

    for ax in (ax1, ax2):
        ax.set_xlim(-1, N_WINDOW + num_steps)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    xticks = list(range(0, N_WINDOW + num_steps, max(1, (N_WINDOW + num_steps) // 20)))
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([str(t) for t in xticks])

    fig.suptitle(
        f"Recursive prediction pipeline  |  Input → Output  |  "
        f"MAE High: {mae_high_pips:.2f} pips  |  MAE Low: {mae_low_pips:.2f} pips",
        fontsize=12, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to: {save_path}")
    else:
        plt.show()


def draw_candles_ghost(ax, candles, offset, color_up, color_down, alpha=0.3, bar_width=0.6):
    import matplotlib.patches as mpatches

    for i, c in enumerate(candles):
        x = offset + i
        o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
        color = color_up if cl >= o else color_down

        ax.plot([x, x], [l, h], color=color, linewidth=0.8, alpha=alpha)

        body_bottom = min(o, cl)
        body_height = abs(cl - o) or 1e-8
        rect = mpatches.Rectangle(
            (x - bar_width / 2, body_bottom),
            bar_width,
            body_height,
            linewidth=0.5,
            edgecolor=color,
            facecolor=color,
            alpha=alpha,
        )
        ax.add_patch(rect)


if __name__ == "__main__":
    from src.config import DATA_PATH
    visualize_recursive_pipeline(DATA_PATH, num_steps=30, save_path="recursive_pipeline.png")
