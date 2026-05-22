import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.config import N_WINDOW, TRAIN_SPLIT, BATCH_SIZE, PIP_MULTIPLIER, SEED


def load_raw_data(csv_path):
    df = pd.read_csv(csv_path)
    timestamp_col = df.columns[0]
    df = df.rename(columns={timestamp_col: "timestamp"})
    ohlc = df[["Open", "High", "Low", "Close"]].values.astype(np.float64)
    return ohlc


def normalize_pips(ohlc):
    n = len(ohlc)
    prev_close = np.empty(n)
    prev_close[0] = ohlc[0, 0]
    prev_close[1:] = ohlc[:-1, 3]

    norm = np.empty_like(ohlc)
    for j in range(4):
        norm[:, j] = (ohlc[:, j] / prev_close - 1.0) * PIP_MULTIPLIER
    return norm, prev_close


def build_sequences(norm):
    n = len(norm)
    features = []
    targets = []
    for i in range(N_WINDOW, n):
        hist = norm[i - N_WINDOW:i, 1:4]
        cur_close = norm[i, 3:4]
        feat = np.concatenate([hist.flatten(), cur_close])
        features.append(feat)
        target = norm[i, 1:3]
        targets.append(target)
    X = np.array(features, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    return X, y


def prepare_dataloaders(csv_path):
    ohlc = load_raw_data(csv_path)
    norm, prev_close = normalize_pips(ohlc)
    X, y = build_sequences(norm)

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(X))
    split_idx = int(len(X) * TRAIN_SPLIT)

    for part, idx in [("train", perm[:split_idx]), ("val", perm[split_idx:])]:
        X_sub = X[idx]
        y_sub = y[idx]

        hist = torch.from_numpy(X_sub[:, :-1].reshape(-1, N_WINDOW, 3))
        cur_close = torch.from_numpy(X_sub[:, -1:].copy())
        target = torch.from_numpy(y_sub)

        dataset = TensorDataset(hist, cur_close, target)
        shuffle = part == "train"
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

        if part == "train":
            train_loader = loader
        else:
            val_loader = loader

    return train_loader, val_loader, prev_close


def unnormalize(value_pips, prev_close):
    return prev_close * (1.0 + value_pips / PIP_MULTIPLIER)
