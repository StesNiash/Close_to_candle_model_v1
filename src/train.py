import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.config import EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MODELS_DIR
from src.data_loader import prepare_dataloaders
from src.model import CandleHLPredictor
from src.config import DATA_PATH


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _ = prepare_dataloaders(DATA_PATH)

    model = CandleHLPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for hist, cur_close, target in train_loader:
            hist = hist.to(device)
            cur_close = cur_close.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = model(hist, cur_close)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * hist.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for hist, cur_close, target in val_loader:
                hist = hist.to(device)
                cur_close = cur_close.to(device)
                target = target.to(device)
                pred = model(hist, cur_close)
                val_loss += criterion(pred, target).item() * hist.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pt"))

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to: {os.path.join(MODELS_DIR, 'best_model.pt')}")


if __name__ == "__main__":
    train()
