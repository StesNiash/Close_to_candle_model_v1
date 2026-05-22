import os

N_WINDOW = 10
BATCH_SIZE = 64
EPOCHS = 300
LEARNING_RATE = 0.0005
TRAIN_SPLIT = 0.8
HIDDEN_SIZE = 64
NUM_LSTM_LAYERS = 2
DROPOUT = 0.3
WEIGHT_DECAY = 1e-5
SEED = 42
PIP_MULTIPLIER = 10000.0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "EUR-USD_1Minute_BID_2026-05-21_00_00-23_59_Etc_UTC.csv")
