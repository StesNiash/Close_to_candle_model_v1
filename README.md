# Close_to_candle_model_v1

ML Model that transforms close values to whole candle (decorative use ONLY).

## Usage

### Train

```bash
python -m src.train
```

### Evaluate

```bash
python -m src.inference
```

### Inference API

```python
from src.inference import CandleDecorator

decorator = CandleDecorator()

# historical_candles: (N_WINDOW, 4) array of [Open, High, Low, Close]
# current_close: float — predicted close from external model
candle = decorator.predict(historical_candles, current_close)
# => {"open": ..., "high": ..., "low": ..., "close": ...}
```

High/low are post-processed: `high = max(open, close, predicted_high)`, `low = min(open, close, predicted_low)`.
