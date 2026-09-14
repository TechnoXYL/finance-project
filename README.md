# finance-project

## Stock model

This project uses `uv` to train and evaluate the supplied causal CNN-LSTM
next-trading-day log-return model from all forward-adjusted stock exports in
the working directory.

```bash
uv sync
uv run python model.py
```

The script pools valid rows from all `*.txt` stock files while keeping each
stock's sequences separate. It uses a smaller CNN/LSTM (32 channels, 32 hidden
units, 30-step lookback, 30% dropout), stronger weight decay, a lower learning
rate, and early stopping to reduce overfitting. The script preserves the
supplied model's two causal dilated convolutions,
LSTM, Gaussian negative log-likelihood uncertainty head, AdamW training,
learning-rate scheduling, early stopping, AR(10) comparison, and Ljung-Box
residual tests. It reads the GB18030 comma-separated export, validates OHLCV
rows, and uses `Close` and `Volume` to calculate log returns and volatility
features. Metadata and malformed rows are excluded and reported.
Invalid metadata or malformed OHLCV rows are excluded and reported. Use
`--epochs` and `--patience` to control training:

```bash
uv run python model.py --data path/to/file-or-directory --epochs 100 --patience 12
```
