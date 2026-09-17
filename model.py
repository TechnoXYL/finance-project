"""CNN-LSTM stock model adapted from the supplied implementation."""

from __future__ import annotations

import argparse
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as scipy_stats
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf
from torch.utils.data import DataLoader, Dataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def load_csv_data(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read and validate the uploaded GB18030 comma-separated stock export."""
    columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Amount"]
    frame = pd.read_csv(
        path,
        skiprows=2,
        header=None,
        names=columns,
        encoding="gb18030",
    )
    raw_rows = len(frame)
    frame["Date"] = pd.to_datetime(
        frame["Date"], format="%Y/%m/%d", errors="coerce"
    )
    for column in ["Open", "High", "Low", "Close", "Volume", "Amount"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    parsed = frame.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume", "Amount"])
    valid = parsed[
        (parsed[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
        & (parsed["Volume"] >= 0)
        & (parsed["Amount"] >= 0)
        & (parsed["High"] >= parsed[["Open", "Close"]].max(axis=1))
        & (parsed["Low"] <= parsed[["Open", "Close"]].min(axis=1))
    ].copy()
    valid = valid.sort_values("Date").drop_duplicates("Date", keep="last")
    return (
        valid[["Date", "Open", "High", "Low", "Close", "Volume", "Amount"]].reset_index(drop=True),
        {
            "raw_rows": raw_rows,
            "parsed_rows": len(parsed),
            "valid_rows": len(valid),
            "invalid_rows": raw_rows - len(valid),
        },
    )


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.rename(columns={"Close": "price", "Volume": "volume"}).copy()
    price = frame["price"]
    volume = frame["volume"]
    log_price = np.log(price)
    log_volume = np.log(volume + 1.0)

    # Returns and volume dynamics
    frame["log_ret"] = log_price.diff()
    frame["vol_chg"] = log_volume.diff()

    # Realized volatility at several horizons
    frame["roll_vol_5"] = frame["log_ret"].rolling(5).std()
    frame["roll_vol_10"] = frame["log_ret"].rolling(10).std()
    frame["roll_vol_20"] = frame["log_ret"].rolling(20).std()
    frame["roll_vol_60"] = frame["log_ret"].rolling(60).std()

    # Short/medium momentum
    frame["mom_5"] = frame["log_ret"].rolling(5).mean()
    frame["mom_20"] = frame["log_ret"].rolling(20).mean()

    # Intraday and overnight structure
    frame["intraday_range"] = np.log(frame["High"] / frame["Low"])
    frame["close_open"] = np.log(price / frame["Open"])
    frame["overnight_gap"] = np.log(frame["Open"] / price.shift(1))
    high_low = frame["High"] - frame["Low"]
    frame["close_position"] = np.where(
        high_low > 0,
        (price - frame["Low"]) / high_low.replace(0, np.nan),
        0.5,
    )
    frame["range_5"] = frame["intraday_range"].rolling(5).mean()

    # Volume relative measures
    volume_ma_20 = volume.rolling(20).mean()
    frame["volume_ratio"] = volume / volume_ma_20.replace(0, np.nan)
    frame["volume_std_20"] = log_volume.rolling(20).std()

    # RSI(14)
    delta = price.diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    frame["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)

    # Normalized trend (MACD-like) and Bollinger position
    ema_fast = price.ewm(span=12, adjust=False).mean()
    ema_slow = price.ewm(span=26, adjust=False).mean()
    frame["macd_norm"] = (ema_fast - ema_slow) / price
    roll_mean_20 = price.rolling(20).mean()
    roll_std_20 = price.rolling(20).std()
    frame["bb_position"] = (price - roll_mean_20) / (2.0 * roll_std_20)

    # Longer-horizon momentum, volatility, and rate of change
    frame["mom_60"] = frame["log_ret"].rolling(60).mean()
    frame["roll_vol_120"] = frame["log_ret"].rolling(120).std()
    frame["roc_10"] = price.pct_change(10)

    # Average True Range (normalized by price)
    prev_close = price.shift(1)
    true_range = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - prev_close).abs(),
            (frame["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr_14"] = true_range.rolling(14).mean() / price

    # Stochastic oscillator and Williams %R
    low_14 = frame["Low"].rolling(14).min()
    high_14 = frame["High"].rolling(14).max()
    stoch_k = 100.0 * (price - low_14) / (high_14 - low_14).replace(0, np.nan)
    frame["stoch_k"] = stoch_k
    frame["stoch_d"] = stoch_k.rolling(3).mean()
    frame["williams_r"] = (
        -100.0 * (high_14 - price) / (high_14 - low_14).replace(0, np.nan)
    )

    # Commodity Channel Index (20)
    typical = (frame["High"] + frame["Low"] + price) / 3.0
    tp_sma = typical.rolling(20).mean()
    tp_mad = (typical - tp_sma).abs().rolling(20).mean()
    frame["cci_20"] = (typical - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))

    # Price relative to short/long moving averages
    frame["price_ma_5"] = price / price.rolling(5).mean().replace(0, np.nan) - 1.0
    frame["price_ma_60"] = price / price.rolling(60).mean().replace(0, np.nan) - 1.0
    ema_50 = price.ewm(span=50, adjust=False).mean()
    frame["ema_ratio_50"] = price / ema_50.replace(0, np.nan) - 1.0

    # Volume trend and higher moments
    frame["volume_trend"] = (
        volume.rolling(5).mean() / volume.rolling(20).mean().replace(0, np.nan)
    )
    frame["volume_skew_20"] = log_volume.rolling(20).skew()

    # Downside volatility and realized higher moments of returns
    frame["downside_vol_20"] = frame["log_ret"].clip(upper=0).rolling(20).std()
    frame["realized_skew_20"] = frame["log_ret"].rolling(20).skew()
    frame["realized_kurt_20"] = frame["log_ret"].rolling(20).kurt()

    # On-balance volume flow (normalized by average volume)
    direction = np.sign(frame["log_ret"]).fillna(0.0)
    obv = (direction * volume).cumsum()
    frame["obv_norm"] = obv.diff() / volume.rolling(20).mean().replace(0, np.nan)

    # Calendar seasonality (cyclic encodings)
    frame["day_of_week"] = frame["Date"].dt.dayofweek.astype(float) / 4.0
    day_of_year = frame["Date"].dt.dayofyear.astype(float)
    frame["day_sin"] = np.sin(2 * np.pi * day_of_year / 365.0)
    frame["day_cos"] = np.cos(2 * np.pi * day_of_year / 365.0)
    month = frame["Date"].dt.month.astype(float)
    frame["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    frame["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # Amount / turnover dynamics
    amount = frame["Amount"]
    log_amount = np.log(amount + 1.0)
    frame["amount_ratio_20"] = amount / amount.rolling(20).mean().replace(0, np.nan)
    frame["amount_std_20"] = log_amount.rolling(20).std()
    vwap = amount / volume.replace(0, np.nan)
    frame["vwap_close"] = np.log(vwap / price.replace(0, np.nan))

    # Absolute return magnitude (realized volatility alternative)
    frame["ret_abs_5"] = frame["log_ret"].abs().rolling(5).mean()
    frame["ret_abs_20"] = frame["log_ret"].abs().rolling(20).mean()

    # Volume/return co-movement and intraday-range volatility
    frame["corr_ret_vol_20"] = frame["log_ret"].rolling(20).corr(frame["vol_chg"])
    frame["range_vol_20"] = frame["intraday_range"].rolling(20).std()

    # Fast/slow EMA cross normalized by price
    ema_5 = price.ewm(span=5, adjust=False).mean()
    ema_20 = price.ewm(span=20, adjust=False).mean()
    frame["ema_cross_5_20"] = (ema_5 - ema_20) / price

    # Lagged returns give the network the same information as the AR baseline,
    # so it can learn short-horizon autoregressive structure explicitly.
    log_ret = frame["log_ret"]
    for lag in range(1, 11):
        frame[f"ret_lag_{lag}"] = log_ret.shift(lag)

    # Exponential-weighted realized volatility.
    frame["ewma_vol_20"] = (log_ret ** 2).ewm(span=20, adjust=False).mean().pow(0.5)

    # Range-based volatility estimators (Parkinson and Garman-Klass).
    log_hl = np.log(frame["High"] / frame["Low"])
    log_co = np.log(price / frame["Open"])
    frame["parkinson_vol"] = (log_hl ** 2 / (4.0 * np.log(2.0))).pow(0.5)
    frame["garman_klass"] = (
        0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
    ).clip(lower=0.0).pow(0.5)

    # Amihud illiquidity: absolute return per unit of traded value.
    frame["amihud_20"] = (
        frame["log_ret"].abs() / amount.replace(0, np.nan)
    ).rolling(20).mean()

    # Position within a trailing price range and distance from recent high.
    high_120 = frame["High"].rolling(120).max()
    low_120 = frame["Low"].rolling(120).min()
    frame["price_pos_120"] = (price - low_120) / (high_120 - low_120).replace(0, np.nan)
    frame["dist_high_120"] = price / high_120.replace(0, np.nan) - 1.0

    # Volume rate of change.
    frame["volume_roc_5"] = volume.pct_change(5)
    frame["volume_roc_20"] = volume.pct_change(20)

    # Short-window signed volume flow (money-flow style).
    signed_vol = np.sign(log_ret).fillna(0.0) * volume
    frame["signed_volume_ratio_10"] = (
        signed_vol.rolling(10).sum() / volume.rolling(10).sum().replace(0, np.nan)
    )

    # --- Additional features ---
    # Extra autoregressive lags (short-horizon continuation).
    for lag in range(11, 21):
        frame[f"ret_lag_{lag}"] = log_ret.shift(lag)

    # Realized variance over the trailing month.
    frame["rv_20"] = (log_ret**2).rolling(20).sum()

    # Money Flow Index (14).
    typical_price = (frame["High"] + frame["Low"] + price) / 3.0
    raw_money_flow = typical_price * volume
    flow_direction = typical_price.diff()
    positive_flow = raw_money_flow.where(flow_direction > 0, 0.0)
    negative_flow = raw_money_flow.where(flow_direction < 0, 0.0)
    frame["mfi_14"] = 100.0 - 100.0 / (
        1.0
        + positive_flow.rolling(14).sum()
        / negative_flow.rolling(14).sum().replace(0, np.nan)
    )

    # Chaikin Money Flow (20).
    hl_range = (frame["High"] - frame["Low"]).replace(0, np.nan)
    mfm = ((price - frame["Low"]) - (frame["High"] - price)) / hl_range
    frame["cmf_20"] = (mfm * volume).rolling(20).sum() / volume.rolling(20).sum().replace(
        0, np.nan
    )

    # Wilder ADX / directional movement (14).
    up_move = frame["High"].diff()
    down_move = -frame["Low"].diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=frame.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=frame.index,
    )
    atr_wilder = true_range.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    plus_di = (
        100.0
        * plus_dm.ewm(alpha=1.0 / 14.0, adjust=False).mean()
        / atr_wilder.replace(0, np.nan)
    )
    minus_di = (
        100.0
        * minus_dm.ewm(alpha=1.0 / 14.0, adjust=False).mean()
        / atr_wilder.replace(0, np.nan)
    )
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    frame["adx_14"] = dx.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    frame["plus_di_14"] = plus_di
    frame["minus_di_14"] = minus_di

    # Price-volume co-movement.
    frame["corr_price_vol_20"] = price.rolling(20).corr(volume)

    # Additional rate-of-change horizons and accelerations.
    frame["roc_5"] = price.pct_change(5)
    frame["roc_20"] = price.pct_change(20)
    frame["price_accel"] = frame["roc_5"] - frame["roc_20"]
    frame["volume_accel"] = frame["volume_roc_5"] - frame["volume_roc_20"]

    # Mean intraday range over a longer horizon.
    frame["range_20"] = frame["intraday_range"].rolling(20).mean()

    # Medium-term EMA cross (20 vs 50).
    frame["ema_cross_20_50"] = (ema_20 - ema_50) / price

    frame["target"] = frame["log_ret"].shift(-1)
    return frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def _static_features(frame: pd.DataFrame, train_end: int) -> np.ndarray:
    """Per-stock summary statistics computed from the training window only."""
    train = frame.iloc[:train_end]
    if len(train) == 0:
        return np.zeros(12, dtype=np.float64)
    price = train["price"].to_numpy(dtype=np.float64)
    volume = train["volume"].to_numpy(dtype=np.float64)
    amount = train["Amount"].to_numpy(dtype=np.float64)
    log_price = np.log(price)
    log_ret = np.diff(log_price, prepend=log_price[0])
    log_volume = np.log(volume + 1.0)
    intraday_range = np.log(
        train["High"].to_numpy(dtype=np.float64)
        / train["Low"].to_numpy(dtype=np.float64)
    )
    high_low = train["High"].to_numpy(dtype=np.float64) - train["Low"].to_numpy(dtype=np.float64)
    close_position = np.where(
        high_low > 0,
        (price - train["Low"].to_numpy(dtype=np.float64)) / np.where(high_low > 0, high_low, 1.0),
        0.5,
    )
    amihud = np.abs(log_ret[1:]) / np.maximum(amount[1:], 1.0)
    features = np.array(
        [
            float(np.mean(log_ret)),
            float(np.std(log_ret)),
            float(np.mean(np.abs(log_ret))),
            float(np.mean(log_volume)),
            float(np.std(log_volume)),
            float(np.mean(intraday_range)),
            float(np.mean(close_position)),
            float(scipy_stats.skew(log_ret)),
            float(scipy_stats.kurtosis(log_ret, fisher=True)),
            float(np.mean(amihud)),
            float(np.log(price[-1] / price[0])),
            float(np.mean(volume)),
        ],
        dtype=np.float64,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def create_sequences(
    frame: pd.DataFrame, feature_cols: list[str], target_col: str, seq_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = frame[feature_cols].to_numpy(dtype=np.float32)
    target = frame[target_col].to_numpy(dtype=np.float32)
    X = np.lib.stride_tricks.sliding_window_view(data, seq_len, axis=0).transpose(0, 2, 1)
    y = target[seq_len - 1 :]
    indices = np.arange(seq_len - 1, len(target))
    return X, y, indices


def _load_file(
    args: tuple[Path, list[str], int],
) -> tuple[str, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, int, np.ndarray]:
    """Load, validate, featurize, and per-file normalize a single export."""
    path, feature_cols, stock_id = args
    data, file_quality = load_csv_data(path)
    frame = make_features(data)
    n = len(frame)
    train_end = int(n * 0.8)
    stats = frame.iloc[:train_end][feature_cols]
    mean = stats.mean().to_numpy(dtype=np.float64)
    std = stats.std().replace(0, 1.0).to_numpy(dtype=np.float64)
    feats = (frame[feature_cols].to_numpy(dtype=np.float32) - mean) / std
    target = frame["target"].to_numpy(dtype=np.float32)
    log_ret = frame["log_ret"].to_numpy(dtype=np.float32)
    dates = frame["Date"].to_numpy()
    static_feats = _static_features(frame, train_end)
    return path.name, stock_id, feats, target, log_ret, dates, file_quality, n, static_feats


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, ids: np.ndarray | None = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.ids = (
            torch.tensor(ids, dtype=torch.long) if ids is not None else None
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        if self.ids is not None:
            return self.X[index], self.y[index], self.ids[index]
        return self.X[index], self.y[index]


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.padding, 0)))


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_features,
        cnn_channels=64,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        num_stocks=0,
        stock_embed_dim=16,
    ):
        super().__init__()
        self.stock_embedding = (
            nn.Embedding(num_stocks, stock_embed_dim) if num_stocks > 0 else None
        )
        self.stock_bias = (
            nn.Embedding(num_stocks, 1) if num_stocks > 0 else None
        )
        input_channels = n_features + (
            stock_embed_dim if self.stock_embedding is not None else 0
        )
        self.conv1 = CausalConv1d(input_channels, cnn_channels, 3, 1)
        self.conv2 = CausalConv1d(cnn_channels, cnn_channels, 3, 2)
        self.conv3 = CausalConv1d(cnn_channels, cnn_channels, 3, 4)
        self.norm1 = nn.LayerNorm(cnn_channels)
        self.norm2 = nn.LayerNorm(cnn_channels)
        self.norm3 = nn.LayerNorm(cnn_channels)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            cnn_channels,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        head_in = hidden_size + (stock_embed_dim if self.stock_embedding is not None else 0)
        self.head_mu = nn.Linear(head_in, 1)
        self.head_logvar = nn.Linear(head_in, 1)

    def forward(self, x, stock_id=None):
        x = x.transpose(1, 2)
        if self.stock_embedding is not None and stock_id is not None:
            emb = self.stock_embedding(stock_id)
            emb = emb.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x = torch.cat([x, emb], dim=1)
        x = self.dropout(self.norm1(F.gelu(self.conv1(x)).transpose(1, 2))).transpose(1, 2)
        x = self.dropout(self.norm2(F.gelu(self.conv2(x)).transpose(1, 2))).transpose(1, 2)
        x = self.dropout(self.norm3(F.gelu(self.conv3(x)).transpose(1, 2))).transpose(1, 2)
        out, _ = self.lstm(x.transpose(1, 2))
        last = self.dropout(out[:, -1, :])
        if self.stock_embedding is not None and stock_id is not None:
            last = torch.cat([last, self.stock_embedding(stock_id)], dim=-1)
        mu = self.head_mu(last)
        if self.stock_bias is not None and stock_id is not None:
            mu = mu + self.stock_bias(stock_id)
        return mu, self.head_logvar(last).clamp(-6.0, 2.0)


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (0.5 * (logvar + (y - mu) ** 2 / torch.exp(logvar) + math.log(2 * math.pi))).mean()


def train_model(
    model: CNNLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    patience: int,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    use_amp: bool = True,
    use_compile: bool = False,
) -> CNNLSTM:
    if use_compile:
        try:
            model = torch.compile(model)
        except Exception as exc:  # fall back to eager if compile is unavailable
            print(f"torch.compile failed ({exc}); continuing in eager mode")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and DEVICE.type == "cuda")
    autocast = torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp and DEVICE.type == "cuda")
    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if len(batch) == 3:
                X, y, sid = batch
                sid = sid.to(DEVICE, non_blocking=True)
            else:
                X, y = batch
                sid = None
            X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                mu, logvar = model(X, sid)
                loss = gaussian_nll(mu, logvar, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.detach() * len(X)
        train_loss = (train_loss / len(train_loader.dataset)).item()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    X, y, sid = batch
                    sid = sid.to(DEVICE, non_blocking=True)
                else:
                    X, y = batch
                    sid = None
                X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                with autocast:
                    val_loss += gaussian_nll(*model(X, sid), y).detach() * len(X)
        val_loss = (val_loss / len(val_loader.dataset)).item()
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | train NLL {train_loss:.6f} | val NLL {val_loss:.6f}")
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}, best val NLL = {best_val:.6f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model: CNNLSTM, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    mus, logvars, ys = [], [], []
    for batch in loader:
        if len(batch) == 3:
            X, y, sid = batch
            sid = sid.to(DEVICE)
        else:
            X, y = batch
            sid = None
        mu, logvar = model(X.to(DEVICE), sid)
        mus.append(mu.cpu().numpy())
        logvars.append(logvar.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(mus).ravel(), np.concatenate(logvars).ravel(), np.concatenate(ys).ravel()


def ar_predict(history: np.ndarray, const: float, phis: np.ndarray) -> float:
    return const + float(np.dot(phis, history[-len(phis) :][::-1]))


def fit_configuration(
    frame: pd.DataFrame,
    feature_cols: list[str],
    train_end: int,
    config: dict,
    epochs: int,
    patience: int,
) -> tuple[CNNLSTM, float, dict]:
    train_stats = frame.iloc[:train_end][feature_cols]
    frame_norm = frame.copy()
    frame_norm[feature_cols] = (
        (frame[feature_cols] - train_stats.mean())
        / train_stats.std().replace(0, 1.0)
    )
    X, y, indices = create_sequences(
        frame_norm, feature_cols, "target", config["seq_len"]
    )
    train_mask = indices < train_end
    X_all, y_all = X[train_mask], y[train_mask]
    val_size = max(1, int(len(X_all) * 0.1))
    train_loader = DataLoader(
        TimeSeriesDataset(X_all[:-val_size], y_all[:-val_size]),
        batch_size=config["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        TimeSeriesDataset(X_all[-val_size:], y_all[-val_size:]),
        batch_size=config["batch_size"],
    )
    torch.manual_seed(SEED)
    model = CNNLSTM(
        len(feature_cols),
        cnn_channels=config["cnn_channels"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(DEVICE)
    model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        patience=patience,
        learning_rate=config["learning_rate"],
    )
    val_mu, val_logvar, val_y = predict(model, val_loader)
    val_nll = float(
        np.mean(
            0.5
            * (
                val_logvar
                + (val_y - val_mu) ** 2 / np.exp(val_logvar)
                + np.log(2 * np.pi)
            )
        )
    )
    return model, val_nll, {
        "frame_norm": frame_norm,
        "indices": indices,
        "X": X,
        "y": y,
    }


def scan_parameters(
    frame: pd.DataFrame,
    feature_cols: list[str],
    train_end: int,
    scan_epochs: int,
    scan_patience: int,
) -> dict:
    configs = [
        {
            "seq_len": seq_len,
            "cnn_channels": cnn_channels,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": 64,
        }
        for seq_len in (30, 60)
        for cnn_channels in (32, 64)
        for hidden_size in (32, 64)
        for num_layers in (1, 2)
        for dropout in ((0.0,) if num_layers == 1 else (0.2,))
        for learning_rate in (1e-3, 3e-4)
    ]
    best = None
    for number, config in enumerate(configs, 1):
        _, validation_nll, _ = fit_configuration(
            frame, feature_cols, train_end, config, scan_epochs, scan_patience
        )
        print(
            f"scan {number:02d}/{len(configs)} {config} "
            f"validation proxy NLL={validation_nll:.6f}"
        )
        if best is None or validation_nll < best["validation_nll"]:
            best = {"validation_nll": validation_nll, "config": config}
    return best


def ljung_box_fft(residuals, lags=(10, 20)):
    """Ljung-Box test using FFT-based ACF (avoids O(n^2) autocorrelation)."""
    residuals = np.asarray(residuals, dtype=float)
    nobs = residuals.shape[0]
    lags = np.atleast_1d(lags).astype(int)
    maxlag = int(lags.max())
    sacf = acf(residuals, nlags=maxlag, fft=True)
    sacf2 = sacf[1 : maxlag + 1] ** 2 / (nobs - np.arange(1, maxlag + 1))
    q_stat = nobs * (nobs + 2) * np.cumsum(sacf2)[lags - 1]
    pvals = scipy_stats.chi2.sf(q_stat, lags)
    return pd.DataFrame({"lb_stat": q_stat, "lb_pvalue": pvals}, index=lags)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("20260914_A"))
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--ar-lags", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--load-workers", type=int, default=None)
    parser.add_argument("--cnn-channels", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--stock-embed-dim", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--compile", action="store_true", default=False)
    args = parser.parse_args()
    feature_cols = [
        "log_ret",
        "vol_chg",
        "roll_vol_5",
        "roll_vol_10",
        "roll_vol_20",
        "roll_vol_60",
        "mom_5",
        "mom_20",
        "intraday_range",
        "close_open",
        "overnight_gap",
        "close_position",
        "range_5",
        "volume_ratio",
        "volume_std_20",
        "rsi_14",
        "macd_norm",
        "bb_position",
        "mom_60",
        "roll_vol_120",
        "roc_10",
        "atr_14",
        "stoch_k",
        "stoch_d",
        "williams_r",
        "cci_20",
        "price_ma_5",
        "price_ma_60",
        "ema_ratio_50",
        "volume_trend",
        "volume_skew_20",
        "downside_vol_20",
        "realized_skew_20",
        "realized_kurt_20",
        "obv_norm",
        "day_of_week",
        "day_sin",
        "day_cos",
        "month_sin",
        "month_cos",
        "amount_ratio_20",
        "amount_std_20",
        "vwap_close",
        "ret_abs_5",
        "ret_abs_20",
        "corr_ret_vol_20",
        "range_vol_20",
        "ema_cross_5_20",
        "ret_lag_1",
        "ret_lag_2",
        "ret_lag_3",
        "ret_lag_4",
        "ret_lag_5",
        "ret_lag_6",
        "ret_lag_7",
        "ret_lag_8",
        "ret_lag_9",
        "ret_lag_10",
        "ewma_vol_20",
        "parkinson_vol",
        "garman_klass",
        "amihud_20",
        "price_pos_120",
        "dist_high_120",
        "volume_roc_5",
        "volume_roc_20",
        "signed_volume_ratio_10",
        "ret_lag_11",
        "ret_lag_12",
        "ret_lag_13",
        "ret_lag_14",
        "ret_lag_15",
        "ret_lag_16",
        "ret_lag_17",
        "ret_lag_18",
        "ret_lag_19",
        "ret_lag_20",
        "rv_20",
        "mfi_14",
        "cmf_20",
        "adx_14",
        "plus_di_14",
        "minus_di_14",
        "corr_price_vol_20",
        "roc_5",
        "roc_20",
        "price_accel",
        "volume_accel",
        "range_20",
        "ema_cross_20_50",
    ]
    config = {
        "seq_len": args.seq_len,
        "cnn_channels": args.cnn_channels,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "learning_rate": 1e-3,
        "batch_size": args.batch_size,
    }
    paths = sorted(args.data.glob("*.txt")) if args.data.is_dir() else [args.data]
    if not paths:
        raise FileNotFoundError("No .txt stock data files found")
    load_workers = args.load_workers or min(64, os.cpu_count() or 1)
    seq_len = config["seq_len"]
    train_Xs, train_ys, train_dates, train_ids = [], [], [], []
    val_Xs, val_ys, val_ids = [], [], []
    test_parts, test_ids = [], []
    ar_parts = []
    quality = []
    static_by_stock: dict[int, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=load_workers) as pool:
        for name, stock_id, feats, target, log_ret, dates, file_quality, n, static_feats in pool.map(
            _load_file, [(path, feature_cols, idx) for idx, path in enumerate(paths)], chunksize=8
        ):
            quality.append((name, file_quality, n))
            static_by_stock[stock_id] = static_feats
            if n <= seq_len:
                continue
            train_end = int(n * 0.8)
            X = np.lib.stride_tricks.sliding_window_view(feats, seq_len, axis=0).transpose(0, 2, 1)
            y = target[seq_len - 1 :]
            indices = np.arange(seq_len - 1, n)
            seq_dates = dates[seq_len - 1 :]
            sid = np.full(len(X), stock_id, dtype=np.int64)
            train_mask = indices < train_end
            val_size = max(1, int(train_mask.sum() * 0.1))
            if train_mask.sum() <= val_size or (~train_mask).sum() == 0:
                continue
            train_Xs.append(X[train_mask][:-val_size])
            train_ys.append(y[train_mask][:-val_size])
            train_dates.append(seq_dates[train_mask][:-val_size])
            train_ids.append(sid[train_mask][:-val_size])
            val_Xs.append(X[train_mask][-val_size:])
            val_ys.append(y[train_mask][-val_size:])
            val_ids.append(sid[train_mask][-val_size:])
            test_mask = indices >= train_end
            test_parts.append((X[test_mask], y[test_mask]))
            test_ids.append(sid[test_mask])
            test_indices = indices[test_mask]
            ar_parts.append((log_ret[:train_end], log_ret, target, test_indices))
    n_static = next(iter(static_by_stock.values())).shape[0]
    static_matrix = np.zeros((len(paths), n_static), dtype=np.float32)
    for sid, sf in static_by_stock.items():
        static_matrix[sid] = sf.astype(np.float32)
    static_mean = static_matrix.mean(axis=0)
    static_std = static_matrix.std(axis=0)
    static_std[static_std == 0.0] = 1.0
    static_matrix = ((static_matrix - static_mean) / static_std).astype(np.float32)

    def add_static(X: np.ndarray, ids: np.ndarray) -> np.ndarray:
        seq_len = X.shape[1]
        stat = static_matrix[ids][:, None, :]
        stat = np.repeat(stat, seq_len, axis=1)
        return np.concatenate([X, stat], axis=2)

    X_train = add_static(np.concatenate(train_Xs), np.concatenate(train_ids))
    y_train = np.concatenate(train_ys)
    train_dates = np.concatenate(train_dates)
    train_ids = np.concatenate(train_ids)
    X_val = add_static(np.concatenate(val_Xs), np.concatenate(val_ids))
    y_val = np.concatenate(val_ys)
    val_ids = np.concatenate(val_ids)
    X_test = add_static(np.concatenate([part[0] for part in test_parts]), np.concatenate(test_ids))
    y_test = np.concatenate([part[1] for part in test_parts])
    test_ids = np.concatenate(test_ids)
    order = np.argsort(train_dates, kind="stable")
    X_train = X_train[order]
    y_train = y_train[order]
    train_ids = train_ids[order]
    train_loader = DataLoader(
        TimeSeriesDataset(X_train, y_train, train_ids),
        config["batch_size"],
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
        drop_last=False,
    )
    val_loader = DataLoader(
        TimeSeriesDataset(X_val, y_val, val_ids),
        config["batch_size"],
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        TimeSeriesDataset(X_test, y_test, test_ids),
        config["batch_size"],
        num_workers=args.workers,
        pin_memory=True,
    )
    torch.manual_seed(SEED)
    model = CNNLSTM(
        len(feature_cols) + n_static,
        config["cnn_channels"],
        config["hidden_size"],
        config["num_layers"],
        config["dropout"],
        num_stocks=len(paths),
        stock_embed_dim=args.stock_embed_dim,
    ).to(DEVICE)
    model = train_model(
        model, train_loader, val_loader, args.epochs, args.patience,
        config["learning_rate"], config["weight_decay"],
        use_amp=args.amp, use_compile=args.compile,
    )
    mu, logvar, y_true = predict(model, test_loader)
    sigma2 = np.exp(logvar)
    positions = (mu > 0).astype(float)
    strategy_returns = positions * (np.exp(y_true) - 1.0)
    buy_hold_returns = np.exp(y_true) - 1.0
    offsets = np.cumsum([0] + [len(part[1]) for part in test_parts])
    strategy_growths = [
        float(np.prod(1.0 + strategy_returns[start:end]))
        for start, end in zip(offsets[:-1], offsets[1:])
    ]
    buy_hold_growths = [
        float(np.prod(1.0 + buy_hold_returns[start:end]))
        for start, end in zip(offsets[:-1], offsets[1:])
    ]
    strategy_growth = float(np.mean(strategy_growths))
    buy_hold_growth = float(np.mean(buy_hold_growths))
    test_years = np.mean([len(part[1]) for part in test_parts]) / 252.0
    print("\n========== Long-only trading backtest ==========")
    print("Rule: invest next day when predicted log return > 0; otherwise hold cash")
    print("Assumptions: no fees, slippage, taxes, or position-size limits")
    print(f"Strategy cumulative return: {strategy_growth - 1.0:.2%}")
    print(f"Strategy annualized return: {strategy_growth ** (1.0 / test_years) - 1.0:.2%}")
    print(f"Invested days: {int(positions.sum())}/{len(positions)}")
    print(f"Position changes: {int(np.abs(np.diff(np.r_[0.0, positions])).sum())}")
    print(f"Buy-and-hold cumulative return: {buy_hold_growth - 1.0:.2%}")
    print(f"Buy-and-hold annualized return: {buy_hold_growth ** (1.0 / test_years) - 1.0:.2%}")
    log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + (y_true - mu) ** 2 / sigma2)
    ar_preds_parts, ar_true_parts = [], []
    for train_ret, log_ret, target, test_indices in ar_parts:
        ar_model = AutoReg(train_ret, lags=args.ar_lags).fit()
        const, phis = float(ar_model.params[0]), np.asarray(ar_model.params[1:])
        ar_preds_parts.append(np.asarray([
            ar_predict(log_ret[:i + 1], const, phis)
            for i in test_indices
        ]))
        ar_true_parts.append(target[test_indices])
    ar_preds = np.concatenate(ar_preds_parts)
    ar_true = np.concatenate(ar_true_parts)
    ar_var = np.var(np.concatenate([part[0] for part in ar_parts]), ddof=1)
    ar_log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * ar_var) + (ar_true - ar_preds) ** 2 / ar_var)
    print(f"Device: {DEVICE}")
    print(
        f"Files: {len(paths)} | "
        f"valid rows: {sum(item[1]['valid_rows'] for item in quality)} | "
        f"invalid rows: {sum(item[1]['invalid_rows'] for item in quality)}"
    )
    print(f"Train samples: {len(X_train)} | validation samples: {len(X_val)} | test samples: {len(X_test)}")
    print("\n========== Test likelihood comparison ==========")
    print(f"CNN-LSTM average NLL: {-log_likelihood / len(y_true):.6f}")
    print(f"AR({args.ar_lags}) average NLL: {-ar_log_likelihood / len(ar_true):.6f}")
    print(f"CNN-LSTM total logL: {log_likelihood:.4f}")
    print(f"AR total logL: {ar_log_likelihood:.4f}")
    print(f"Reference LR: {2 * (log_likelihood - ar_log_likelihood):.4f}")
    print("\n========== Ljung-Box residual test ==========")
    print("CNN-LSTM residuals:")
    print(ljung_box_fft(y_true - mu, lags=[10, 20]))
    print("\nAR residuals:")
    print(ljung_box_fft(ar_true - ar_preds, lags=[10, 20]))


if __name__ == "__main__":
    main()
