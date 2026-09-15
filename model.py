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
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    parsed = frame.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
    valid = parsed[
        (parsed[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
        & (parsed["Volume"] >= 0)
        & (parsed["High"] >= parsed[["Open", "Close"]].max(axis=1))
        & (parsed["Low"] <= parsed[["Open", "Close"]].min(axis=1))
    ].copy()
    valid = valid.sort_values("Date").drop_duplicates("Date", keep="last")
    return (
        valid[["Date", "Open", "High", "Low", "Close", "Volume"]].reset_index(drop=True),
        {
            "raw_rows": raw_rows,
            "parsed_rows": len(parsed),
            "valid_rows": len(valid),
            "invalid_rows": raw_rows - len(valid),
        },
    )


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.rename(columns={"Close": "price", "Volume": "volume"}).copy()
    frame["log_ret"] = np.log(frame["price"]).diff()
    frame["vol_chg"] = np.log(frame["volume"] + 1.0).diff()
    frame["roll_vol_5"] = frame["log_ret"].rolling(5).std()
    frame["roll_vol_20"] = frame["log_ret"].rolling(20).std()
    frame["target"] = frame["log_ret"].shift(-1)
    return frame.dropna().reset_index(drop=True)


def create_sequences(
    frame: pd.DataFrame, feature_cols: list[str], target_col: str, seq_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = frame[feature_cols].to_numpy(dtype=np.float32)
    target = frame[target_col].to_numpy(dtype=np.float32)
    X = np.lib.stride_tricks.sliding_window_view(data, seq_len, axis=0).transpose(0, 2, 1)
    y = target[seq_len - 1 :]
    indices = np.arange(seq_len - 1, len(target))
    return X, y, indices


def _load_file(args: tuple[Path, list[str]]) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, dict, int]:
    """Load, validate, featurize, and per-file normalize a single export."""
    path, feature_cols = args
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
    return path.name, feats, target, log_ret, file_quality, n


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
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
        n_features: int,
        cnn_channels: int = 64,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(n_features, cnn_channels, 3, 1)
        self.conv2 = CausalConv1d(cnn_channels, cnn_channels, 3, 2)
        self.norm1 = nn.LayerNorm(cnn_channels)
        self.norm2 = nn.LayerNorm(cnn_channels)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            cnn_channels,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head_mu = nn.Linear(hidden_size, 1)
        self.head_logvar = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        x = self.dropout(self.norm1(F.gelu(self.conv1(x)).transpose(1, 2))).transpose(1, 2)
        x = self.dropout(self.norm2(F.gelu(self.conv2(x)).transpose(1, 2))).transpose(1, 2)
        out, _ = self.lstm(x.transpose(1, 2))
        last = out[:, -1, :]
        return self.head_mu(last), self.head_logvar(last).clamp(-6.0, 2.0)


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (0.5 * (logvar + (y - mu) ** 2 / torch.exp(logvar) + math.log(2 * math.pi))).mean()


def train_model(
    model: CNNLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    patience: int,
    learning_rate: float = 1e-3,
    use_amp: bool = True,
    use_compile: bool = False,
) -> CNNLSTM:
    if use_compile:
        try:
            model = torch.compile(model)
        except Exception as exc:  # fall back to eager if compile is unavailable
            print(f"torch.compile failed ({exc}); continuing in eager mode")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and DEVICE.type == "cuda")
    autocast = torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp and DEVICE.type == "cuda")
    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                mu, logvar = model(X)
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
            for X, y in val_loader:
                X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                with autocast:
                    val_loss += gaussian_nll(*model(X), y).detach() * len(X)
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
    for X, y in loader:
        mu, logvar = model(X.to(DEVICE))
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
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--ar-lags", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--load-workers", type=int, default=None)
    parser.add_argument("--cnn-channels", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--compile", action="store_true", default=False)
    args = parser.parse_args()
    feature_cols = ["log_ret", "vol_chg", "roll_vol_5", "roll_vol_20"]
    config = {
        "seq_len": args.seq_len,
        "cnn_channels": args.cnn_channels,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": 0.3,
        "learning_rate": 5e-4,
        "batch_size": args.batch_size,
    }
    paths = sorted(args.data.glob("*.txt")) if args.data.is_dir() else [args.data]
    if not paths:
        raise FileNotFoundError("No .txt stock data files found")
    load_workers = args.load_workers or min(64, os.cpu_count() or 1)
    seq_len = config["seq_len"]
    train_parts, val_parts, test_parts = [], [], []
    ar_parts = []
    quality = []
    with ProcessPoolExecutor(max_workers=load_workers) as pool:
        for name, feats, target, log_ret, file_quality, n in pool.map(
            _load_file, [(path, feature_cols) for path in paths], chunksize=8
        ):
            quality.append((name, file_quality, n))
            if n <= seq_len:
                continue
            train_end = int(n * 0.8)
            X = np.lib.stride_tricks.sliding_window_view(feats, seq_len, axis=0).transpose(0, 2, 1)
            y = target[seq_len - 1 :]
            indices = np.arange(seq_len - 1, n)
            train_mask = indices < train_end
            val_size = max(1, int(train_mask.sum() * 0.1))
            if train_mask.sum() <= val_size or (~train_mask).sum() == 0:
                continue
            train_X, train_y = X[train_mask][:-val_size], y[train_mask][:-val_size]
            val_X, val_y = X[train_mask][-val_size:], y[train_mask][-val_size:]
            train_parts.append((train_X, train_y))
            val_parts.append((val_X, val_y))
            test_mask = indices >= train_end
            test_parts.append((X[test_mask], y[test_mask]))
            test_indices = indices[test_mask]
            ar_parts.append((log_ret[:train_end], log_ret, target, test_indices))
    X_train = np.concatenate([part[0] for part in train_parts])
    y_train = np.concatenate([part[1] for part in train_parts])
    X_val = np.concatenate([part[0] for part in val_parts])
    y_val = np.concatenate([part[1] for part in val_parts])
    X_test = np.concatenate([part[0] for part in test_parts])
    y_test = np.concatenate([part[1] for part in test_parts])
    train_loader = DataLoader(
        TimeSeriesDataset(X_train, y_train),
        config["batch_size"],
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4,
        drop_last=True,
    )
    val_loader = DataLoader(
        TimeSeriesDataset(X_val, y_val),
        config["batch_size"],
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        TimeSeriesDataset(X_test, y_test),
        config["batch_size"],
        num_workers=args.workers,
        pin_memory=True,
    )
    torch.manual_seed(SEED)
    model = CNNLSTM(
        len(feature_cols),
        config["cnn_channels"],
        config["hidden_size"],
        config["num_layers"],
        config["dropout"],
    ).to(DEVICE)
    model = train_model(
        model, train_loader, val_loader, args.epochs, args.patience,
        config["learning_rate"], use_amp=args.amp, use_compile=args.compile,
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
