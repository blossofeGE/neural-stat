"""
neural_arima_selector_torch.py

Экспериментальная версия селектора ARIMA/SARIMA:
- 42 класса: 18 ARIMA + 8 сезонных шаблонов для s = 4, 8, 12;
- коэффициенты AR/MA не выбираются близкими к нулю;
- проверяются корни объединённых AR/MA-полиномов;
- длина рядов: 96–192 наблюдения;
- по умолчанию 4200 рядов (100 на класс);
- сохраняются модель, scaler, каталог, тестовая выборка и предсказания;
- сохраняются исходные временные ряды для честного теста LLM;
- автоматически создаются blind-пакеты и отдельный ключ ответов;
- bic_winner оставлен в основном наборе признаков;
- дополнительно обучается контрольная MLP без bic_winner;
- результаты автоматически раскладываются по папкам data/, model/,
  artifacts/, results/ и sarima_llm_benchmark/.


Переменные окружения:
    N_SAMPLES=4200
    MAXITER=20
    SEED=42
    RESUME_CHECKPOINT=0  # принудительно сгенерировать датасет заново

Примечание:
Это исследовательский симулятор. Результаты следует проверять на независимых
режимах симуляции и реальных рядах; каталог не является точным воспроизведением
каталога из исходной статьи.
"""

from __future__ import annotations

import json
import os
import zipfile
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Limit native numerical-library threads to avoid oversubscription when
# independent SARIMAX fits run concurrently. Set before importing NumPy/SciPy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import lfilter
from statsmodels.tsa.stattools import adfuller, acf, pacf, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    log_loss,
)
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import binomtest, chi2

warnings.filterwarnings("ignore")

# Project directories. All paths are anchored to this script, not the shell cwd.
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
BENCHMARK_DIR = ROOT / "sarima_llm_benchmark"

for _directory in (DATA_DIR, ARTIFACTS_DIR, MODEL_DIR, RESULTS_DIR, BENCHMARK_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

SEED = int(os.getenv("SEED", "42"))
N_SAMPLES = int(os.getenv("N_SAMPLES", "4200"))
MAXITER = int(os.getenv("MAXITER", "20"))
N_WORKERS = max(1, int(os.getenv("N_WORKERS", "6")))
CHECKPOINT_EVERY = max(10, int(os.getenv("CHECKPOINT_EVERY", "100")))
MIN_LENGTH = 96
MAX_LENGTH = 192

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    # Keep allocator behaviour predictable; batch sizes below control peak VRAM.
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------
# 1. Каталог из 42 спецификаций
# ---------------------------------------------------------------------
def build_catalog():
    catalog = []

    nonseasonal_orders = [
        (1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 0), (0, 0, 2),
        (2, 0, 1), (1, 0, 2), (2, 0, 2), (1, 1, 0), (0, 1, 1),
        (1, 1, 1), (2, 1, 0), (0, 1, 2), (2, 1, 1), (1, 1, 2),
        (2, 1, 2), (0, 2, 1), (1, 2, 1),
    ]
    for order in nonseasonal_orders:
        catalog.append({
            "order": order,
            "seasonal_order": (0, 0, 0, 0),
            "family": "ARIMA",
        })

    templates = [
        ((0, 0, 0), (1, 0, 0)),
        ((0, 0, 0), (0, 0, 1)),
        ((1, 0, 0), (1, 0, 0)),
        ((0, 0, 1), (0, 0, 1)),
        ((1, 0, 1), (1, 0, 0)),
        ((1, 1, 0), (0, 1, 1)),
        ((0, 1, 1), (1, 1, 0)),
        ((1, 1, 1), (0, 1, 1)),
    ]
    for s in (4, 8, 12):
        for order, seasonal in templates:
            P, D, Q = seasonal
            catalog.append({
                "order": order,
                "seasonal_order": (P, D, Q, s),
                "family": f"SARIMA_s{s}",
            })

    keys = [(tuple(c["order"]), tuple(c["seasonal_order"])) for c in catalog]
    assert len(catalog) == 42
    assert len(keys) == len(set(keys)), "Duplicate specification"
    return catalog


CATALOG = build_catalog()
N_CLASSES = len(CATALOG)
LAGS = (1, 2, 3, 4, 7, 8, 12, 16, 24)
FEATURE_NAMES = (
    ["log_n", "adf_c", "adf_ct", "adf_ctt", "kpss_c", "kpss_ct"]
    + [f"acf_{lag}" for lag in LAGS]
    + [f"pacf_{lag}" for lag in LAGS]
    + [
        "variance_ratio_1", "variance_ratio_2", "trend_strength",
        "seasonal_strength_4", "seasonal_strength_8",
        "seasonal_strength_12", "bic_winner",
    ]
)
N_FEATURES = len(FEATURE_NAMES)


def spec_name(spec):
    order = tuple(spec["order"])
    seas = tuple(spec["seasonal_order"])
    if seas[3] == 0:
        return f"ARIMA{order}"
    return f"SARIMA{order}x{seas}"


# ---------------------------------------------------------------------
# 2. Генерация коэффициентов и проверка устойчивости
# ---------------------------------------------------------------------
def draw_nonzero_coefficients(count, rng, low=0.15, high=0.65):
    """Коэффициенты с |phi|/|theta| в [low, high], без значений около нуля."""
    if count == 0:
        return np.array([], dtype=float)
    magnitudes = rng.uniform(low, high, size=count)
    signs = rng.choice([-1.0, 1.0], size=count)
    return magnitudes * signs


def ar_poly(params):
    return np.r_[1.0, -np.asarray(params, dtype=float)]


def ma_poly(params):
    return np.r_[1.0, np.asarray(params, dtype=float)]


def roots_outside_unit_circle(poly, margin=1.01):
    """Для стационарности/обратимости корни соответствующего полинома снаружи."""
    if len(poly) <= 1:
        return True
    roots = np.roots(poly)
    return bool(np.all(np.abs(roots) > margin))


def draw_stable_polynomials(p, q, P, Q, s, rng, max_tries=200):
    """
    Сэмплирует ненулевые коэффициенты, проверяя объединённые полиномы.
    Возвращает коэффициенты AR, MA, сезонные AR и сезонные MA.
    """
    for _ in range(max_tries):
        ar = draw_nonzero_coefficients(p, rng, 0.15, 0.65)
        ma = draw_nonzero_coefficients(q, rng, 0.15, 0.65)
        sar = draw_nonzero_coefficients(P, rng, 0.15, 0.55)
        sma = draw_nonzero_coefficients(Q, rng, 0.15, 0.55)

        ar_combined = ar_poly(ar)
        ma_combined = ma_poly(ma)

        if s:
            seasonal_ar_poly = np.zeros(P * s + 1)
            seasonal_ar_poly[::s] = ar_poly(sar)
            seasonal_ma_poly = np.zeros(Q * s + 1)
            seasonal_ma_poly[::s] = ma_poly(sma)
            ar_combined = np.convolve(ar_combined, seasonal_ar_poly)
            ma_combined = np.convolve(ma_combined, seasonal_ma_poly)

        if (roots_outside_unit_circle(ar_combined)
                and roots_outside_unit_circle(ma_combined)):
            return ar, ma, sar, sma

    # Не делаем бесконечный цикл при сложной спецификации.
    # Более консервативные коэффициенты повышают вероятность стабильности.
    ar = draw_nonzero_coefficients(p, rng, 0.15, 0.30)
    ma = draw_nonzero_coefficients(q, rng, 0.15, 0.30)
    sar = draw_nonzero_coefficients(P, rng, 0.15, 0.25)
    sma = draw_nonzero_coefficients(Q, rng, 0.15, 0.25)
    return ar, ma, sar, sma


def simulate_series(class_id, n, rng):
    """Генерирует приближённый процесс указанного класса."""
    spec = CATALOG[int(class_id)]
    p, d, q = spec["order"]
    P, D, Q, s = spec["seasonal_order"]

    ar, ma, sar, sma = draw_stable_polynomials(p, q, P, Q, s, rng)

    ar_combined = ar_poly(ar)
    ma_combined = ma_poly(ma)

    if s:
        sar_poly = np.zeros(P * s + 1)
        sar_poly[::s] = ar_poly(sar)
        sma_poly = np.zeros(Q * s + 1)
        sma_poly[::s] = ma_poly(sma)
        ar_combined = np.convolve(ar_combined, sar_poly)
        ma_combined = np.convolve(ma_combined, sma_poly)

    burn = max(200, 3 * n)
    eps = rng.normal(0, 1, n + burn)
    x = lfilter(ma_combined, ar_combined, eps)[burn:]

    # Обычное дифференцирование / интегрирование.
    for _ in range(d):
        x = np.cumsum(x)

    # Сезонное интегрирование: (1 - B^s)^(-D).
    for _ in range(D):
        for t in range(s, len(x)):
            x[t] += x[t - s]

    # Масштабирование без искусственного дополнительного тренда.
    x = x * rng.uniform(0.5, 2.0)
    sd = np.std(x)
    if not np.isfinite(sd) or sd < 1e-8 or not np.all(np.isfinite(x)):
        return simulate_series(class_id, n, rng)
    return (x - np.mean(x)) / sd


# ---------------------------------------------------------------------
# 3. Признаки и BIC
# ---------------------------------------------------------------------
def safe_adf(x, regression):
    try:
        return float(adfuller(x, regression=regression, autolag="AIC")[1])
    except Exception:
        return 1.0


def safe_kpss(x, regression):
    try:
        return float(kpss(x, regression=regression, nlags="auto")[1])
    except Exception:
        return 0.0


def seasonal_strength(x, period):
    if len(x) <= 2 * period:
        return 0.0
    phase_means = np.array([np.mean(x[i::period]) for i in range(period)])
    seasonal = np.tile(phase_means, int(np.ceil(len(x) / period)))[:len(x)]
    denom = np.var(x)
    if denom < 1e-12:
        return 0.0
    return float(np.clip(1.0 - np.var(x - seasonal) / denom, 0.0, 1.0))


def extract_diagnostics(x, maxiter=MAXITER):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < MIN_LENGTH:
        raise ValueError(f"At least {MIN_LENGTH} finite observations are required.")

    adfs = [safe_adf(x, r) for r in ("c", "ct", "ctt")]
    kpss_vals = [safe_kpss(x, r) for r in ("c", "ct")]

    max_lag = max(LAGS)
    acf_vals = acf(x, nlags=min(max_lag, len(x) - 1), fft=True, missing="drop")
    pacf_vals = pacf(x, nlags=min(max_lag, (len(x) - 1) // 2), method="ywm")
    acf_features = [float(acf_vals[k]) if k < len(acf_vals) else 0.0 for k in LAGS]
    pacf_features = [float(pacf_vals[k]) if k < len(pacf_vals) else 0.0 for k in LAGS]

    dx = np.diff(x)
    d2x = np.diff(x, n=2)
    variance_ratio_1 = float(np.log((np.var(dx) + 1e-9) / (np.var(x) + 1e-9)))
    variance_ratio_2 = float(np.log((np.var(d2x) + 1e-9) / (np.var(dx) + 1e-9)))
    t = np.arange(len(x), dtype=float)
    slope = np.polyfit(t, x, 1)[0]
    trend_strength = float(np.clip(abs(slope) * len(x) / (np.std(x) + 1e-9), 0, 20))
    seasonal_features = [seasonal_strength(x, s) for s in (4, 8, 12)]

    bics = []
    for spec in CATALOG:
        try:
            fitted = SARIMAX(
                x,
                order=tuple(spec["order"]),
                seasonal_order=tuple(spec["seasonal_order"]),
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=maxiter)
            bic = float(fitted.bic)
            bics.append(bic if np.isfinite(bic) else np.inf)
        except Exception:
            bics.append(np.inf)

    bic_winner = int(np.argmin(bics)) if np.isfinite(bics).any() else 0
    features = np.asarray(
        [np.log(len(x)), *adfs, *kpss_vals]
        + acf_features + pacf_features
        + [variance_ratio_1, variance_ratio_2, trend_strength]
        + seasonal_features + [float(bic_winner)],
        dtype=float,
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    assert len(features) == N_FEATURES, (len(features), N_FEATURES)
    return features, bic_winner


# ---------------------------------------------------------------------
# 4. Создание датасета
# ---------------------------------------------------------------------
def _build_one_sample(task):
    """Worker function; per-row RNG makes results independent of scheduling."""
    i, class_id, n, row_seed = task
    local_rng = np.random.default_rng(row_seed)
    series = simulate_series(class_id, n, local_rng)
    feats, bic_winner = extract_diagnostics(series)
    # Fixed-width representation; values after n are NaN and are ignored via lengths.
    padded = np.full(MAX_LENGTH, np.nan, dtype=np.float32)
    padded[:n] = series.astype(np.float32)
    return i, feats, class_id, bic_winner, n, padded


def make_dataset(n_samples=N_SAMPLES):
    if n_samples < N_CLASSES * 2:
        raise ValueError(f"N_SAMPLES должен быть не меньше {N_CLASSES * 2}.")

    checkpoint = DATA_DIR / "sarima_selector_dataset_checkpoint.npz"
    X, y, bic_pred, lengths, series_rows = [], [], [], [], []
    start_i = 0

    # Resume only if checkpoint metadata matches the current run.
    if checkpoint.exists() and os.getenv("RESUME_CHECKPOINT", "1") == "1":
        try:
            with np.load(checkpoint, allow_pickle=False) as saved:
                if (int(saved["n_samples"]) == n_samples
                        and int(saved["seed"]) == SEED
                        and int(saved["maxiter"]) == MAXITER
                        and int(saved["n_classes"]) == N_CLASSES
                        and "series_padded" in saved.files):
                    X = saved["X"].tolist()
                    y = saved["y"].tolist()
                    bic_pred = saved["bic_pred"].tolist()
                    lengths = saved["lengths"].tolist()
                    series_rows = saved["series_padded"].tolist()
                    start_i = len(y)
                    if start_i < n_samples:
                        print(f"Resuming from checkpoint: {start_i}/{n_samples} rows")
                    else:
                        print(f"Complete dataset checkpoint found: {start_i}/{n_samples} rows")
                else:
                    print("Existing checkpoint does not match settings; starting fresh.")
        except Exception as exc:
            print(f"Could not read checkpoint ({exc}); starting fresh.")
            X, y, bic_pred, lengths, series_rows, start_i = [], [], [], [], [], 0

    if start_i < n_samples:
        # Create deterministic tasks up front; executor.map preserves input order.
        tasks = []
        for i in range(start_i, n_samples):
            class_id = i % N_CLASSES
            # Row-local seed keeps each sample reproducible even after resuming.
            row_rng = np.random.default_rng(np.random.SeedSequence([SEED, i]))
            n = int(row_rng.integers(MIN_LENGTH, MAX_LENGTH + 1))
            row_seed = int(row_rng.integers(0, np.iinfo(np.int32).max))
            tasks.append((i, class_id, n, row_seed))

        print(f"Generating {len(tasks)} rows with {N_WORKERS} worker threads.")
        try:
            from threadpoolctl import threadpool_limits
            limits_context = threadpool_limits(limits=1)
        except Exception:
            from contextlib import nullcontext
            limits_context = nullcontext()

        with limits_context:
            if N_WORKERS == 1:
                results_iter = map(_build_one_sample, tasks)
                for done, result in enumerate(results_iter, start=start_i + 1):
                    _, feats, class_id, bic_winner, n, padded = result
                    X.append(feats); y.append(class_id); bic_pred.append(bic_winner)
                    lengths.append(n); series_rows.append(padded)
                    if done % CHECKPOINT_EVERY == 0 or done == n_samples:
                        np.savez_compressed(checkpoint, X=np.asarray(X, dtype=np.float32),
                            y=np.asarray(y, dtype=np.int64), bic_pred=np.asarray(bic_pred, dtype=np.int64),
                            lengths=np.asarray(lengths, dtype=np.int64),
                            series_padded=np.asarray(series_rows, dtype=np.float32),
                            n_samples=n_samples, seed=SEED, maxiter=MAXITER, n_classes=N_CLASSES)
                        print(f"Built {done}/{n_samples} samples; checkpoint saved")
            else:
                with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
                    for done, result in enumerate(pool.map(_build_one_sample, tasks), start=start_i + 1):
                        _, feats, class_id, bic_winner, n, padded = result
                        X.append(feats); y.append(class_id); bic_pred.append(bic_winner)
                        lengths.append(n); series_rows.append(padded)
                        if done % CHECKPOINT_EVERY == 0 or done == n_samples:
                            np.savez_compressed(checkpoint, X=np.asarray(X, dtype=np.float32),
                                y=np.asarray(y, dtype=np.int64), bic_pred=np.asarray(bic_pred, dtype=np.int64),
                                lengths=np.asarray(lengths, dtype=np.int64),
                                series_padded=np.asarray(series_rows, dtype=np.float32),
                                n_samples=n_samples, seed=SEED, maxiter=MAXITER, n_classes=N_CLASSES)
                            print(f"Built {done}/{n_samples} samples; checkpoint saved")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    bic_pred = np.asarray(bic_pred, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    series_padded = np.asarray(series_rows, dtype=np.float32)
    order = rng.permutation(len(y))
    X, y, bic_pred, lengths, series_padded = (
        X[order], y[order], bic_pred[order], lengths[order], series_padded[order]
    )
    np.savez_compressed(
        DATA_DIR / "sarima_selector_full_dataset.npz",
        X=X, y=y, bic_pred=bic_pred, lengths=lengths,
        series_padded=series_padded,
    )
    return X, y, bic_pred, lengths, series_padded


class SelectorMLP(nn.Module):
    """PyTorch equivalent of the original Keras 64 -> 32 -> 42 MLP."""

    def __init__(self, n_features=N_FEATURES):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, N_CLASSES),
        )

    def forward(self, x):
        return self.network(x)


def train_mlp(X_scaled, y, n_features, class_weights, seed):
    """Train with Adam, batch size 64 and early stopping on held-out validation loss."""
    model = SelectorMLP(n_features).to(DEVICE)
    y = np.asarray(y, dtype=np.int64)

    # Match Keras validation_split=0.15 semantics: last 15% is validation.
    split_at = int(len(X_scaled) * (1.0 - 0.15))
    if split_at <= 0 or split_at >= len(X_scaled):
        raise ValueError("Not enough training rows for a 15% validation split.")

    x_train = torch.as_tensor(X_scaled[:split_at], dtype=torch.float32)
    y_train = torch.as_tensor(y[:split_at], dtype=torch.long)
    x_val = torch.as_tensor(X_scaled[split_at:], dtype=torch.float32, device=DEVICE)
    y_val = torch.as_tensor(y[split_at:], dtype=torch.long, device=DEVICE)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=64,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    weights = torch.ones(N_CLASSES, dtype=torch.float32)
    for class_id, weight in class_weights.items():
        weights[int(class_id)] = float(weight)
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")
    best_state = None
    patience = 12
    stale_epochs = 0

    for epoch in range(1, 151):
        model.train()
        running_loss = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(yb)
            seen += len(yb)

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = float(criterion(val_logits, y_val).item())
            val_acc = float((val_logits.argmax(dim=1) == y_val).float().mean().item())

        print(
            f"Epoch {epoch:03d}/150 - "
            f"loss: {running_loss / max(seen, 1):.4f} "
            f"- val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}"
        )

        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch}; best val_loss={best_loss:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_probabilities(model, X_scaled, batch_size=1024):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(X_scaled), batch_size):
            xb = torch.as_tensor(
                X_scaled[start:start + batch_size],
                dtype=torch.float32,
                device=DEVICE,
            )
            probs = torch.softmax(model(xb), dim=1)
            chunks.append(probs.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def evaluate_method(name, y_true, y_pred, probabilities=None):
    result = {
        "method": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if probabilities is not None:
        try:
            result["macro_roc_auc_ovr"] = roc_auc_score(
                y_true, probabilities, multi_class="ovr", average="macro",
                labels=np.arange(N_CLASSES),
            )
        except ValueError:
            result["macro_roc_auc_ovr"] = np.nan
        try:
            result["log_loss"] = log_loss(
                y_true, probabilities, labels=np.arange(N_CLASSES)
            )
        except ValueError:
            result["log_loss"] = np.nan
    else:
        result["macro_roc_auc_ovr"] = np.nan
        result["log_loss"] = np.nan
    return result



def feature_dictionary():
    """Human-readable definitions matching extract_diagnostics()."""
    descriptions = {
        "log_n": "Натуральный логарифм длины временного ряда.",
        "adf_c": "p-value теста ADF с константой (regression='c').",
        "adf_ct": "p-value теста ADF с константой и линейным трендом (regression='ct').",
        "adf_ctt": "p-value теста ADF с константой, линейным и квадратичным трендом (regression='ctt').",
        "kpss_c": "p-value теста KPSS с константой (regression='c').",
        "kpss_ct": "p-value теста KPSS с константой и линейным трендом (regression='ct').",
        "variance_ratio_1": "log((var(diff(x))+1e-9)/(var(x)+1e-9)).",
        "variance_ratio_2": "log((var(diff(x, n=2))+1e-9)/(var(diff(x))+1e-9)).",
        "trend_strength": "clip(abs(linear_slope)*n/(std(x)+1e-9), 0, 20).",
        "seasonal_strength_4": "Сезонная сила для периода 4, рассчитанная по средним значениям фаз.",
        "seasonal_strength_8": "Сезонная сила для периода 8, рассчитанная по средним значениям фаз.",
        "seasonal_strength_12": "Сезонная сила для периода 12, рассчитанная по средним значениям фаз.",
        "bic_winner": "Индекс класса с минимальным BIC среди 42 SARIMAX-кандидатов. Это сильная дополнительная подсказка, поэтому отдельно предусмотрен тест без этого признака.",
    }
    for lag in LAGS:
        descriptions[f"acf_{lag}"] = f"Значение ACF на лаге {lag}."
        descriptions[f"pacf_{lag}"] = f"Значение PACF на лаге {lag}."
    return [{"feature": name, "description": descriptions.get(name, "")}
            for name in FEATURE_NAMES]


def export_llm_benchmark(X_test, y_test, bic_test, len_test, series_test_padded,
                         nn_pred, nn_pred_no_bic=None):
    """Create a blind LLM input package and a separate private evaluation key."""
    out = BENCHMARK_DIR
    public_dir = out / "llm_input"
    private_dir = out / "private_evaluation"
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    ids = [f"series_{i + 1:04d}" for i in range(len(y_test))]
    catalog_rows = []
    for class_id, spec in enumerate(CATALOG):
        catalog_rows.append({
            "class_id": class_id,
            "specification": spec_name(spec),
            "order": json.dumps(list(spec["order"])),
            "seasonal_order": json.dumps(list(spec["seasonal_order"])),
            "family": spec["family"],
        })

    pd.DataFrame(catalog_rows).to_csv(public_dir / "class_catalog.csv", index=False)
    (public_dir / "class_catalog.json").write_text(
        json.dumps(catalog_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(feature_dictionary()).to_csv(
        public_dir / "feature_dictionary.csv", index=False, encoding="utf-8-sig"
    )

    feature_df = pd.DataFrame(X_test, columns=FEATURE_NAMES)
    feature_df.insert(0, "series_id", ids)
    feature_df.to_csv(public_dir / "blind_test_features_31.csv", index=False, float_format="%.9g")
    feature_df.drop(columns=["bic_winner"]).to_csv(
        public_dir / "blind_test_features_30.csv", index=False, float_format="%.9g"
    )

    # Long format avoids confusing padded values with real observations.
    raw_rows = []
    for i, (series_id, n) in enumerate(zip(ids, len_test)):
        values = series_test_padded[i, :int(n)]
        raw_rows.extend(
            {"series_id": series_id, "time_index": t, "value": float(value)}
            for t, value in enumerate(values)
        )
    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(
        public_dir / "blind_test_raw_series.csv", index=False, float_format="%.9g"
    )

    # Batch exports make the 1,050-object benchmark practical for LLM context limits.
    batch_root = public_dir / "batches"
    for track in ("raw_series", "features_31", "features_30"):
        (batch_root / track).mkdir(parents=True, exist_ok=True)
    batch_size = 25
    feature_df_30 = feature_df.drop(columns=["bic_winner"])
    for batch_no, start in enumerate(range(0, len(ids), batch_size), start=1):
        batch_ids = ids[start:start + batch_size]
        batch_tag = f"batch_{batch_no:03d}.csv"
        raw_df[raw_df["series_id"].isin(batch_ids)].to_csv(
            batch_root / "raw_series" / batch_tag, index=False, float_format="%.9g"
        )
        feature_df[feature_df["series_id"].isin(batch_ids)].to_csv(
            batch_root / "features_31" / batch_tag, index=False, float_format="%.9g"
        )
        feature_df_30[feature_df_30["series_id"].isin(batch_ids)].to_csv(
            batch_root / "features_30" / batch_tag, index=False, float_format="%.9g"
        )

    response_template = pd.DataFrame({
        "series_id": ids,
        "predicted_class_id": ["" for _ in ids],
        "predicted_specification": ["" for _ in ids],
        "confidence": ["" for _ in ids],
    })
    response_template.to_csv(public_dir / "llm_response_template.csv", index=False)

    instructions = """SARIMA / ARIMA LLM BLIND BENCHMARK

The blind test contains held-out synthetic time series. There are 42 permitted classes.
Use class_catalog.csv (or class_catalog.json) to map class_id to a specification.
Do not invent class IDs outside 0..41.

Three separate tracks are provided. Run them independently:
A) RAW SERIES: blind_test_raw_series.csv
   Each object is a sequence of (time_index, value) rows. Infer one class per series_id.
B) FEATURES-31: blind_test_features_31.csv plus feature_dictionary.csv
   Uses all 31 diagnostics, including bic_winner.
C) FEATURES-30: blind_test_features_30.csv plus feature_dictionary.csv
   Same diagnostics with bic_winner removed.

For context limits, batches of 25 objects are also provided under batches/
in separate raw_series, features_31, and features_30 folders. Process every
batch in the chosen track and concatenate the prediction CSVs.

Return one CSV per track with exactly these columns:
series_id,predicted_class_id,predicted_specification,confidence
You may use llm_response_template.csv as the starting template.
Use every series_id exactly once. predicted_class_id must be an integer 0..41.
confidence is optional, from 0 to 1. Do not omit objects.

Do not ask for or infer the hidden ground-truth labels.
Do not combine results from separate tracks.
The test set is balanced in the original generation design, but do not assume labels.
"""
    (public_dir / "LLM_INSTRUCTIONS.txt").write_text(instructions, encoding="utf-8")

    # Hidden truth and model predictions are kept out of the LLM input ZIP.
    key_df = pd.DataFrame({
        "series_id": ids,
        "true_class_id": y_test.astype(int),
        "true_specification": [spec_name(CATALOG[int(v)]) for v in y_test],
        "bic_class_id": bic_test.astype(int),
        "bic_specification": [spec_name(CATALOG[int(v)]) for v in bic_test],
        "neural_class_id_features31": nn_pred.astype(int),
    })
    if nn_pred_no_bic is not None:
        key_df["neural_class_id_features30"] = nn_pred_no_bic.astype(int)
    key_df.to_csv(private_dir / "answer_key_and_baselines.csv", index=False)

    # Useful NPZ for reproducible offline scoring and later analyses.
    np.savez_compressed(
        private_dir / "test_arrays_private.npz",
        X_test=X_test, y_test=y_test, bic_test=bic_test,
        lengths=len_test, series_padded=series_test_padded,
    )

    readme = """PACKAGE LAYOUT

llm_input.zip:
  Upload ONLY this archive's contents to the LLM being evaluated.
  Contains raw test series, feature-only variants, feature dictionary,
  the 42-class catalog, and response instructions.
  Does NOT contain true labels or neural predictions.

private_evaluation.zip:
  Keep private. Contains ground-truth labels, BIC baseline predictions,
  neural predictions, and aligned test arrays.

Fair-comparison notes:
- FEATURES-31 is the direct same-feature comparison with the original MLP.
- FEATURES-30 must be compared with the separately trained 30-feature MLP.
- RAW SERIES gives the LLM the original sequence. The current MLP consumes
  engineered diagnostics, so this is an end-to-end task comparison, not a
  strictly identical-input comparison.
- bic_winner is derived by fitting all 42 candidate SARIMAX models and may
  strongly reveal the target. Report results both with and without it.
"""
    (out / "README.txt").write_text(readme, encoding="utf-8")

    def make_zip(zip_path, files, base):
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in files:
                zf.write(file_path, arcname=str(Path(file_path).relative_to(base)))

    public_zip = BENCHMARK_DIR / "sarima_llm_input_only.zip"
    public_files = [p for p in public_dir.rglob("*") if p.is_file()]
    make_zip(public_zip, public_files, public_dir)

    private_zip = BENCHMARK_DIR / "sarima_llm_private_evaluation.zip"
    private_files = [p for p in private_dir.rglob("*") if p.is_file()]
    make_zip(private_zip, private_files, private_dir)

    print("\nLLM benchmark files created:")
    print("-", public_zip, "(safe to upload to the LLM)")
    print("-", private_zip, "(KEEP PRIVATE; contains correct answers)")
    print("-", out / "README.txt")


def main():
    print(f"Catalog: {N_CLASSES} classes; features: {N_FEATURES}")
    print(f"Samples: {N_SAMPLES}; lengths: {MIN_LENGTH}-{MAX_LENGTH}; MAXITER={MAXITER}; workers={N_WORKERS}")
    print("Each row fits 42 SARIMAX models; parallel workers and checkpoints are enabled.")
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("Compute device:", DEVICE)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

    X, y, bic_pred, lengths, series_padded = make_dataset()

    X_train, X_test, y_train, y_test, bic_train, bic_test, len_train, len_test, series_train, series_test = train_test_split(
        X, y, bic_pred, lengths, series_padded,
        test_size=0.25,
        random_state=SEED,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    print(f"Training primary MLP on {DEVICE} ...")
    model = train_mlp(
        X_train_scaled, y_train, X_train_scaled.shape[1],
        class_weights, seed=SEED,
    )
    probs = predict_probabilities(model, X_test_scaled)
    nn_pred = probs.argmax(axis=1)
    bic_probs = np.eye(N_CLASSES)[bic_test]

    # Ablation model: train a second MLP without bic_winner (last feature).
    X_train_30 = X_train[:, :-1]
    X_test_30 = X_test[:, :-1]
    scaler_30 = StandardScaler()
    X_train_30_scaled = scaler_30.fit_transform(X_train_30)
    X_test_30_scaled = scaler_30.transform(X_test_30)
    print(f"Training ablation MLP (without bic_winner) on {DEVICE} ...")
    model_30 = train_mlp(
        X_train_30_scaled, y_train, X_train_30_scaled.shape[1],
        class_weights, seed=SEED + 1,
    )
    probs_30 = predict_probabilities(model_30, X_test_30_scaled)
    nn_pred_30 = probs_30.argmax(axis=1)

    results_df = pd.DataFrame([
        evaluate_method("Neural MLP (31 features)", y_test, nn_pred, probs),
        evaluate_method("Neural MLP (30 features, no bic_winner)", y_test, nn_pred_30, probs_30),
        evaluate_method("Minimum BIC", y_test, bic_test, bic_probs),
    ])
    print("\n=== Comparative metrics ===")
    print(results_df.to_string(index=False, float_format=lambda z: f"{z:.4f}"))

    print("\n=== Per-class report: neural MLP ===")
    print(classification_report(
        y_test, nn_pred,
        labels=np.arange(N_CLASSES),
        target_names=[f"{i}: {spec_name(s)}" for i, s in enumerate(CATALOG)],
        zero_division=0,
    ))

    nn_correct = nn_pred == y_test
    bic_correct = bic_test == y_test
    n10 = int(np.sum(nn_correct & ~bic_correct))
    n01 = int(np.sum(~nn_correct & bic_correct))
    discordant = n10 + n01
    p_exact = float(binomtest(min(n10, n01), discordant, p=0.5).pvalue) if discordant else 1.0
    stat_cc = ((abs(n10 - n01) - 1) ** 2 / discordant) if discordant else 0.0
    p_cc = float(chi2.sf(stat_cc, 1)) if discordant else 1.0

    print("\n=== McNemar test (exact two-sided) ===")
    print(f"n10 (NN correct, BIC wrong): {n10}")
    print(f"n01 (NN wrong, BIC correct): {n01}")
    print(f"Exact p-value: {p_exact:.8g}")
    print(f"Continuity-corrected chi-square p-value: {p_cc:.8g}")

    # Save artifacts so future evaluation does not require regeneration.
    torch.save({
        "state_dict": model.state_dict(),
        "n_features": int(X_train_scaled.shape[1]),
        "n_classes": N_CLASSES,
        "feature_names": FEATURE_NAMES,
        "torch_version": torch.__version__,
    }, MODEL_DIR / "sarima_selector.pt")
    np.savez(ARTIFACTS_DIR / "sarima_selector_scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
    torch.save({
        "state_dict": model_30.state_dict(),
        "n_features": int(X_train_30_scaled.shape[1]),
        "n_classes": N_CLASSES,
        "feature_names": FEATURE_NAMES[:-1],
        "torch_version": torch.__version__,
    }, MODEL_DIR / "sarima_selector_no_bic.pt")
    np.savez(ARTIFACTS_DIR / "sarima_selector_no_bic_scaler.npz", mean=scaler_30.mean_, scale=scaler_30.scale_)
    Path("sarima_selector_catalog.json").write_text(
        json.dumps([
            {
                "class_id": i,
                "name": spec_name(spec),
                "order": list(spec["order"]),
                "seasonal_order": list(spec["seasonal_order"]),
                "family": spec["family"],
            }
            for i, spec in enumerate(CATALOG)
        ], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    Path("sarima_selector_feature_names.json").write_text(
        json.dumps(FEATURE_NAMES, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        ARTIFACTS_DIR / "sarima_selector_test_data.npz",
        X_test=X_test, y_test=y_test, bic_test=bic_test,
        lengths=len_test, series_padded=series_test,
    )
    results_df.to_csv(RESULTS_DIR / "sarima_selector_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "n10_nn_correct_bic_wrong": n10,
        "n01_nn_wrong_bic_correct": n01,
        "mcnemar_exact_p": p_exact,
        "mcnemar_chi2_cc_statistic": stat_cc,
        "mcnemar_chi2_cc_p": p_cc,
    }]).to_csv(RESULTS_DIR / "sarima_selector_mcnemar.csv", index=False, encoding="utf-8-sig")

    pred_rows = []
    for i, (truth, nn_class, bic_class, row_probs) in enumerate(zip(y_test, nn_pred, bic_test, probs)):
        top3 = np.argsort(row_probs)[::-1][:3]
        pred_rows.append({
            "row": i,
            "true_class_id": int(truth),
            "true_specification": spec_name(CATALOG[int(truth)]),
            "neural_class_id": int(nn_class),
            "neural_specification": spec_name(CATALOG[int(nn_class)]),
            "neural_correct": bool(nn_class == truth),
            "bic_class_id": int(bic_class),
            "bic_specification": spec_name(CATALOG[int(bic_class)]),
            "bic_correct": bool(bic_class == truth),
            "series_length": int(len_test[i]),
            "top1_score": float(row_probs[top3[0]]),
            "top2_class_id": int(top3[1]),
            "top2_specification": spec_name(CATALOG[int(top3[1])]),
            "top2_score": float(row_probs[top3[1]]),
            "top3_class_id": int(top3[2]),
            "top3_specification": spec_name(CATALOG[int(top3[2])]),
            "top3_score": float(row_probs[top3[2]]),
        })
    pd.DataFrame(pred_rows).to_csv(
        RESULTS_DIR / "sarima_selector_test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(classification_report(
        y_test, nn_pred_30,
        labels=np.arange(N_CLASSES),
        target_names=[f"{i}: {spec_name(s)}" for i, s in enumerate(CATALOG)],
        zero_division=0, output_dict=True,
    )).transpose().to_csv(RESULTS_DIR / "sarima_selector_no_bic_classification_report.csv", encoding="utf-8-sig")

    export_llm_benchmark(
        X_test, y_test, bic_test, len_test, series_test,
        nn_pred=nn_pred, nn_pred_no_bic=nn_pred_30,
    )

    print("\nSaved files:")
    saved_files = [
        MODEL_DIR / "sarima_selector.pt",
        ARTIFACTS_DIR / "sarima_selector_scaler.npz",
        MODEL_DIR / "sarima_selector_no_bic.pt",
        ARTIFACTS_DIR / "sarima_selector_no_bic_scaler.npz",
        BENCHMARK_DIR / "sarima_llm_input_only.zip",
        BENCHMARK_DIR / "sarima_llm_private_evaluation.zip",
        ROOT / "sarima_selector_catalog.json",
        ROOT / "sarima_selector_feature_names.json",
        ARTIFACTS_DIR / "sarima_selector_test_data.npz",
        DATA_DIR / "sarima_selector_full_dataset.npz",
        RESULTS_DIR / "sarima_selector_no_bic_classification_report.csv",
        RESULTS_DIR / "sarima_selector_metrics.csv",
        RESULTS_DIR / "sarima_selector_mcnemar.csv",
        RESULTS_DIR / "sarima_selector_test_predictions.csv",
        DATA_DIR / "sarima_selector_dataset_checkpoint.npz",
        BENCHMARK_DIR / "private_evaluation" / "answer_key_and_baselines.csv",
        BENCHMARK_DIR / "private_evaluation" / "test_arrays_private.npz",
        BENCHMARK_DIR / "README.txt",
    ]
    for path in saved_files:
        print("-", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
