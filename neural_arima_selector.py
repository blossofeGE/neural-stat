"""
Universal neural ARIMA/SARIMA specification selector (expanded ~50-class catalog).

Purpose
-------
Classify the data-generating specification from interpretable time-series
diagnostics. This is specification identification, NOT direct forecast optimization.

Default catalog: 50 candidate classes spanning:
- nonseasonal ARIMA orders
- seasonal SARIMA orders for s in {4, 12, 7, 24}
The exact catalog is defined programmatically below and saved to JSON.

Install:
    pip install numpy pandas scipy statsmodels scikit-learn tensorflow

Run:
    python neural_arima_selector_v2.py
Optional environment variables:
    N_SAMPLES=3000       number of synthetic training examples (default 2000)
    MAXITER=35           optimizer iterations per candidate model
    SEED=42              random seed

Caveats
-------
1. The catalog is a configurable research design, not the exact class list in
   the uploaded paper (its table's mathematical notation was not recoverable).
2. Synthetic labels are only meaningful to the extent the simulator matches
   the intended process family. Validate on independent simulation regimes and
   real series where possible.
3. Simulation uses a practical approximate ARMA/SARMA filter. It is not a
   universal exact sampler for every nonstationary seasonal specification.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from statsmodels.tsa.stattools import adfuller, acf, pacf, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, log_loss
)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

warnings.filterwarnings("ignore")

SEED = int(os.getenv("SEED", "42"))
N_SAMPLES = int(os.getenv("N_SAMPLES", "20000"))
MAXITER = int(os.getenv("MAXITER", "35"))
rng = np.random.default_rng(SEED)
tf.keras.utils.set_random_seed(SEED)

# ---------------------------------------------------------------------------
# 1. Create an explicit catalog with exactly 50 classes.
# ---------------------------------------------------------------------------
def build_catalog():
    catalog = []

    # 18 nonseasonal ARIMA candidates.
    for order in [
        (1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 0), (0, 0, 2),
        (2, 0, 1), (1, 0, 2), (2, 0, 2), (1, 1, 0), (0, 1, 1),
        (1, 1, 1), (2, 1, 0), (0, 1, 2), (2, 1, 1), (1, 1, 2),
        (2, 1, 2), (0, 2, 1), (1, 2, 1)
    ]:
        catalog.append({"order": order, "seasonal_order": (0, 0, 0, 0),
                        "family": "ARIMA"})

    # 8 seasonal templates for each of four common periods = 32 classes.
    templates = [
        ((0, 0, 0), (1, 0, 0)),  # seasonal AR
        ((0, 0, 0), (0, 0, 1)),  # seasonal MA
        ((1, 0, 0), (1, 0, 0)),
        ((0, 0, 1), (0, 0, 1)),
        ((1, 0, 1), (1, 0, 0)),
        ((1, 1, 0), (0, 1, 1)),
        ((0, 1, 1), (1, 1, 0)),
        ((1, 1, 1), (0, 1, 1)),
    ]
    for s in (4, 12, 7, 24):
        for order, seas in templates:
            P, D, Q = seas
            catalog.append({
                "order": order,
                "seasonal_order": (P, D, Q, s),
                "family": f"SARIMA_s{s}"
            })

    assert len(catalog) == 50
    # Deduplicate accidental duplicate definitions if catalog is edited.
    keys = [(tuple(c["order"]), tuple(c["seasonal_order"])) for c in catalog]
    assert len(keys) == len(set(keys)), "Duplicate specification in catalog"
    return catalog


CATALOG = build_catalog()
N_CLASSES = len(CATALOG)
LAGS = (1, 2, 3, 4, 7, 12, 24)
FEATURE_NAMES = (
    ["log_n", "adf_c", "adf_ct", "adf_ctt", "kpss_c", "kpss_ct"]
    + [f"acf_{lag}" for lag in LAGS]
    + [f"pacf_{lag}" for lag in LAGS]
    + ["variance_ratio_1", "variance_ratio_2", "trend_strength",
       "seasonal_strength_4", "seasonal_strength_7",
       "seasonal_strength_12", "seasonal_strength_24", "bic_winner"]
)
N_FEATURES = len(FEATURE_NAMES)


def spec_name(spec):
    order = tuple(spec["order"])
    seas = tuple(spec["seasonal_order"])
    if seas[3] == 0:
        return f"ARIMA{order}"
    return f"SARIMA{order}x{seas}"


def _ar_poly(params):
    return np.r_[1.0, -np.asarray(params, dtype=float)]


def _ma_poly(params):
    return np.r_[1.0, np.asarray(params, dtype=float)]


def simulate_series(class_id, n, rng):
    """Generate an approximate process for the specified candidate class."""
    spec = CATALOG[int(class_id)]
    p, d, q = spec["order"]
    P, D, Q, s = spec["seasonal_order"]

    # Random coefficients kept moderate to avoid frequent explosive processes.
    ar = rng.uniform(-0.35, 0.35, size=p)
    ma = rng.uniform(-0.35, 0.35, size=q)
    sar = rng.uniform(-0.30, 0.30, size=P)
    sma = rng.uniform(-0.30, 0.30, size=Q)

    ar_poly = _ar_poly(ar)
    ma_poly = _ma_poly(ma)
    if s:
        sar_poly = np.zeros(P * s + 1)
        sar_poly[::s] = _ar_poly(sar)
        sma_poly = np.zeros(Q * s + 1)
        sma_poly[::s] = _ma_poly(sma)
        ar_poly = np.convolve(ar_poly, sar_poly)
        ma_poly = np.convolve(ma_poly, sma_poly)

    burn = max(100, 2 * n)
    eps = rng.normal(0, 1, n + burn)
    x = lfilter(ma_poly, ar_poly, eps)[burn:]

    # Ordinary and seasonal differencing integration.
    for _ in range(d):
        x = np.cumsum(x)
    for _ in range(D):
        if s:
            for i in range(s, len(x)):
                x[i] += x[i - s]

    # Random scale and optional weak trend/level to broaden the simulator.
    x = x * rng.uniform(0.5, 2.0)
    if rng.random() < 0.25:
        x = x + np.linspace(0, rng.normal(0, 0.25 * np.std(x) + 1e-6), len(x))
    sd = np.std(x)
    if not np.isfinite(sd) or sd < 1e-8 or not np.all(np.isfinite(x)):
        return simulate_series(class_id, n, rng)
    return (x - np.mean(x)) / sd


def _safe_adf(x, regression):
    try:
        return float(adfuller(x, regression=regression, autolag="AIC")[1])
    except Exception:
        return 1.0


def _safe_kpss(x, regression):
    try:
        return float(kpss(x, regression=regression, nlags="auto")[1])
    except Exception:
        return 0.0


def _seasonal_strength(x, period):
    if len(x) <= 2 * period:
        return 0.0
    # Simple variance-reduction proxy after subtracting phase-specific means.
    phase_means = np.array([np.mean(x[i::period]) for i in range(period)])
    seasonal = np.tile(phase_means, int(np.ceil(len(x) / period)))[:len(x)]
    denom = np.var(x)
    if denom < 1e-12:
        return 0.0
    return float(np.clip(1.0 - np.var(x - seasonal) / denom, 0.0, 1.0))


def extract_diagnostics(x, maxiter=MAXITER):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 40:
        raise ValueError("At least 40 finite observations are required.")

    adfs = [_safe_adf(x, r) for r in ("c", "ct", "ctt")]
    kpss_vals = [_safe_kpss(x, r) for r in ("c", "ct")]
    # ACF can be computed at larger lags, but statsmodels PACF requires
    # nlags < 50% of the sample size. Use the same fixed feature layout for
    # every series and pad unavailable PACF lags with 0.0.
    max_lag = max(LAGS)
    acf_nlags = min(max_lag, len(x) - 1)
    acf_vals = acf(x, nlags=acf_nlags, fft=True, missing="drop")

    # PACF maximum must be strictly less than nobs // 2.
    pacf_nlags = min(max_lag, (len(x) - 1) // 2)
    pacf_vals = pacf(x, nlags=pacf_nlags, method="ywm")

    acf_features = [
        float(acf_vals[k]) if k < len(acf_vals) else 0.0 for k in LAGS
    ]
    pacf_features = [
        float(pacf_vals[k]) if k < len(pacf_vals) else 0.0 for k in LAGS
    ]

    # Robust-ish scale and trend summaries.
    dx = np.diff(x)
    d2x = np.diff(x, n=2)
    variance_ratio_1 = float(np.log((np.var(dx) + 1e-9) / (np.var(x) + 1e-9)))
    variance_ratio_2 = float(np.log((np.var(d2x) + 1e-9) / (np.var(dx) + 1e-9)))
    t = np.arange(len(x), dtype=float)
    slope = np.polyfit(t, x, 1)[0]
    trend_strength = float(np.clip(abs(slope) * len(x) / (np.std(x) + 1e-9), 0, 20))
    seasonal_features = [_seasonal_strength(x, s) for s in (4, 7, 12, 24)]

    bics = []
    for spec in CATALOG:
        try:
            fitted = SARIMAX(
                x, order=tuple(spec["order"]),
                seasonal_order=tuple(spec["seasonal_order"]),
                trend="n", enforce_stationarity=False,
                enforce_invertibility=False
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
        dtype=float
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    assert len(features) == N_FEATURES, (len(features), N_FEATURES)
    return features, bic_winner


def make_dataset(n_samples=N_SAMPLES):
    X, y, bic_pred, n_values = [], [], [], []
    for i in range(n_samples):
        # Stratified-by-construction: cycle through classes, then shuffle split.
        class_id = i % N_CLASSES
        n = int(rng.integers(40, 241))
        series = simulate_series(class_id, n, rng)
        feats, bic_winner = extract_diagnostics(series)
        X.append(feats)
        y.append(class_id)
        bic_pred.append(bic_winner)
        n_values.append(n)
        if (i + 1) % 25 == 0:
            print(f"Built {i + 1}/{n_samples} samples")
    X, y, bic_pred, n_values = map(np.asarray, (X, y, bic_pred, n_values))
    order = rng.permutation(len(y))
    return X[order], y[order], bic_pred[order], n_values[order]


def build_mlp(n_features=N_FEATURES):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.10),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(N_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


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
                y_true, probabilities, multi_class="ovr", average="macro"
            )
        except ValueError:
            result["macro_roc_auc_ovr"] = np.nan
        try:
            result["log_loss"] = log_loss(
                y_true, probabilities, labels=np.arange(N_CLASSES)
            )
        except ValueError:
            result["log_loss"] = np.nan
    return result


def main():
    print(f"Catalog: {N_CLASSES} classes; diagnostic features: {N_FEATURES}")
    print(f"Samples: {N_SAMPLES}; max optimizer iterations per candidate: {MAXITER}")
    X, y, bic_pred, lengths = make_dataset()

    X_train, X_test, y_train, y_test, bic_train, bic_test = train_test_split(
        X, y, bic_pred, test_size=0.25, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_mlp(X_train_scaled.shape[1])
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    model.fit(
        X_train_scaled, y_train,
        validation_split=0.15,
        epochs=200, batch_size=64,
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=1,
    )
    probs = model.predict(X_test_scaled, verbose=0)
    nn_pred = probs.argmax(axis=1)

    # BIC is a hard class choice; one-hot scores can technically produce AUC,
    # but this is not comparable to calibrated probabilistic scores.
    bic_probs = np.eye(N_CLASSES)[bic_test]
    results = [
        evaluate_method("Neural MLP", y_test, nn_pred, probs),
        evaluate_method("Minimum BIC", y_test, bic_test, bic_probs),
    ]
    results_df = pd.DataFrame(results)
    print("\n=== Comparative metrics ===")
    print(results_df.to_string(index=False, float_format=lambda z: f"{z:.4f}"))

    print("\n=== Per-class report: neural MLP ===")
    print(classification_report(
        y_test, nn_pred, labels=np.arange(N_CLASSES),
        target_names=[f"{i}: {spec_name(s)}" for i, s in enumerate(CATALOG)],
        zero_division=0
    ))
    print("\n=== Confusion matrix: neural MLP ===")
    print(confusion_matrix(y_test, nn_pred))
    print("\n=== Confusion matrix: BIC ===")
    print(confusion_matrix(y_test, bic_test))

    # Save reproducible artifacts.
    model.save("sarima_selector_v2.keras")
    np.savez(
        "sarima_selector_v2_scaler.npz",
        mean=scaler.mean_, scale=scaler.scale_
    )
    Path("sarima_selector_v2_catalog.json").write_text(
        json.dumps([
            {"class_id": i, "name": spec_name(s),
             "order": list(s["order"]),
             "seasonal_order": list(s["seasonal_order"]),
             "family": s["family"]}
            for i, s in enumerate(CATALOG)
        ], indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    results_df.to_csv("sarima_selector_v2_metrics.csv", index=False)
    # Keep the held-out test split so metrics can be recomputed later without
    # regenerating data or retraining the network.
    np.savez_compressed(
        "sarima_selector_v2_test_data.npz",
        X_test=X_test, y_test=y_test, bic_test=bic_test
    )
    # Save per-example predictions and top-three model scores for inspection.
    pred_rows = []
    for row_i, (truth, nn_class, bic_class, row_probs) in enumerate(
        zip(y_test, nn_pred, bic_test, probs)
    ):
        top3 = np.argsort(row_probs)[::-1][:3]
        pred_rows.append({
            "row": row_i,
            "true_class_id": int(truth),
            "true_specification": spec_name(CATALOG[int(truth)]),
            "neural_class_id": int(nn_class),
            "neural_specification": spec_name(CATALOG[int(nn_class)]),
            "neural_correct": bool(nn_class == truth),
            "bic_class_id": int(bic_class),
            "bic_specification": spec_name(CATALOG[int(bic_class)]),
            "bic_correct": bool(bic_class == truth),
            "top1_score": float(row_probs[top3[0]]),
            "top2_class_id": int(top3[1]),
            "top2_specification": spec_name(CATALOG[int(top3[1])]),
            "top2_score": float(row_probs[top3[1]]),
            "top3_class_id": int(top3[2]),
            "top3_specification": spec_name(CATALOG[int(top3[2])]),
            "top3_score": float(row_probs[top3[2]]),
        })
    pd.DataFrame(pred_rows).to_csv(
        "sarima_selector_v2_test_predictions.csv", index=False
    )
    Path("sarima_selector_v2_feature_names.json").write_text(
        json.dumps(FEATURE_NAMES, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nSaved artifacts:")
    print("- sarima_selector_v2.keras")
    print("- sarima_selector_v2_scaler.npz")
    print("- sarima_selector_v2_catalog.json")
    print("- sarima_selector_v2_metrics.csv")
    print("- sarima_selector_v2_test_data.npz")
    print("- sarima_selector_v2_test_predictions.csv")
    print("- sarima_selector_v2_feature_names.json")


def predict_csv(csv_path, value_column):
    """Load trained model and rank the three most likely classes for a CSV series."""
    model = tf.keras.models.load_model("sarima_selector_v2.keras")
    scaler_data = np.load("sarima_selector_v2_scaler.npz")
    scaler = StandardScaler()
    scaler.mean_ = scaler_data["mean"]
    scaler.scale_ = scaler_data["scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    df = pd.read_csv(csv_path)
    x = pd.to_numeric(df[value_column], errors="coerce").dropna().to_numpy()
    feats, bic_winner = extract_diagnostics(x)
    probabilities = model.predict(scaler.transform(feats.reshape(1, -1)), verbose=0)[0]
    top = np.argsort(probabilities)[::-1][:3]

    print(f"BIC winner: {bic_winner} — {spec_name(CATALOG[bic_winner])}")
    print("Top neural candidates:")
    for idx in top:
        print(f"{idx}: {spec_name(CATALOG[idx])}; score={probabilities[idx]:.4f}")

    # Do not call these scores calibrated probabilities without calibration tests.
    best_id = int(top[0])
    best = CATALOG[best_id]
    fitted = SARIMAX(
        x, order=tuple(best["order"]),
        seasonal_order=tuple(best["seasonal_order"]),
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)
    print("\nSelected model summary:")
    print(fitted.summary())
    return fitted, probabilities


if __name__ == "__main__":
    main()
