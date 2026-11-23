from __future__ import annotations
import warnings
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .catalog import CATALOG

LAGS = (1, 2, 3, 4, 7, 8, 12, 16, 24)
CONTINUOUS_FEATURES = (
    ["log_n", "adf_c", "adf_ct", "adf_ctt", "kpss_c", "kpss_ct"]
    + [f"acf_{lag}" for lag in LAGS]
    + [f"pacf_{lag}" for lag in LAGS]
    + ["variance_ratio_1", "variance_ratio_2", "trend_strength",
       "seasonal_strength_4", "seasonal_strength_8", "seasonal_strength_12"]
)
FEATURE_NAMES = [*CONTINUOUS_FEATURES, "bic_winner"]
N_FEATURES_RAW = len(FEATURE_NAMES)  # 31 raw diagnostics; bic_winner is categorical


def _safe_test(function, x, regression: str, fallback: float) -> tuple[float, bool]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pvalue = float(function(x, regression=regression, **(
                {"autolag": "AIC"} if function is adfuller else {"nlags": "auto"}
            ))[1])
        if not np.isfinite(pvalue):
            return fallback, True
        return float(np.clip(pvalue, 0.0, 1.0)), False
    except Exception:
        return fallback, True


def _seasonal_strength(x: np.ndarray, period: int) -> float:
    if len(x) <= 2 * period:
        return 0.0
    phase_means = np.array([np.mean(x[i::period]) for i in range(period)])
    seasonal_component = np.tile(phase_means, int(np.ceil(len(x) / period)))[:len(x)]
    denominator = float(np.var(x))
    if denominator < 1e-12:
        return 0.0
    return float(np.clip(1.0 - np.var(x - seasonal_component) / denominator, 0.0, 1.0))


def _bic_winner(x: np.ndarray, maxiter: int) -> tuple[int, dict]:
    """Select the minimum finite BIC and separately record optimizer convergence.

    A non-converged but finite likelihood is not silently discarded, because doing
    so would change the candidate set from row to row. Instead, convergence of the
    winning candidate and the convergence counts are exported for sensitivity
    analysis and explicit reporting.
    """
    bic_values = np.full(len(CATALOG), np.inf, dtype=float)
    converged_flags = np.zeros(len(CATALOG), dtype=bool)
    converged_count = 0
    finite_nonconverged_count = 0
    warning_count = 0
    exception_count = 0
    for class_id, spec in enumerate(CATALOG):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = SARIMAX(
                    x,
                    order=tuple(spec["order"]),
                    seasonal_order=tuple(spec["seasonal_order"]),
                    trend="n",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    concentrate_scale=True,
                ).fit(disp=False, maxiter=maxiter, method="lbfgs", cov_type="none")
            warning_count += len(caught)
            bic = float(fit.bic)
            converged = bool(getattr(fit, "mle_retvals", {}).get("converged", False))
            if np.isfinite(bic):
                bic_values[class_id] = bic
                converged_flags[class_id] = converged
                if converged:
                    converged_count += 1
                else:
                    finite_nonconverged_count += 1
        except Exception:
            exception_count += 1
    finite_count = int(np.isfinite(bic_values).sum())
    if finite_count == 0:
        raise RuntimeError("No finite BIC among 42 SARIMAX candidates; sample is regenerated.")
    winner = int(np.argmin(bic_values))
    return winner, {
        "bic_converged_candidates": converged_count,
        "bic_finite_nonconverged_candidates": finite_nonconverged_count,
        "bic_exception_candidates": exception_count,
        "bic_fit_warnings": warning_count,
        "bic_candidate_count": len(CATALOG),
        "bic_finite_candidates": finite_count,
        "bic_winner_converged": int(converged_flags[winner]),
        "bic_winner_value": float(bic_values[winner]),
    }


def extract_features(series: np.ndarray, maxiter: int = 50) -> tuple[np.ndarray, dict]:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 60:
        raise ValueError("At least 60 finite observations are required for diagnostics.")

    adf_vals, adf_failures = [], 0
    for reg in ("c", "ct", "ctt"):
        val, failed = _safe_test(adfuller, x, reg, 1.0)
        adf_vals.append(val)
        adf_failures += int(failed)

    kpss_vals, kpss_failures = [], 0
    for reg in ("c", "ct"):
        val, failed = _safe_test(kpss, x, reg, 1.0)
        kpss_vals.append(val)
        kpss_failures += int(failed)

    max_lag = max(LAGS)
    try:
        acf_values = acf(x, nlags=max_lag, fft=True, missing="raise")
    except Exception as exc:
        raise RuntimeError(f"ACF calculation failed: {exc}") from exc
    try:
        pacf_values = pacf(x, nlags=max_lag, method="ywm")
    except Exception as exc:
        raise RuntimeError(f"PACF calculation failed: {exc}") from exc

    acf_features = [float(acf_values[k]) for k in LAGS]
    pacf_features = [float(pacf_values[k]) for k in LAGS]
    dx = np.diff(x)
    d2x = np.diff(x, n=2)
    vr1 = float(np.log((np.var(dx) + 1e-9) / (np.var(x) + 1e-9)))
    vr2 = float(np.log((np.var(d2x) + 1e-9) / (np.var(dx) + 1e-9)))
    t = np.arange(len(x), dtype=float)
    slope = float(np.polyfit(t, x, 1)[0])
    trend_strength = float(np.clip(abs(slope) * len(x) / (np.std(x) + 1e-9), 0, 20))
    seasonal = [_seasonal_strength(x, period) for period in (4, 8, 12)]
    winner, bic_diagnostics = _bic_winner(x, maxiter=maxiter)

    continuous = np.asarray(
        [np.log(len(x)), *adf_vals, *kpss_vals]
        + acf_features + pacf_features
        + [vr1, vr2, trend_strength, *seasonal],
        dtype=np.float32,
    )
    if not np.isfinite(continuous).all():
        raise RuntimeError("Feature vector contains non-finite values.")
    raw_features = np.r_[continuous, np.float32(winner)]
    diagnostics = {
        **bic_diagnostics,
        "adf_test_failures": adf_failures,
        "kpss_test_failures": kpss_failures,
    }
    if len(raw_features) != 31:
        raise AssertionError(f"Expected 31 raw diagnostics, got {len(raw_features)}")
    return raw_features, diagnostics


def feature_dictionary() -> list[dict]:
    descriptions = {
        "log_n": "Natural logarithm of observed series length.",
        "adf_c": "ADF p-value with intercept (regression='c').",
        "adf_ct": "ADF p-value with intercept and linear trend (regression='ct').",
        "adf_ctt": "ADF p-value with intercept, linear and quadratic trend (regression='ctt').",
        "kpss_c": "KPSS p-value with level-stationarity null (regression='c').",
        "kpss_ct": "KPSS p-value with trend-stationarity null (regression='ct').",
        "variance_ratio_1": "log((var(diff(x))+1e-9)/(var(x)+1e-9)).",
        "variance_ratio_2": "log((var(diff(x, n=2))+1e-9)/(var(diff(x))+1e-9)).",
        "trend_strength": "clip(abs(linear_slope)*n/(std(x)+1e-9), 0, 20).",
        "seasonal_strength_4": "Variance-based seasonal strength from phase means at period 4.",
        "seasonal_strength_8": "Variance-based seasonal strength from phase means at period 8.",
        "seasonal_strength_12": "Variance-based seasonal strength from phase means at period 12.",
        "bic_winner": "Categorical class ID with the minimum BIC among the 42 candidate SARIMAX specifications. Treat as a category, not as an ordered numeric measurement.",
    }
    result = [{"feature": name, "description": descriptions.get(name, "")}
              for name in CONTINUOUS_FEATURES]
    for lag in LAGS:
        result.extend([
            {"feature": f"acf_{lag}", "description": f"Sample ACF at lag {lag}."},
            {"feature": f"pacf_{lag}", "description": f"Sample PACF at lag {lag}."},
        ])
    # Reorder exactly to FEATURE_NAMES after the descriptive additions.
    mapping = {item["feature"]: item for item in result}
    mapping["bic_winner"] = {"feature": "bic_winner", "description": descriptions["bic_winner"]}
    return [mapping[name] for name in FEATURE_NAMES]
