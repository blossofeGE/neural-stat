from __future__ import annotations
import numpy as np
from scipy.signal import lfilter
from .catalog import CATALOG


def _poly_ar(params: np.ndarray) -> np.ndarray:
    return np.r_[1.0, -np.asarray(params, dtype=float)]


def _poly_ma(params: np.ndarray) -> np.ndarray:
    return np.r_[1.0, np.asarray(params, dtype=float)]


def _roots_outside(poly: np.ndarray, margin: float = 1.001) -> bool:
    if len(poly) <= 1:
        return True
    roots = np.roots(np.asarray(poly, dtype=float)[::-1])
    return bool(np.all(np.isfinite(roots)) and np.all(np.abs(roots) > margin))


def _draw_nonzero(count: int, rng: np.random.Generator, low: float, high: float) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=float)
    return rng.choice([-1.0, 1.0], size=count) * rng.uniform(low, high, size=count)


def draw_stable_coefficients(p: int, q: int, P: int, Q: int, s: int,
                             rng: np.random.Generator, max_tries: int = 2000):
    """Draw AR/MA coefficients and accept only stable/invertible combined polynomials.

    Unlike the earlier prototype, the conservative fallback is also validated;
    no unverified coefficient vector is returned.
    """
    for _ in range(max_tries):
        ar = _draw_nonzero(p, rng, 0.10, 0.60)
        ma = _draw_nonzero(q, rng, 0.10, 0.60)
        sar = _draw_nonzero(P, rng, 0.10, 0.45)
        sma = _draw_nonzero(Q, rng, 0.10, 0.45)
        ar_poly = _poly_ar(ar)
        ma_poly = _poly_ma(ma)
        if s:
            sar_poly = np.zeros(P * s + 1)
            sar_poly[::s] = _poly_ar(sar)
            sma_poly = np.zeros(Q * s + 1)
            sma_poly[::s] = _poly_ma(sma)
            ar_poly = np.convolve(ar_poly, sar_poly)
            ma_poly = np.convolve(ma_poly, sma_poly)
        if _roots_outside(ar_poly) and _roots_outside(ma_poly):
            return ar, ma, sar, sma
    raise RuntimeError(f"Could not draw stable/invertible coefficients for orders {(p,q,P,Q,s)}")


def simulate_series(class_id: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate a Gaussian ARIMA/SARIMA process for a catalogue class.

    The stationary/invertible ARMA component is filtered first; ordinary and
    seasonal integration operators are then applied. The output is centred and
    scaled only after the integration step.
    """
    spec = CATALOG[int(class_id)]
    p, d, q = map(int, spec["order"])
    P, D, Q, s = map(int, spec["seasonal_order"])
    ar, ma, sar, sma = draw_stable_coefficients(p, q, P, Q, s, rng)

    ar_poly = _poly_ar(ar)
    ma_poly = _poly_ma(ma)
    if s:
        seasonal_ar = np.zeros(P * s + 1)
        seasonal_ar[::s] = _poly_ar(sar)
        seasonal_ma = np.zeros(Q * s + 1)
        seasonal_ma[::s] = _poly_ma(sma)
        ar_poly = np.convolve(ar_poly, seasonal_ar)
        ma_poly = np.convolve(ma_poly, seasonal_ma)

    burn = max(500, 5 * n)
    innovations = rng.normal(0.0, 1.0, n + burn)
    series = lfilter(ma_poly, ar_poly, innovations)[burn:]

    for _ in range(d):
        series = np.cumsum(series)
    for _ in range(D):
        for t in range(s, len(series)):
            series[t] += series[t - s]

    series = series * rng.uniform(0.5, 2.0)
    sd = float(np.std(series))
    if not np.isfinite(sd) or sd < 1e-10 or not np.isfinite(series).all():
        raise RuntimeError("Non-finite or degenerate series generated.")
    series = (series - float(np.mean(series))) / sd
    if not np.isfinite(series).all():
        raise RuntimeError("Generated series contains non-finite values.")
    return series.astype(np.float64)
