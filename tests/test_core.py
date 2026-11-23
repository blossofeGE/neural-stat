import numpy as np
from neural_sarima.catalog import CATALOG, N_CLASSES
from neural_sarima.simulation import simulate_series
from neural_sarima.features import FEATURE_NAMES, CONTINUOUS_FEATURES
from neural_sarima.modeling import encode_features


def test_catalog_size_and_unique():
    assert N_CLASSES == 42
    assert len({(tuple(s['order']), tuple(s['seasonal_order'])) for s in CATALOG}) == 42


def test_simulator_finite():
    rng = np.random.default_rng(1)
    for class_id in (0, 8, 17, 18, 25, 41):
        x = simulate_series(class_id, 120, rng)
        assert len(x) == 120
        assert np.isfinite(x).all()


def test_feature_schema_and_bic_onehot():
    assert len(FEATURE_NAMES) == 31
    assert len(CONTINUOUS_FEATURES) == 30
    x = np.zeros((3, 31), dtype=np.float32)
    x[:, -1] = [0, 17, 41]
    encoded, scaler = encode_features(x, fit=True)
    assert encoded.shape == (3, 72)
    assert np.all(encoded[:, 30:].sum(axis=1) == 1)
    assert np.argmax(encoded[:, 30:], axis=1).tolist() == [0, 17, 41]


def test_ar_polynomial_root_orientation():
    from neural_sarima.simulation import _roots_outside
    # Coefficients are stored in ascending powers of the lag operator B.
    assert _roots_outside(np.array([1.0, -0.60]))
    assert not _roots_outside(np.array([1.0, -1.10]))


def test_feature_dictionary_matches_feature_order():
    from neural_sarima.features import feature_dictionary
    assert [item["feature"] for item in feature_dictionary()] == FEATURE_NAMES
