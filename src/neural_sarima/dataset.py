from __future__ import annotations
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from .catalog import N_CLASSES
from .config import ExperimentConfig, ROOT
from .features import extract_features
from .simulation import simulate_series

DATA_DIR = ROOT / "data"
CHECKPOINT_PATH = DATA_DIR / "generation_checkpoint.npz"
DATASET_PATH = DATA_DIR / "dataset.npz"

_THREAD_LIMITER = None

def _worker_init():
    global _THREAD_LIMITER
    from threadpoolctl import threadpool_limits
    _THREAD_LIMITER = threadpool_limits(limits=1)


def _fingerprint(config: ExperimentConfig) -> str:
    relevant = {
        "seed": config.seed, "n_samples": config.n_samples,
        "min_length": config.min_length, "max_length": config.max_length,
        "sarimax_maxiter": config.sarimax_maxiter, "catalog_classes": N_CLASSES,
        "generator_version": "stable-roots-v2",
        "feature_version": "diagnostics-v3-onehot-bic-convergence-audit",
    }
    payload = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_save_npz(path: Path, **arrays) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def _build_one_sample(task):
    index, class_id, n, row_seed, maxiter = task
    for attempt in range(5):
        rng = np.random.default_rng(np.random.SeedSequence([row_seed, attempt]))
        try:
            series = simulate_series(class_id, n, rng)
            features, diag = extract_features(series, maxiter=maxiter)
            return {
                "index": index,
                "features": features,
                "class_id": class_id,
                "bic_winner": int(features[-1]),
                "length": n,
                "series": series.astype(np.float32),
                "diagnostics": diag,
                "attempts": attempt + 1,
            }
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to generate sample {index}, class {class_id}: {last_error}")


def _load_checkpoint(path: Path, fingerprint: str):
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["fingerprint"].item()) != fingerprint:
                print("Checkpoint fingerprint does not match current configuration; starting from zero.")
                return None
            return {k: data[k] for k in data.files if k != "fingerprint"}
    except Exception as exc:
        print(f"Could not read checkpoint ({exc}); starting from zero.")
        return None


def generate_dataset(config: ExperimentConfig, force: bool = False) -> dict:
    config.validate(N_CLASSES)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(config)

    if DATASET_PATH.exists() and not force:
        try:
            with np.load(DATASET_PATH, allow_pickle=False) as data:
                if str(data["fingerprint"].item()) == fingerprint:
                    print(f"Reusing compatible dataset: {DATASET_PATH}")
                    return {k: data[k] for k in data.files if k != "fingerprint"}
        except Exception:
            pass

    saved = None if force else _load_checkpoint(CHECKPOINT_PATH, fingerprint)
    if saved is None:
        arrays = {
            "X": np.empty((0, 31), dtype=np.float32),
            "y": np.empty(0, dtype=np.int64),
            "lengths": np.empty(0, dtype=np.int64),
            "series_padded": np.empty((0, config.max_length), dtype=np.float32),
            "bic_fit_counts": np.empty((0, 6), dtype=np.int32),
            "generation_attempts": np.empty(0, dtype=np.int16),
        }
    else:
        arrays = saved
    start = len(arrays["y"])
    print(f"Generating {config.n_samples - start} additional rows; {config.workers} workers.")

    all_tasks = []
    for i in range(start, config.n_samples):
        row_rng = np.random.default_rng(np.random.SeedSequence([config.seed, i]))
        class_id = i % N_CLASSES
        n = int(row_rng.integers(config.min_length, config.max_length + 1))
        row_seed = int(row_rng.integers(0, np.iinfo(np.int32).max))
        all_tasks.append((i, class_id, n, row_seed, config.sarimax_maxiter))

    # Process isolation avoids shared warning-filter state during concurrent
    # statsmodels fits. Results are committed in index order, in atomic batches.
    with ProcessPoolExecutor(max_workers=config.workers, initializer=_worker_init) as executor:
        for batch_start in range(0, len(all_tasks), config.checkpoint_every):
            task_batch = all_tasks[batch_start:batch_start + config.checkpoint_every]
            results = list(executor.map(_build_one_sample, task_batch, chunksize=1))
            results.sort(key=lambda item: item["index"])
            feature_rows, class_rows, length_rows = [], [], []
            series_rows, stats_rows, attempts_rows = [], [], []
            for result in results:
                row = np.full(config.max_length, np.nan, dtype=np.float32)
                row[:result["length"]] = result["series"]
                d = result["diagnostics"]
                stats = np.array([
                    d["bic_converged_candidates"], d["bic_finite_nonconverged_candidates"],
                    d["bic_exception_candidates"], d["bic_fit_warnings"],
                    d["adf_test_failures"] + d["kpss_test_failures"],
                    d["bic_winner_converged"],
                ], dtype=np.int32)
                feature_rows.append(result["features"])
                class_rows.append(result["class_id"])
                length_rows.append(result["length"])
                series_rows.append(row)
                stats_rows.append(stats)
                attempts_rows.append(result["attempts"])

            arrays["X"] = np.vstack([arrays["X"], np.asarray(feature_rows, dtype=np.float32)])
            arrays["y"] = np.r_[arrays["y"], np.asarray(class_rows, dtype=np.int64)]
            arrays["lengths"] = np.r_[arrays["lengths"], np.asarray(length_rows, dtype=np.int64)]
            arrays["series_padded"] = np.vstack([arrays["series_padded"], np.stack(series_rows)])
            arrays["bic_fit_counts"] = np.vstack([arrays["bic_fit_counts"], np.stack(stats_rows)])
            arrays["generation_attempts"] = np.r_[arrays["generation_attempts"], np.asarray(attempts_rows, dtype=np.int16)]
            absolute_done = start + batch_start + len(task_batch)
            _atomic_save_npz(CHECKPOINT_PATH, **arrays, fingerprint=np.array(fingerprint))
            print(f"Generated {absolute_done}/{config.n_samples}; checkpoint saved.", flush=True)

    if len(arrays["y"]) != config.n_samples:
        raise RuntimeError(f"Dataset incomplete: {len(arrays['y'])}/{config.n_samples}")
    rng = np.random.default_rng(config.seed + 99_991)
    order = rng.permutation(config.n_samples)
    arrays = {k: v[order] for k, v in arrays.items()}
    _atomic_save_npz(DATASET_PATH, **arrays, fingerprint=np.array(fingerprint))
    print(f"Dataset saved: {DATASET_PATH}")
    return arrays
