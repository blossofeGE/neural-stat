#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from neural_sarima.catalog import N_CLASSES, CATALOG, specification_name, save_catalog
from neural_sarima.config import ExperimentConfig, ROOT
from neural_sarima.dataset import generate_dataset
from neural_sarima.features import FEATURE_NAMES
from neural_sarima.modeling import run_training
from neural_sarima.benchmark import export_benchmark


def parse_args():
    p = argparse.ArgumentParser(description="Regenerate synthetic ARIMA/SARIMA data, train one MLP, and export LLM benchmark.")
    p.add_argument("--samples", type=int, default=4200, help="Total balanced synthetic series; must be divisible by 42. Suggested publication run: 8400 or more.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-length", type=int, default=96)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--maxiter", type=int, default=20, help="Maximum L-BFGS iterations for each SARIMAX candidate fit.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--force-regenerate", action="store_true", help="Ignore existing dataset/checkpoint and regenerate from scratch.")
    p.add_argument("--smoke-test", action="store_true", help="Run a small technical test; does not create publication results.")
    return p.parse_args()


def smoke_test():
    from neural_sarima.simulation import simulate_series, draw_stable_coefficients
    from neural_sarima.features import extract_features
    from neural_sarima.modeling import encode_features
    rng = np.random.default_rng(123)
    print(f"Catalog has {N_CLASSES} classes.")
    seen = set()
    for i, spec in enumerate(CATALOG):
        x = simulate_series(i, 120, rng)
        if len(x) != 120 or not np.isfinite(x).all():
            raise AssertionError(f"Simulator failed at class {i}")
        seen.add(tuple(spec["order"]) + tuple(spec["seasonal_order"]))
    if len(seen) != 42:
        raise AssertionError("Catalogue specifications are not unique.")
    # Exercise the full feature/BIC path on one series (42 candidate SARIMAX fits).
    x = simulate_series(0, 120, np.random.default_rng(321))
    features, diag = extract_features(x, maxiter=12)
    if features.shape != (31,) or not 0 <= int(features[-1]) < 42:
        raise AssertionError("Feature extraction produced an invalid 31-value vector.")
    encoded, scaler = encode_features(features.reshape(1, -1), fit=True)
    if encoded.shape != (1, 72):
        raise AssertionError(f"Expected 72 encoded model inputs, got {encoded.shape}")
    print("SMOKE TEST PASSED")
    print(f"Raw features: {len(FEATURE_NAMES)} (30 continuous + categorical bic_winner)")
    print(f"Encoded neural inputs: {encoded.shape[1]} (30 scaled + 42 one-hot)")
    print(f"BIC fit diagnostics for smoke sample: {diag}")


def main():
    args = parse_args()
    if args.smoke_test:
        smoke_test()
        return
    cfg = ExperimentConfig(
        seed=args.seed, n_samples=args.samples,
        min_length=args.min_length, max_length=args.max_length,
        sarimax_maxiter=args.maxiter, workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        bootstrap_replicates=args.bootstrap,
    )
    cfg.validate(N_CLASSES)
    (ROOT / "logs").mkdir(exist_ok=True)
    cfg.save(ROOT / "experiment_config.json")
    save_catalog(ROOT / "catalog_42_classes.json")
    (ROOT / "feature_names.json").write_text(json.dumps(FEATURE_NAMES, indent=2), encoding="utf-8")

    print("=" * 72)
    print("NEURAL ARIMA/SARIMA SPECIFICATION SELECTION — REPRODUCIBLE PIPELINE")
    print(f"Classes: {N_CLASSES}; balanced samples: {cfg.n_samples} ({cfg.n_samples // N_CLASSES}/class)")
    print(f"Raw predictors: 31; model inputs after preprocessing: 72")
    print(f"Series length: {cfg.min_length}-{cfg.max_length}; SARIMAX maxiter: {cfg.sarimax_maxiter}")
    print(f"Workers: {cfg.workers}; device: {'CUDA' if __import__('torch').cuda.is_available() else 'CPU'}")
    print("One neural network is trained. bic_winner is retained and one-hot encoded.")
    print("=" * 72)

    dataset = generate_dataset(cfg, force=args.force_regenerate)
    # Store aggregate diagnostics so SARIMAX estimation quality can be reported.
    diag = np.asarray(dataset["bic_fit_counts"])
    diag_summary = {
        "n_series": int(len(diag)),
        "mean_converged_bic_candidates_per_series": float(diag[:, 0].mean()),
        "mean_finite_nonconverged_candidates_per_series": float(diag[:, 1].mean()),
        "mean_exception_candidates_per_series": float(diag[:, 2].mean()),
        "mean_optimizer_warnings_per_series": float(diag[:, 3].mean()),
        "mean_unit_root_test_failures_per_series": float(diag[:, 4].mean()),
        "share_bic_winner_nonconverged": float(1.0 - diag[:, 5].mean()),
        "series_requiring_more_than_one_generation_attempt": int(np.sum(dataset["generation_attempts"] > 1)),
    }
    (ROOT / "results" / "generation_diagnostics.json").write_text(
        json.dumps(diag_summary, indent=2), encoding="utf-8")
    print("Generation diagnostics:", json.dumps(diag_summary, indent=2))

    trained = run_training(dataset, cfg)
    export_benchmark(trained["test_data"], cfg)
    print("\nPIPELINE COMPLETE")
    print("Metrics:", ROOT / "results" / "metrics_summary.csv")
    print("Model:", ROOT / "models" / "sarima_selector_mlp.pt")
    print("Public LLM input:", ROOT / "benchmark" / "llm_benchmark_input_only.zip")
    print("PRIVATE key (never upload):", ROOT / "benchmark" / "private_evaluation" / "answer_key_and_baselines.csv")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
