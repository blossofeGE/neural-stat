#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix,
)
from neural_sarima.catalog import N_CLASSES
from neural_sarima.modeling import _metrics
from neural_sarima.config import ROOT

BENCH = ROOT / "benchmark"
KEY = BENCH / "private_evaluation" / "answer_key_and_baselines.csv"
PRIVATE_ARRAYS = BENCH / "private_evaluation" / "test_arrays_private.npz"


def parse_args():
    p = argparse.ArgumentParser(description="Validate and score blind LLM predictions against the private test key.")
    p.add_argument("--track", choices=["features31", "raw_series"], required=True)
    p.add_argument("--predictions", type=Path, required=True, help="LLM CSV with series_id and predicted_class_id.")
    p.add_argument("--metadata", type=Path, required=True, help="Completed evaluation_metadata_template.json.")
    p.add_argument("--run-name", default=None, help="Optional label for this model/run.")
    return p.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def mcnemar(y, a, b, a_name, b_name):
    a_ok = np.asarray(a) == np.asarray(y)
    b_ok = np.asarray(b) == np.asarray(y)
    n10 = int(np.sum(a_ok & ~b_ok))
    n01 = int(np.sum(~a_ok & b_ok))
    n = n10 + n01
    p = float(binomtest(min(n10, n01), n, p=0.5).pvalue) if n else 1.0
    stat = float((max(abs(n10 - n01) - 1, 0) ** 2) / n) if n else 0.0
    return {"method_a": a_name, "method_b": b_name,
            "a_correct_b_wrong": n10, "a_wrong_b_correct": n01,
            "discordant_pairs": n, "mcnemar_exact_p_two_sided": p,
            "mcnemar_chi2_cc": stat,
            "mcnemar_chi2_cc_p": float(chi2.sf(stat, 1)) if n else 1.0}


def main():
    args = parse_args()
    pred_path, metadata_path = resolve(args.predictions), resolve(args.metadata)
    if not pred_path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not KEY.is_file() or not PRIVATE_ARRAYS.is_file():
        raise FileNotFoundError("Private benchmark key not found. Run run_experiment.py first; never upload private_evaluation to the LLM.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required_meta = ["provider", "model_name", "model_version_or_snapshot", "track"]
    for field in required_meta:
        if not metadata.get(field) or str(metadata[field]).startswith("FILL"):
            raise ValueError(f"Complete required metadata field: {field}")
    if metadata["track"] != args.track:
        raise ValueError(f"Metadata track={metadata['track']} does not match --track {args.track}.")

    # Keep ID strings intact and reject duplicate/missing predictions rather than silently fixing them.
    pred = pd.read_csv(pred_path, dtype={"series_id": str}, encoding="utf-8-sig")
    key = pd.read_csv(KEY, dtype={"series_id": str}, encoding="utf-8-sig")
    if not {"series_id", "predicted_class_id"}.issubset(pred.columns):
        raise ValueError("Prediction CSV must include series_id and predicted_class_id columns.")
    if pred["series_id"].isna().any() or pred["series_id"].duplicated().any():
        raise ValueError("Prediction CSV contains missing or duplicated series_id values.")
    if key["series_id"].isna().any() or key["series_id"].duplicated().any():
        raise ValueError("Private key contains missing or duplicated series_id values.")
    expected_ids, observed_ids = set(key.series_id), set(pred.series_id)
    if expected_ids != observed_ids:
        missing, extra = expected_ids - observed_ids, observed_ids - expected_ids
        raise ValueError(f"ID mismatch: missing={len(missing)}, extra={len(extra)}. "
                         f"Examples missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}")

    merged = key.merge(pred, on="series_id", validate="one_to_one", how="inner")
    class_ids = pd.to_numeric(merged["predicted_class_id"], errors="coerce")
    if class_ids.isna().any() or not np.all(class_ids.to_numpy() == np.floor(class_ids.to_numpy())):
        raise ValueError("predicted_class_id must contain integer class IDs, with no blanks or text.")
    llm = class_ids.astype(int).to_numpy()
    if not np.isin(llm, np.arange(N_CLASSES)).all():
        raise ValueError("predicted_class_id values must be in 0..41.")
    y = merged["true_class_id"].astype(int).to_numpy()
    nn = merged["neural_class_id"].astype(int).to_numpy()
    bic = merged["bic_class_id"].astype(int).to_numpy()

    confidence = None
    if "confidence" in merged.columns:
        confidence = pd.to_numeric(merged["confidence"], errors="coerce").to_numpy(dtype=float)
        nonmissing = np.isfinite(confidence)
        if np.any((confidence[nonmissing] < 0) | (confidence[nonmissing] > 1)):
            raise ValueError("confidence values must be between 0 and 1.")

    with np.load(PRIVATE_ARRAYS, allow_pickle=False) as private:
        probs = np.asarray(private["nn_probabilities"])
    # Key and private arrays are produced in the same deterministic test order.
    if len(probs) != len(y):
        raise ValueError("Private model probabilities and answer key have different lengths.")

    rows = [
        {"method": f"LLM ({args.track})", **_metrics(y, llm, None)},
        {"method": "Neural MLP (same test set)", **_metrics(y, nn, probs)},
        {"method": "Minimum BIC", **_metrics(y, bic, np.eye(N_CLASSES)[bic])},
    ]
    result_dir = BENCH / "results" / (args.run_name or f"{args.track}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(result_dir / "metrics_summary.csv", index=False)
    pd.DataFrame([
        mcnemar(y, llm, nn, f"LLM ({args.track})", "Neural MLP"),
        mcnemar(y, llm, bic, f"LLM ({args.track})", "Minimum BIC"),
    ]).to_csv(result_dir / "mcnemar_tests.csv", index=False)

    class_names = merged[["true_class_id", "true_specification"]].drop_duplicates().set_index("true_class_id")["true_specification"].to_dict()
    labels = np.arange(N_CLASSES)
    names = [f"{i}: {class_names.get(i, f'class_{i}')}" for i in labels]
    for method, values, stem in [("LLM", llm, "llm"), ("Neural MLP", nn, "neural"), ("Minimum BIC", bic, "bic")]:
        report = classification_report(y, values, labels=labels, target_names=names,
                                       output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(result_dir / f"{stem}_classification_report.csv")
        pd.DataFrame(confusion_matrix(y, values, labels=labels), index=names, columns=names).to_csv(
            result_dir / f"{stem}_confusion_matrix.csv")

    per_object = merged.copy()
    per_object["llm_correct"] = llm == y
    per_object["neural_correct"] = nn == y
    per_object["bic_correct"] = bic == y
    per_object.to_csv(result_dir / "per_object_comparison.csv", index=False)
    if confidence is not None and np.isfinite(confidence).any():
        conf_df = pd.DataFrame({"confidence": confidence, "correct": llm == y})
        bins = [-np.inf, .5, .6, .7, .8, .9, 1.0000001]
        conf_df["confidence_bin"] = pd.cut(conf_df.confidence, bins=bins,
                                           labels=["<=0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"])
        conf_df.groupby("confidence_bin", observed=False).agg(
            n=("correct", "size"), mean_confidence=("confidence", "mean"),
            empirical_accuracy=("correct", "mean")).reset_index().to_csv(
                result_dir / "confidence_calibration.csv", index=False)

    metadata["scored_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["track_comparison_note"] = (
        "Same-information benchmark: LLM and MLP receive the same 31 raw diagnostic features, including bic_winner."
        if args.track == "features31" else
        "Raw-series track: LLM sees the numeric series; MLP uses engineered diagnostic features. Not a same-input comparison."
    )
    (result_dir / "evaluation_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Validated {len(merged)} unique blinded test objects.")
    print(f"Results saved to: {result_dir}")
    print(pd.DataFrame(rows)[["method", "accuracy", "balanced_accuracy", "macro_f1", "macro_roc_auc_ovr"]].to_string(index=False))
    if args.track == "raw_series":
        print("NOTE: raw-series results are an end-to-end comparison, not an equal-input comparison with the feature-based MLP.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
