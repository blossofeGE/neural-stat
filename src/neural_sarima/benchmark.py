from __future__ import annotations
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from .catalog import CATALOG, catalog_rows, specification_name
from .config import ROOT, ExperimentConfig
from .features import FEATURE_NAMES, feature_dictionary

BENCHMARK_DIR = ROOT / "benchmark"
PUBLIC_DIR = BENCHMARK_DIR / "llm_input"
PRIVATE_DIR = BENCHMARK_DIR / "private_evaluation"


def _write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.9g", encoding="utf-8-sig")


def export_benchmark(test_data: dict, config: ExperimentConfig) -> None:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    for child in list(PUBLIC_DIR.iterdir()):
        if child.is_dir():
            import shutil
            shutil.rmtree(child)
        else:
            child.unlink()
    for child in list(PRIVATE_DIR.iterdir()):
        if child.is_dir():
            import shutil
            shutil.rmtree(child)
        else:
            child.unlink()

    X = np.asarray(test_data["X_test_raw"])
    y = np.asarray(test_data["y_test"], dtype=int)
    bic = np.asarray(test_data["bic_test"], dtype=int)
    nn = np.asarray(test_data["nn_test"], dtype=int)
    lengths = np.asarray(test_data["lengths"], dtype=int)
    series_padded = np.asarray(test_data["series_padded"])
    ids = [f"series_{i + 1:05d}" for i in range(len(y))]

    catalog_df = pd.DataFrame(catalog_rows())
    _write_csv(catalog_df, PUBLIC_DIR / "class_catalog.csv")
    (PUBLIC_DIR / "class_catalog.json").write_text(
        json.dumps(catalog_rows(), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(pd.DataFrame(feature_dictionary()), PUBLIC_DIR / "feature_dictionary.csv")

    feature_df = pd.DataFrame(X, columns=FEATURE_NAMES)
    feature_df.insert(0, "series_id", ids)
    _write_csv(feature_df, PUBLIC_DIR / "blind_test_features_31.csv")

    raw_rows = []
    for series_id, n, row in zip(ids, lengths, series_padded):
        raw_rows.extend({"series_id": series_id, "time_index": int(t), "value": float(v)}
                        for t, v in enumerate(row[:int(n)]))
    raw_df = pd.DataFrame(raw_rows, columns=["series_id", "time_index", "value"])
    _write_csv(raw_df, PUBLIC_DIR / "blind_test_raw_series.csv")

    feature_template = pd.DataFrame({
        "series_id": ids, "predicted_class_id": [""] * len(ids),
        "predicted_specification": [""] * len(ids), "confidence": [""] * len(ids),
    })
    raw_template = feature_template.copy()
    _write_csv(feature_template, PUBLIC_DIR / "response_template_features31.csv")
    _write_csv(raw_template, PUBLIC_DIR / "response_template_raw_series.csv")

    # Separate batch folders; each batch keeps the exact same blinded IDs.
    feature_batch = PUBLIC_DIR / "batches" / "features31"
    raw_batch = PUBLIC_DIR / "batches" / "raw_series"
    for track_dir in (feature_batch, raw_batch):
        track_dir.mkdir(parents=True, exist_ok=True)
    for bno, start in enumerate(range(0, len(ids), config.llm_batch_size_features), start=1):
        stop = min(start + config.llm_batch_size_features, len(ids))
        _write_csv(feature_df.iloc[start:stop].copy(), feature_batch / f"batch_{bno:03d}.csv")
    for bno, start in enumerate(range(0, len(ids), config.llm_batch_size_raw), start=1):
        stop = min(start + config.llm_batch_size_raw, len(ids))
        batch_ids = set(ids[start:stop])
        _write_csv(raw_df[raw_df["series_id"].isin(batch_ids)], raw_batch / f"batch_{bno:03d}.csv")

    instructions = f"""# BLIND LLM BENCHMARK: ARIMA/SARIMA SPECIFICATION IDENTIFICATION

This benchmark evaluates a model's ability to infer the generating class of held-out synthetic time series.
There are {len(CATALOG)} permitted classes (IDs 0-41). See `class_catalog.csv` and `class_catalog.json`.
All objects in the two tracks refer to the SAME held-out test set.

## Track A — FEATURES31 (primary, same-information comparison)
Input: `blind_test_features_31.csv` and `feature_dictionary.csv`.
There are 30 continuous diagnostic features plus `bic_winner` (the class ID selected by minimum BIC).
Treat `bic_winner` as a CATEGORICAL label, not an ordered numerical measurement.
The specialist neural network receives precisely these 31 raw pieces of information; internally it
standardizes the 30 continuous variables and one-hot encodes `bic_winner`.
This is the primary comparison between the LLM and the neural network on the same test objects and same information.

## Track B — RAW_SERIES (secondary, different-input comparison)
Input: `blind_test_raw_series.csv`, represented in long format with `series_id,time_index,value`.
Reconstruct each series by sorting on `time_index`; infer its class from the numeric observations.
This is an end-to-end LLM experiment. It is NOT a same-input comparison with the neural network,
which receives engineered diagnostics. Report it separately.

## Required prediction format
Return a CSV with exactly one row per `series_id` and these columns:
`series_id,predicted_class_id,predicted_specification,confidence`
- `predicted_class_id`: integer from 0 to 41. This is the scored field.
- `predicted_specification`: exact class name from the catalogue, if possible.
- `confidence`: optional numeric value from 0 to 1.
- Do not add rows, omit IDs, or duplicate IDs.
- Do not return multiple alternatives. Select one best class per object.
- Do not request or attempt to infer hidden labels. The answer key is private.

## Batch processing
For large contexts, process files in `batches/features31/` OR `batches/raw_series/`, not both in one call.
Each batch is independent. Concatenate all prediction files for the chosen track before scoring.
Use the corresponding response template as a schema example.

## Reproducibility record
Before scoring, save the provider, exact model/version, access mode, date, decoding parameters,
prompt text, and any context/reasoning settings in `evaluation_metadata_template.json`.
Keep the full prompt and raw response for every batch. Do not silently repair incorrect class IDs.
"""
    (PUBLIC_DIR / "LLM_BENCHMARK_PROTOCOL.md").write_text(instructions, encoding="utf-8")
    prompt_features = """You are performing a blind 42-class ARIMA/SARIMA specification-identification task.
Use ONLY the attached class_catalog.csv, feature_dictionary.csv, and this batch CSV. For each series_id, select exactly one class_id in 0..41. Treat bic_winner as a categorical class ID, not an ordered numeric feature. Use all 31 diagnostic fields. Do not invent classes, omit IDs, duplicate IDs, or ask for hidden labels.
Return ONLY CSV with columns: series_id,predicted_class_id,predicted_specification,confidence. Use one row for every input series_id. predicted_class_id must be an integer 0..41; predicted_specification must match the catalogue; confidence must be between 0 and 1 (or blank). Do not provide explanations or alternative classes in the CSV.
"""
    prompt_raw = """You are performing a blind 42-class ARIMA/SARIMA specification-identification task from raw numeric series.
Use ONLY the attached class_catalog.csv and this batch CSV. For each series_id, reconstruct the sequence by sorting its rows by time_index and reading value. Select exactly one class_id in 0..41. Do not invent classes, omit IDs, duplicate IDs, or ask for hidden labels.
Return ONLY CSV with columns: series_id,predicted_class_id,predicted_specification,confidence. Use one row for every input series_id. predicted_class_id must be an integer 0..41; predicted_specification must match the catalogue; confidence must be between 0 and 1 (or blank). Do not provide explanations or alternative classes in the CSV.
"""
    (PUBLIC_DIR / "PROMPT_FEATURES31.txt").write_text(prompt_features, encoding="utf-8")
    (PUBLIC_DIR / "PROMPT_RAW_SERIES.txt").write_text(prompt_raw, encoding="utf-8")
    (PUBLIC_DIR / "evaluation_metadata_template.json").write_text(json.dumps({
        "provider": "FILL_IN", "model_name": "FILL_IN", "model_version_or_snapshot": "FILL_IN",
        "access_mode": "web_ui_or_api", "run_date_utc": "YYYY-MM-DD",
        "track": "features31_or_raw_series", "temperature": None, "top_p": None,
        "seed": None, "reasoning_effort_or_mode": "FILL_IN",
        "prompt_version": "sarima-llm-benchmark-v2", "batch_size": None,
        "notes": "Record any retries, tool use, context truncation, or manual edits."
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # Strictly private key. Never include this folder in the LLM input ZIP.
    key_df = pd.DataFrame({
        "series_id": ids,
        "true_class_id": y,
        "true_specification": [specification_name(CATALOG[int(v)]) for v in y],
        "bic_class_id": bic,
        "bic_specification": [specification_name(CATALOG[int(v)]) for v in bic],
        "neural_class_id": nn,
    })
    _write_csv(key_df, PRIVATE_DIR / "answer_key_and_baselines.csv")
    np.savez_compressed(PRIVATE_DIR / "test_arrays_private.npz",
                        X_test_raw=X, y_test=y, bic_test=bic, nn_test=nn,
                        nn_probabilities=np.asarray(test_data["nn_probabilities"]),
                        lengths=lengths, series_padded=series_padded)
    (PRIVATE_DIR / "DO_NOT_UPLOAD.txt").write_text(
        "PRIVATE EVALUATION DATA. Contains hidden test labels and baseline predictions. "
        "Never upload this folder/archive to any evaluated LLM.\n", encoding="utf-8")

    public_zip = BENCHMARK_DIR / "llm_benchmark_input_only.zip"
    with zipfile.ZipFile(public_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in PUBLIC_DIR.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(PUBLIC_DIR)))
    print(f"LLM public input archive created: {public_zip}")
    print(f"Private answer key saved separately: {PRIVATE_DIR / 'answer_key_and_baselines.csv'}")
