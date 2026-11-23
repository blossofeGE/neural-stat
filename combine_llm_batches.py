#!/usr/bin/env python3
"""Combine per-batch LLM prediction CSVs and validate the blind object IDs."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
from neural_sarima.config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=["features31", "raw_series"], required=True)
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing one LLM response CSV per batch.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_dir = args.input_dir if args.input_dir.is_absolute() else ROOT / args.input_dir
    output = args.output if args.output.is_absolute() else ROOT / args.output
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV response files found in {input_dir}")
    frames = [pd.read_csv(p, dtype={"series_id": str}, encoding="utf-8-sig") for p in files]
    for path, frame in zip(files, frames):
        if not {"series_id", "predicted_class_id"}.issubset(frame.columns):
            raise ValueError(f"{path.name} must include series_id and predicted_class_id")
    combined = pd.concat(frames, ignore_index=True)
    if combined.series_id.isna().any() or combined.series_id.duplicated().any():
        duplicates = combined.loc[combined.series_id.duplicated(keep=False), "series_id"].astype(str).unique().tolist()
        raise ValueError(f"Duplicate or missing series_id values across batches: {duplicates[:10]}")
    template_name = "response_template_features31.csv" if args.track == "features31" else "response_template_raw_series.csv"
    template_path = ROOT / "benchmark" / "llm_input" / template_name
    if not template_path.is_file():
        raise FileNotFoundError(f"Benchmark template not found: {template_path}. Generate the dataset and benchmark first.")
    template = pd.read_csv(template_path, dtype={"series_id": str}, encoding="utf-8-sig")
    expected, observed = set(template.series_id), set(combined.series_id)
    if expected != observed:
        raise ValueError(f"ID mismatch: missing={len(expected-observed)}, extra={len(observed-expected)}")
    combined["predicted_class_id"] = pd.to_numeric(combined["predicted_class_id"], errors="raise").astype(int)
    if not combined.predicted_class_id.between(0, 41).all():
        raise ValueError("predicted_class_id values must be between 0 and 41")
    combined["_order"] = combined.series_id.map({sid: i for i, sid in enumerate(template.series_id)})
    combined = combined.sort_values("_order").drop(columns="_order")
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Combined {len(files)} batch files / {len(combined)} unique predictions: {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
