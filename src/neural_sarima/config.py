from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    n_samples: int = 4200          # balanced: 100 observations per each of 42 classes
    min_length: int = 96
    max_length: int = 192
    sarimax_maxiter: int = 20
    workers: int = 4
    checkpoint_every: int = 50
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_epochs: int = 200
    early_stopping_patience: int = 18
    dropout_1: float = 0.10
    dropout_2: float = 0.05
    bootstrap_replicates: int = 1000
    llm_batch_size_features: int = 25
    llm_batch_size_raw: int = 10

    def validate(self, n_classes: int) -> None:
        if self.n_samples < n_classes * 7:
            raise ValueError(f"n_samples must be at least 7 observations/class ({n_classes * 7}) so each stratified split can include every class.")
        if self.n_samples % n_classes != 0:
            raise ValueError(f"n_samples must be divisible by {n_classes} to guarantee class balance.")
        if self.min_length < 60 or self.max_length < self.min_length:
            raise ValueError("Invalid series-length interval.")
        if not abs(self.train_fraction + self.validation_fraction + self.test_fraction - 1.0) < 1e-9:
            raise ValueError("Train/validation/test fractions must sum to 1.")
        if min(self.train_fraction, self.validation_fraction, self.test_fraction) <= 0:
            raise ValueError("All data split fractions must be positive.")
        if self.workers < 1 or self.sarimax_maxiter < 1:
            raise ValueError("workers and sarimax_maxiter must be positive.")

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
