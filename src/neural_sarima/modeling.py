from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import binomtest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score, log_loss, top_k_accuracy_score,
)
from .catalog import CATALOG, N_CLASSES, specification_name
from .config import ExperimentConfig, ROOT
from .features import CONTINUOUS_FEATURES, FEATURE_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SelectorMLP(nn.Module):
    """The single primary classifier; BIC winner enters as a one-hot categorical feature."""
    def __init__(self, n_input: int = len(CONTINUOUS_FEATURES) + N_CLASSES):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_input, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, N_CLASSES),
        )

    def forward(self, x):
        return self.network(x)


def stratified_indices(y: np.ndarray, config: ExperimentConfig):
    indices = np.arange(len(y))
    train_val, test_idx = train_test_split(
        indices, test_size=config.test_fraction, random_state=config.seed,
        stratify=y,
    )
    val_share_of_trainval = config.validation_fraction / (1.0 - config.test_fraction)
    train_idx, val_idx = train_test_split(
        train_val, test_size=val_share_of_trainval, random_state=config.seed + 1,
        stratify=y[train_val],
    )
    return train_idx, val_idx, test_idx


def encode_features(X_raw: np.ndarray, scaler: StandardScaler | None = None, fit: bool = False):
    """Scale continuous diagnostics and one-hot encode categorical bic_winner.

    Raw X columns are 30 continuous features followed by the integer class ID
    selected by BIC. The one-hot representation prevents imposing an artificial
    numerical ordering on class IDs.
    """
    X_raw = np.asarray(X_raw, dtype=np.float32)
    continuous = X_raw[:, :-1]
    bic_ids = X_raw[:, -1].astype(int)
    if not np.all(np.isin(bic_ids, np.arange(N_CLASSES))):
        raise ValueError("bic_winner contains IDs outside the 42-class catalogue.")
    if scaler is None:
        scaler = StandardScaler()
    x_cont = scaler.fit_transform(continuous) if fit else scaler.transform(continuous)
    x_bic = np.eye(N_CLASSES, dtype=np.float32)[bic_ids]
    encoded = np.concatenate([x_cont.astype(np.float32), x_bic], axis=1)
    return encoded, scaler


def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                config: ExperimentConfig):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    model = SelectorMLP(n_input=X_train.shape[1]).to(DEVICE)
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                        generator=generator, num_workers=0)
    x_val_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        rows_seen = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(yb)
            rows_seen += len(yb)
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            val_loss = float(criterion(val_logits, y_val_t).item())
            val_acc = float((val_logits.argmax(1) == y_val_t).float().mean().item())
        train_loss = train_loss_sum / max(rows_seen, 1)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "validation_loss": val_loss, "validation_accuracy": val_acc})
        print(f"Epoch {epoch:03d}/{config.max_epochs}: train_loss={train_loss:.4f}; "
              f"val_loss={val_loss:.4f}; val_accuracy={val_acc:.4f}")
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.early_stopping_patience:
                print(f"Early stopping: best epoch {best_epoch} (val_loss={best_loss:.5f}).")
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def predict_probabilities(model: SelectorMLP, X: np.ndarray, batch_size: int = 1024):
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=DEVICE)
            chunks.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.vstack(chunks)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray | None = None):
    row = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "top2_accuracy": float(top_k_accuracy_score(y_true, probabilities, k=2,
                                                       labels=np.arange(N_CLASSES))) if probabilities is not None else np.nan,
    }
    if probabilities is not None:
        row["macro_roc_auc_ovr"] = float(roc_auc_score(
            y_true, probabilities, labels=np.arange(N_CLASSES),
            multi_class="ovr", average="macro"))
        row["log_loss"] = float(log_loss(y_true, probabilities, labels=np.arange(N_CLASSES)))
    else:
        row["macro_roc_auc_ovr"] = np.nan
        row["log_loss"] = np.nan
    return row


def _stratified_bootstrap_ci(y_true, y_pred, rng, n_boot: int = 1000):
    per_class = [np.flatnonzero(y_true == c) for c in range(N_CLASSES)]
    accuracies, macro_f1s = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(class_idx, size=len(class_idx), replace=True)
                              for class_idx in per_class if len(class_idx)])
        accuracies.append(accuracy_score(y_true[idx], y_pred[idx]))
        macro_f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    return {
        "accuracy_ci95_low": float(np.quantile(accuracies, 0.025)),
        "accuracy_ci95_high": float(np.quantile(accuracies, 0.975)),
        "macro_f1_ci95_low": float(np.quantile(macro_f1s, 0.025)),
        "macro_f1_ci95_high": float(np.quantile(macro_f1s, 0.975)),
    }


def _mcnemar_exact(y_true, pred_a, pred_b, name_a="Neural MLP", name_b="Minimum BIC"):
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    n10 = int(np.sum(a_ok & ~b_ok))
    n01 = int(np.sum(~a_ok & b_ok))
    discordant = n10 + n01
    p = float(binomtest(min(n10, n01), discordant, p=0.5).pvalue) if discordant else 1.0
    chi2_cc = float((max(abs(n10 - n01) - 1, 0) ** 2) / discordant) if discordant else 0.0
    return {
        "method_a": name_a, "method_b": name_b,
        "a_correct_b_wrong": n10, "a_wrong_b_correct": n01,
        "discordant_pairs": discordant, "mcnemar_exact_p_two_sided": p,
        "mcnemar_chi2_cc": chi2_cc,
        "mcnemar_chi2_cc_p": float(chi2.sf(chi2_cc, 1)) if discordant else 1.0,
    }


def run_training(dataset: dict, config: ExperimentConfig) -> dict:
    X = np.asarray(dataset["X"], dtype=np.float32)
    y = np.asarray(dataset["y"], dtype=np.int64)
    bic_pred = X[:, -1].astype(np.int64)
    train_idx, val_idx, test_idx = stratified_indices(y, config)

    # Assert split integrity: no observation is shared across partitions.
    if len(set(train_idx) & set(val_idx) | set(train_idx) & set(test_idx) | set(val_idx) & set(test_idx)):
        raise AssertionError("Train/validation/test indices overlap.")
    for label, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        counts = np.bincount(y[idx], minlength=N_CLASSES)
        print(f"{label}: n={len(idx)}, min/class={counts.min()}, max/class={counts.max()}")

    X_train, scaler = encode_features(X[train_idx], fit=True)
    X_val, _ = encode_features(X[val_idx], scaler=scaler, fit=False)
    X_test, _ = encode_features(X[test_idx], scaler=scaler, fit=False)
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    model, history, best_epoch = train_model(X_train, y_train, X_val, y_val, config)
    probs = predict_probabilities(model, X_test)
    nn_pred = probs.argmax(axis=1)
    bic_test = bic_pred[test_idx]
    bic_probs = np.eye(N_CLASSES, dtype=np.float32)[bic_test]

    rows = [
        {"method": "Neural MLP (30 continuous + one-hot BIC winner)", **_metrics(y_test, nn_pred, probs)},
        {"method": "Minimum BIC", **_metrics(y_test, bic_test, bic_probs)},
        {"method": "Uniform random reference", "n": len(y_test),
         "accuracy": 1.0 / N_CLASSES, "balanced_accuracy": 1.0 / N_CLASSES,
         "macro_precision": 1.0 / N_CLASSES, "macro_recall": 1.0 / N_CLASSES,
         "macro_f1": 1.0 / N_CLASSES, "weighted_f1": 1.0 / N_CLASSES,
         "top2_accuracy": 2.0 / N_CLASSES, "macro_roc_auc_ovr": 0.5, "log_loss": float(np.log(N_CLASSES))},
    ]
    metrics_df = pd.DataFrame(rows)
    ci = _stratified_bootstrap_ci(y_test, nn_pred,
                                  np.random.default_rng(config.seed + 83),
                                  config.bootstrap_replicates)
    mcnemar = _mcnemar_exact(y_test, nn_pred, bic_test)

    ROOT.joinpath("models").mkdir(exist_ok=True)
    ROOT.joinpath("artifacts").mkdir(exist_ok=True)
    ROOT.joinpath("results").mkdir(exist_ok=True)
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_features": int(X_train.shape[1]),
        "continuous_feature_names": CONTINUOUS_FEATURES,
        "bic_categories": list(range(N_CLASSES)),
        "class_count": N_CLASSES,
        "best_epoch": best_epoch,
        "torch_version": torch.__version__,
        "seed": config.seed,
    }, ROOT / "models" / "sarima_selector_mlp.pt")
    np.savez_compressed(ROOT / "artifacts" / "preprocessor.npz",
                        mean=scaler.mean_, scale=scaler.scale_,
                        continuous_feature_names=np.asarray(CONTINUOUS_FEATURES))
    (ROOT / "artifacts" / "class_catalog.json").write_text(
        json.dumps([{"class_id": i, "specification": specification_name(spec)}
                   for i, spec in enumerate(CATALOG)], ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(ROOT / "results" / "training_history.csv", index=False)
    metrics_df.to_csv(ROOT / "results" / "metrics_summary.csv", index=False)
    (ROOT / "results" / "neural_bootstrap_ci95.json").write_text(json.dumps(ci, indent=2), encoding="utf-8")
    pd.DataFrame([mcnemar]).to_csv(ROOT / "results" / "mcnemar_nn_vs_bic.csv", index=False)

    class_names = [f"{i}: {specification_name(spec)}" for i, spec in enumerate(CATALOG)]
    report = classification_report(y_test, nn_pred, labels=np.arange(N_CLASSES),
                                   target_names=class_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(ROOT / "results" / "neural_classification_report.csv")
    cm = confusion_matrix(y_test, nn_pred, labels=np.arange(N_CLASSES))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(ROOT / "results" / "neural_confusion_matrix.csv")

    lengths = np.asarray(dataset["lengths"])[test_idx]
    series_padded = np.asarray(dataset["series_padded"])[test_idx]
    bic_diag = np.asarray(dataset["bic_fit_counts"])[test_idx]
    test_X_raw = X[test_idx]
    test_data = {
        "X_test_raw": test_X_raw,
        "y_test": y_test,
        "bic_test": bic_test,
        "nn_test": nn_pred,
        "nn_probabilities": probs,
        "test_indices": test_idx,
        "lengths": lengths,
        "series_padded": series_padded,
        "bic_fit_counts": bic_diag,
    }
    np.savez_compressed(ROOT / "artifacts" / "private_test_arrays.npz", **test_data)
    np.savez_compressed(ROOT / "artifacts" / "data_split_indices.npz",
                        train_indices=train_idx, validation_indices=val_idx, test_indices=test_idx)

    pred_rows = []
    for i, (truth, nn_class, bic_class, p) in enumerate(zip(y_test, nn_pred, bic_test, probs)):
        top = np.argsort(p)[::-1][:3]
        pred_rows.append({
            "series_id": f"series_{i + 1:05d}",
            "true_class_id": int(truth),
            "true_specification": specification_name(CATALOG[int(truth)]),
            "nn_class_id": int(nn_class),
            "nn_specification": specification_name(CATALOG[int(nn_class)]),
            "nn_correct": bool(nn_class == truth),
            "bic_class_id": int(bic_class),
            "bic_specification": specification_name(CATALOG[int(bic_class)]),
            "bic_correct": bool(bic_class == truth),
            "nn_top1_probability": float(p[top[0]]),
            "nn_top2_class_id": int(top[1]),
            "nn_top2_probability": float(p[top[1]]),
            "series_length": int(lengths[i]),
        })
    pd.DataFrame(pred_rows).to_csv(ROOT / "results" / "test_predictions.csv", index=False)

    return {
        "model": model, "scaler": scaler, "test_data": test_data,
        "metrics": metrics_df, "mcnemar": mcnemar, "ci": ci,
        "train_indices": train_idx, "validation_indices": val_idx, "test_indices": test_idx,
    }
