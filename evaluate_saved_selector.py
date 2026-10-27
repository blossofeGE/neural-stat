"""Recalculate evaluation metrics for a previously trained ARIMA/SARIMA selector.
Run this script from the folder containing the saved model artifacts:
    python evaluate_saved_selector.py
It does not retrain the model or regenerate synthetic data.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, log_loss
)

MODEL_PATH = "sarima_selector_v2.keras"
SCALER_PATH = "sarima_selector_v2_scaler.npz"
TEST_PATH = "sarima_selector_v2_test_data.npz"
CATALOG_PATH = "sarima_selector_v2_catalog.json"

for required in (MODEL_PATH, SCALER_PATH, TEST_PATH, CATALOG_PATH):
    if not Path(required).exists():
        raise FileNotFoundError(
            f"Не найден файл {required}. Запустите обновлённый "
            "neural_arima_selector_v2_fixed.py хотя бы один раз, чтобы сохранить тестовые данные."
        )

model = tf.keras.models.load_model(MODEL_PATH)
scaler_data = np.load(SCALER_PATH)
test_data = np.load(TEST_PATH)
X_test = test_data["X_test"]
y_test = test_data["y_test"].astype(int)
bic_test = test_data["bic_test"].astype(int)

scaler = StandardScaler()
scaler.mean_ = scaler_data["mean"]
scaler.scale_ = scaler_data["scale"]
scaler.var_ = scaler.scale_ ** 2
scaler.n_features_in_ = len(scaler.mean_)
X_test_scaled = scaler.transform(X_test)
probs = model.predict(X_test_scaled, verbose=0)
nn_pred = probs.argmax(axis=1)

catalog = json.loads(Path(CATALOG_PATH).read_text(encoding="utf-8"))
def name(class_id):
    return catalog[int(class_id)]["name"]

rows = []
for method, pred, prob in [
    ("Neural MLP", nn_pred, probs),
    ("Minimum BIC", bic_test, np.eye(probs.shape[1])[bic_test]),
]:
    row = {
        "method": method,
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0),
    }
    try:
        row["macro_roc_auc_ovr"] = roc_auc_score(
            y_test, prob, multi_class="ovr", average="macro"
        )
    except ValueError:
        row["macro_roc_auc_ovr"] = np.nan
    try:
        row["log_loss"] = log_loss(y_test, prob, labels=np.arange(probs.shape[1]))
    except ValueError:
        row["log_loss"] = np.nan
    rows.append(row)

metrics = pd.DataFrame(rows)
metrics.to_csv("sarima_selector_v2_metrics_recalculated.csv", index=False)

class_ids = np.arange(probs.shape[1])
report = classification_report(
    y_test, nn_pred, labels=class_ids,
    target_names=[f'{i}: {name(i)}' for i in class_ids],
    zero_division=0, output_dict=True
)
pd.DataFrame(report).transpose().to_csv(
    "sarima_selector_v2_classification_report.csv", index=True
)
pd.DataFrame(
    confusion_matrix(y_test, nn_pred, labels=class_ids),
    index=[f"true_{i}_{name(i)}" for i in class_ids],
    columns=[f"pred_{i}_{name(i)}" for i in class_ids],
).to_csv("sarima_selector_v2_confusion_matrix_nn.csv")
pd.DataFrame(
    confusion_matrix(y_test, bic_test, labels=class_ids),
    index=[f"true_{i}_{name(i)}" for i in class_ids],
    columns=[f"pred_{i}_{name(i)}" for i in class_ids],
).to_csv("sarima_selector_v2_confusion_matrix_bic.csv")

pred_rows = []
for i, (truth, nn_class, bic_class, p) in enumerate(zip(y_test, nn_pred, bic_test, probs)):
    top = np.argsort(p)[::-1][:3]
    pred_rows.append({
        "row": i,
        "true_class_id": int(truth), "true_specification": name(truth),
        "neural_class_id": int(nn_class), "neural_specification": name(nn_class),
        "neural_correct": bool(nn_class == truth),
        "bic_class_id": int(bic_class), "bic_specification": name(bic_class),
        "bic_correct": bool(bic_class == truth),
        "top1_score": float(p[top[0]]),
        "top2_class_id": int(top[1]), "top2_specification": name(top[1]),
        "top2_score": float(p[top[1]]),
        "top3_class_id": int(top[2]), "top3_specification": name(top[2]),
        "top3_score": float(p[top[2]]),
    })
pd.DataFrame(pred_rows).to_csv("sarima_selector_v2_test_predictions_recalculated.csv", index=False)

print("\n=== Метрики на сохранённой тестовой выборке ===")
print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nФайлы сохранены:")
for f in [
    "sarima_selector_v2_metrics_recalculated.csv",
    "sarima_selector_v2_classification_report.csv",
    "sarima_selector_v2_confusion_matrix_nn.csv",
    "sarima_selector_v2_confusion_matrix_bic.csv",
    "sarima_selector_v2_test_predictions_recalculated.csv",
]:
    print("-", f)
