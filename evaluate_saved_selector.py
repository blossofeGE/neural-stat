
"""
evaluate_saved_selector.py

Повторная оценка сохранённой нейросети для выбора спецификации ARIMA/SARIMA.

Скрипт:
1. Загружает сохранённую модель и scaler.
2. Загружает тестовую выборку.
3. Рассчитывает метрики нейросети и базового метода BIC.
4. Выполняет точный тест Мак-Немара и вариант с поправкой на непрерывность.
5. Сохраняет результаты в CSV и TXT.

Важно:
- Скрипт не обучает модель заново.
- Тестовые данные должны быть получены на этапе обучения.
- nn_pred и bic_pred должны соответствовать одним и тем же рядам.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    log_loss,
)

from scipy.stats import binomtest, chi2


# ============================================================
# 1. НАСТРОЙКИ
# ============================================================

MODEL_PATH = Path("sarima_selector_v2.keras")
SCALER_PATH = Path("sarima_selector_v2_scaler.npz")
TEST_PATH = Path("sarima_selector_v2_test_data.npz")
CATALOG_PATH = Path("sarima_selector_v2_catalog.json")

OUTPUT_METRICS_CSV = Path("sarima_selector_v2_metrics.csv")
OUTPUT_REPORT_TXT = Path("sarima_selector_v2_report.txt")
OUTPUT_MCNEMAR_CSV = Path("sarima_selector_v2_mcnemar_test.csv")

ALPHA = 0.05


# ============================================================
# 2. ПРОВЕРКА ФАЙЛОВ
# ============================================================

required_files = [
    MODEL_PATH,
    SCALER_PATH,
    TEST_PATH,
    CATALOG_PATH,
]

missing_files = [str(path) for path in required_files if not path.exists()]

if missing_files:
    raise FileNotFoundError(
        "Не найдены необходимые файлы:\n"
        + "\n".join(missing_files)
        + "\n\nПоложите их в папку со скриптом "
          "или измените пути в начале файла."
    )

print("=" * 70)
print("ОЦЕНКА СОХРАНЁННОЙ НЕЙРОСЕТИ ARIMA/SARIMA")
print("=" * 70)


# ============================================================
# 3. ЗАГРУЗКА МОДЕЛИ И ТЕСТОВЫХ ДАННЫХ
# ============================================================

print("\nЗагрузка нейросети...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Модель загружена:", MODEL_PATH)

print("\nЗагрузка масштабатора...")

with np.load(SCALER_PATH, allow_pickle=False) as scaler_data:
    # Поддерживаются два распространённых формата сохранения.
    if "mean" in scaler_data and "scale" in scaler_data:
        scaler_mean = scaler_data["mean"]
        scaler_scale = scaler_data["scale"]
    elif "mean_" in scaler_data and "scale_" in scaler_data:
        scaler_mean = scaler_data["mean_"]
        scaler_scale = scaler_data["scale_"]
    else:
        raise KeyError(
            "В файле scaler не найдены массивы mean/scale "
            "или mean_/scale_. Проверьте формат сохранения."
        )

print("Масштабатор загружен.")

print("\nЗагрузка тестовой выборки...")

with np.load(TEST_PATH, allow_pickle=False) as test_data:
    available_keys = list(test_data.files)
    print("Массивы в файле:", available_keys)

    required_keys = ["X_test", "y_test", "bic_test"]
    missing_keys = [key for key in required_keys if key not in test_data]

    if missing_keys:
        raise KeyError(
            "В тестовом NPZ отсутствуют массивы: "
            + ", ".join(missing_keys)
            + ". Имеющиеся массивы: "
            + ", ".join(available_keys)
        )

    X_test = test_data["X_test"].astype(np.float32)
    y_test = test_data["y_test"].astype(int)
    bic_test = test_data["bic_test"].astype(int)

with open(CATALOG_PATH, "r", encoding="utf-8") as file:
    catalog = json.load(file)

print("Количество тестовых наблюдений:", len(y_test))
print("Размерность X_test:", X_test.shape)


# ============================================================
# 4. ПОДГОТОВКА ДАННЫХ
# ============================================================

if X_test.ndim != 2:
    raise ValueError(
        f"Ожидался двумерный X_test, получена форма {X_test.shape}."
    )

if len(X_test) != len(y_test) or len(y_test) != len(bic_test):
    raise ValueError(
        "X_test, y_test и bic_test должны содержать "
        "одинаковое количество наблюдений."
    )

if X_test.shape[1] != len(scaler_mean):
    raise ValueError(
        "Количество признаков X_test не совпадает с размерностью scaler: "
        f"{X_test.shape[1]} против {len(scaler_mean)}."
    )

if len(scaler_mean) != len(scaler_scale):
    raise ValueError("Размерности mean и scale масштабатора не совпадают.")

if np.any(scaler_scale == 0):
    raise ValueError(
        "В scaler найдены нулевые значения scale. "
        "Проверьте сохранённый масштабатор."
    )

# Стандартизация должна соответствовать той, которая использовалась
# при обучении модели.
X_test_scaled = (X_test - scaler_mean) / scaler_scale

# Защита от NaN и бесконечностей.
if not np.isfinite(X_test_scaled).all():
    raise ValueError(
        "После стандартизации в X_test обнаружены NaN или бесконечности."
    )

# Индексы классов должны быть неотрицательными.
if np.any(y_test < 0) or np.any(bic_test < 0):
    raise ValueError("Индексы классов не должны быть отрицательными.")


# ============================================================
# 5. ПРЕДСКАЗАНИЯ НЕЙРОСЕТИ
# ============================================================

print("\nВычисление предсказаний нейросети...")

probabilities = model.predict(
    X_test_scaled,
    verbose=0,
)

if probabilities.ndim != 2:
    raise ValueError(
        "Модель должна возвращать матрицу вероятностей классов."
    )

nn_pred = np.argmax(probabilities, axis=1)

n_classes = probabilities.shape[1]

if np.any(y_test >= n_classes) or np.any(bic_test >= n_classes):
    raise ValueError(
        "В y_test или bic_test обнаружен индекс класса, "
        "который отсутствует в выходах модели."
    )

if not np.isfinite(probabilities).all():
    raise ValueError(
        "Предсказанные вероятности содержат NaN или бесконечности."
    )

# Нормируем вероятности на случай небольших погрешностей.
row_sums = probabilities.sum(axis=1, keepdims=True)

if np.any(row_sums <= 0):
    raise ValueError("Некорректная сумма предсказанных вероятностей.")

probabilities = probabilities / row_sums

print("Предсказания готовы.")


# ============================================================
# 6. МЕТРИКИ КАЧЕСТВА
# ============================================================

def calculate_metrics(y_true, y_pred, model_name, probs=None):
    """Рассчитать основные метрики классификации."""

    result = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }

    if probs is not None:
        try:
            result["log_loss"] = log_loss(
                y_true,
                probs,
                labels=np.arange(probs.shape[1]),
            )
        except ValueError:
            result["log_loss"] = np.nan

        try:
            result["macro_roc_auc_ovr"] = roc_auc_score(
                y_true,
                probs,
                multi_class="ovr",
                average="macro",
                labels=np.arange(probs.shape[1]),
            )
        except ValueError:
            # Например, если в тесте отсутствует один из классов.
            result["macro_roc_auc_ovr"] = np.nan
    else:
        result["log_loss"] = np.nan
        result["macro_roc_auc_ovr"] = np.nan

    return result


nn_metrics = calculate_metrics(
    y_test,
    nn_pred,
    "Neural network",
    probabilities,
)

bic_metrics = calculate_metrics(
    y_test,
    bic_test,
    "BIC baseline",
)

metrics_df = pd.DataFrame([nn_metrics, bic_metrics])

print("\nМЕТРИКИ КАЧЕСТВА")
print("=" * 70)
print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

metrics_df.to_csv(OUTPUT_METRICS_CSV, index=False, encoding="utf-8-sig")
print("\nМетрики сохранены в:", OUTPUT_METRICS_CSV)


# ============================================================
# 7. ТЕСТ МАК-НЕМАРА
# ============================================================

print("\nТЕСТ МАК-НЕМАРА: НЕЙРОСЕТЬ ПРОТИВ BIC")
print("=" * 70)

# Для Мак-Немара сравниваются два бинарных исхода для каждого ряда:
# правильно ли классифицирован этот ряд каждым из методов.

nn_correct = (nn_pred == y_test)
bic_correct = (bic_test == y_test)

# n10: нейросеть верна, BIC ошибся.
n10 = int(np.sum(nn_correct & ~bic_correct))

# n01: нейросеть ошиблась, BIC верен.
n01 = int(np.sum(~nn_correct & bic_correct))

# Оба метода либо правы, либо ошибаются — такие случаи
# не определяют статистику Мак-Немара.
n11 = int(np.sum(nn_correct & bic_correct))
n00 = int(np.sum(~nn_correct & ~bic_correct))

n_discordant = n10 + n01

# Точный двусторонний биномиальный тест — точный вариант Мак-Немара.
if n_discordant > 0:
    exact_result = binomtest(
        k=min(n10, n01),
        n=n_discordant,
        p=0.5,
        alternative="two-sided",
    )
    p_exact = float(exact_result.pvalue)

    # Асимптотический вариант с поправкой на непрерывность.
    statistic_cc = (
        (abs(n10 - n01) - 1) ** 2 / n_discordant
    )
    p_chi2_cc = float(chi2.sf(statistic_cc, df=1))

    # Асимптотический вариант без поправки.
    statistic = (n10 - n01) ** 2 / n_discordant
    p_chi2 = float(chi2.sf(statistic, df=1))
else:
    # Все парные исходы совпадают: данных для различия нет.
    p_exact = 1.0
    statistic_cc = 0.0
    p_chi2_cc = 1.0
    statistic = 0.0
    p_chi2 = 1.0

print(f"Оба метода классифицировали правильно (n11): {n11}")
print(f"Оба метода классифицировали неправильно (n00): {n00}")
print(f"Нейросеть права, BIC ошибся (n10): {n10}")
print(f"Нейросеть ошиблась, BIC прав (n01): {n01}")
print(f"Всего несовпадающих пар: {n_discordant}")
print()
print(f"Точный тест Мак-Немара: p-value = {p_exact:.8f}")
print(
    "Мак-Немар, хи-квадрат с поправкой на непрерывность: "
    f"statistic = {statistic_cc:.6f}, p-value = {p_chi2_cc:.8f}"
)
print(
    "Мак-Немар, хи-квадрат без поправки: "
    f"statistic = {statistic:.6f}, p-value = {p_chi2:.8f}"
)

if p_exact < ALPHA:
    if n10 > n01:
        conclusion = (
            "Различие статистически значимо; нейросеть чаще "
            "классифицирует правильно, чем BIC, на парных наблюдениях."
        )
    elif n01 > n10:
        conclusion = (
            "Различие статистически значимо; BIC чаще "
            "классифицирует правильно, чем нейросеть, на парных наблюдениях."
        )
    else:
        conclusion = (
            "Получено p-value ниже порога, но направления расхождений "
            "равны; проверьте корректность данных и расчёта."
        )
else:
    conclusion = (
        "При выбранном уровне значимости статистически значимое "
        "различие долей правильных классификаций не установлено."
    )

print("\nВывод при alpha =", ALPHA)
print(conclusion)

mcnemar_df = pd.DataFrame([{
    "n11_both_correct": n11,
    "n00_both_wrong": n00,
    "n10_nn_correct_bic_wrong": n10,
    "n01_nn_wrong_bic_correct": n01,
    "n_discordant": n_discordant,
    "mcnemar_exact_p": p_exact,
    "mcnemar_chi2_cc_statistic": statistic_cc,
    "mcnemar_chi2_cc_p": p_chi2_cc,
    "mcnemar_chi2_statistic": statistic,
    "mcnemar_chi2_p": p_chi2,
    "alpha": ALPHA,
    "significant_exact": p_exact < ALPHA,
    "conclusion": conclusion,
}])

mcnemar_df.to_csv(
    OUTPUT_MCNEMAR_CSV,
    index=False,
    encoding="utf-8-sig",
)

print("\nРезультаты теста Мак-Немара сохранены в:", OUTPUT_MCNEMAR_CSV)


# ============================================================
# 8. ПОДРОБНЫЙ ОТЧЁТ ПО КЛАССАМ
# ============================================================

print("\nОТЧЁТ ПО КЛАССАМ: НЕЙРОСЕТЬ")
print("=" * 70)

report = classification_report(
    y_test,
    nn_pred,
    labels=np.arange(n_classes),
    zero_division=0,
)

print(report)

report_bic = classification_report(
    y_test,
    bic_test,
    labels=np.arange(n_classes),
    zero_division=0,
)

print("\nОТЧЁТ ПО КЛАССАМ: BIC")
print("=" * 70)
print(report_bic)


# ============================================================
# 9. МАТРИЦЫ ОШИБОК
# ============================================================

cm_nn = confusion_matrix(
    y_test,
    nn_pred,
    labels=np.arange(n_classes),
)

cm_bic = confusion_matrix(
    y_test,
    bic_test,
    labels=np.arange(n_classes),
)

print("\nМатрица ошибок нейросети:")
print(cm_nn)

print("\nМатрица ошибок BIC:")
print(cm_bic)


# ============================================================
# 10. СОХРАНЕНИЕ ТЕКСТОВОГО ОТЧЁТА
# ============================================================

with open(OUTPUT_REPORT_TXT, "w", encoding="utf-8") as file:
    file.write("ОЦЕНКА НЕЙРОСЕТИ ARIMA/SARIMA\n")
    file.write("=" * 70 + "\n\n")

    file.write("Метрики качества:\n")
    file.write(metrics_df.to_string(index=False))
    file.write("\n\n")

    file.write("Тест Мак-Немара:\n")
    file.write(f"n11 (оба верны): {n11}\n")
    file.write(f"n00 (оба ошиблись): {n00}\n")
    file.write(f"n10 (нейросеть верна, BIC ошибся): {n10}\n")
    file.write(f"n01 (нейросеть ошиблась, BIC верен): {n01}\n")
    file.write(f"Точный p-value: {p_exact:.8f}\n")
    file.write(f"Хи-квадрат с поправкой: {statistic_cc:.6f}\n")
    file.write(f"p-value с поправкой: {p_chi2_cc:.8f}\n")
    file.write(f"Уровень значимости alpha: {ALPHA}\n")
    file.write("Вывод: " + conclusion + "\n\n")

    file.write("Отчёт по классам — нейросеть:\n")
    file.write(report)
    file.write("\n\nОтчёт по классам — BIC:\n")
    file.write(report_bic)

    file.write("\n\nМатрица ошибок нейросети:\n")
    file.write(np.array2string(cm_nn))
    file.write("\n\nМатрица ошибок BIC:\n")
    file.write(np.array2string(cm_bic))
    file.write("\n")

print("Текстовый отчёт сохранён в:", OUTPUT_REPORT_TXT)

print("\n" + "=" * 70)
print("ОЦЕНКА ЗАВЕРШЕНА")
print("=" * 70)
