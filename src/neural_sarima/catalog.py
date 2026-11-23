from __future__ import annotations
import json
from pathlib import Path

NONSEASONAL_ORDERS = [
    (1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 0), (0, 0, 2),
    (2, 0, 1), (1, 0, 2), (2, 0, 2), (1, 1, 0), (0, 1, 1),
    (1, 1, 1), (2, 1, 0), (0, 1, 2), (2, 1, 1), (1, 1, 2),
    (2, 1, 2), (0, 2, 1), (1, 2, 1),
]
SEASONAL_TEMPLATES = [
    ((0, 0, 0), (1, 0, 0)),
    ((0, 0, 0), (0, 0, 1)),
    ((1, 0, 0), (1, 0, 0)),
    ((0, 0, 1), (0, 0, 1)),
    ((1, 0, 1), (1, 0, 0)),
    ((1, 1, 0), (0, 1, 1)),
    ((0, 1, 1), (1, 1, 0)),
    ((1, 1, 1), (0, 1, 1)),
]


def build_catalog() -> list[dict]:
    result = []
    for order in NONSEASONAL_ORDERS:
        result.append({"order": list(order), "seasonal_order": [0, 0, 0, 0], "family": "ARIMA"})
    for period in (4, 8, 12):
        for order, seasonal in SEASONAL_TEMPLATES:
            result.append({
                "order": list(order),
                "seasonal_order": [*seasonal, period],
                "family": f"SARIMA_s{period}",
            })
    keys = [(tuple(x["order"]), tuple(x["seasonal_order"])) for x in result]
    if len(result) != 42 or len(keys) != len(set(keys)):
        raise RuntimeError("Catalog must contain 42 unique specifications.")
    return result

CATALOG = build_catalog()
N_CLASSES = len(CATALOG)


def specification_name(spec: dict) -> str:
    order = tuple(spec["order"])
    seas = tuple(spec["seasonal_order"])
    if seas[3] == 0:
        return f"ARIMA{order}"
    return f"SARIMA{order}x{seas}"


def catalog_rows() -> list[dict]:
    rows = []
    for class_id, spec in enumerate(CATALOG):
        rows.append({
            "class_id": class_id,
            "specification": specification_name(spec),
            "order": json.dumps(spec["order"]),
            "seasonal_order": json.dumps(spec["seasonal_order"]),
            "family": spec["family"],
        })
    return rows


def save_catalog(path: Path) -> None:
    path.write_text(json.dumps(catalog_rows(), ensure_ascii=False, indent=2), encoding="utf-8")
