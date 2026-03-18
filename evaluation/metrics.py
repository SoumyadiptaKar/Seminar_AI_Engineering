import json
import os
from typing import Dict, Any


def baseline_metrics(metrics_json_path: str) -> Dict[str, Any]:
    if not os.path.exists(metrics_json_path):
        return {}
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_metric_cards(
    baseline: Dict[str, Any],
    unlearned: Dict[str, Any],
) -> Dict[str, Any]:
    if not baseline:
        return {"note": "No baseline metrics found"}

    card = {
        "baseline_available": True,
        "unlearned_available": bool(unlearned),
    }

    try:
        card["baseline_test_map50"] = baseline["test"]["map50"]
        card["baseline_test_map50_95"] = baseline["test"]["map50_95"]
    except Exception:
        card["note"] = "Baseline format did not match expected schema"

    if unlearned:
        card.update({
            "retain_map50": unlearned.get("retain_map50"),
            "forget_map50": unlearned.get("forget_map50"),
            "runtime_seconds": unlearned.get("runtime_seconds"),
        })

    return card
