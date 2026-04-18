import json
import csv
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    model_path = root / "metrics_reports_annotation" / "model_comparison_summary.json"
    energy_path = root / "outputs" / "comparison" / "energy_comparison_annotation.json"
    out_json = root / "outputs" / "comparison" / "tableau_unlearning_combined.json"
    out_csv = root / "outputs" / "comparison" / "tableau_unlearning_combined.csv"

    model_data = json.loads(model_path.read_text())
    energy_data = json.loads(energy_path.read_text())

    energy_by_alg = {item["algorithm"]: item for item in energy_data.get("algorithms", [])}

    rows = []
    models = model_data.get("models", {})
    for algorithm, payload in models.items():
        energy = energy_by_alg.get(algorithm, {})
        for split_name in ("validation", "test"):
            split = payload.get(split_name, {})
            per_map50 = split.get("per_class_map50", {})
            per_map95 = split.get("per_class_map50_95", {})
            per_prec50 = split.get("per_class_precision50", {})
            per_rec50 = split.get("per_class_recall50", {})

            rows.append(
                {
                    "algorithm": algorithm,
                    "split": split_name,
                    "weights_path": payload.get("weights"),
                    "timestamp": model_data.get("timestamp"),
                    "precision": split.get("precision"),
                    "recall": split.get("recall"),
                    "map50": split.get("map50"),
                    "map50_95": split.get("map50_95"),
                    "num_images": split.get("num_images"),
                    "num_detections": split.get("num_detections"),
                    "map50_nothing": per_map50.get("nothing"),
                    "map50_stomata": per_map50.get("stomata"),
                    "map50_trichome": per_map50.get("trichome"),
                    "map50_vein": per_map50.get("vein"),
                    "map50_95_nothing": per_map95.get("nothing"),
                    "map50_95_stomata": per_map95.get("stomata"),
                    "map50_95_trichome": per_map95.get("trichome"),
                    "map50_95_vein": per_map95.get("vein"),
                    "precision50_nothing": per_prec50.get("nothing"),
                    "precision50_stomata": per_prec50.get("stomata"),
                    "precision50_trichome": per_prec50.get("trichome"),
                    "precision50_vein": per_prec50.get("vein"),
                    "recall50_nothing": per_rec50.get("nothing"),
                    "recall50_stomata": per_rec50.get("stomata"),
                    "recall50_trichome": per_rec50.get("trichome"),
                    "recall50_vein": per_rec50.get("vein"),
                    "success": energy.get("success"),
                    "runtime_seconds": energy.get("runtime_seconds"),
                    "duration_seconds": energy.get("duration_seconds"),
                    "energy_kwh": energy.get("energy_kwh"),
                    "co2_kg": energy.get("co2_kg"),
                    "tracking_backend": energy.get("tracking_backend"),
                    "assumed_watts": energy.get("assumed_watts"),
                }
            )

    model_algorithms = set(models.keys())
    for algorithm, energy in energy_by_alg.items():
        if algorithm in model_algorithms:
            continue
        rows.append(
            {
                "algorithm": algorithm,
                "split": "aggregate",
                "weights_path": None,
                "timestamp": model_data.get("timestamp"),
                "precision": None,
                "recall": None,
                "map50": None,
                "map50_95": None,
                "num_images": None,
                "num_detections": None,
                "map50_nothing": None,
                "map50_stomata": None,
                "map50_trichome": None,
                "map50_vein": None,
                "map50_95_nothing": None,
                "map50_95_stomata": None,
                "map50_95_trichome": None,
                "map50_95_vein": None,
                "precision50_nothing": None,
                "precision50_stomata": None,
                "precision50_trichome": None,
                "precision50_vein": None,
                "recall50_nothing": None,
                "recall50_stomata": None,
                "recall50_trichome": None,
                "recall50_vein": None,
                "success": energy.get("success"),
                "runtime_seconds": energy.get("runtime_seconds"),
                "duration_seconds": energy.get("duration_seconds"),
                "energy_kwh": energy.get("energy_kwh"),
                "co2_kg": energy.get("co2_kg"),
                "tracking_backend": energy.get("tracking_backend"),
                "assumed_watts": energy.get("assumed_watts"),
            }
        )

    output_payload = {
        "dataset_name": "tableau_unlearning_combined",
        "description": "Combined model metrics and energy/runtime metrics for Tableau",
        "source_files": {
            "model_summary": str(model_path),
            "energy_summary": str(energy_path),
        },
        "row_count": len(rows),
        "rows": rows,
    }

    out_json.write_text(json.dumps(output_payload, indent=2))

    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows")
    print(out_json)
    print(out_csv)


if __name__ == "__main__":
    main()
