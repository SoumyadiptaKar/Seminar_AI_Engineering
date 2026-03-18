import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import baseline_metrics, compare_metric_cards
from evaluation.report import write_experiment_summary
from unlearning.common.device import resolve_device, torch_device_for_log
from unlearning.common.registry import get_unlearner_class
from unlearning.common.types import UnlearningConfig
from unlearning.common.utils import now_iso


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(project_root: str, maybe_relative: str) -> str:
    path = Path(maybe_relative)
    if path.is_absolute():
        return str(path)
    return str((Path(project_root) / path).resolve())


def run_once(config_path: str) -> Dict[str, Any]:
    cfg = load_config(config_path)

    project_root = resolve_path(".", cfg["project_root"])
    original_weights = resolve_path(project_root, cfg["original_weights"])
    dataset_root = resolve_path(project_root, cfg["dataset_root"])
    output_dir = resolve_path(project_root, cfg["output_dir"])
    baseline_json = resolve_path(project_root, cfg["baseline_metrics_json"])

    run_cfg = cfg["run"]
    algorithm = run_cfg["algorithm"]
    resolved_device = resolve_device(run_cfg.get("device", "auto"))
    try:
        import torch

        torch_version = torch.__version__
    except Exception:
        torch_version = "unknown"

    unlearn_cfg = UnlearningConfig(
        algorithm=algorithm,
        project_root=project_root,
        original_weights=original_weights,
        output_dir=output_dir,
        dataset_root=dataset_root,
        device=resolved_device,
        forget_class=run_cfg.get("forget_class", "trichome"),
        epochs=run_cfg.get("epochs", 5),
        batch_size=run_cfg.get("batch_size", 8),
        learning_rate=run_cfg.get("learning_rate", 1e-4),
        seed=run_cfg.get("seed", 42),
        extra={
            "sisa_shards": cfg.get("sisa", {}).get("shards", 10),
            "sisa_slices_per_shard": cfg.get("sisa", {}).get("slices_per_shard", 5),
        },
    )

    unlearner_cls = get_unlearner_class(algorithm)
    result = unlearner_cls(unlearn_cfg).run()

    baseline = baseline_metrics(baseline_json)
    unlearned_metrics_stub = {
        "retain_map50": None,
        "forget_map50": None,
        "runtime_seconds": result.runtime_seconds,
    }
    metric_card = compare_metric_cards(baseline, unlearned_metrics_stub)

    summary = {
        "timestamp": now_iso(),
        "config_path": str(Path(config_path).resolve()),
        "algorithm": result.algorithm,
        "success": result.success,
        "output_weights": result.output_weights,
        "runtime_seconds": result.runtime_seconds,
        "notes": result.notes,
        "device": torch_device_for_log(resolved_device, torch_version),
        "metric_card": metric_card,
    }

    run_dir = os.path.join(output_dir, "runs", result.algorithm)
    out_json = write_experiment_summary(run_dir, summary)
    summary["summary_path"] = out_json
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unlearning experiment scaffold")
    parser.add_argument("--config", default="experiments/config.yaml")
    args = parser.parse_args()

    summary = run_once(args.config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
