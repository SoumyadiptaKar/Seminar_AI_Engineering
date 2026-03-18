import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, List

import yaml
from ultralytics import YOLO
from ultralytics.data.converter import convert_coco

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learning.common.tracking import track_execution
from unlearning.common.device import resolve_device, torch_device_for_log


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def _category_names(coco: Dict[str, Any]) -> List[str]:
    categories = coco.get("categories", [])
    if not categories:
        return []

    category_ids = sorted({int(c["id"]) for c in categories})
    names = []
    seen = {}
    for cid in category_ids:
        cat = next(c for c in categories if int(c["id"]) == cid)
        name = str(cat.get("name", f"class_{cid}"))
        count = seen.get(name, 0)
        seen[name] = count + 1
        if count > 0:
            name = f"{name}_{cid}"
        names.append(name)
    return names


def _prepare_retain_dataset(dataset_root: Path, split_manifest: Dict[str, Any], out_dir: Path) -> str:
    ann_dir = out_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    split_map = {"train": "train", "valid": "val", "test": "test"}
    for split, yolo_split in split_map.items():
        split_info = split_manifest.get("splits", {}).get(split)
        if not split_info:
            continue
        retain_ann = split_info.get("retain_annotations")
        if not retain_ann or not os.path.exists(retain_ann):
            continue
        shutil.copy2(retain_ann, ann_dir / f"{yolo_split}.json")

    converted_dir = out_dir / "converted"
    convert_coco(
        labels_dir=str(ann_dir),
        save_dir=str(converted_dir),
        use_segments=False,
        cls91to80=False,
    )

    for split, yolo_split in split_map.items():
        src_split = dataset_root / split
        if not src_split.exists():
            continue
        dst_split = converted_dir / "images" / yolo_split
        dst_split.mkdir(parents=True, exist_ok=True)
        for img in src_split.glob("*.jpg"):
            _safe_symlink(img, dst_split / img.name)

    first_split = next(iter(split_manifest.get("splits", {}).keys()), None)
    if not first_split:
        raise ValueError("No splits found in split manifest")
    retain_ann_path = split_manifest["splits"][first_split]["retain_annotations"]
    names = _category_names(_read_json(retain_ann_path))

    data_yaml = {
        "path": str(converted_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    yaml_path = out_dir / "retain_data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    return str(yaml_path)


def _load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(base: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    return p if p.is_absolute() else (base / p).resolve()


def run(config_path: str) -> Dict[str, Any]:
    cfg = _load_cfg(config_path)
    base = PROJECT_ROOT

    project_root = _resolve(base, cfg.get("project_root", "."))
    dataset_root = _resolve(project_root, cfg["dataset_root"])
    split_manifest_path = _resolve(project_root, cfg["split_manifest"])
    output_dir = _resolve(project_root, cfg["output_dir"])

    run_cfg = cfg["baseline_retrain"]
    tracking_cfg = cfg.get("tracking", {})
    forget_class = run_cfg.get("forget_class", "trichome")
    resolved_device = resolve_device(run_cfg.get("device", "auto"))

    try:
        import torch

        torch_version = torch.__version__
    except Exception:
        torch_version = "unknown"

    split_manifest = _read_json(str(split_manifest_path))

    run_dir = output_dir / f"retrain_excluding_{forget_class}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prepared_dir = run_dir / "prepared_data"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = _prepare_retain_dataset(dataset_root, split_manifest, prepared_dir)

    model = YOLO(run_cfg.get("model_config", "yolov8n-seg.yaml"))
    dry_run = bool(run_cfg.get("dry_run", True))

    with track_execution(
        project_name=f"retrain-excluding-{forget_class}",
        device=resolved_device,
        enable_codecarbon=bool(tracking_cfg.get("use_codecarbon", False)),
        estimated_watts=tracking_cfg.get("estimated_watts"),
    ) as tracking:
        if not dry_run:
            model.train(
                data=data_yaml,
                epochs=int(run_cfg.get("epochs", 50)),
                batch=int(run_cfg.get("batch_size", 8)),
                imgsz=int(run_cfg.get("imgsz", 640)),
                device=resolved_device,
                lr0=float(run_cfg.get("learning_rate", 1e-3)),
                workers=int(run_cfg.get("workers", 4)),
                seed=int(run_cfg.get("seed", 42)),
                project=str(run_dir),
                name="train",
                exist_ok=True,
                verbose=False,
            )

    weights_out = run_dir / "retrained.pt"
    model.save(str(weights_out))

    summary = {
        "config_path": str(Path(config_path).resolve()),
        "forget_class": forget_class,
        "dry_run": dry_run,
        "device": torch_device_for_log(resolved_device, torch_version),
        "data_yaml": data_yaml,
        "weights": str(weights_out.resolve()),
        "tracking": tracking,
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary["summary_path"] = str(summary_path.resolve())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train retain-only baseline from scratch")
    parser.add_argument("--config", default="learning/config.yaml")
    args = parser.parse_args()

    summary = run(args.config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
