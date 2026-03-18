import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unlearning.common.data_prep import (
    load_manifest,
    prepare_forget_empty_dataset,
    prepare_retain_dataset,
    read_json,
    read_yaml,
)


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(project_root: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    return p if p.is_absolute() else (project_root / p).resolve()


def _scan_yolo_labels(label_root: Path) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for split in ["train", "val", "test"]:
        split_dir = label_root / split
        files = list(split_dir.glob("*.txt")) if split_dir.exists() else []

        empty_files = 0
        bad_format_lines = 0
        oob_lines = 0
        valid_lines = 0
        class_counts = Counter()

        for label_file in files:
            content = label_file.read_text(encoding="utf-8").strip()
            if not content:
                empty_files += 1
                continue
            for line in content.splitlines():
                parts = line.strip().split()
                if len(parts) < 7 or len(parts) % 2 == 0:
                    bad_format_lines += 1
                    continue
                try:
                    class_id = int(float(parts[0]))
                    coords = [float(v) for v in parts[1:]]
                except Exception:
                    bad_format_lines += 1
                    continue
                if any((v < 0 or v > 1) for v in coords):
                    oob_lines += 1
                    continue
                class_counts[class_id] += 1
                valid_lines += 1

        stats[split] = {
            "files": len(files),
            "empty_files": empty_files,
            "valid_lines": valid_lines,
            "bad_format_lines": bad_format_lines,
            "oob_lines": oob_lines,
            "class_counts": dict(class_counts),
        }
    return stats


def _collect_source_dataset_diagnostics(manifest: Dict[str, Any]) -> Dict[str, Any]:
    by_split: Dict[str, Any] = {}
    for split, info in manifest.get("splits", {}).items():
        source_path = info.get("source_annotations")
        if not source_path or not os.path.exists(source_path):
            by_split[split] = {"error": "missing source annotations"}
            continue

        coco = read_json(source_path)
        categories = coco.get("categories", [])
        names = [str(c.get("name", "")) for c in categories]
        name_counts = Counter(names)
        duplicate_names = {k: v for k, v in name_counts.items() if v > 1}

        seg_type_counts = Counter()
        anns = coco.get("annotations", [])
        for ann in anns[:5000]:
            seg = ann.get("segmentation")
            if isinstance(seg, dict):
                seg_type_counts["dict_rle"] += 1
            elif isinstance(seg, list):
                if not seg:
                    seg_type_counts["list_empty"] += 1
                elif isinstance(seg[0], list):
                    seg_type_counts["list_of_lists"] += 1
                else:
                    seg_type_counts["flat_list"] += 1
            elif seg is None:
                seg_type_counts["none"] += 1
            else:
                seg_type_counts[type(seg).__name__] += 1

        by_split[split] = {
            "images": len(coco.get("images", [])),
            "annotations": len(anns),
            "duplicate_category_names": duplicate_names,
            "sample_segmentation_types": dict(seg_type_counts),
        }

    return by_split


def run_preflight(config_path: str, clean: bool = True) -> Dict[str, Any]:
    cfg = _load_config(config_path)
    project_root = _resolve_path(PROJECT_ROOT, cfg.get("project_root", "."))

    split_manifest_path = _resolve_path(project_root, cfg.get("split_manifest", "outputs/splits/split_manifest.json"))
    output_dir = _resolve_path(project_root, cfg.get("output_dir", "outputs"))

    blocking_issues: List[str] = []
    warnings: List[str] = []

    if not split_manifest_path.exists():
        blocking_issues.append(f"Missing split manifest: {split_manifest_path}")
        return {
            "ready_to_run": False,
            "config_path": str(Path(config_path).resolve()),
            "blocking_issues": blocking_issues,
            "warnings": warnings,
        }

    manifest = load_manifest(str(split_manifest_path))

    for split in ["train", "valid", "test"]:
        split_info = manifest.get("splits", {}).get(split)
        if not split_info:
            blocking_issues.append(f"Missing split in manifest: {split}")
            continue
        for key in ["source_annotations", "forget_annotations", "retain_annotations"]:
            p = split_info.get(key)
            if not p or not Path(p).exists():
                blocking_issues.append(f"Missing file for {split}.{key}: {p}")

    source_diag = _collect_source_dataset_diagnostics(manifest)
    for split, diag in source_diag.items():
        dup = diag.get("duplicate_category_names", {})
        if dup:
            warnings.append(f"{split}: duplicate category names detected {dup}; preprocessor will uniquify names")
        seg_types = diag.get("sample_segmentation_types", {})
        if seg_types.get("dict_rle", 0) > 0:
            warnings.append(f"{split}: source masks include COCO RLE dicts; conversion preserves masks to segment labels")

    preflight_root = output_dir / "preflight"
    if clean and preflight_root.exists():
        shutil.rmtree(preflight_root)
    preflight_root.mkdir(parents=True, exist_ok=True)

    prepared_dir = preflight_root / "prepared_data"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    try:
        retain_yaml = Path(prepare_retain_dataset(manifest, prepared_dir))
        forget_yaml = Path(prepare_forget_empty_dataset(manifest, prepared_dir))
    except Exception as exc:
        blocking_issues.append(f"Dataset preparation failed: {exc}")
        return {
            "ready_to_run": False,
            "config_path": str(Path(config_path).resolve()),
            "manifest_path": str(split_manifest_path),
            "blocking_issues": blocking_issues,
            "warnings": warnings,
        }

    retain_cfg = read_yaml(str(retain_yaml))
    forget_cfg = read_yaml(str(forget_yaml))

    retain_root = Path(retain_cfg["path"]) / "labels"
    forget_root = Path(forget_cfg["path"]) / "labels"

    retain_stats = _scan_yolo_labels(retain_root)
    forget_stats = _scan_yolo_labels(forget_root)

    for split, s in retain_stats.items():
        if s["files"] == 0:
            blocking_issues.append(f"retain/{split}: no label files generated")
        if s["valid_lines"] == 0:
            warnings.append(f"retain/{split}: no valid labels found")
        if s["bad_format_lines"] > 0:
            blocking_issues.append(f"retain/{split}: {s['bad_format_lines']} malformed label lines")
        if s["oob_lines"] > 0:
            blocking_issues.append(f"retain/{split}: {s['oob_lines']} out-of-bounds label lines")

    for split, s in forget_stats.items():
        if s["files"] == 0:
            warnings.append(f"forget/{split}: no files (split may be empty)")
        if s["valid_lines"] > 0:
            warnings.append(f"forget/{split}: has non-empty labels (expected mostly empty for suppression stage)")

    report = {
        "ready_to_run": len(blocking_issues) == 0,
        "config_path": str(Path(config_path).resolve()),
        "manifest_path": str(split_manifest_path),
        "prepared_retain_yaml": str(retain_yaml),
        "prepared_forget_yaml": str(forget_yaml),
        "source_diagnostics": source_diag,
        "retain_label_stats": retain_stats,
        "forget_label_stats": forget_stats,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
    }

    out_path = preflight_root / "preflight_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(out_path.resolve())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight checks for unlearning pipeline")
    parser.add_argument("--config", default="experiments/config.yaml")
    parser.add_argument("--no-clean", action="store_true", help="Do not clean previous preflight outputs")
    args = parser.parse_args()

    payload = run_preflight(args.config, clean=not args.no_clean)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
