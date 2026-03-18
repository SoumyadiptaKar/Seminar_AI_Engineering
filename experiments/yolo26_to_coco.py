from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from PIL import Image


def _polygon_area(points: List[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _load_data_yaml(dataset_root: Path) -> Dict[str, Any]:
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml: {data_yaml}")
    with open(data_yaml, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.width, img.height


def convert_split(dataset_root: Path, split: str, names: List[str]) -> Path:
    split_dir = dataset_root / split
    image_dir = split_dir / "images"
    label_dir = split_dir / "labels"
    out_path = split_dir / "_annotations.coco.json"

    image_files = sorted([p for p in image_dir.iterdir() if p.is_file()])

    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    categories = [{"id": idx, "name": name} for idx, name in enumerate(names)]

    ann_id = 1
    for image_id, image_path in enumerate(image_files, start=1):
        width, height = _image_size(image_path)
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        content = label_path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) < 7 or len(parts) % 2 == 0:
                continue

            try:
                class_id = int(float(parts[0]))
                coords = [float(v) for v in parts[1:]]
            except Exception:
                continue

            if class_id < 0 or class_id >= len(names):
                continue

            pixel_points: List[Tuple[float, float]] = []
            flat_pixels: List[float] = []
            for i in range(0, len(coords), 2):
                x = min(max(coords[i], 0.0), 1.0) * width
                y = min(max(coords[i + 1], 0.0), 1.0) * height
                pixel_points.append((x, y))
                flat_pixels.extend([x, y])

            if len(pixel_points) < 3:
                continue

            xs = [p[0] for p in pixel_points]
            ys = [p[1] for p in pixel_points]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw <= 0 or bh <= 0:
                continue

            area = _polygon_area(pixel_points)
            if area <= 0:
                area = bw * bh

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x1, y1, bw, bh],
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": [flat_pixels],
                }
            )
            ann_id += 1

    payload = {
        "info": {"description": f"YOLO26-to-COCO converted split: {split}"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert YOLO26 labels into per-split COCO JSON files")
    parser.add_argument("--dataset-root", default="stomata-batch-1-18")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    cfg = _load_data_yaml(dataset_root)
    names = cfg.get("names", [])
    if not isinstance(names, list) or not names:
        raise ValueError("Invalid or missing class names in data.yaml")

    out = {}
    for split in ["train", "valid", "test"]:
        out[split] = str(convert_split(dataset_root, split, names))

    print(json.dumps({"dataset_root": str(dataset_root), "coco_files": out}, indent=2))


if __name__ == "__main__":
    main()
