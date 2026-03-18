import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from pycocotools import mask as mask_utils
from ultralytics import YOLO


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def image_name_to_label_name(image_name: str) -> str:
    return Path(image_name).with_suffix(".txt").name


def resolve_image_path_in_split(split_root: Path, file_name: str) -> Optional[Path]:
    direct = split_root / file_name
    if direct.exists():
        return direct

    nested = split_root / "images" / file_name
    if nested.exists():
        return nested

    stem = Path(file_name).stem
    for candidate in (split_root / "images").glob(f"{stem}.*") if (split_root / "images").exists() else []:
        if candidate.is_file():
            return candidate

    for candidate in split_root.glob(f"{stem}.*"):
        if candidate.is_file():
            return candidate

    return None


def category_names(coco: Dict[str, Any]) -> List[str]:
    categories = coco.get("categories", [])
    if not categories:
        return []
    max_id = max(int(c["id"]) for c in categories)
    names = [f"class_{idx}" for idx in range(max_id + 1)]
    for c in categories:
        cid = int(c["id"])
        cname = str(c.get("name", f"class_{cid}"))
        names[cid] = cname
    return names


def _unique_category_names(categories: List[Dict[str, Any]]) -> List[str]:
    seen: Dict[str, int] = {}
    names: List[str] = []
    for cat in categories:
        cid = int(cat["id"])
        base = str(cat.get("name", f"class_{cid}"))
        count = seen.get(base, 0)
        seen[base] = count + 1
        names.append(base if count == 0 else f"{base}_{cid}")
    return names


def build_global_category_schema(manifest: Dict[str, Any]) -> Tuple[Dict[int, int], List[str]]:
    first_split = next(iter(manifest.get("splits", {}).keys()), None)
    if not first_split:
        raise ValueError("No splits found in manifest")

    source_ann = manifest["splits"][first_split].get("source_annotations")
    if not source_ann or not os.path.exists(source_ann):
        raise FileNotFoundError("Source annotations not found in split manifest")

    source_coco = read_json(source_ann)
    categories = sorted(source_coco.get("categories", []), key=lambda c: int(c["id"]))
    if not categories:
        raise ValueError("No categories found in source annotations")

    id_map = {int(cat["id"]): idx for idx, cat in enumerate(categories)}
    names = _unique_category_names(categories)
    return id_map, names


def _sanitize_segmentation(segmentation: Any, width: float, height: float) -> Any:
    if isinstance(segmentation, dict):
        seg = dict(segmentation)
        seg.setdefault("size", [int(height), int(width)])
        return seg

    if isinstance(segmentation, list):
        polys: List[List[float]] = segmentation if segmentation and isinstance(segmentation[0], list) else [segmentation]
        cleaned_polys: List[List[float]] = []

        for poly in polys:
            if not isinstance(poly, list):
                continue
            if len(poly) < 6 or len(poly) % 2 != 0:
                continue

            cleaned: List[float] = []
            valid = True
            for i, v in enumerate(poly):
                try:
                    fv = float(v)
                except Exception:
                    valid = False
                    break
                if i % 2 == 0:
                    fv = min(max(fv, 0.0), width)
                else:
                    fv = min(max(fv, 0.0), height)
                cleaned.append(fv)

            if valid and len(cleaned) >= 6:
                cleaned_polys.append(cleaned)

        if cleaned_polys:
            return cleaned_polys

    return None


def _normalized_polygon_line(class_id: int, poly: List[float], width: float, height: float) -> str:
    coords: List[float] = []
    for i, v in enumerate(poly):
        if i % 2 == 0:
            coords.append(min(max(float(v) / width, 0.0), 1.0))
        else:
            coords.append(min(max(float(v) / height, 0.0), 1.0))
    return f"{class_id} " + " ".join(f"{c:.6f}" for c in coords)


def _rle_to_polygons(segmentation: Dict[str, Any], width: int, height: int) -> List[List[float]]:
    seg = dict(segmentation)
    seg.setdefault("size", [height, width])

    rle = seg
    if isinstance(seg.get("counts"), list):
        rle = mask_utils.frPyObjects(seg, height, width)

    decoded = mask_utils.decode(rle)
    if decoded is None:
        return []

    mask = decoded[:, :, 0] if decoded.ndim == 3 else decoded
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[List[float]] = []
    for contour in contours:
        if contour is None or len(contour) < 3:
            continue
        flat = contour.reshape(-1, 2).astype(float).flatten().tolist()
        if len(flat) >= 6:
            polygons.append(flat)
    return polygons


def sanitize_coco_for_detection(
    src_json: str,
    dst_json: Path,
    category_id_map: Dict[int, int],
    category_names_list: List[str],
) -> Path:
    coco = read_json(src_json)
    images = coco.get("images", [])
    images_by_id = {int(img.get("id")): img for img in images if img.get("id") is not None}

    kept_annotations: List[Dict[str, Any]] = []
    kept_image_ids: set = set()

    for ann in coco.get("annotations", []):
        image_id = int(ann.get("image_id", -1))
        img = images_by_id.get(image_id)
        if not img:
            continue

        raw_category_id = int(ann.get("category_id", -1))
        if raw_category_id not in category_id_map:
            continue

        width = float(img.get("width", 0) or 0)
        height = float(img.get("height", 0) or 0)
        if width <= 1 or height <= 1:
            continue

        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        try:
            x, y, w, h = [float(v) for v in bbox]
        except Exception:
            continue

        x1 = min(max(x, 0.0), width)
        y1 = min(max(y, 0.0), height)
        x2 = min(max(x + w, 0.0), width)
        y2 = min(max(y + h, 0.0), height)
        new_w = x2 - x1
        new_h = y2 - y1
        if new_w <= 1e-6 or new_h <= 1e-6:
            continue

        clean_ann = dict(ann)
        clean_ann["category_id"] = int(category_id_map[raw_category_id])
        clean_ann["bbox"] = [x1, y1, new_w, new_h]

        segmentation = _sanitize_segmentation(ann.get("segmentation"), width, height)
        if segmentation is None:
            segmentation = [[x1, y1, x2, y1, x2, y2, x1, y2]]
        clean_ann["segmentation"] = segmentation

        kept_annotations.append(clean_ann)
        kept_image_ids.add(image_id)

    filtered_images = [img for img in images if int(img.get("id", -1)) in kept_image_ids]
    filtered_categories = [{"id": idx, "name": name} for idx, name in enumerate(category_names_list)]

    sanitized = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": filtered_categories,
        "images": filtered_images,
        "annotations": kept_annotations,
    }

    dst_json.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(sanitized, f)
    return dst_json


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    if not manifest_path:
        raise ValueError("Missing split manifest path")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    return read_json(manifest_path)


def prepare_retain_dataset(manifest: Dict[str, Any], work_dir: Path) -> str:
    ann_dir = work_dir / "retain_annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    category_id_map, names = build_global_category_schema(manifest)

    split_name_map = {"train": "train", "valid": "val", "test": "test"}
    for split, yolo_split in split_name_map.items():
        split_info = manifest.get("splits", {}).get(split)
        if not split_info:
            continue
        retain_ann = split_info.get("retain_annotations")
        if not retain_ann or not os.path.exists(retain_ann):
            continue
        sanitize_coco_for_detection(
            retain_ann,
            ann_dir / f"{yolo_split}.json",
            category_id_map=category_id_map,
            category_names_list=names,
        )

    converted_dir = work_dir / "retain_yolo"
    if converted_dir.exists():
        shutil.rmtree(converted_dir)

    for split, yolo_split in split_name_map.items():
        ann_path = ann_dir / f"{yolo_split}.json"
        if not ann_path.exists():
            continue
        write_yolo_segment_labels_from_coco(
            coco_json_path=ann_path,
            labels_out_dir=converted_dir / "labels" / yolo_split,
        )

    dataset_root = Path(manifest["dataset_root"])
    for split, yolo_split in split_name_map.items():
        dst_split_dir = converted_dir / "images" / yolo_split
        ann_path = ann_dir / f"{yolo_split}.json"
        if not ann_path.exists():
            continue
        retain_coco = read_json(str(ann_path))
        image_names = [img.get("file_name") for img in retain_coco.get("images", []) if img.get("file_name")]

        src_split_dir = dataset_root / split
        if not src_split_dir.exists():
            continue
        dst_split_dir.mkdir(parents=True, exist_ok=True)
        for image_name in image_names:
            src_image = resolve_image_path_in_split(src_split_dir, str(image_name))
            if src_image is None:
                continue
            safe_symlink(src_image, dst_split_dir / str(image_name))

    data_yaml = {
        "path": str(converted_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    yaml_path = work_dir / "retain_data.yaml"
    write_yaml(yaml_path, data_yaml)
    return str(yaml_path)


def write_yolo_segment_labels_from_coco(coco_json_path: Path, labels_out_dir: Path) -> None:
    coco = read_json(str(coco_json_path))
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    anns_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in annotations:
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    labels_out_dir.mkdir(parents=True, exist_ok=True)

    for image in images:
        image_id = int(image["id"])
        width = int(float(image.get("width", 0) or 0))
        height = int(float(image.get("height", 0) or 0))
        if width <= 1 or height <= 1:
            continue

        lines: List[str] = []
        for ann in anns_by_image.get(image_id, []):
            class_id = int(ann["category_id"])
            seg = ann.get("segmentation")

            polygons: List[List[float]] = []
            if isinstance(seg, dict):
                try:
                    polygons = _rle_to_polygons(seg, width=width, height=height)
                except Exception:
                    polygons = []
            elif isinstance(seg, list):
                candidate_polys = seg if seg and isinstance(seg[0], list) else [seg]
                for poly in candidate_polys:
                    if not isinstance(poly, list) or len(poly) < 6 or len(poly) % 2 != 0:
                        continue
                    clipped: List[float] = []
                    valid = True
                    for i, v in enumerate(poly):
                        try:
                            fv = float(v)
                        except Exception:
                            valid = False
                            break
                        if i % 2 == 0:
                            fv = min(max(fv, 0.0), float(width))
                        else:
                            fv = min(max(fv, 0.0), float(height))
                        clipped.append(fv)
                    if valid and len(clipped) >= 6:
                        polygons.append(clipped)

            if not polygons:
                bbox = ann.get("bbox")
                if isinstance(bbox, list) and len(bbox) == 4:
                    x, y, w, h = [float(v) for v in bbox]
                    x1 = min(max(x, 0.0), float(width))
                    y1 = min(max(y, 0.0), float(height))
                    x2 = min(max(x + w, 0.0), float(width))
                    y2 = min(max(y + h, 0.0), float(height))
                    if x2 > x1 and y2 > y1:
                        polygons = [[x1, y1, x2, y1, x2, y2, x1, y2]]

            for poly in polygons:
                if len(poly) < 6 or len(poly) % 2 != 0:
                    continue
                lines.append(_normalized_polygon_line(class_id, poly, float(width), float(height)))

        label_path = labels_out_dir / image_name_to_label_name(str(image.get("file_name", "")))
        label_path.write_text("\n".join(lines), encoding="utf-8")


def prepare_forget_empty_dataset(manifest: Dict[str, Any], work_dir: Path) -> str:
    dataset_root = Path(manifest["dataset_root"])
    forget_dir = work_dir / "forget_yolo"
    split_name_map = {"train": "train", "valid": "val", "test": "test"}
    _, names = build_global_category_schema(manifest)

    for split, yolo_split in split_name_map.items():
        split_info = manifest.get("splits", {}).get(split)
        if not split_info:
            continue
        forget_ann_path = split_info.get("forget_annotations")
        if not forget_ann_path or not os.path.exists(forget_ann_path):
            continue

        forget_coco = read_json(forget_ann_path)
        image_names = [img.get("file_name") for img in forget_coco.get("images", []) if img.get("file_name")]

        image_dst = forget_dir / "images" / yolo_split
        label_dst = forget_dir / "labels" / yolo_split
        image_dst.mkdir(parents=True, exist_ok=True)
        label_dst.mkdir(parents=True, exist_ok=True)

        src_split_dir = dataset_root / split
        for image_name in image_names:
            src_image = resolve_image_path_in_split(src_split_dir, str(image_name))
            if src_image is None:
                continue
            safe_symlink(src_image, image_dst / str(image_name))
            (label_dst / image_name_to_label_name(str(image_name))).write_text("", encoding="utf-8")

    data_yaml = {
        "path": str(forget_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    yaml_path = work_dir / "forget_data.yaml"
    write_yaml(yaml_path, data_yaml)
    return str(yaml_path)


def build_teacher_pseudolabel_dataset(
    teacher_weights: str,
    source_data_yaml: str,
    out_dir: Path,
    conf: float,
    device: str,
) -> str:
    source_cfg = read_yaml(source_data_yaml)
    base_path = Path(source_cfg["path"])
    pseudo_root = out_dir / "retain_pseudo_yolo"

    teacher = YOLO(teacher_weights)
    split_names = ["train", "val", "test"]
    for split_name in split_names:
        rel_images_dir = source_cfg.get(split_name)
        if not rel_images_dir:
            continue
        src_images_dir = base_path / rel_images_dir
        if not src_images_dir.exists():
            continue

        dst_images_dir = pseudo_root / "images" / split_name
        dst_labels_dir = pseudo_root / "labels" / split_name
        dst_images_dir.mkdir(parents=True, exist_ok=True)
        dst_labels_dir.mkdir(parents=True, exist_ok=True)

        for image_file in src_images_dir.glob("*.jpg"):
            safe_symlink(image_file, dst_images_dir / image_file.name)
            preds = teacher.predict(str(image_file), conf=conf, device=device, verbose=False)
            result = preds[0]
            label_path = dst_labels_dir / image_name_to_label_name(image_file.name)
            lines = []
            if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
                xywhn = result.boxes.xywhn.cpu().numpy()
                cls = result.boxes.cls.cpu().numpy().astype(int)
                for box, class_idx in zip(xywhn, cls):
                    x, y, w, h = box.tolist()
                    lines.append(f"{class_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            label_path.write_text("\n".join(lines), encoding="utf-8")

    pseudo_cfg = {
        "path": str(pseudo_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": source_cfg.get("names", []),
    }
    pseudo_yaml = out_dir / "retain_pseudo_data.yaml"
    write_yaml(pseudo_yaml, pseudo_cfg)
    return str(pseudo_yaml)


def build_shard_data_yaml(
    retain_data_yaml: str,
    shard_train_images: List[str],
    out_dir: Path,
    shard_name: str,
) -> str:
    src_cfg = read_yaml(retain_data_yaml)
    base_path = Path(src_cfg["path"])

    shard_root = out_dir / shard_name
    split_names = ["train", "val", "test"]
    for split_name in split_names:
        src_img_rel = src_cfg.get(split_name)
        if not src_img_rel:
            continue
        src_img_dir = base_path / src_img_rel
        src_lbl_dir = base_path / "labels" / split_name
        dst_img_dir = shard_root / "images" / split_name
        dst_lbl_dir = shard_root / "labels" / split_name
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        if split_name == "train":
            target_images = shard_train_images
            for image_name in target_images:
                src_image = src_img_dir / image_name
                src_label = src_lbl_dir / image_name_to_label_name(image_name)
                if src_image.exists():
                    safe_symlink(src_image, dst_img_dir / image_name)
                if src_label.exists():
                    safe_symlink(src_label, dst_lbl_dir / src_label.name)
        else:
            for src_image in src_img_dir.glob("*.jpg"):
                safe_symlink(src_image, dst_img_dir / src_image.name)
                src_label = src_lbl_dir / image_name_to_label_name(src_image.name)
                if src_label.exists():
                    safe_symlink(src_label, dst_lbl_dir / src_label.name)

    out_cfg = {
        "path": str(shard_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": src_cfg.get("names", []),
    }
    out_yaml = shard_root / "data.yaml"
    write_yaml(out_yaml, out_cfg)
    return str(out_yaml)
