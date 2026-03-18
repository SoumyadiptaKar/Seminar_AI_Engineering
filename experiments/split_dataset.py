import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple


def _load_coco(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_output_coco(
    source: Dict[str, Any],
    selected_image_ids: set,
    selected_annotations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    image_id_set = set(selected_image_ids)
    images = [img for img in source.get("images", []) if img.get("id") in image_id_set]
    return {
        "info": source.get("info", {}),
        "licenses": source.get("licenses", []),
        "categories": source.get("categories", []),
        "images": images,
        "annotations": selected_annotations,
    }


def _resolve_forget_category_ids(coco: Dict[str, Any], forget_class: str) -> List[int]:
    target = forget_class.strip().lower()
    ids = [
        int(cat["id"])
        for cat in coco.get("categories", [])
        if str(cat.get("name", "")).strip().lower() == target
    ]
    return sorted(set(ids))


def _split_single_coco(
    coco: Dict[str, Any],
    forget_cat_ids: List[int],
    split_mode: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, int]]:
    annotations = coco.get("annotations", [])

    image_to_annotations: Dict[int, List[Dict[str, Any]]] = {}
    for ann in annotations:
        image_to_annotations.setdefault(int(ann["image_id"]), []).append(ann)

    forget_image_ids = set()
    for image_id, anns in image_to_annotations.items():
        if any(int(a["category_id"]) in forget_cat_ids for a in anns):
            forget_image_ids.add(image_id)

    all_image_ids = {int(img["id"]) for img in coco.get("images", [])}

    if split_mode == "image":
        retain_image_ids = all_image_ids - forget_image_ids
        forget_annotations = [ann for ann in annotations if int(ann["image_id"]) in forget_image_ids]
        retain_annotations = [ann for ann in annotations if int(ann["image_id"]) in retain_image_ids]
    else:
        retain_image_ids = all_image_ids
        forget_annotations = [ann for ann in annotations if int(ann["category_id"]) in forget_cat_ids]
        retain_annotations = [ann for ann in annotations if int(ann["category_id"]) not in forget_cat_ids]

    forget_coco = _build_output_coco(coco, forget_image_ids, forget_annotations)
    retain_coco = _build_output_coco(coco, retain_image_ids, retain_annotations)

    stats = {
        "total_images": len(all_image_ids),
        "total_annotations": len(annotations),
        "forget_images": len(forget_coco.get("images", [])),
        "forget_annotations_count": len(forget_annotations),
        "retain_images": len(retain_coco.get("images", [])),
        "retain_annotations_count": len(retain_annotations),
    }
    return forget_coco, retain_coco, stats


def build_class_split(
    dataset_root: str,
    forget_class: str,
    out_dir: str,
    split_mode: str,
) -> Dict[str, Any]:
    dataset_path = Path(dataset_root)
    out_path = Path(out_dir)
    os.makedirs(out_path, exist_ok=True)

    split_names = ["train", "valid", "test"]
    split_results: Dict[str, Any] = {}
    found_splits = []

    for split_name in split_names:
        ann_path = dataset_path / split_name / "_annotations.coco.json"
        if not ann_path.exists():
            continue

        found_splits.append(split_name)
        coco = _load_coco(ann_path)
        forget_cat_ids = _resolve_forget_category_ids(coco, forget_class)
        if not forget_cat_ids:
            category_names = [c.get("name", "") for c in coco.get("categories", [])]
            raise ValueError(
                f"Class '{forget_class}' not found in split '{split_name}'. "
                f"Available names: {sorted(set(category_names))}"
            )

        forget_coco, retain_coco, stats = _split_single_coco(coco, forget_cat_ids, split_mode=split_mode)

        split_out_dir = out_path / split_name
        os.makedirs(split_out_dir, exist_ok=True)
        forget_ann_path = split_out_dir / "forget_annotations.coco.json"
        retain_ann_path = split_out_dir / "retain_annotations.coco.json"

        with open(forget_ann_path, "w", encoding="utf-8") as f:
            json.dump(forget_coco, f, indent=2)
        with open(retain_ann_path, "w", encoding="utf-8") as f:
            json.dump(retain_coco, f, indent=2)

        split_results[split_name] = {
            "source_annotations": str(ann_path.resolve()),
            "forget_annotations": str(forget_ann_path.resolve()),
            "retain_annotations": str(retain_ann_path.resolve()),
            "forget_category_ids": forget_cat_ids,
            **stats,
        }

    manifest = {
        "dataset_root": str(dataset_path.resolve()),
        "forget_class": forget_class,
        "split_mode": split_mode,
        "splits_found": found_splits,
        "splits": split_results,
    }

    manifest_path = out_path / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate class-based forget/retain COCO splits")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--forget-class", default="trichome")
    parser.add_argument("--out-dir", default="outputs/splits")
    parser.add_argument(
        "--split-mode",
        choices=["image", "annotation"],
        default="image",
        help="image: remove full images containing forget class from retain; annotation: keep image but remove only forget-class annotations",
    )
    args = parser.parse_args()

    payload = build_class_split(
        dataset_root=args.dataset_root,
        forget_class=args.forget_class,
        out_dir=args.out_dir,
        split_mode=args.split_mode,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
