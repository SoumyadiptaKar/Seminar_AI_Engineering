import os
import shutil
import time
import json
from pathlib import Path
from typing import Dict, Any, List

from ultralytics import YOLO
from ultralytics.data.converter import convert_coco
import yaml

from ..common.base import BaseUnlearner
from ..common.types import UnlearningResult
from ..common.utils import ensure_dir


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def _image_name_to_label_name(image_name: str) -> str:
    return Path(image_name).with_suffix(".txt").name


def _category_names(coco: Dict[str, Any]) -> List[str]:
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


class GradientAscentUnlearner(BaseUnlearner):
    def _load_manifest(self) -> Dict[str, Any]:
        manifest_path = self.config.extra.get("split_manifest")
        if not manifest_path:
            raise ValueError("Missing split manifest path in config.extra['split_manifest']")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
        return _read_json(manifest_path)

    def _prepare_retain_dataset(self, manifest: Dict[str, Any], work_dir: Path) -> str:
        ann_dir = work_dir / "retain_annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)

        split_name_map = {"train": "train", "valid": "val", "test": "test"}
        for split, yolo_split in split_name_map.items():
            split_info = manifest.get("splits", {}).get(split)
            if not split_info:
                continue
            retain_ann = split_info.get("retain_annotations")
            if not retain_ann or not os.path.exists(retain_ann):
                continue
            shutil.copy2(retain_ann, ann_dir / f"{yolo_split}.json")

        converted_dir = work_dir / "retain_yolo"
        convert_coco(
            labels_dir=str(ann_dir),
            save_dir=str(converted_dir),
            use_segments=False,
            cls91to80=False,
        )

        dataset_root = Path(manifest["dataset_root"])
        for split, yolo_split in split_name_map.items():
            src_split_dir = dataset_root / split
            dst_split_dir = converted_dir / "images" / yolo_split
            if not src_split_dir.exists():
                continue
            dst_split_dir.mkdir(parents=True, exist_ok=True)
            for image_file in src_split_dir.glob("*.jpg"):
                _safe_symlink(image_file, dst_split_dir / image_file.name)

        first_split = next(iter(manifest.get("splits", {}).keys()), None)
        if not first_split:
            raise ValueError("No splits found in manifest")
        first_retain_ann = manifest["splits"][first_split]["retain_annotations"]
        names = _category_names(_read_json(first_retain_ann))

        data_yaml = {
            "path": str(converted_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": names,
        }
        yaml_path = work_dir / "retain_data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data_yaml, f, sort_keys=False)
        return str(yaml_path)

    def _prepare_forget_dataset(self, manifest: Dict[str, Any], work_dir: Path) -> str:
        dataset_root = Path(manifest["dataset_root"])
        forget_dir = work_dir / "forget_yolo"
        split_name_map = {"train": "train", "valid": "val", "test": "test"}

        for split, yolo_split in split_name_map.items():
            split_info = manifest.get("splits", {}).get(split)
            if not split_info:
                continue
            forget_ann_path = split_info.get("forget_annotations")
            if not forget_ann_path or not os.path.exists(forget_ann_path):
                continue

            forget_coco = _read_json(forget_ann_path)
            image_names = [img.get("file_name") for img in forget_coco.get("images", []) if img.get("file_name")]

            image_dst = forget_dir / "images" / yolo_split
            label_dst = forget_dir / "labels" / yolo_split
            image_dst.mkdir(parents=True, exist_ok=True)
            label_dst.mkdir(parents=True, exist_ok=True)

            src_split_dir = dataset_root / split
            for image_name in image_names:
                src_image = src_split_dir / image_name
                if src_image.exists():
                    _safe_symlink(src_image, image_dst / image_name)
                    (label_dst / _image_name_to_label_name(image_name)).write_text("", encoding="utf-8")

        first_split = next(iter(manifest.get("splits", {}).keys()), None)
        if not first_split:
            raise ValueError("No splits found in manifest")
        first_forget_ann = manifest["splits"][first_split]["forget_annotations"]
        names = _category_names(_read_json(first_forget_ann))

        data_yaml = {
            "path": str(forget_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": names,
        }
        yaml_path = work_dir / "forget_data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data_yaml, f, sort_keys=False)
        return str(yaml_path)


    def run(self) -> UnlearningResult:
        started = time.time()
        out_dir = ensure_dir(os.path.join(self.config.output_dir, "gradient_ascent"))
        output_weights = os.path.join(out_dir, "unlearned.pt")

        ga_dry_run = bool(self.config.extra.get("ga_dry_run", False))
        if ga_dry_run:
            shutil.copy2(self.config.original_weights, output_weights)
            notes = (
                "Dry-run enabled: copied original weights without training. "
                f"Selected device: {self.config.device}."
            )
            return UnlearningResult(
                algorithm="gradient_ascent",
                success=True,
                output_weights=output_weights,
                runtime_seconds=time.time() - started,
                notes=notes,
            )

        manifest = self._load_manifest()
        work_dir = Path(out_dir) / "prepared_data"
        work_dir.mkdir(parents=True, exist_ok=True)

        retain_data_yaml = self._prepare_retain_dataset(manifest, work_dir)
        forget_data_yaml = self._prepare_forget_dataset(manifest, work_dir)

        forget_epochs = int(self.config.extra.get("ga_forget_epochs", 1))
        retain_epochs = int(self.config.extra.get("ga_retain_epochs", self.config.epochs))
        imgsz = int(self.config.extra.get("imgsz", 640))
        workers = int(self.config.extra.get("workers", 4))
        save_period = int(self.config.extra.get("save_period", -1))

        model = YOLO(self.config.original_weights)

        model.train(
            data=forget_data_yaml,
            epochs=max(1, forget_epochs),
            batch=self.config.batch_size,
            imgsz=imgsz,
            device=self.config.device,
            lr0=self.config.learning_rate,
            workers=workers,
            save_period=save_period,
            project=out_dir,
            name="forget_stage",
            exist_ok=True,
            seed=self.config.seed,
            verbose=False,
        )

        model.train(
            data=retain_data_yaml,
            epochs=max(1, retain_epochs),
            batch=self.config.batch_size,
            imgsz=imgsz,
            device=self.config.device,
            lr0=self.config.learning_rate,
            workers=workers,
            save_period=save_period,
            project=out_dir,
            name="retain_stage",
            exist_ok=True,
            seed=self.config.seed,
            verbose=False,
        )

        model.save(output_weights)

        notes = (
            "Executed two-stage GA-style unlearning: forget-stage suppression on forget set "
            "(empty labels) followed by retain-stage recovery fine-tuning on retain set. "
            f"Selected device: {self.config.device}."
        )
        return UnlearningResult(
            algorithm="gradient_ascent",
            success=True,
            output_weights=output_weights,
            runtime_seconds=time.time() - started,
            notes=notes,
        )
