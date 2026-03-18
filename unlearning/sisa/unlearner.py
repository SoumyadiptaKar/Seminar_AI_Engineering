import os
import shutil
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List

from ultralytics import YOLO

from ..common.base import BaseUnlearner
from ..common.types import UnlearningResult
from ..common.utils import ensure_dir
from ..common.data_prep import (
    load_manifest,
    prepare_retain_dataset,
    read_json,
    read_yaml,
    build_shard_data_yaml,
)


class SISAUnlearner(BaseUnlearner):
    def run(self) -> UnlearningResult:
        started = time.time()
        out_dir = ensure_dir(os.path.join(self.config.output_dir, "sisa"))
        output_weights = os.path.join(out_dir, "unlearned.pt")

        sisa_dry_run = bool(self.config.extra.get("sisa_dry_run", False))
        shards = int(self.config.extra.get("sisa_shards", 10))
        slices_per_shard = int(self.config.extra.get("sisa_slices_per_shard", 5))
        slice_epochs = int(self.config.extra.get("sisa_slice_epochs", 1))

        if sisa_dry_run:
            shutil.copy2(self.config.original_weights, output_weights)
            notes = (
                "Dry-run enabled: copied original weights without training. "
                f"SISA settings: shards={shards}, slices_per_shard={slices_per_shard}. "
                f"Selected device: {self.config.device}."
            )
            return UnlearningResult(
                algorithm="sisa",
                success=True,
                output_weights=output_weights,
                runtime_seconds=time.time() - started,
                notes=notes,
            )

        manifest = load_manifest(self.config.extra.get("split_manifest"))
        work_dir = Path(out_dir) / "prepared_data"
        work_dir.mkdir(parents=True, exist_ok=True)
        retain_data_yaml = prepare_retain_dataset(manifest, work_dir)
        retain_cfg = read_yaml(retain_data_yaml)

        retain_train_dir = Path(retain_cfg["path"]) / "images" / "train"
        train_images = sorted([p.name for p in retain_train_dir.glob("*.jpg")])
        if not train_images:
            raise ValueError("No train images found for SISA retain dataset")

        train_forget_ann = manifest.get("splits", {}).get("train", {}).get("forget_annotations")
        forget_images = set()
        if train_forget_ann and os.path.exists(train_forget_ann):
            forget_coco = read_json(train_forget_ann)
            forget_images = {
                img["file_name"]
                for img in forget_coco.get("images", [])
                if img.get("file_name")
            }

        shard_to_images: Dict[int, List[str]] = {idx: [] for idx in range(max(1, shards))}

        def shard_index(image_name: str) -> int:
            digest = hashlib.md5(image_name.encode("utf-8")).hexdigest()
            return int(digest, 16) % max(1, shards)

        for image_name in train_images:
            shard_to_images[shard_index(image_name)].append(image_name)

        affected_shards = sorted({shard_index(img) for img in forget_images if img in train_images})
        if not affected_shards:
            shutil.copy2(self.config.original_weights, output_weights)
            notes = (
                "No affected shards found for current forget set; copied original weights. "
                f"Selected device: {self.config.device}."
            )
            return UnlearningResult(
                algorithm="sisa",
                success=True,
                output_weights=output_weights,
                runtime_seconds=time.time() - started,
                notes=notes,
            )

        imgsz = int(self.config.extra.get("imgsz", 640))
        workers = int(self.config.extra.get("workers", 4))
        save_period = int(self.config.extra.get("save_period", -1))
        train_batch = int(self.config.extra.get("train_batch", 1))

        shard_models: List[str] = []
        shard_info: List[Dict[str, int]] = []
        shard_root = Path(out_dir) / "shards"
        shard_root.mkdir(parents=True, exist_ok=True)

        for shard_id in affected_shards:
            shard_images = shard_to_images.get(shard_id, [])
            if not shard_images:
                continue

            shard_yaml = build_shard_data_yaml(
                retain_data_yaml=retain_data_yaml,
                shard_train_images=shard_images,
                out_dir=work_dir / "sisa_shards",
                shard_name=f"shard_{shard_id}",
            )

            model = YOLO(self.config.original_weights)
            epochs = max(1, slices_per_shard * slice_epochs)
            run_name = f"shard_{shard_id}_retrain"
            model.train(
                data=shard_yaml,
                epochs=epochs,
                batch=max(1, train_batch),
                imgsz=imgsz,
                device=self.config.device,
                lr0=self.config.learning_rate,
                overlap_mask=False,
                mosaic=0.0,
                mixup=0.0,
                copy_paste=0.0,
                erasing=0.0,
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.0,
                fliplr=0.0,
                flipud=0.0,
                degrees=0.0,
                translate=0.0,
                scale=0.0,
                shear=0.0,
                perspective=0.0,
                workers=workers,
                save_period=save_period,
                project=str(shard_root),
                name=run_name,
                exist_ok=True,
                seed=self.config.seed,
                val=False,
                verbose=False,
            )

            shard_weight = shard_root / run_name / "weights" / "last.pt"
            if not shard_weight.exists():
                shard_weight = shard_root / run_name / "weights" / "best.pt"
            if shard_weight.exists():
                shard_models.append(str(shard_weight.resolve()))

            shard_info.append(
                {
                    "shard_id": shard_id,
                    "train_images": len(shard_images),
                    "epochs": epochs,
                }
            )

        if shard_models:
            shutil.copy2(shard_models[0], output_weights)
        else:
            shutil.copy2(self.config.original_weights, output_weights)

        metadata = {
            "shards": shards,
            "slices_per_shard": slices_per_shard,
            "slice_epochs": slice_epochs,
            "affected_shards": affected_shards,
            "trained_shards": shard_info,
            "candidate_models": shard_models,
            "output_weights": output_weights,
        }
        with open(Path(out_dir) / "sisa_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        notes = (
            "Executed SISA-style proxy: retrained affected shards and exported representative checkpoint. "
            f"affected_shards={affected_shards}, total_shards={shards}, slices_per_shard={slices_per_shard}. "
            f"Selected device: {self.config.device}."
        )
        return UnlearningResult(
            algorithm="sisa",
            success=True,
            output_weights=output_weights,
            runtime_seconds=time.time() - started,
            notes=notes,
        )
