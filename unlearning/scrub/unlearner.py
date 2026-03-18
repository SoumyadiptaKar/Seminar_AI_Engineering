import os
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

from ..common.base import BaseUnlearner
from ..common.types import UnlearningResult
from ..common.utils import ensure_dir
from ..common.data_prep import (
    load_manifest,
    prepare_retain_dataset,
    prepare_forget_empty_dataset,
    build_teacher_pseudolabel_dataset,
)


class ScrubUnlearner(BaseUnlearner):
    @staticmethod
    def _stage_weights(base_dir: str, stage_name: str) -> str:
        last_path = os.path.join(base_dir, stage_name, "weights", "last.pt")
        best_path = os.path.join(base_dir, stage_name, "weights", "best.pt")
        if os.path.exists(last_path):
            return last_path
        if os.path.exists(best_path):
            return best_path
        raise FileNotFoundError(f"No weights found for stage '{stage_name}'")

    def run(self) -> UnlearningResult:
        started = time.time()
        out_dir = ensure_dir(os.path.join(self.config.output_dir, "scrub"))
        output_weights = os.path.join(out_dir, "unlearned.pt")

        scrub_dry_run = bool(self.config.extra.get("scrub_dry_run", False))
        if scrub_dry_run:
            shutil.copy2(self.config.original_weights, output_weights)
            notes = (
                "Dry-run enabled: copied original weights without training. "
                f"Selected device: {self.config.device}."
            )
            return UnlearningResult(
                algorithm="scrub",
                success=True,
                output_weights=output_weights,
                runtime_seconds=time.time() - started,
                notes=notes,
            )

        manifest = load_manifest(self.config.extra.get("split_manifest"))
        work_dir = Path(out_dir) / "prepared_data"
        work_dir.mkdir(parents=True, exist_ok=True)

        retain_data_yaml = prepare_retain_dataset(manifest, work_dir)
        forget_data_yaml = prepare_forget_empty_dataset(manifest, work_dir)

        use_pseudo = bool(self.config.extra.get("scrub_use_pseudo", False))
        pseudo_conf = float(self.config.extra.get("scrub_pseudo_conf", 0.35))
        if use_pseudo:
            retain_stage_yaml = build_teacher_pseudolabel_dataset(
                teacher_weights=self.config.original_weights,
                source_data_yaml=retain_data_yaml,
                out_dir=work_dir,
                conf=pseudo_conf,
                device=self.config.device,
            )
        else:
            retain_stage_yaml = retain_data_yaml

        imgsz = int(self.config.extra.get("imgsz", 640))
        workers = int(self.config.extra.get("workers", 4))
        save_period = int(self.config.extra.get("save_period", -1))
        forget_epochs = int(self.config.extra.get("scrub_forget_epochs", 1))
        retain_epochs = int(self.config.extra.get("scrub_retain_epochs", 2))
        retain_lr = float(self.config.extra.get("scrub_retain_lr", self.config.learning_rate * 0.5))

        train_batch = int(self.config.extra.get("train_batch", 1))
        forget_stage = "forget_divergence_stage"
        retain_stage = "retain_distill_stage"
        student = YOLO(self.config.original_weights)

        student.train(
            data=forget_data_yaml,
            epochs=max(1, forget_epochs),
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
            project=out_dir,
            name=forget_stage,
            exist_ok=True,
            seed=self.config.seed,
            val=False,
            verbose=False,
        )

        forget_weights = self._stage_weights(out_dir, forget_stage)
        student = YOLO(forget_weights)

        student.train(
            data=retain_stage_yaml,
            epochs=max(1, retain_epochs),
            batch=max(1, train_batch),
            imgsz=imgsz,
            device=self.config.device,
            lr0=retain_lr,
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
            project=out_dir,
            name=retain_stage,
            exist_ok=True,
            seed=self.config.seed,
            val=False,
            verbose=False,
        )

        retain_weights = self._stage_weights(out_dir, retain_stage)
        shutil.copy2(retain_weights, output_weights)

        notes = (
            "Executed SCRUB-style proxy: forget divergence stage + retain distillation stage "
            f"using {'teacher pseudo labels' if use_pseudo else 'retain ground-truth labels'}. "
            f"pseudo_conf={pseudo_conf}, forget_epochs={forget_epochs}, retain_epochs={retain_epochs}. "
            f"Selected device: {self.config.device}."
        )
        return UnlearningResult(
            algorithm="scrub",
            success=True,
            output_weights=output_weights,
            runtime_seconds=time.time() - started,
            notes=notes,
        )
