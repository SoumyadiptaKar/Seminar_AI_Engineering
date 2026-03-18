import os
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

from ..common.base import BaseUnlearner
from ..common.types import UnlearningResult
from ..common.utils import ensure_dir
from ..common.data_prep import load_manifest, prepare_retain_dataset, prepare_forget_empty_dataset


class GradientDifferenceUnlearner(BaseUnlearner):
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
        out_dir = ensure_dir(os.path.join(self.config.output_dir, "gradient_difference"))
        output_weights = os.path.join(out_dir, "unlearned.pt")

        gd_dry_run = bool(self.config.extra.get("gd_dry_run", False))
        if gd_dry_run:
            shutil.copy2(self.config.original_weights, output_weights)
            notes = (
                "Dry-run enabled: copied original weights without training. "
                f"Selected device: {self.config.device}."
            )
            return UnlearningResult(
                algorithm="gradient_difference",
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

        imgsz = int(self.config.extra.get("imgsz", 640))
        workers = int(self.config.extra.get("workers", 4))
        save_period = int(self.config.extra.get("save_period", -1))
        cycles = int(self.config.extra.get("gd_cycles", 2))
        forget_epochs_per_cycle = int(self.config.extra.get("gd_forget_epochs_per_cycle", 1))
        retain_epochs_per_cycle = int(self.config.extra.get("gd_retain_epochs_per_cycle", 1))

        train_batch = int(self.config.extra.get("train_batch", 1))
        current_weights = self.config.original_weights
        for cycle in range(max(1, cycles)):
            model = YOLO(current_weights)
            forget_stage_name = f"cycle_{cycle+1}_forget"
            model.train(
                data=forget_data_yaml,
                epochs=max(1, forget_epochs_per_cycle),
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
                name=forget_stage_name,
                exist_ok=True,
                seed=self.config.seed,
                val=False,
                verbose=False,
            )
            current_weights = self._stage_weights(out_dir, forget_stage_name)

            model = YOLO(current_weights)
            retain_stage_name = f"cycle_{cycle+1}_retain"
            model.train(
                data=retain_data_yaml,
                epochs=max(1, retain_epochs_per_cycle),
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
                name=retain_stage_name,
                exist_ok=True,
                seed=self.config.seed,
                val=False,
                verbose=False,
            )
            current_weights = self._stage_weights(out_dir, retain_stage_name)

        shutil.copy2(current_weights, output_weights)

        notes = (
            "Executed alternating forget/retain optimization cycles as a practical "
            "gradient-difference proxy. "
            f"cycles={cycles}, forget_epochs_per_cycle={forget_epochs_per_cycle}, "
            f"retain_epochs_per_cycle={retain_epochs_per_cycle}. "
            f"Selected device: {self.config.device}."
        )
        return UnlearningResult(
            algorithm="gradient_difference",
            success=True,
            output_weights=output_weights,
            runtime_seconds=time.time() - started,
            notes=notes,
        )
