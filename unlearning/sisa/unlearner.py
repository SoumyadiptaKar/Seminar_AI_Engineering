import os
import shutil
import time

from ..common.base import BaseUnlearner
from ..common.types import UnlearningResult
from ..common.utils import ensure_dir


class SISAUnlearner(BaseUnlearner):
    def run(self) -> UnlearningResult:
        started = time.time()
        out_dir = ensure_dir(os.path.join(self.config.output_dir, "sisa"))
        output_weights = os.path.join(out_dir, "unlearned.pt")

        shutil.copy2(self.config.original_weights, output_weights)

        shards = self.config.extra.get("sisa_shards", 10)
        slices_per_shard = self.config.extra.get("sisa_slices_per_shard", 5)
        notes = (
            "Scaffold run only: copied original weights. "
            f"Replace with SISA retraining workflow (shards={shards}, slices_per_shard={slices_per_shard}). "
            f"Selected device: {self.config.device}."
        )
        return UnlearningResult(
            algorithm="sisa",
            success=True,
            output_weights=output_weights,
            runtime_seconds=time.time() - started,
            notes=notes,
        )
