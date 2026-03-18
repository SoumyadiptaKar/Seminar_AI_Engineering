import os
import shutil
import time

from ..common.base import BaseUnlearner
from ..common.types import UnlearningResult
from ..common.utils import ensure_dir


class GradientDifferenceUnlearner(BaseUnlearner):
    def run(self) -> UnlearningResult:
        started = time.time()
        out_dir = ensure_dir(os.path.join(self.config.output_dir, "gradient_difference"))
        output_weights = os.path.join(out_dir, "unlearned.pt")

        shutil.copy2(self.config.original_weights, output_weights)

        notes = (
            "Scaffold run only: copied original weights. "
            "Replace with forget-gradient ascent + retain-gradient descent updates. "
            f"Selected device: {self.config.device}."
        )
        return UnlearningResult(
            algorithm="gradient_difference",
            success=True,
            output_weights=output_weights,
            runtime_seconds=time.time() - started,
            notes=notes,
        )
