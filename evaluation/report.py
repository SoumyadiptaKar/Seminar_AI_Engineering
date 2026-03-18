import os
from typing import Dict, Any

from unlearning.common.utils import write_json


def write_experiment_summary(output_dir: str, summary: Dict[str, Any]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "experiment_summary.json")
    write_json(out_path, summary)
    return out_path
