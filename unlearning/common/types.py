from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class UnlearningConfig:
    algorithm: str
    project_root: str
    original_weights: str
    output_dir: str
    dataset_root: str
    device: str = "auto"
    forget_class: str = "trichome"
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-4
    seed: int = 42
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnlearningResult:
    algorithm: str
    success: bool
    output_weights: str
    runtime_seconds: float
    notes: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
