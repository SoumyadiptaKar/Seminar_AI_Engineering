from typing import Dict, Type

from .base import BaseUnlearner
from ..gradient_ascent.unlearner import GradientAscentUnlearner
from ..gradient_difference.unlearner import GradientDifferenceUnlearner
from ..scrub.unlearner import ScrubUnlearner
from ..ssd.unlearner import SSDUnlearner
from ..sisa.unlearner import SISAUnlearner


UNLEARNER_REGISTRY: Dict[str, Type[BaseUnlearner]] = {
    "gradient_ascent": GradientAscentUnlearner,
    "gradient_difference": GradientDifferenceUnlearner,
    "scrub": ScrubUnlearner,
    "ssd": SSDUnlearner,
    "sisa": SISAUnlearner,
}


def get_unlearner_class(name: str) -> Type[BaseUnlearner]:
    if name not in UNLEARNER_REGISTRY:
        allowed = ", ".join(sorted(UNLEARNER_REGISTRY.keys()))
        raise ValueError(f"Unknown algorithm '{name}'. Allowed: {allowed}")
    return UNLEARNER_REGISTRY[name]
