from abc import ABC, abstractmethod
from .types import UnlearningConfig, UnlearningResult


class BaseUnlearner(ABC):
    def __init__(self, config: UnlearningConfig):
        self.config = config

    @abstractmethod
    def run(self) -> UnlearningResult:
        raise NotImplementedError
