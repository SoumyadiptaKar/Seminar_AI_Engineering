import threading
from ultralytics import YOLO

class ModelManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._model = None

    def load_model(self, weights_path: str):
        with self._lock:
            if self._model is None:
                self._model = YOLO(weights_path)

    def is_loaded(self) -> bool:
        return self._model is not None

    def get_model(self):
        if self._model is None:
            raise RuntimeError("Model not loaded")
        return self._model
