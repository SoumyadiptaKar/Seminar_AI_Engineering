from typing import Optional


def resolve_device(preferred: str = "auto") -> str:
    pref = (preferred or "auto").strip().lower()
    if pref in {"cpu", "mps", "cuda"}:
        return pref

    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def torch_device_for_log(device: str, torch_version: Optional[str]) -> str:
    tv = torch_version or "unknown"
    return f"{device} (torch={tv})"
