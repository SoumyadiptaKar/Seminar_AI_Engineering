import time
import os
from contextlib import contextmanager
from typing import Dict, Any


@contextmanager
def track_execution(
    project_name: str = "seminar-ai-unlearning",
    device: str = "cpu",
    enable_codecarbon: bool = False,
    estimated_watts: float = None,
):
    start = time.perf_counter()
    tracker = None
    emissions = None
    tracking_backend = "estimate"

    default_watts_by_device = {
        "cpu": 20.0,
        "mps": 35.0,
        "cuda": 150.0,
    }
    assumed_watts = float(estimated_watts) if estimated_watts is not None else default_watts_by_device.get(device, 25.0)

    use_codecarbon = enable_codecarbon or os.getenv("ENABLE_CODECARBON", "0") == "1"

    if use_codecarbon:
        try:
            from codecarbon import EmissionsTracker

            tracker = EmissionsTracker(project_name=project_name, save_to_file=False, log_level="error")
            tracker.start()
            tracking_backend = "codecarbon"
        except Exception:
            tracker = None
            tracking_backend = "estimate"

    payload: Dict[str, Any] = {
        "tracking_backend": tracking_backend,
        "assumed_watts": assumed_watts,
        "duration_seconds": None,
        "energy_kwh": None,
        "co2_kg": None,
        "error": None,
    }

    try:
        yield payload
    except Exception as exc:
        payload["error"] = str(exc)
        raise
    finally:
        end = time.perf_counter()
        duration_seconds = end - start
        payload["duration_seconds"] = duration_seconds

        payload["energy_kwh"] = (assumed_watts * duration_seconds) / (1000.0 * 3600.0)
        payload["co2_kg"] = payload["energy_kwh"] * 0.4

        if tracker is not None:
            try:
                emissions = tracker.stop()
            except Exception as exc:
                payload["error"] = payload["error"] or str(exc)
                emissions = None

        if emissions is not None:
            payload["co2_kg"] = float(emissions)
            energy_kwh = None
            try:
                if hasattr(tracker, "final_emissions_data") and tracker.final_emissions_data is not None:
                    fed = tracker.final_emissions_data
                    if hasattr(fed, "energy_consumed") and fed.energy_consumed is not None:
                        energy_kwh = float(fed.energy_consumed)
            except Exception:
                energy_kwh = None

            if energy_kwh is None:
                try:
                    energy_kwh = float(tracker._total_energy.kWh)
                except Exception:
                    energy_kwh = None

            payload["energy_kwh"] = energy_kwh
