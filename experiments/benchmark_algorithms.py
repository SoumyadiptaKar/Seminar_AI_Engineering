import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import yaml

from run_unlearning import load_config, run_once


def _write_temp_config(base_cfg: Dict[str, Any], algorithm: str) -> str:
    cfg = dict(base_cfg)
    cfg["run"] = dict(base_cfg.get("run", {}))
    cfg["run"]["algorithm"] = algorithm

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    with tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
    return tmp.name


def _write_temp_config_with_device(base_cfg: Dict[str, Any], algorithm: str, device: str) -> str:
    cfg = dict(base_cfg)
    cfg["run"] = dict(base_cfg.get("run", {}))
    cfg["run"]["algorithm"] = algorithm
    cfg["run"]["device"] = device

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    with tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
    return tmp.name


def benchmark(config_path: str, algorithms: List[str], out_path: str) -> Dict[str, Any]:
    base_cfg = load_config(config_path)
    results = []

    for algorithm in algorithms:
        temp_cfg_path = _write_temp_config(base_cfg, algorithm)
        try:
            summary = run_once(temp_cfg_path)
            results.append(
                {
                    "algorithm": algorithm,
                    "success": bool(summary.get("success", False)),
                    "runtime_seconds": summary.get("runtime_seconds"),
                    "tracking": summary.get("tracking", {}),
                    "output_weights": summary.get("output_weights"),
                    "summary_path": summary.get("summary_path"),
                    "error": None,
                }
            )
        except Exception as exc:
            err = str(exc)
            if "MPS Tensor to float64" in err:
                try:
                    cpu_cfg_path = _write_temp_config_with_device(base_cfg, algorithm, "cpu")
                    summary = run_once(cpu_cfg_path)
                    results.append(
                        {
                            "algorithm": algorithm,
                            "success": bool(summary.get("success", False)),
                            "runtime_seconds": summary.get("runtime_seconds"),
                            "tracking": summary.get("tracking", {}),
                            "output_weights": summary.get("output_weights"),
                            "summary_path": summary.get("summary_path"),
                            "error": None,
                            "fallback_device": "cpu",
                            "fallback_reason": "MPS float64 incompatibility",
                        }
                    )
                    continue
                except Exception as retry_exc:
                    err = f"{err}; cpu_fallback_failed: {retry_exc}"

            results.append(
                {
                    "algorithm": algorithm,
                    "success": False,
                    "runtime_seconds": None,
                    "tracking": {},
                    "output_weights": None,
                    "summary_path": None,
                    "error": err,
                }
            )

    payload = {
        "config_path": str(Path(config_path).resolve()),
        "algorithms": results,
        "all_success": all(x["success"] for x in results),
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["out_path"] = str(out.resolve())
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark multiple unlearning algorithms")
    parser.add_argument("--config", default="experiments/config.yaml")
    parser.add_argument("--algorithms", nargs="+", required=True)
    parser.add_argument("--out", default="outputs/comparison/algorithms_benchmark.json")
    args = parser.parse_args()

    payload = benchmark(args.config, args.algorithms, args.out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
