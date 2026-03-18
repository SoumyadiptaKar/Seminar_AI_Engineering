from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _bar_labels(ax, bars, fmt: str = "{:.4f}") -> None:
    labels = [fmt.format(bar.get_height()) for bar in bars]
    ax.bar_label(bars, labels=labels, padding=2, fontsize=8)


def build_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in payload.get("algorithms", []):
        tracking = row.get("tracking", {}) or {}
        rows.append(
            {
                "algorithm": row.get("algorithm", "unknown"),
                "success": bool(row.get("success", False)),
                "runtime_seconds": _safe_float(row.get("runtime_seconds"), 0.0),
                "duration_seconds": _safe_float(tracking.get("duration_seconds"), 0.0),
                "energy_kwh": _safe_float(tracking.get("energy_kwh"), 0.0),
                "co2_kg": _safe_float(tracking.get("co2_kg"), 0.0),
                "tracking_backend": tracking.get("tracking_backend", "unknown"),
                "assumed_watts": tracking.get("assumed_watts"),
            }
        )
    return rows


def append_retraining_row(rows: List[Dict[str, Any]], retraining_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    tracking = retraining_summary.get("tracking", {}) or {}
    rows.append(
        {
            "algorithm": "retraining_baseline",
            "success": not retraining_summary.get("dry_run", True),
            "runtime_seconds": _safe_float(tracking.get("duration_seconds"), 0.0),
            "duration_seconds": _safe_float(tracking.get("duration_seconds"), 0.0),
            "energy_kwh": _safe_float(tracking.get("energy_kwh"), 0.0),
            "co2_kg": _safe_float(tracking.get("co2_kg"), 0.0),
            "tracking_backend": tracking.get("tracking_backend", "unknown"),
            "assumed_watts": tracking.get("assumed_watts"),
        }
    )
    return rows


def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "algorithm",
        "success",
        "runtime_seconds",
        "duration_seconds",
        "energy_kwh",
        "co2_kg",
        "tracking_backend",
        "assumed_watts",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: List[Dict[str, Any]], benchmark_path: Path, out_json: Path) -> None:
    fastest = min(rows, key=lambda r: r["runtime_seconds"]) if rows else None
    lowest_energy = min(rows, key=lambda r: r["energy_kwh"]) if rows else None
    lowest_co2 = min(rows, key=lambda r: r["co2_kg"]) if rows else None

    payload = {
        "benchmark_path": str(benchmark_path.resolve()),
        "algorithms": rows,
        "rankings": {
            "fastest_runtime": fastest,
            "lowest_energy": lowest_energy,
            "lowest_co2": lowest_co2,
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot(rows: List[Dict[str, Any]], out_png: Path) -> None:
    names = [r["algorithm"] for r in rows]
    runtime = [r["runtime_seconds"] for r in rows]
    energy = [r["energy_kwh"] for r in rows]
    co2 = [r["co2_kg"] for r in rows]

    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    bars_rt = axes[0].bar(x, runtime)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right")
    axes[0].set_title("Runtime (seconds)")
    axes[0].grid(axis="y", alpha=0.3)
    _bar_labels(axes[0], bars_rt, fmt="{:.1f}")

    bars_en = axes[1].bar(x, energy)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].set_title("Energy (kWh)")
    axes[1].grid(axis="y", alpha=0.3)
    _bar_labels(axes[1], bars_en, fmt="{:.5f}")

    bars_co2 = axes[2].bar(x, co2)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=20, ha="right")
    axes[2].set_title("CO₂ (kg)")
    axes[2].grid(axis="y", alpha=0.3)
    _bar_labels(axes[2], bars_co2, fmt="{:.5f}")

    fig.suptitle("Energy Efficiency Comparison: Unlearning vs Retraining")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create energy/runtime/CO2 comparison from benchmark output")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSON")
    parser.add_argument(
        "--retraining-summary",
        default="",
        help="Optional path to retraining summary JSON to include as baseline",
    )
    parser.add_argument("--out-dir", default="outputs/comparison", help="Output directory")
    parser.add_argument("--prefix", default="energy_comparison", help="Output filename prefix")
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark).resolve()
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark JSON not found: {benchmark_path}")

    payload = _load_json(benchmark_path)
    rows = build_rows(payload)

    if args.retraining_summary:
        retraining_path = Path(args.retraining_summary).resolve()
        if not retraining_path.exists():
            raise FileNotFoundError(f"Retraining summary JSON not found: {retraining_path}")
        retraining_summary = _load_json(retraining_path)
        rows = append_retraining_row(rows, retraining_summary)

    if not rows:
        raise ValueError("No algorithm rows found in benchmark JSON")

    out_dir = Path(args.out_dir).resolve()
    out_json = out_dir / f"{args.prefix}.json"
    out_csv = out_dir / f"{args.prefix}.csv"
    out_png = out_dir / f"{args.prefix}.png"

    write_summary(rows, benchmark_path, out_json)
    write_csv(rows, out_csv)
    plot(rows, out_png)

    print(json.dumps({
        "benchmark": str(benchmark_path),
        "summary_json": str(out_json),
        "summary_csv": str(out_csv),
        "plot_png": str(out_png),
    }, indent=2))


if __name__ == "__main__":
    main()
