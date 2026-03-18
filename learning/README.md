# Learning Pipeline Scaffold

This folder contains scaffolding for **retain-only retraining from scratch** and execution tracking.

## Goal
Train a reference model from scratch on all classes **except** the forget class (`trichome`) to compare:
- retrain-from-scratch baseline vs
- unlearning approach runtime/energy/utility.

## Files
- `config.yaml`: learning pipeline config
- `train_retain_baseline.py`: prepares retain-only dataset and trains YOLO from scratch
- `common/tracking.py`: time + energy tracking helper (CodeCarbon when available)

## Run
```bash
python learning/train_retain_baseline.py --config learning/config.yaml
```

## Outputs
- model weights in `learning/runs/retrain_excluding_trichome/`
- summary JSON with timing and energy estimates

## Notes on Energy
- Uses `codecarbon` if installed to estimate `energy_kwh` and `co2_kg`.
- If unavailable, still records accurate wall-clock duration.
- On macOS, this is usually an estimate (no universal direct hardware wattmeter API without elevated tooling).

## Enabling Real CodeCarbon Mode (macOS)
Set in `learning/config.yaml` (or `experiments/config.yaml`):

```yaml
tracking:
	use_codecarbon: true
	estimated_watts: null
```

Then run the script and provide your macOS password when prompted by `powermetrics`.
If you prefer non-interactive runs, keep `use_codecarbon: false` and the estimator backend will be used.
