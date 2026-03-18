# Learning Diary

Project: Seminar AI Engineering — Machine Unlearning on Vision Models

## 2026-03-18

### Entry 1 — Project setup and baseline
- Created root virtual environment `.venv`.
- Added root dependency file `requirements.txt` for experiment code.
- Added `.gitignore` with Python, model, dataset, run-output, and env ignores.
- Added/updated `README.md` with project overview, setup, evaluation, and unlearning plan.
- Fixed `test.py` model path from `../backend/models/weights.pt` to `backend/models/weights.pt`.
- Installed dependencies and ran baseline evaluation script.
- Downloaded Roboflow dataset (`stomata-batch-1-18`) and generated baseline reports in `metrics_reports/`.

### Entry 2 — Unlearning scaffold (modular)
- Created unlearning package structure:
  - `unlearning/common/`
  - `unlearning/gradient_ascent/`
  - `unlearning/gradient_difference/`
  - `unlearning/scrub/`
  - `unlearning/ssd/`
  - `unlearning/sisa/`
- Added shared interfaces and registry:
  - `unlearning/common/base.py`
  - `unlearning/common/types.py`
  - `unlearning/common/registry.py`
  - `unlearning/common/utils.py`
- Added scaffold unlearner implementations for all algorithms (currently copy weights + metadata note).

### Entry 3 — Experiments and evaluation scaffolding
- Added experiment config and runners:
  - `experiments/config.yaml`
  - `experiments/run_unlearning.py`
  - `experiments/split_dataset.py`
- Added evaluation helpers:
  - `evaluation/metrics.py`
  - `evaluation/report.py`
- Added missing dependency: `pyyaml` in root `requirements.txt`.
- Validated `experiments/run_unlearning.py` end-to-end output JSON generation.

### Entry 4 — Real split generator + Metal GPU support
- Replaced split stub with real COCO class-based split logic in `experiments/split_dataset.py`.
- Implemented split modes:
  - `image` mode (remove full images containing forget class from retain set)
  - `annotation` mode (remove only forget-class annotations, keep images)
- Added Apple Metal auto-device detection utility:
  - `unlearning/common/device.py`
- Wired device selection into pipeline:
  - `experiments/config.yaml` (`run.device`)
  - `experiments/run_unlearning.py` (device resolve + logging)
  - all unlearner stubs now include selected device in notes.
- Ran and validated both split modes:
  - `outputs/splits/split_manifest.json`
  - `outputs/splits_annotation/split_manifest.json`
- Confirmed unlearning runner detects MPS on MacBook: `mps (torch=2.8.0)`.

### Entry 5 — Actual GA implementation + per-algorithm docs
- Implemented a runnable unlearning algorithm in `unlearning/gradient_ascent/unlearner.py`.
  - Added real two-stage workflow:
    1) forget-stage suppression training on forget images with empty labels,
    2) retain-stage recovery training on retain dataset.
  - Added COCO→YOLO dataset preparation inside GA flow using Ultralytics `convert_coco`.
  - Added generation of `retain_data.yaml` and `forget_data.yaml`.
- Updated config plumbing to pass split manifest and GA hyperparameters:
  - `experiments/config.yaml`
  - `experiments/run_unlearning.py`
- Added per-algorithm mini docs with architecture/inputs/outputs/hyperparameters:
  - `unlearning/gradient_ascent/ALGORITHM.md`
  - `unlearning/gradient_difference/ALGORITHM.md`
  - `unlearning/scrub/ALGORITHM.md`
  - `unlearning/ssd/ALGORITHM.md`
  - `unlearning/sisa/ALGORITHM.md`
- Validation performed:
  - Syntax check: `python -m py_compile experiments/run_unlearning.py unlearning/gradient_ascent/unlearner.py experiments/split_dataset.py`
  - Dry-run execution through runner (temporary config with `gradient_ascent.dry_run=true`):
    - output: `ok gradient_ascent True`
    - device resolved: `mps (torch=2.8.0)`
    - output weights path produced: `outputs/gradient_ascent/unlearned.pt`

### Entry 6 — Energy/time tracking + retain-only retrain scaffold
- Added `learning/` scaffolding for retraining baseline from scratch excluding forget class (`trichome`):
  - `learning/config.yaml`
  - `learning/train_retain_baseline.py`
  - `learning/README.md`
  - `learning/common/tracking.py`
- Added non-blocking execution tracking to both unlearning and learning pipelines:
  - `experiments/run_unlearning.py` now includes `tracking` payload in summary JSON.
  - `learning/train_retain_baseline.py` writes `summary.json` with `tracking`.
- Tracking behavior:
  - Default uses deterministic **estimate** backend (no password prompts on macOS).
  - Tracks `duration_seconds`, estimated `energy_kwh`, and estimated `co2_kg`.
  - Optional CodeCarbon backend can be enabled via env var `ENABLE_CODECARBON=1`.
- Fixed split manifest schema bug in `experiments/split_dataset.py`:
  - preserved annotation paths (`forget_annotations`, `retain_annotations`)
  - moved counts to `forget_annotations_count`, `retain_annotations_count`.
- Added dependency `codecarbon` in `requirements.txt`.
- Validation performed:
  - `python experiments/split_dataset.py --dataset-root stomata-batch-1-18 --forget-class trichome --out-dir outputs/splits_annotation --split-mode annotation`
  - `python learning/train_retain_baseline.py --config learning/config.yaml` (dry-run path)
  - `python -m py_compile experiments/split_dataset.py learning/train_retain_baseline.py experiments/run_unlearning.py learning/common/tracking.py`
  - unlearning dry-run tracking check returned backend `estimate` with non-null duration/energy.

### Entry 7 — Password-backed CodeCarbon mode toggle
- Added explicit tracking config controls:
  - `experiments/config.yaml` → `tracking.use_codecarbon`, `tracking.estimated_watts`
  - `learning/config.yaml` → `tracking.use_codecarbon`, `tracking.estimated_watts`
- Wired config toggles into runtime tracking calls:
  - `experiments/run_unlearning.py`
  - `learning/train_retain_baseline.py`
- Updated docs with operational guidance for macOS password prompt mode:
  - `learning/README.md`
  - `README.md`
- Purpose:
  - enable real CodeCarbon mode when user can provide password,
  - keep deterministic estimate mode for non-interactive runs.

---

## Update policy (from now on)
- Every code/config/documentation change will be appended here as a new dated entry.
- Each entry will include:
  - files changed
  - what was changed
  - why it was changed
  - validation command/output (if executed)
