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

---

## Update policy (from now on)
- Every code/config/documentation change will be appended here as a new dated entry.
- Each entry will include:
  - files changed
  - what was changed
  - why it was changed
  - validation command/output (if executed)
