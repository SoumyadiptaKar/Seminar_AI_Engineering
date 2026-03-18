# Machine Unlearning on Vision Models — Stomata Segmentation

Seminar in AI Engineering · University of Helsinki · Spring 2026

## Overview

This project investigates **machine unlearning** applied to a YOLOv8 instance-segmentation model trained to detect stomata, trichomes, and veins in plant leaf microscopy images. The goal is to compare how different unlearning algorithms selectively remove knowledge from the model, and to evaluate the trade-off between forgetting quality and retained utility.

The model (`backend/models/weights.pt`) was originally trained on the [Stomata Batch-1 dataset on Roboflow](https://universe.roboflow.com/stomata-project-qurl1/stomata-batch-1-m63eo).

---

## Repository Structure

```
.
├── backend/                    # Inference API (FastAPI + YOLOv8)
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── models/weights.pt       # Pre-trained YOLOv8 weights
│   └── utils/
│       ├── inference.py
│       └── model_utils.py
│
├── unlearning/                 # Unlearning framework
│   ├── common/
│   │   ├── base.py
│   │   ├── registry.py
│   │   └── types.py
│   ├── gradient_ascent/
│   │   └── unlearner.py        # Algorithm 1: GA
│   ├── gradient_difference/
│   │   └── unlearner.py        # Algorithm 2: NegGrad+
│   ├── scrub/
│   │   └── unlearner.py        # Algorithm 3: SCRUB
│   ├── ssd/
│   │   └── unlearner.py        # Algorithm 4: SSD
│   └── sisa/
│       └── unlearner.py        # SISA-style unlearning scaffold
│
├── evaluation/                 # Evaluation scaffold
│   ├── metrics.py              # baseline comparison card
│   └── report.py               # summary writer
│
├── experiments/                # Orchestration + config
│   ├── config.yaml
│   ├── run_unlearning.py
│   └── split_dataset.py
│
├── test.py                     # Baseline evaluation: downloads dataset + computes COCO metrics
├── requirements.txt            # Research/experiment dependencies
├── Writeups/                   # Paper drafts (IEEE format)
└── .gitignore
```

---

## Setup

### 1. Create and activate the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> The `backend/` service has its own `backend/requirements.txt` for the FastAPI API container.

---

## Baseline Evaluation

Run `test.py` to download the dataset from Roboflow and compute COCO metrics (mAP50, mAP50-95, per-class precision/recall) on the original model:

```bash
python test.py
```

Outputs are written to `metrics_reports/`:
- `metrics_summary.json` — full metrics as JSON
- `metrics_report.txt` — human-readable summary
- `metrics_scores.png` — val/test bar chart
- `classwise_map50_valid_vs_test.png` — per-class mAP50
- `classwise_precision_recall_valid_vs_test.png` — per-class P/R

---

## Unlearning Experiment Plan

### Generate Forget/Retain Splits

Use the splitter before running unlearning:

```bash
python experiments/split_dataset.py \
	--dataset-root stomata-batch-1-18 \
	--forget-class trichome \
	--split-mode annotation \
	--out-dir outputs/splits_annotation
```

- `split-mode image`: remove full images that contain forget class from retain set
- `split-mode annotation`: keep images but remove forget-class annotations from retain set

### Device (Apple Metal)

Set `run.device: auto` in `experiments/config.yaml` to prefer `mps` on MacBook Metal GPUs.
You can also force a device with `run.device: mps`, `cuda`, or `cpu`.

### Forget Set Definition

The model is asked to **forget all detections of the `trichome` class** (class forgetting). This gives clear visual validation — the unlearned model should produce zero trichome detections on held-out images.

| Split | Description |
|---|---|
| **Forget Set** | All images containing `trichome` annotations |
| **Retain Set** | All other training images |
| **Gold Standard** | Model retrained from scratch on retain set only |

### Algorithms

| # | Algorithm | Description |
|---|---|---|
| 1 | **Gradient Ascent (GA)** | Maximise loss on forget set (negative fine-tuning) |
| 2 | **Gradient Difference (NegGrad+)** | GA on forget set + GD on retain set simultaneously |
| 3 | **SCRUB** | Teacher-student: match teacher on retain, diverge on forget |
| 4 | **Selective Synaptic Dampening (SSD)** | Fisher-information-guided weight perturbation |

### Evaluation Metrics

| Metric | Purpose |
|---|---|
| mAP50 on Retain Set | Does the model still work? |
| mAP50 on Forget Set | Did it forget? (should be ~0) |
| MIA Score | Membership Inference Attack — can attacker detect forget data? |
| Activation Distance | L2 distance to gold-standard model in feature space |
| Compute Time | Wall-clock time of the unlearning phase |

---

## Running the API (Docker)

```bash
cd backend
docker build -t stomata-api .
docker run -p 8080:8080 stomata-api
```

Then POST microscopy images to `http://localhost:8080/predict`.

---

## References

- Kurmanji et al. (2023). *Towards Unbounded Machine Unlearning.* NeurIPS.
- Liu et al. (2022). *Continual Learning with Gradient Episodic Memory.*
- Foster et al. (2024). *Fast Machine Unlearning Without Retraining.* AAAI.
- Jocher et al. (2023). *Ultralytics YOLOv8.*
