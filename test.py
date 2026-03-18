import importlib
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def install_and_import(package_name: str, module_name: str = None):
    target_module = module_name or package_name
    try:
        return importlib.import_module(target_module)
    except ImportError:
        print(f"Installing {package_name}...")
        os.system(f'"{sys.executable}" -m pip install {package_name}')
        return importlib.import_module(target_module)


roboflow_module = install_and_import("roboflow")
Roboflow = roboflow_module.Roboflow
ultralytics_module = install_and_import("ultralytics")
YOLO = ultralytics_module.YOLO
np = install_and_import("numpy")
plt = install_and_import("matplotlib", "matplotlib.pyplot")
pycoco_coco = install_and_import("pycocotools", "pycocotools.coco")
pycoco_eval = install_and_import("pycocotools", "pycocotools.cocoeval")
COCO = pycoco_coco.COCO
COCOeval = pycoco_eval.COCOeval


def download_roboflow_dataset() -> str:
    print("=" * 80)
    print("DOWNLOADING ROBOFLOW DATASET (COCO)")
    print("=" * 80)

    rf = Roboflow(api_key="3DDbJxx4ALMlWCXf8n8j")
    project = rf.workspace("stomata-project-qurl1").project("stomata-batch-1-m63eo")
    version = project.version(18)
    dataset = version.download("coco")

    dataset_path = dataset.location if hasattr(dataset, "location") else str(dataset)
    if not os.path.exists(dataset_path):
        fallback = os.path.join(os.getcwd(), "stomata-batch-1-18")
        if os.path.exists(fallback):
            dataset_path = fallback
        else:
            raise FileNotFoundError(f"Could not resolve dataset path: {dataset_path}")

    print(f"✓ Dataset ready at: {dataset_path}")
    return dataset_path


def _slugify_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_") or "model"


def _resolve_existing_image_path(split_dir: str, file_name: str) -> str:
    direct = os.path.join(split_dir, file_name)
    if os.path.exists(direct):
        return direct

    nested = os.path.join(split_dir, "images", file_name)
    if os.path.exists(nested):
        return nested

    return direct


def _label_bars(ax, containers, fmt: str = "{:.3f}"):
    for container in containers:
        labels = [fmt.format(bar.get_height()) for bar in container]
        ax.bar_label(container, labels=labels, padding=2, fontsize=8)


def discover_models(benchmark_json: str = None, baseline_weights: str = None, explicit_weights=None):
    models = []
    seen_paths = set()

    if baseline_weights:
        bpath = os.path.abspath(baseline_weights)
        if os.path.exists(bpath):
            models.append(("baseline", bpath))
            seen_paths.add(bpath)

    if explicit_weights:
        for weight_path in explicit_weights:
            abs_path = os.path.abspath(weight_path)
            if not os.path.exists(abs_path) or abs_path in seen_paths:
                continue
            models.append((Path(abs_path).stem, abs_path))
            seen_paths.add(abs_path)

    if benchmark_json and os.path.exists(benchmark_json):
        with open(benchmark_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for row in payload.get("algorithms", []):
            output_weights = row.get("output_weights")
            algorithm = row.get("algorithm", "model")
            if not output_weights:
                continue
            abs_path = os.path.abspath(output_weights)
            if not os.path.exists(abs_path) or abs_path in seen_paths:
                continue
            models.append((algorithm, abs_path))
            seen_paths.add(abs_path)

    return models


def load_model(weights_path: str):
    print("\n" + "=" * 80)
    print("LOADING YOLO26 MODEL")
    print("=" * 80)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    model = YOLO(weights_path)
    print(f"✓ Model loaded: {weights_path}")
    return model


def display_model_summary(model):
    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    print(f"Model type: {type(model).__name__}")
    print(f"Model source: {getattr(model, 'model_name', 'N/A')}")
    try:
        model.info(verbose=True)
    except Exception:
        pass


def _build_category_mapping(coco_gt, model_names):
    category_ids = coco_gt.getCatIds()
    categories = coco_gt.loadCats(category_ids)
    cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}

    ann_counts = {}
    for ann in coco_gt.dataset.get("annotations", []):
        cid = ann["category_id"]
        ann_counts[cid] = ann_counts.get(cid, 0) + 1

    name_to_cat_ids = {}
    for cat in categories:
        name_to_cat_ids.setdefault(cat["name"], []).append(cat["id"])

    class_index_to_cat_id = {}
    for class_index, class_name in model_names.items():
        candidate_ids = name_to_cat_ids.get(class_name, [])
        if not candidate_ids:
            continue
        best_cat_id = max(candidate_ids, key=lambda cid: ann_counts.get(cid, 0))
        class_index_to_cat_id[int(class_index)] = int(best_cat_id)

    return class_index_to_cat_id, cat_id_to_name


def _compute_precision_recall_at_iou50(coco_eval):
    iou_thresholds = coco_eval.params.iouThrs
    iou_50_idx = int(np.argmin(np.abs(iou_thresholds - 0.5)))

    precision = coco_eval.eval["precision"]
    recall = coco_eval.eval["recall"]

    p = precision[iou_50_idx, :, :, 0, -1]
    p_valid = p[p > -1]
    precision_50 = float(np.mean(p_valid)) if p_valid.size > 0 else 0.0

    r = recall[iou_50_idx, :, 0, -1]
    r_valid = r[r > -1]
    recall_50 = float(np.mean(r_valid)) if r_valid.size > 0 else 0.0

    return precision_50, recall_50


def _compute_per_class_ap(coco_eval, cat_id_to_name):
    precision = coco_eval.eval["precision"]
    iou_thresholds = coco_eval.params.iouThrs
    iou_50_idx = int(np.argmin(np.abs(iou_thresholds - 0.5)))

    per_class_map50_95 = {}
    per_class_map50 = {}

    for k, cat_id in enumerate(coco_eval.params.catIds):
        name = cat_id_to_name.get(cat_id, f"class_{cat_id}")

        ap_all = precision[:, :, k, 0, -1]
        ap_all_valid = ap_all[ap_all > -1]
        per_class_map50_95[name] = float(np.mean(ap_all_valid)) if ap_all_valid.size > 0 else 0.0

        ap_50 = precision[iou_50_idx, :, k, 0, -1]
        ap_50_valid = ap_50[ap_50 > -1]
        per_class_map50[name] = float(np.mean(ap_50_valid)) if ap_50_valid.size > 0 else 0.0

    return per_class_map50, per_class_map50_95


def _compute_per_class_precision_recall(coco_eval, cat_id_to_name):
    precision = coco_eval.eval["precision"]
    recall = coco_eval.eval["recall"]
    iou_thresholds = coco_eval.params.iouThrs
    iou_50_idx = int(np.argmin(np.abs(iou_thresholds - 0.5)))

    per_class_precision50 = {}
    per_class_recall50 = {}

    for k, cat_id in enumerate(coco_eval.params.catIds):
        class_name = cat_id_to_name.get(cat_id, f"class_{cat_id}")

        p = precision[iou_50_idx, :, k, 0, -1]
        p_valid = p[p > -1]
        per_class_precision50[class_name] = float(np.mean(p_valid)) if p_valid.size > 0 else 0.0

        r = recall[iou_50_idx, k, 0, -1]
        per_class_recall50[class_name] = float(r) if r > -1 else 0.0

    return per_class_precision50, per_class_recall50


def evaluate_split(model, dataset_path: str, split: str):
    print("\n" + "=" * 80)
    print(f"EVALUATING {split.upper()} SPLIT (TRUE COCO METRICS)")
    print("=" * 80)

    split_dir = os.path.join(dataset_path, split)
    ann_file = os.path.join(split_dir, "_annotations.coco.json")
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"Missing annotation file: {ann_file}")

    coco_gt = COCO(ann_file)
    image_ids = coco_gt.getImgIds()
    images = coco_gt.loadImgs(image_ids)
    filename_to_id = {img["file_name"]: img["id"] for img in images}

    model_names = getattr(model, "names", {}) or {}
    class_index_to_cat_id, cat_id_to_name = _build_category_mapping(coco_gt, model_names)

    detections = []
    total_images = len(images)
    for idx, image in enumerate(images, start=1):
        image_path = _resolve_existing_image_path(split_dir, image["file_name"])
        if not os.path.exists(image_path):
            continue

        pred = model.predict(image_path, conf=0.001, iou=0.7, verbose=False)
        result = pred[0]

        if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, score, class_index in zip(xyxy, confs, classes):
                if class_index not in class_index_to_cat_id:
                    continue

                x1, y1, x2, y2 = box.tolist()
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue

                detections.append(
                    {
                        "image_id": filename_to_id[image["file_name"]],
                        "category_id": class_index_to_cat_id[class_index],
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score),
                    }
                )

        if idx % max(1, total_images // 10) == 0:
            print(f"  Progress: {idx}/{total_images}")

    if not detections:
        print("WARNING: No detections produced, metrics will be zeros.")
        return {
            "split": split,
            "precision": 0.0,
            "recall": 0.0,
            "map50": 0.0,
            "map50_95": 0.0,
            "per_class_map50": {cat_id_to_name[cid]: 0.0 for cid in sorted(cat_id_to_name.keys())},
            "per_class_map50_95": {cat_id_to_name[cid]: 0.0 for cid in sorted(cat_id_to_name.keys())},
            "per_class_precision50": {cat_id_to_name[cid]: 0.0 for cid in sorted(cat_id_to_name.keys())},
            "per_class_recall50": {cat_id_to_name[cid]: 0.0 for cid in sorted(cat_id_to_name.keys())},
            "num_images": total_images,
            "num_detections": 0,
        }

    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.params.catIds = sorted(set(class_index_to_cat_id.values()))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    precision_50, recall_50 = _compute_precision_recall_at_iou50(coco_eval)
    per_class_map50, per_class_map50_95 = _compute_per_class_ap(coco_eval, cat_id_to_name)
    per_class_precision50, per_class_recall50 = _compute_per_class_precision_recall(coco_eval, cat_id_to_name)

    metrics = {
        "split": split,
        "precision": precision_50,
        "recall": recall_50,
        "map50": float(coco_eval.stats[1]),
        "map50_95": float(coco_eval.stats[0]),
        "per_class_map50": per_class_map50,
        "per_class_map50_95": per_class_map50_95,
        "per_class_precision50": per_class_precision50,
        "per_class_recall50": per_class_recall50,
        "num_images": total_images,
        "num_detections": len(detections),
    }

    print("Overall:")
    print(f"  Precision@0.5: {metrics['precision']:.4f}")
    print(f"  Recall@0.5: {metrics['recall']:.4f}")
    print(f"  mAP50: {metrics['map50']:.4f}")
    print(f"  mAP50-95: {metrics['map50_95']:.4f}")

    print("Per-class mAP50:")
    for class_name, value in metrics["per_class_map50"].items():
        print(f"  {class_name}: {value:.4f}")

    print("Per-class Precision@0.5:")
    for class_name, value in metrics["per_class_precision50"].items():
        print(f"  {class_name}: {value:.4f}")

    print("Per-class Recall@0.5:")
    for class_name, value in metrics["per_class_recall50"].items():
        print(f"  {class_name}: {value:.4f}")

    return metrics


def generate_reports(val_metrics: dict, test_metrics: dict, output_dir: str, model_label: str = "model"):
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "validation": val_metrics,
        "test": test_metrics,
    }

    summary_json = os.path.join(output_dir, "metrics_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_txt = os.path.join(output_dir, "metrics_report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        def write_split(title: str, m: dict):
            f.write(f"{title}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Images: {m['num_images']}\n")
            f.write(f"Detections: {m['num_detections']}\n")
            f.write(f"Precision@0.5: {m['precision']:.6f}\n")
            f.write(f"Recall@0.5: {m['recall']:.6f}\n")
            f.write(f"mAP50: {m['map50']:.6f}\n")
            f.write(f"mAP50-95: {m['map50_95']:.6f}\n")
            f.write("Per-class mAP50:\n")
            for class_name, score in m["per_class_map50"].items():
                f.write(f"  {class_name}: {score:.6f}\n")
            f.write("Per-class mAP50-95:\n")
            for class_name, score in m["per_class_map50_95"].items():
                f.write(f"  {class_name}: {score:.6f}\n")
            f.write("Per-class Precision@0.5:\n")
            for class_name, score in m["per_class_precision50"].items():
                f.write(f"  {class_name}: {score:.6f}\n")
            f.write("Per-class Recall@0.5:\n")
            for class_name, score in m["per_class_recall50"].items():
                f.write(f"  {class_name}: {score:.6f}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write(f"YOLO26 TRUE ACCURACY REPORT (COCO EVALUATION) — {model_label}\n")
        f.write("=" * 80 + "\n\n")
        write_split("VALIDATION SET", val_metrics)
        write_split("TEST SET", test_metrics)

    metrics_plot = os.path.join(output_dir, "metrics_scores.png")
    metric_names = ["Precision@0.5", "Recall@0.5", "mAP50", "mAP50-95"]
    val_values = [val_metrics["precision"], val_metrics["recall"], val_metrics["map50"], val_metrics["map50_95"]]
    test_values = [test_metrics["precision"], test_metrics["recall"], test_metrics["map50"], test_metrics["map50_95"]]

    x = np.arange(len(metric_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_val = ax.bar(x - width / 2, val_values, width=width, label="Validation")
    bars_test = ax.bar(x + width / 2, test_values, width=width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"YOLO26 Accuracy Metrics on Validation/Test — {model_label}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    _label_bars(ax, [bars_val, bars_test])
    fig.tight_layout()
    fig.savefig(metrics_plot, dpi=300)
    plt.close(fig)

    class_names = sorted(set(val_metrics["per_class_map50"].keys()) | set(test_metrics["per_class_map50"].keys()))
    x_cls = np.arange(len(class_names))
    width_cls = 0.35

    # Plot class-wise mAP50 comparison
    val_map50_vals = [val_metrics["per_class_map50"].get(c, 0.0) for c in class_names]
    test_map50_vals = [test_metrics["per_class_map50"].get(c, 0.0) for c in class_names]

    class_map50_plot = os.path.join(output_dir, "classwise_map50_valid_vs_test.png")
    fig, ax = plt.subplots(figsize=(12, 6))
    bars_val = ax.bar(x_cls - width_cls / 2, val_map50_vals, width=width_cls, label="Validation")
    bars_test = ax.bar(x_cls + width_cls / 2, test_map50_vals, width=width_cls, label="Test")
    ax.set_xticks(x_cls)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("mAP50")
    ax.set_title(f"Class-wise mAP50 (Validation vs Test) — {model_label}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    _label_bars(ax, [bars_val, bars_test])
    fig.tight_layout()
    fig.savefig(class_map50_plot, dpi=300)
    plt.close(fig)

    # Plot class-wise Precision@0.5 and Recall@0.5 comparison
    val_prec_vals = [val_metrics["per_class_precision50"].get(c, 0.0) for c in class_names]
    test_prec_vals = [test_metrics["per_class_precision50"].get(c, 0.0) for c in class_names]
    val_rec_vals = [val_metrics["per_class_recall50"].get(c, 0.0) for c in class_names]
    test_rec_vals = [test_metrics["per_class_recall50"].get(c, 0.0) for c in class_names]

    class_pr_plot = os.path.join(output_dir, "classwise_precision_recall_valid_vs_test.png")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    bars_p_val = axes[0].bar(x_cls - width_cls / 2, val_prec_vals, width=width_cls, label="Validation")
    bars_p_test = axes[0].bar(x_cls + width_cls / 2, test_prec_vals, width=width_cls, label="Test")
    axes[0].set_xticks(x_cls)
    axes[0].set_xticklabels(class_names)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Class-wise Precision@0.5")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    bars_r_val = axes[1].bar(x_cls - width_cls / 2, val_rec_vals, width=width_cls, label="Validation")
    bars_r_test = axes[1].bar(x_cls + width_cls / 2, test_rec_vals, width=width_cls, label="Test")
    axes[1].set_xticks(x_cls)
    axes[1].set_xticklabels(class_names)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Class-wise Recall@0.5")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()
    _label_bars(axes[0], [bars_p_val, bars_p_test])
    _label_bars(axes[1], [bars_r_val, bars_r_test])

    fig.suptitle(f"Class-wise Precision/Recall (Validation vs Test) — {model_label}")
    fig.tight_layout()
    fig.savefig(class_pr_plot, dpi=300)
    plt.close(fig)

    print("\n" + "=" * 80)
    print("REPORTS GENERATED")
    print("=" * 80)
    print(f"✓ {summary_json}")
    print(f"✓ {report_txt}")
    print(f"✓ {metrics_plot}")
    print(f"✓ {class_map50_plot}")
    print(f"✓ {class_pr_plot}")


def write_model_comparison(all_metrics: dict, output_root: str):
    os.makedirs(output_root, exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "models": all_metrics,
    }
    out_json = os.path.join(output_root, "model_comparison_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    model_names = list(all_metrics.keys())
    val_map50 = [all_metrics[m]["validation"]["map50"] for m in model_names]
    test_map50 = [all_metrics[m]["test"]["map50"] for m in model_names]
    val_tri = [all_metrics[m]["validation"].get("per_class_map50", {}).get("trichome", 0.0) for m in model_names]
    test_tri = [all_metrics[m]["test"].get("per_class_map50", {}).get("trichome", 0.0) for m in model_names]

    x = np.arange(len(model_names))
    width = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    bars_m_val = axes[0].bar(x - width / 2, val_map50, width=width, label="Val mAP50")
    bars_m_test = axes[0].bar(x + width / 2, test_map50, width=width, label="Test mAP50")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=20, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Overall mAP50 by Model")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    bars_t_val = axes[1].bar(x - width / 2, val_tri, width=width, label="Val trichome mAP50")
    bars_t_test = axes[1].bar(x + width / 2, test_tri, width=width, label="Test trichome mAP50")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=20, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Trichome mAP50 by Model")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()
    _label_bars(axes[0], [bars_m_val, bars_m_test])
    _label_bars(axes[1], [bars_t_val, bars_t_test])

    fig.suptitle("Model Comparison: Baseline vs Unlearned")
    fig.tight_layout()
    out_plot = os.path.join(output_root, "model_comparison_metrics.png")
    fig.savefig(out_plot, dpi=300)
    plt.close(fig)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON GENERATED")
    print("=" * 80)
    print(f"✓ {out_json}")
    print(f"✓ {out_plot}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline and unlearned models on COCO metrics")
    parser.add_argument("--dataset-path", default=None, help="Dataset root path (uses Roboflow download if omitted)")
    parser.add_argument("--weights", nargs="*", default=None, help="Explicit model weight paths to evaluate")
    parser.add_argument("--benchmark-json", default="outputs/comparison/algorithms_benchmark.json", help="Benchmark JSON with unlearned weight paths")
    parser.add_argument("--baseline-weights", default=os.path.join("backend", "models", "weights.pt"))
    parser.add_argument("--output-root", default=os.path.join(os.getcwd(), "metrics_reports"))
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("YOLO26 TRUE EVALUATION PIPELINE")
    print("=" * 80)

    dataset_path = args.dataset_path if args.dataset_path else download_roboflow_dataset()
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    model_specs = discover_models(
        benchmark_json=args.benchmark_json,
        baseline_weights=args.baseline_weights,
        explicit_weights=args.weights,
    )

    if not model_specs:
        raise ValueError("No model weights found to evaluate")

    all_metrics = {}
    for model_name, weights_path in model_specs:
        safe_name = _slugify_name(model_name)
        print("\n" + "#" * 80)
        print(f"EVALUATING MODEL: {model_name}")
        print("#" * 80)

        model = load_model(weights_path)
        display_model_summary(model)

        val_metrics = evaluate_split(model, dataset_path, "valid")
        test_metrics = evaluate_split(model, dataset_path, "test")

        model_out_dir = os.path.join(args.output_root, safe_name)
        generate_reports(val_metrics, test_metrics, model_out_dir, model_label=model_name)

        all_metrics[model_name] = {
            "weights": os.path.abspath(weights_path),
            "validation": val_metrics,
            "test": test_metrics,
            "report_dir": os.path.abspath(model_out_dir),
        }

    write_model_comparison(all_metrics, args.output_root)


if __name__ == "__main__":
    main()
