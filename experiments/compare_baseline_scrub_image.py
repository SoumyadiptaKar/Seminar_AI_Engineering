import argparse
import json
from pathlib import Path
from collections import Counter

import cv2
from ultralytics import YOLO


def summarize_result(result) -> dict:
    names = result.names
    classes = result.boxes.cls.tolist() if result.boxes is not None else []
    confidences = result.boxes.conf.tolist() if result.boxes is not None else []

    class_ids = [int(class_id) for class_id in classes]
    counts = Counter(class_ids)
    class_counts = {names[class_id]: int(count) for class_id, count in sorted(counts.items())}

    trichome_count = 0
    for class_id, count in counts.items():
        if names[class_id].lower() == "trichome":
            trichome_count = int(count)
            break

    return {
        "total_instances": int(len(class_ids)),
        "trichome_instances": trichome_count,
        "class_counts": class_counts,
        "mean_confidence": float(sum(confidences) / len(confidences)) if confidences else None,
        "max_confidence": float(max(confidences)) if confidences else None,
        "min_confidence": float(min(confidences)) if confidences else None,
    }


def add_title_strip(image_bgr, title: str) -> any:
    strip_height = 48
    strip = 255 * (image_bgr[:strip_height, :, :].copy() * 0 + 1)
    cv2.putText(
        strip,
        title,
        (14, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return cv2.vconcat([strip, image_bgr])


def make_side_by_side(left_bgr, right_bgr):
    target_height = max(left_bgr.shape[0], right_bgr.shape[0])

    def resize_to_height(img, height):
        h, w = img.shape[:2]
        if h == height:
            return img
        scale = height / h
        return cv2.resize(img, (int(w * scale), height), interpolation=cv2.INTER_LINEAR)

    left_resized = resize_to_height(left_bgr, target_height)
    right_resized = resize_to_height(right_bgr, target_height)

    gap = 14
    spacer = 255 * (left_resized[:, :gap, :].copy() * 0 + 1)
    return cv2.hconcat([left_resized, spacer, right_resized])


def run_comparison(
    image_path: Path,
    baseline_weights: Path,
    scrub_weights: Path,
    output_dir: Path,
    conf: float,
    iou: float,
    imgsz: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_model = YOLO(str(baseline_weights))
    scrub_model = YOLO(str(scrub_weights))

    baseline_result = baseline_model.predict(
        source=str(image_path), conf=conf, iou=iou, imgsz=imgsz, verbose=False
    )[0]
    scrub_result = scrub_model.predict(
        source=str(image_path), conf=conf, iou=iou, imgsz=imgsz, verbose=False
    )[0]

    baseline_plot_bgr = baseline_result.plot()
    scrub_plot_bgr = scrub_result.plot()

    baseline_plot_bgr = add_title_strip(baseline_plot_bgr, "Baseline model")
    scrub_plot_bgr = add_title_strip(scrub_plot_bgr, "SCRUB unlearned model")
    combined = make_side_by_side(baseline_plot_bgr, scrub_plot_bgr)

    image_stem = image_path.stem
    comparison_image_path = output_dir / f"{image_stem}_baseline_vs_scrub.png"
    cv2.imwrite(str(comparison_image_path), combined)

    baseline_summary = summarize_result(baseline_result)
    scrub_summary = summarize_result(scrub_result)

    summary = {
        "input_image": str(image_path),
        "baseline_weights": str(baseline_weights),
        "scrub_weights": str(scrub_weights),
        "settings": {"conf": conf, "iou": iou, "imgsz": imgsz},
        "baseline": baseline_summary,
        "scrub": scrub_summary,
        "delta": {
            "total_instances": scrub_summary["total_instances"] - baseline_summary["total_instances"],
            "trichome_instances": scrub_summary["trichome_instances"] - baseline_summary["trichome_instances"],
        },
        "output_image": str(comparison_image_path),
    }

    summary_path = output_dir / f"{image_stem}_baseline_vs_scrub_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Saved image:", comparison_image_path)
    print("Saved summary:", summary_path)
    print("Baseline trichome instances:", baseline_summary["trichome_instances"])
    print("SCRUB trichome instances:", scrub_summary["trichome_instances"])


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    return argparse.ArgumentParser(
        description="Create side-by-side baseline vs SCRUB prediction comparison for one image."
    ), project_root


def main():
    parser, project_root = parse_args()

    parser.add_argument("--image", required=True, type=Path, help="Path to input image")
    parser.add_argument(
        "--baseline-weights",
        type=Path,
        default=project_root / "backend" / "models" / "weights.pt",
        help="Path to baseline weights",
    )
    parser.add_argument(
        "--scrub-weights",
        type=Path,
        default=project_root / "outputs" / "scrub" / "unlearned.pt",
        help="Path to SCRUB unlearned weights",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "comparison" / "image_comparison",
        help="Directory for output image and summary json",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.25, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")

    args = parser.parse_args()

    image_path = args.image.resolve()
    baseline_weights = args.baseline_weights.resolve()
    scrub_weights = args.scrub_weights.resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not baseline_weights.exists():
        raise FileNotFoundError(f"Baseline weights not found: {baseline_weights}")
    if not scrub_weights.exists():
        raise FileNotFoundError(f"SCRUB weights not found: {scrub_weights}")

    run_comparison(
        image_path=image_path,
        baseline_weights=baseline_weights,
        scrub_weights=scrub_weights,
        output_dir=args.output_dir,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
