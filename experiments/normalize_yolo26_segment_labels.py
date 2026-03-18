from __future__ import annotations

import argparse
from pathlib import Path


def _is_valid_segment_row(parts: list[str]) -> bool:
    if len(parts) < 7 or len(parts) % 2 == 0:
        return False
    try:
        int(float(parts[0]))
        coords = [float(v) for v in parts[1:]]
    except Exception:
        return False
    return all(0.0 <= v <= 1.0 for v in coords)


def normalize_dataset(dataset_root: Path) -> tuple[int, int, int]:
    files_touched = 0
    rows_converted = 0
    rows_dropped = 0

    for split in ["train", "valid", "test"]:
        labels_dir = dataset_root / split / "labels"
        if not labels_dir.exists():
            continue

        for label_file in labels_dir.glob("*.txt"):
            lines = label_file.read_text(encoding="utf-8").splitlines()
            out_lines: list[str] = []
            changed = False

            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    out_lines.append(line)
                    continue

                parts = line.split()
                if len(parts) == 5:
                    try:
                        class_id = int(float(parts[0]))
                        cx, cy, w, h = [float(v) for v in parts[1:]]
                    except Exception:
                        out_lines.append(line)
                        continue

                    x1 = max(0.0, min(1.0, cx - w / 2.0))
                    y1 = max(0.0, min(1.0, cy - h / 2.0))
                    x2 = max(0.0, min(1.0, cx + w / 2.0))
                    y2 = max(0.0, min(1.0, cy + h / 2.0))

                    if x2 > x1 and y2 > y1:
                        new_row = f"{class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}"
                        out_lines.append(new_row)
                        rows_converted += 1
                        changed = True
                        continue

                if _is_valid_segment_row(parts):
                    out_lines.append(line)
                else:
                    rows_dropped += 1
                    changed = True

            if changed:
                label_file.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
                files_touched += 1

    return files_touched, rows_converted, rows_dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize YOLO26 segmentation labels")
    parser.add_argument("--dataset", default="stomata-batch-1-18", help="Dataset root path")
    args = parser.parse_args()

    dataset_root = Path(args.dataset).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

    files_touched, rows_converted, rows_dropped = normalize_dataset(dataset_root)
    print(f"dataset={dataset_root}")
    print(f"files_touched={files_touched}")
    print(f"rows_converted_box_to_polygon={rows_converted}")
    print(f"rows_dropped_invalid={rows_dropped}")


if __name__ == "__main__":
    main()
