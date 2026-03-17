import io
import logging
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

COUNTABLE_OVERLAP_THRESHOLD = 0.5 # so a stomata is excluded from the total if its are overlaps >50% with a non countable region
NON_COUNTABLE_CLASSES = {"nothing", "trichome", "vein"}

def mask_to_bbox(binary_mask: np.ndarray):
    """
    takes a binary mask and returns the tightest bounding box around it by finding the min/max row and column indices of all True pixels.
    """
    ys, xs = np.where(binary_mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("mask_to_bbox received an empty mask")
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def add_closest_stomata_distance(stomata_metadata: list[dict], um_per_px: float):
    """
    Adds nearest-neighbor distance for each stomata based on centroid (!!!!!!!!) distance.
    distance is stored in pixels, micrometers, and millimeters.
    """
    if not stomata_metadata:
        logger.debug("No stomata metadata provided for nearest-neighbor distance calculation")
        return stomata_metadata

    centroids = np.array([[s["centroid_x"], s["centroid_y"]] for s in stomata_metadata], dtype=float)

    for i, stomata in enumerate(stomata_metadata):
        diff = centroids - centroids[i]
        distances = np.sqrt(np.sum(diff**2, axis=1))
        distances[i] = np.inf  # ignore distance to itself
        min_dist_px = distances.min()
        if np.isinf(min_dist_px):
            min_dist_px = None

        stomata["closest_stomata_distance_px"] = float(min_dist_px) if min_dist_px is not None else None
        stomata["closest_stomata_distance_um"] = float(min_dist_px * um_per_px) if min_dist_px is not None else None
        stomata["closest_stomata_distance_mm"] = float(min_dist_px * um_per_px / 1000) if min_dist_px is not None else None

    return stomata_metadata


def get_ellipse_length(binary_mask: np.ndarray):
    ys, xs = np.where(binary_mask)
    points = np.column_stack([xs, ys]).astype(np.float32)
    if len(points) < 5: # Just in case, this should never happen with a real stomata
        logger.debug(f"Too few points for ellipse fitting: {len(points)}")
        return None, None
    ellipse = cv2.fitEllipse(points)
    (cx, cy), (minor_axis, major_axis), angle = ellipse
    return float(major_axis), float(minor_axis)


def build_metadata(binary_mask: np.ndarray, instance_id: int, class_id: int, class_name: str, confidence: float, um_per_px: float):
    """
    takes a single instance's binary mask and associated detection info, and returns a dict of everything we want to
    know about that stomata:
    area, bbox, dimensions, centroid, all in both pixel and um units.
    """
    pixel_area = int(binary_mask.sum())
    if pixel_area == 0:
        raise ValueError(f"Empty mask encountered for instance_id={instance_id}")

    x1, y1, x2, y2 = mask_to_bbox(binary_mask)
    ys, xs = np.where(binary_mask)
    ellipse_length_px, ellipse_width_px = get_ellipse_length(binary_mask)

    return {
        "instance_id": instance_id,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": float(confidence),
        "pixel_area": pixel_area,
        "area_um2": float(pixel_area * um_per_px ** 2),
        "stomata_length_um": ellipse_length_px * um_per_px if ellipse_length_px is not None else None, # tbh not sure if they want in mm or um?
        "stomata_length_mm": (ellipse_length_px * um_per_px)/1000 if ellipse_length_px is not None else None,
        "bbox_xyxy": [x1, y1, x2, y2],
        "width_px": float(x2 - x1),
        "height_px": float(y2 - y1),
        "centroid_x": float(xs.mean()),
        "centroid_y": float(ys.mean()),
    }


def density_per_mm2(stomata_count: int, countable_pixels: int, um_per_px: float):
    """
    converts the pixel-based density into mm^2 using the know value for pixels per 10 um, then dividing stomata count by that area.
    """
    if not um_per_px or countable_pixels == 0:
        logger.warning(f"Invalid or empty pixel based density")
        return None
    countable_mm2 = countable_pixels * (um_per_px ** 2) / 1e6
    return stomata_count / countable_mm2


def process_detections(result, um_per_px):
    """
    Takes a raw YOLO result, iterates over all detections, separates non-countable area masks from stomata, 
    builds the union non-countable mask, then calls build_metadata for each stomata instance and assembles the density_info summary.
    """
    stomata_metadata = []

    if result.masks is None:
        logger.info(f"No masks found in the result")
        return stomata_metadata, {}

    # masks are paired 1-to-1 with boxes, confidence and class come from there mask is float [0,1], at model resolution
    masks_float = result.masks.data.cpu().numpy()  
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()
    img_h, img_w = result.orig_shape

    non_countable_mask = np.zeros((img_h, img_w), dtype=bool) # kind of useful later, now kind of redundant
    stomata_candidates= []

    for mask_float, conf, cls in zip(masks_float, confidences, class_ids):
        cls_id   = int(cls)
        cls_name = result.names[cls_id]
        binary   = mask_float > 0.5  # <- binarise: confident mask pixels only,
        #YOLO outputs each mask as a float array where each pixel value represents how confident the model is that pixel belongs to the instance 

        if binary.shape != (img_h, img_w):
            # NEAREST avoids interpolation introducing values between 0 and 1
            binary = np.array(Image.fromarray(binary).resize((img_w, img_h), Image.NEAREST))

        if cls_name in NON_COUNTABLE_CLASSES:
            non_countable_mask |= binary  # <- union: any pixel flagged by any non-countable instance
            
        elif cls_name == "stomata":
            stomata_candidates.append((binary, cls_id, cls_name, float(conf)))

    countable_mask = ~non_countable_mask # everything not in non countables
    countable_pixels = int(countable_mask.sum()) 

    kept_index = 0
    for binary, cls_id, cls_name, conf in stomata_candidates:
        stomata_pixels = int(binary.sum())
        if stomata_pixels == 0:
            continue

        stomata_in_countable = int((binary & countable_mask).sum())
        overlap_fraction = stomata_in_countable / stomata_pixels

        if overlap_fraction >= COUNTABLE_OVERLAP_THRESHOLD:
            meta = build_metadata(binary, kept_index, cls_id, cls_name, conf, um_per_px)
            meta["fraction_in_countable_area"] = float(overlap_fraction)
            stomata_metadata.append(meta)
            kept_index += 1

    stomata_metadata = add_closest_stomata_distance(stomata_metadata, um_per_px)
    stomata_count = len(stomata_metadata)
    total_pixels = int(img_h * img_w)

    density_info = {
        "total_pixels": total_pixels,
        "non_countable_pixels": int(non_countable_mask.sum()),
        "countable_pixels": countable_pixels,
        "stomata_count": stomata_count,
        "stomatal_density_per_px2": float(stomata_count / countable_pixels) if countable_pixels else None,
        "um_per_px": um_per_px,
        "stomatal_density_per_mm2": density_per_mm2(stomata_count, countable_pixels, um_per_px),
    }

    return stomata_metadata, density_info


def run_inference(model, image_bytes: bytes, conf_threshold: float, um_per_px: float):
    """
    handles a single image end-to-end: decodes the bytes, runs the YOLO model, passes the result to process_detections, and returns the structured response dict.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = model(image, conf=conf_threshold, imgsz=640, retina_masks=True, rect=False, iou = 0.25)[0]
        stomata_metadata, density_info = process_detections(result, um_per_px=um_per_px)

        return {"success": True, "density_info": density_info, "stomata": stomata_metadata}

    except Exception as e:
        logger.exception(f"Inference error: {e}")
        raise


def batch_inference(model, images_data: list[tuple[str, bytes]], conf_threshold: float, um_per_px: float):
    """
    honsetly no need for docstring here
    """
    results_out = []

    for index, (filename, image_bytes) in enumerate(images_data):
        res = run_inference(model, image_bytes, conf_threshold, um_per_px)
        res["filename"] = filename
        res["index"] = index
        results_out.append(res)

    return results_out