from typing import Dict, Tuple

import cv2
import numpy as np


def compute_corridor_polygon(
    frame_h: int,
    frame_w: int,
    bottom_width_frac: float,
    top_width_frac: float,
    height_frac: float,
    center_x_frac: float = 0.5,
) -> np.ndarray:
    bottom_width = int(frame_w * float(np.clip(bottom_width_frac, 0.05, 1.0)))
    top_width = int(frame_w * float(np.clip(top_width_frac, 0.05, 1.0)))
    height = int(frame_h * float(np.clip(height_frac, 0.05, 1.0)))
    center_x = int(frame_w * float(np.clip(center_x_frac, 0.0, 1.0)))

    y_bottom = frame_h - 1
    y_top = max(0, frame_h - height)

    x_bl = max(0, center_x - bottom_width // 2)
    x_br = min(frame_w - 1, center_x + bottom_width // 2)
    x_tl = max(0, center_x - top_width // 2)
    x_tr = min(frame_w - 1, center_x + top_width // 2)

    return np.array([[x_bl, y_bottom], [x_br, y_bottom], [x_tr, y_top], [x_tl, y_top]], dtype=np.int32)


def polygon_mask(frame_h: int, frame_w: int, poly: np.ndarray) -> np.ndarray:
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    return mask


def split_corridor_bands(corridor_mask: np.ndarray) -> Dict[str, np.ndarray]:
    h, w = corridor_mask.shape[:2]
    band_w = max(1, w // 3)
    bands: Dict[str, np.ndarray] = {}

    ranges = {
        "LEFT": (0, band_w),
        "CENTER": (band_w, min(w, 2 * band_w)),
        "RIGHT": (min(w, 2 * band_w), w),
    }
    for name, (x0, x1) in ranges.items():
        band = np.zeros_like(corridor_mask, dtype=np.uint8)
        band[:, x0:x1] = corridor_mask[:, x0:x1]
        bands[name] = band
    return bands


def intersection_score(obstacle_mask: np.ndarray, corridor_mask: np.ndarray) -> float:
    corridor_area = int(np.count_nonzero(corridor_mask))
    if corridor_area <= 0:
        return 0.0
    overlap = cv2.bitwise_and((obstacle_mask > 0).astype(np.uint8) * 255, corridor_mask)
    overlap_area = int(np.count_nonzero(overlap))
    return float(overlap_area / corridor_area)


def band_overlap_scores(obstacle_mask: np.ndarray, corridor_mask: np.ndarray) -> Dict[str, float]:
    bands = split_corridor_bands(corridor_mask)
    obs = (obstacle_mask > 0).astype(np.uint8) * 255
    scores: Dict[str, float] = {}
    for key, band_mask in bands.items():
        denom = max(1, int(np.count_nonzero(band_mask)))
        overlap = cv2.bitwise_and(obs, band_mask)
        scores[key] = float(np.count_nonzero(overlap) / denom)
    return scores


def dominant_direction(obstacle_mask: np.ndarray, corridor_mask: np.ndarray) -> Tuple[str, Dict[str, float]]:
    scores = band_overlap_scores(obstacle_mask, corridor_mask)
    if not scores:
        return "CENTER", {"LEFT": 0.0, "CENTER": 0.0, "RIGHT": 0.0}
    direction = max(scores, key=scores.get)
    if scores[direction] <= 1e-6:
        direction = "CENTER"
    return direction, scores
