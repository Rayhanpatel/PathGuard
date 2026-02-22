from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np


@dataclass
class FallbackParams:
    canny_low: int = 60
    canny_high: int = 160
    min_component_area: int = 250
    trip_component_area: int = 180
    trip_aspect_ratio: float = 6.0
    blur_kernel: int = 5
    morph_kernel: int = 5


def detect_fallback_obstacles(
    frame_bgr: np.ndarray,
    corridor_mask: np.ndarray,
    blur_thresh: float,
    params: FallbackParams | None = None,
) -> Dict[str, object]:
    cfg = params or FallbackParams()

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (cfg.blur_kernel, cfg.blur_kernel), 0)

    lap_var = float(cv2.Laplacian(blur, cv2.CV_64F).var())
    low_quality = lap_var < blur_thresh

    edges = cv2.Canny(blur, cfg.canny_low, cfg.canny_high)
    edges = cv2.bitwise_and(edges, corridor_mask)

    kernel = np.ones((cfg.morph_kernel, cfg.morph_kernel), dtype=np.uint8)
    merged = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    merged = cv2.dilate(merged, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((merged > 0).astype(np.uint8), connectivity=8)

    obstacle_mask = np.zeros_like(corridor_mask, dtype=np.uint8)
    trip_risk_flag = False

    for comp_id in range(1, num_labels):
        x, y, w, h, area = stats[comp_id]
        if area < cfg.min_component_area:
            continue
        obstacle_mask[labels == comp_id] = 255

        if area >= cfg.trip_component_area:
            ratio = max((w / max(1.0, h)), (h / max(1.0, w)))
            if ratio >= cfg.trip_aspect_ratio:
                trip_risk_flag = True

    obstacle_mask = cv2.bitwise_and(obstacle_mask, corridor_mask)
    corridor_area = max(1, int(np.count_nonzero(corridor_mask)))
    obstacle_area = int(np.count_nonzero(obstacle_mask))
    occupancy_score = float(obstacle_area / corridor_area)

    return {
        "obstacle_mask": obstacle_mask,
        "occupancy_score": occupancy_score,
        "trip_risk_flag": bool(trip_risk_flag),
        "low_quality": bool(low_quality),
        "laplacian_var": lap_var,
    }
