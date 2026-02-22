from typing import Optional, Tuple

import cv2
import numpy as np


def rasterize_box_mask(frame_h: int, frame_w: int, box_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0, min(frame_w - 1, int(x1)))
    y1 = max(0, min(frame_h - 1, int(y1)))
    x2 = max(0, min(frame_w - 1, int(x2)))
    y2 = max(0, min(frame_h - 1, int(y2)))
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


class SAM2Segmenter:
    def __init__(self) -> None:
        self.available = False
        self.warning = ""
        self._backend = None
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import sam2  # type: ignore

            self._backend = sam2
            self.available = True
        except Exception as exc:
            self.warning = f"SAM2 unavailable ({exc})"
            self.available = False

    def segment_from_box(self, frame_rgb: np.ndarray, box_xyxy: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        h, w = frame_rgb.shape[:2]
        if not self.available:
            return None

        # SAM2 APIs vary across releases. If unavailable at runtime, gracefully fallback.
        try:
            if hasattr(self._backend, "predict_box_mask"):
                mask = self._backend.predict_box_mask(frame_rgb, box_xyxy)
                if mask is not None:
                    return (mask > 0).astype(np.uint8) * 255
        except Exception as exc:
            self.warning = f"SAM2 inference error ({exc})"

        return rasterize_box_mask(h, w, box_xyxy)
