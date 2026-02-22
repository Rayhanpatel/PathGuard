from typing import Optional, Tuple

import cv2
import numpy as np


class DepthAnythingV2Estimator:
    def __init__(self, model_id: str, device: str = "cuda") -> None:
        self.model_id = model_id
        self.device = device
        self.available = False
        self.warning = ""
        self._torch = None
        self._processor = None
        self._model = None
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation  # type: ignore

            self._torch = torch
            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.to("cuda")
            else:
                self.device = "cpu"
            self._model.eval()
            self.available = True
        except Exception as exc:
            self.warning = f"Depth Anything V2 unavailable ({exc})"
            self.available = False

    def estimate_closeness(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        if not self.available or self._torch is None or self._processor is None or self._model is None:
            return None

        h, w = frame_rgb.shape[:2]
        try:
            with self._torch.no_grad():
                inputs = self._processor(images=frame_rgb, return_tensors="pt")
                if self.device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                outputs = self._model(**inputs)
                pred = outputs.predicted_depth
                pred = self._torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                depth = pred.detach().float().cpu().numpy()
        except Exception as exc:
            self.warning = f"Depth Anything V2 inference error ({exc})"
            return None

        return _to_closeness_map(depth)


def _to_closeness_map(depth_map: np.ndarray) -> np.ndarray:
    depth = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    d_min = float(depth.min())
    d_max = float(depth.max())
    if d_max - d_min < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)

    depth_norm = (depth - d_min) / (d_max - d_min)

    # Auto-select polarity. Bottom is usually closer in forward walking videos.
    h = depth_norm.shape[0]
    top_band = depth_norm[: max(1, h // 5), :]
    bottom_band = depth_norm[max(0, h - h // 5) :, :]
    top_med = float(np.median(top_band))
    bottom_med = float(np.median(bottom_band))
    larger_is_closer = bottom_med > top_med

    closeness = depth_norm if larger_is_closer else (1.0 - depth_norm)
    return np.clip(closeness, 0.0, 1.0).astype(np.float32)


def box_closeness(closeness_map: np.ndarray, box_xyxy: Tuple[int, int, int, int]) -> float:
    h, w = closeness_map.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = closeness_map[y1:y2, x1:x2]
    if patch.size <= 0:
        return 0.0
    return float(np.median(patch))


def closeness_bucket(value: float) -> str:
    if value >= 0.66:
        return "NEAR"
    if value >= 0.33:
        return "MID"
    return "FAR"
