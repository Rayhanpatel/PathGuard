from dataclasses import dataclass
import os
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    box_xyxy: Tuple[int, int, int, int]
    label: str
    score: float


class GroundedDINODetector:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.available = False
        self.warning = ""
        self._show_warnings = os.getenv("PATHGUARD_SHOW_GDINO_WARNINGS", "0") == "1"
        self.backend_name = "none"

        self._gdino_backend = None
        self._gdino_model = None

        self._torch = None
        self._hf_processor = None
        self._hf_model = None

        self._init_backend()

    def _set_warning(self, msg: str) -> None:
        self.warning = msg if self._show_warnings else ""

    def _init_backend(self) -> None:
        # Try groundeddino_vl first
        err_msgs: List[str] = []
        if self._init_groundeddino_vl():
            self.available = True
            self.backend_name = "groundeddino_vl"
            return
        if self.warning:
            err_msgs.append(self.warning)

        # Fallback: Hugging Face GroundingDINO implementation
        if self._init_hf_grounding_dino():
            self.available = True
            self.backend_name = "hf_grounding_dino"
            self.warning = ""
            return
        if self.warning:
            err_msgs.append(self.warning)

        self.available = False
        if err_msgs:
            self.warning = " | ".join(err_msgs)

    def _init_groundeddino_vl(self) -> bool:
        try:
            import groundeddino_vl as gdino  # type: ignore
        except Exception as exc:
            self._set_warning(f"GroundedDINO unavailable ({exc})")
            return False

        self._gdino_backend = gdino
        try:
            model_config_path = os.getenv("PATHGUARD_GDINO_CONFIG", "")
            model_checkpoint_path = os.getenv("PATHGUARD_GDINO_CHECKPOINT", "")

            if hasattr(gdino, "download_model_weights"):
                cfg, ckpt = gdino.download_model_weights()
                if not model_config_path:
                    model_config_path = cfg
                if not model_checkpoint_path:
                    model_checkpoint_path = ckpt

            if (not model_config_path or not model_checkpoint_path) and hasattr(gdino, "weights_manager"):
                try:
                    from groundeddino_vl.weights_manager import ensure_weights  # type: ignore

                    cfg, ckpt = ensure_weights(model_config_path or None, model_checkpoint_path or None)
                    if not model_config_path:
                        model_config_path = cfg
                    if not model_checkpoint_path:
                        model_checkpoint_path = ckpt
                except Exception:
                    pass

            if not model_config_path or not model_checkpoint_path:
                self._set_warning("GroundedDINO model init error (missing config/checkpoint paths)")
                return False

            self._gdino_model = gdino.load_model(
                model_config_path,
                model_checkpoint_path,
                device=self.device,
            )
            return True
        except Exception as exc:
            self._set_warning(f"GroundedDINO model init error ({exc})")
            return False

    def _init_hf_grounding_dino(self) -> bool:
        try:
            import torch  # type: ignore
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # type: ignore

            hf_model_id = os.getenv("PATHGUARD_GDINO_HF_MODEL", "IDEA-Research/grounding-dino-tiny")
            self._torch = torch
            self._hf_processor = AutoProcessor.from_pretrained(hf_model_id)
            target_device = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
            self.device = target_device

            # Avoid meta-tensor move errors: load directly on target device map.
            try:
                if target_device == "cuda":
                    self._hf_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                        hf_model_id,
                        low_cpu_mem_usage=True,
                        device_map="auto",
                    )
                else:
                    self._hf_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                        hf_model_id,
                        low_cpu_mem_usage=True,
                        device_map="cpu",
                    )
            except Exception:
                # Guaranteed fallback path: CPU-only load.
                self.device = "cpu"
                self._hf_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    hf_model_id,
                    low_cpu_mem_usage=True,
                    device_map="cpu",
                )
            self._hf_model.eval()
            return True
        except Exception as exc:
            self._set_warning(f"HF GroundingDINO init error ({exc})")
            return False

    def _as_numpy(self, value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _coerce_output(self, outputs: object, frame_w: int, frame_h: int) -> List[Detection]:
        dets: List[Detection] = []
        boxes = []
        labels = []
        scores = []

        if hasattr(outputs, "boxes") and hasattr(outputs, "labels") and hasattr(outputs, "scores"):
            labels = list(outputs.labels)
            scores = outputs.scores
            if hasattr(outputs, "to_xyxy"):
                boxes = outputs.to_xyxy(denormalize=True)
            else:
                boxes = outputs.boxes
        elif isinstance(outputs, dict):
            boxes = outputs.get("boxes", [])
            labels = outputs.get("labels", outputs.get("phrases", []))
            scores = outputs.get("scores", outputs.get("logits", []))
        elif isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
            boxes, scores, labels = outputs[0], outputs[1], outputs[2]
        else:
            return dets

        boxes = self._as_numpy(boxes)
        scores = self._as_numpy(scores)
        labels = list(labels)

        n = min(len(boxes), len(labels), len(scores))
        for i in range(n):
            box = np.asarray(boxes[i]).astype(float).reshape(-1)
            if box.size != 4:
                continue
            x1, y1, x2, y2 = box.tolist()

            # Convert normalized cxcywh to xyxy pixels if needed.
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5 and x2 <= 1.5 and y2 <= 1.5:
                cx, cy, bw, bh = x1, y1, x2, y2
                x1 = (cx - bw / 2.0) * frame_w
                y1 = (cy - bh / 2.0) * frame_h
                x2 = (cx + bw / 2.0) * frame_w
                y2 = (cy + bh / 2.0) * frame_h

            x1 = int(max(0, min(frame_w - 1, x1)))
            y1 = int(max(0, min(frame_h - 1, y1)))
            x2 = int(max(0, min(frame_w - 1, x2)))
            y2 = int(max(0, min(frame_h - 1, y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append(Detection(box_xyxy=(x1, y1, x2, y2), label=str(labels[i]), score=float(scores[i])))
        return dets

    def _predict_groundeddino_vl(self, frame_rgb: np.ndarray, prompt_text: str, box_threshold: float, text_threshold: float):
        if self._gdino_backend is None or self._gdino_model is None:
            raise RuntimeError("groundeddino_vl backend/model not ready")
        try:
            return self._gdino_backend.predict(
                model=self._gdino_model,
                image=frame_rgb,
                text_prompt=prompt_text,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device,
            )
        except TypeError:
            return self._gdino_backend.predict(
                model=self._gdino_model,
                image=frame_rgb,
                caption=prompt_text,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device,
            )

    def _predict_hf(self, frame_rgb: np.ndarray, prompt_text: str, box_threshold: float, text_threshold: float):
        if self._torch is None or self._hf_processor is None or self._hf_model is None:
            raise RuntimeError("HF GroundingDINO backend/model not ready")
        h, w = frame_rgb.shape[:2]
        with self._torch.no_grad():
            inputs = self._hf_processor(images=frame_rgb, text=prompt_text, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            outputs = self._hf_model(**inputs)
            try:
                # Newer Transformers signature.
                processed = self._hf_processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    input_ids=inputs["input_ids"],
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=[(h, w)],
                )[0]
            except TypeError:
                try:
                    # Older variants use `threshold` instead of `box_threshold`.
                    processed = self._hf_processor.post_process_grounded_object_detection(
                        outputs=outputs,
                        input_ids=inputs["input_ids"],
                        threshold=box_threshold,
                        text_threshold=text_threshold,
                        target_sizes=[(h, w)],
                    )[0]
                except TypeError:
                    # Some releases do not accept text threshold in post-process.
                    processed = self._hf_processor.post_process_grounded_object_detection(
                        outputs=outputs,
                        input_ids=inputs["input_ids"],
                        threshold=box_threshold,
                        target_sizes=[(h, w)],
                    )[0]

        return {
            "boxes": processed["boxes"],
            "scores": processed["scores"],
            "labels": processed["labels"],
        }

    def detect(
        self,
        frame_rgb: np.ndarray,
        prompts: Sequence[str],
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ) -> List[Detection]:
        if not self.available:
            return []

        h, w = frame_rgb.shape[:2]
        prompt_text = ", ".join(prompts)

        try:
            if self.backend_name == "groundeddino_vl":
                outputs = self._predict_groundeddino_vl(frame_rgb, prompt_text, box_threshold, text_threshold)
            elif self.backend_name == "hf_grounding_dino":
                outputs = self._predict_hf(frame_rgb, prompt_text, box_threshold, text_threshold)
            else:
                return []
            return self._coerce_output(outputs, w, h)
        except Exception as exc:
            self._set_warning(f"GroundedDINO inference error ({exc})")
            self.available = False
            return []


def filter_detections_in_corridor(
    detections: Sequence[Detection],
    corridor_mask: np.ndarray,
    min_overlap: float = 0.15,
) -> List[Detection]:
    valid: List[Detection] = []
    h, w = corridor_mask.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        box_area = max(1, (x2 - x1) * (y2 - y1))
        box_mask = np.zeros_like(corridor_mask, dtype=np.uint8)
        cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)
        overlap = cv2.bitwise_and(box_mask, corridor_mask)
        overlap_ratio = float(np.count_nonzero(overlap) / box_area)
        if overlap_ratio >= min_overlap:
            valid.append(det)
    return valid
