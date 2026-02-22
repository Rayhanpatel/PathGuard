import time
from dataclasses import dataclass
from typing import Dict, Generator, Optional

import cv2
import numpy as np

from .config import CorridorParams, RuntimeParams
from .corridor import compute_corridor_polygon, intersection_score, polygon_mask
from .depth import DepthAnythingV2Estimator, box_closeness, closeness_bucket
from .detect import GroundedDINODetector, filter_detections_in_corridor
from .events import EventStateMachine
from .fallback import detect_fallback_obstacles
from .segment import SAM2Segmenter, rasterize_box_mask


@dataclass
class PipelineRuntime:
    corridor: CorridorParams
    runtime: RuntimeParams


def _resize_keep_aspect(frame_bgr: np.ndarray, out_w: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if w <= out_w:
        return frame_bgr
    out_h = int((h / w) * out_w)
    return cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def _render_overlay(
    frame_bgr: np.ndarray,
    corridor_poly: np.ndarray,
    corridor_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    status_text: str,
    detections: list,
    depth_tags: Dict[int, str],
    closeness_map: Optional[np.ndarray],
    show_depth_overlay: bool,
) -> np.ndarray:
    draw = frame_bgr.copy()

    cv2.polylines(draw, [corridor_poly], isClosed=True, color=(0, 255, 255), thickness=2)

    red = np.zeros_like(draw)
    red[:, :, 2] = 255
    alpha_mask = (obstacle_mask > 0).astype(np.uint8) * 120
    for c in range(3):
        draw[:, :, c] = np.where(alpha_mask > 0, (0.55 * draw[:, :, c] + 0.45 * red[:, :, c]).astype(np.uint8), draw[:, :, c])

    if show_depth_overlay and closeness_map is not None:
        depth_u8 = np.clip((closeness_map * 255.0), 0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        depth_region = corridor_mask > 0
        for c in range(3):
            draw[:, :, c] = np.where(
                depth_region,
                (0.75 * draw[:, :, c] + 0.25 * depth_color[:, :, c]).astype(np.uint8),
                draw[:, :, c],
            )

    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        cv2.rectangle(draw, (x1, y1), (x2, y2), (80, 255, 80), 2)
        suffix = depth_tags.get(id(det), "")
        label_text = f"{det.label}:{det.score:.2f}"
        if suffix:
            label_text = f"{label_text} {suffix}"
        cv2.putText(
            draw,
            label_text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 255, 80),
            2,
            cv2.LINE_AA,
        )

    chip_color = (40, 180, 40) if status_text == "CLEAR" else (50, 170, 255) if status_text == "OBSTACLE AHEAD" else (40, 40, 255)
    cv2.rectangle(draw, (16, 16), (360, 86), chip_color, -1)
    cv2.putText(draw, status_text, (28, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    return draw


def process_video(
    video_path: str,
    cfg: PipelineRuntime,
    stop_flag_fn=None,
) -> Generator[Dict[str, object], None, None]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = src_fps if src_fps and src_fps > 1e-6 else 30.0
    frame_dt = 1.0 / src_fps

    runtime = cfg.runtime
    dino = GroundedDINODetector(device="cuda")
    sam2 = SAM2Segmenter() if runtime.use_sam2 else None
    depth = DepthAnythingV2Estimator(model_id=runtime.depth_model_id, device="cuda") if runtime.use_depth else None
    machine = EventStateMachine(persistence_frames=runtime.persistence_frames, debounce_sec=2.0)

    frame_idx = 0
    prev_t = time.perf_counter()
    last_fps = 0.0
    last_distance_bucket = ""
    last_depth_score = 0.0
    warning_msg = ""
    closeness_map: Optional[np.ndarray] = None

    try:
        while True:
            if stop_flag_fn and stop_flag_fn():
                break

            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_idx += 1
            t0 = time.perf_counter()

            frame_bgr = _resize_keep_aspect(frame_bgr, runtime.output_width)
            h, w = frame_bgr.shape[:2]
            corridor_poly = compute_corridor_polygon(
                h,
                w,
                cfg.corridor.bottom_width_frac,
                cfg.corridor.top_width_frac,
                cfg.corridor.height_frac,
                cfg.corridor.center_x_frac,
            )
            corridor_mask = polygon_mask(h, w, corridor_poly)

            fb = detect_fallback_obstacles(frame_bgr, corridor_mask, blur_thresh=runtime.blur_thresh)
            fallback_mask = fb["obstacle_mask"]
            trip_risk_flag = bool(fb["trip_risk_flag"])
            low_quality = bool(fb["low_quality"])

            detections = []
            if runtime.use_dino and dino.available:
                should_run_dino = (frame_idx % max(1, runtime.dino_interval) == 0) or (fb["occupancy_score"] > runtime.occ_thresh * 0.8) or trip_risk_flag
                if should_run_dino:
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    detections = dino.detect(rgb, runtime.prompts)
                    detections = filter_detections_in_corridor(detections, corridor_mask, min_overlap=0.10)

            if runtime.use_depth and depth is not None and depth.available:
                should_run_depth = (frame_idx % max(1, runtime.depth_interval) == 0) or bool(detections)
                if should_run_depth:
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    maybe_map = depth.estimate_closeness(rgb)
                    if maybe_map is not None:
                        closeness_map = maybe_map

            obstacle_mask = fallback_mask.copy()
            if detections:
                det_mask = np.zeros_like(corridor_mask, dtype=np.uint8)
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                for det in detections:
                    if runtime.use_sam2 and sam2 is not None:
                        m = sam2.segment_from_box(rgb, det.box_xyxy)
                        if m is None:
                            m = rasterize_box_mask(h, w, det.box_xyxy)
                    else:
                        m = rasterize_box_mask(h, w, det.box_xyxy)
                    det_mask = cv2.bitwise_or(det_mask, m)
                obstacle_mask = cv2.bitwise_or(obstacle_mask, cv2.bitwise_and(det_mask, corridor_mask))

            occ_score = intersection_score(obstacle_mask, corridor_mask)
            depth_tags: Dict[int, str] = {}
            last_distance_bucket = ""
            last_depth_score = 0.0
            if detections and closeness_map is not None:
                for det in detections:
                    score = box_closeness(closeness_map, det.box_xyxy)
                    bucket = closeness_bucket(score)
                    depth_tags[id(det)] = f"[{bucket}]"
                    last_distance_bucket = bucket
                    last_depth_score = score
            elif closeness_map is not None:
                corridor_vals = closeness_map[corridor_mask > 0]
                if corridor_vals.size > 0:
                    score = float(np.median(corridor_vals))
                    last_depth_score = score
                    last_distance_bucket = closeness_bucket(score)

            warning_parts = []
            if runtime.use_sam2 and sam2 is not None and sam2.warning:
                warning_parts.append(sam2.warning)
            if runtime.use_depth and depth is not None and depth.warning:
                warning_parts.append(depth.warning)
            warning_msg = " | ".join(warning_parts)

            depth_status = "OFF"
            if runtime.use_depth:
                if depth is None:
                    depth_status = "UNAVAILABLE"
                elif depth.available:
                    depth_status = "ON"
                else:
                    depth_status = "UNAVAILABLE"
            ts_sec = frame_idx * frame_dt
            state = machine.update(
                timestamp_s=ts_sec,
                occupancy_score=occ_score,
                occ_thresh=runtime.occ_thresh,
                trip_risk_flag=trip_risk_flag,
                low_quality=low_quality,
            )

            overlay_bgr = _render_overlay(
                frame_bgr,
                corridor_poly,
                corridor_mask,
                obstacle_mask,
                state,
                detections,
                depth_tags,
                closeness_map,
                runtime.show_depth_overlay and runtime.use_depth,
            )
            out_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

            now = time.perf_counter()
            dt = max(1e-6, now - prev_t)
            prev_t = now
            last_fps = 1.0 / dt

            yield {
                "frame_rgb": out_rgb,
                "fps": float(last_fps),
                "occupancy_score": float(occ_score),
                "state": state,
                "last_distance_bucket": last_distance_bucket,
                "last_depth_score": float(last_depth_score),
                "depth_status": depth_status,
                "events": machine.events_as_rows(),
                "warning": warning_msg,
                "timestamp_s": ts_sec,
            }

            elapsed = time.perf_counter() - t0
            if runtime.simulate_realtime and elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
    finally:
        cap.release()
