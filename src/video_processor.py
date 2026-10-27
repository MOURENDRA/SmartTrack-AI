"""Video processing pipeline and visualizer for SmartTrack-AI."""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .detector import PersonDetector
from .exporters import export_mot_format, export_tracking_csv
from .tracker import SmartTracker
from .utils import Timer, generate_color

logger = logging.getLogger(__name__)


def _extract_track_info(track: Any) -> Tuple[int, np.ndarray, str, float]:
    """Extract standard track attributes (id, box, state, confidence) from any Track representation.

    Args:
        track: Track object or dictionary.

    Returns:
        Tuple of (track_id, [x1, y1, x2, y2] bbox array, state_string, confidence).
    """
    # Track ID
    if hasattr(track, "track_id"):
        tid = track.track_id
    elif hasattr(track, "id"):
        tid = track.id
    elif isinstance(track, dict) and "track_id" in track:
        tid = track["track_id"]
    elif isinstance(track, dict) and "id" in track:
        tid = track["id"]
    else:
        tid = 0

    # Box coordinates [x1, y1, x2, y2]
    if hasattr(track, "box") and track.box is not None:
        box = track.box
    elif hasattr(track, "to_tlbr") and callable(track.to_tlbr):
        box = track.to_tlbr()
    elif hasattr(track, "bbox") and track.bbox is not None:
        box = track.bbox
    elif hasattr(track, "tlbr") and track.tlbr is not None:
        box = track.tlbr
    elif hasattr(track, "get_state") and callable(track.get_state):
        box = track.get_state()
    elif isinstance(track, dict) and "bbox" in track:
        box = track["bbox"]
    elif isinstance(track, dict) and "box" in track:
        box = track["box"]
    elif isinstance(track, dict) and all(k in track for k in ("x1", "y1", "x2", "y2")):
        box = [track["x1"], track["y1"], track["x2"], track["y2"]]
    else:
        box = [0.0, 0.0, 0.0, 0.0]

    box_arr = np.asarray(box, dtype=float).ravel()
    if len(box_arr) < 4:
        box_arr = np.pad(box_arr, (0, 4 - len(box_arr)))

    # Track State
    state = "confirmed"
    if hasattr(track, "state"):
        st = track.state
        if hasattr(st, "name"):
            state = st.name.lower()
        elif hasattr(st, "value"):
            state = str(st.value).lower()
        else:
            state = str(st).lower()
    elif isinstance(track, dict) and "state" in track:
        state = str(track["state"]).lower()

    # Confidence score
    conf = 1.0
    if hasattr(track, "confidence") and track.confidence is not None:
        conf = float(track.confidence)
    elif hasattr(track, "score") and track.score is not None:
        conf = float(track.score)
    elif isinstance(track, dict) and "confidence" in track:
        conf = float(track["confidence"])
    elif isinstance(track, dict) and "score" in track:
        conf = float(track["score"])

    return int(tid), box_arr[:4], state, conf


class VideoProcessor:
    """Video processing pipeline for automated multi-object tracking.

    Manages video frame ingestion, detection, tracking association, visual annotation,
    and structured export.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize VideoProcessor with model detector and tracker configuration.

        Args:
            config: Consolidated configuration dictionary.
        """
        self.config = config

        # 1. Initialize Detector with graceful fallback
        detector_config = config.get("detector", {})
        try:
            self.detector = PersonDetector(detector_config)
            if not self.detector.is_ready:
                logger.warning(
                    f"Detector backend is '{self.detector.backend_name}'. "
                    "Ensure ultralytics is installed for real detections."
                )
        except Exception as exc:
            logger.error(f"Failed to initialize PersonDetector: {exc}")
            self.detector = PersonDetector({"model_name": "mock"})

        # 2. Initialize SmartTracker
        self.tracker = SmartTracker(config)

        # 3. Video and display configuration
        video_cfg = config.get("video", {})
        self.max_resolution = video_cfg.get("max_resolution", 1280)
        self.output_fps = video_cfg.get("output_fps", 25.0)
        self.output_codec = video_cfg.get("output_codec", "mp4v")
        self.draw_trajectory = video_cfg.get("draw_trajectory", True)
        self.trajectory_length = int(video_cfg.get("trajectory_length", 30))
        self.show_confidence = video_cfg.get("show_confidence", True)
        self.show_state = video_cfg.get("show_state", True)
        self.line_thickness = int(video_cfg.get("line_thickness", 2))

        # 4. Internal state and trajectories memory
        self._trajectories: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.trajectory_length)
        )
        self.frame_count: int = 0
        self.frame_metrics: List[Dict[str, Any]] = []

        # Color mapping for track states: (B, G, R)
        self.state_colors: Dict[str, Tuple[int, int, int]] = {
            "confirmed": (0, 255, 0),  # Green
            "tentative": (0, 255, 255),  # Yellow
            "lost": (0, 0, 255),  # Red
            "deleted": (128, 128, 128),  # Gray
        }

        logger.info("VideoProcessor initialized successfully.")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Process a single frame: runs detector, tracker, and generates annotated frame.

        Args:
            frame: Input BGR image as numpy array (H, W, 3).

        Returns:
            Tuple of:
                annotated_frame: Frame with drawn bounding boxes, IDs, and trajectories.
                frame_metric: Dictionary of metrics collected for this frame.
        """
        t_start = time.perf_counter()

        # Step 1: Run detector
        try:
            boxes, confidences = self.detector.detect(frame)
        except Exception as exc:
            logger.error(f"Detector error on frame {self.frame_count}: {exc}")
            boxes, confidences = np.empty((0, 4)), np.empty((0,))

        # Step 2: Run tracker
        try:
            tracks = self.tracker.update(frame, boxes, confidences)
        except Exception as exc:
            logger.error(f"Tracker update error on frame {self.frame_count}: {exc}")
            tracks = []

        # Step 3: Draw annotations
        annotated_frame = self.draw_tracks(frame, tracks)

        # Step 4: Collect per-frame metrics
        frame_time = time.perf_counter() - t_start

        active_tracks = 0
        lost_tracks = 0
        track_records: List[Dict[str, Any]] = []

        for trk in tracks:
            tid, box, state, conf = _extract_track_info(trk)
            if state == "lost":
                lost_tracks += 1
            else:
                active_tracks += 1

            track_records.append(
                {
                    "track_id": tid,
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "confidence": float(conf),
                    "state": state,
                }
            )

        frame_metric = {
            "frame_id": self.frame_count,
            "frame_idx": self.frame_count,
            "detections_count": len(boxes),
            "active_tracks": active_tracks,
            "lost_tracks": lost_tracks,
            "frame_time": frame_time,
            "processing_time": frame_time,
            "tracks": track_records,
        }

        self.frame_metrics.append(frame_metric)
        self.frame_count += 1

        return annotated_frame, frame_metric

    def draw_tracks(self, frame: np.ndarray, tracks: List[Any]) -> np.ndarray:
        """Draw bounding boxes, track IDs, states, and trajectories on a video frame.

        Args:
            frame: Input image array (H, W, 3).
            tracks: List of active Track objects or track dictionaries.

        Returns:
            Annotated frame array (H, W, 3).
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for trk in tracks:
            tid, box, state, conf = _extract_track_info(trk)

            x1 = int(np.clip(box[0], 0, w - 1))
            y1 = int(np.clip(box[1], 0, h - 1))
            x2 = int(np.clip(box[2], 0, w - 1))
            y2 = int(np.clip(box[3], 0, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            # Unique deterministic color from track_id
            rgb_color = generate_color(tid)
            # OpenCV uses BGR ordering
            bgr_color = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))

            # 1. Trajectory (motion path history)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            self._trajectories[tid].append((cx, cy))

            if self.draw_trajectory:
                points = list(self._trajectories[tid])
                num_pts = len(points)
                for i in range(1, num_pts):
                    # Progressively thicker and brighter trajectory line
                    thickness = max(1, int(self.line_thickness * (i / num_pts)))
                    cv2.line(annotated, points[i - 1], points[i], bgr_color, thickness)

            # 2. Main Bounding Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr_color, self.line_thickness)

            # 3. Track State Indicator Dot (Green=confirmed, Yellow=tentative, Red=lost)
            state_color = self.state_colors.get(state, (200, 200, 200))
            dot_radius = max(3, self.line_thickness + 1)
            dot_x = max(dot_radius + 2, x1 + 6)
            dot_y = max(dot_radius + 2, y1 - 8)

            if self.show_state:
                cv2.circle(annotated, (dot_x, dot_y), dot_radius, state_color, -1)
                cv2.circle(annotated, (dot_x, dot_y), dot_radius + 1, (0, 0, 0), 1)

            # 4. Text Label with Track ID and optional Confidence
            label_parts = [f"ID: {tid}"]
            if self.show_confidence and 0.0 <= conf <= 1.0:
                label_parts.append(f"{conf:.2f}")
            label = " ".join(label_parts)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )

            text_offset_x = dot_x + dot_radius + 4 if self.show_state else x1 + 2
            text_x = min(text_offset_x, max(0, w - text_w - 4))
            text_y = max(text_h + 4, y1 - 4)

            # Background rectangle for crisp readability
            bg_pt1 = (max(0, text_x - 2), max(0, text_y - text_h - 2))
            bg_pt2 = (min(w, text_x + text_w + 2), min(h, text_y + baseline))
            cv2.rectangle(annotated, bg_pt1, bg_pt2, (20, 20, 20), -1)

            # Text string
            cv2.putText(
                annotated,
                label,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

        return annotated

    def process_video(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int, int, List[Any], float], None]] = None,
    ) -> dict:
        """Process an entire video file through detection, tracking, and export.

        Args:
            input_path: Path to the input video file.
            output_path: Path to save the annotated video file.
            progress_callback: Optional callback called as:
                progress_callback(frame_idx, total_frames, current_tracks, frame_time)

        Returns:
            Dictionary containing processing summary, per-frame metrics, and tracker stats.
        """
        # Validate input file existence
        input_file = Path(input_path)
        if not input_file.is_file():
            raise FileNotFoundError(f"Video input file not found: {input_path}")

        # Open video capture
        cap = cv2.VideoCapture(str(input_file))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file with OpenCV: {input_path}")

        writer: Optional[cv2.VideoWriter] = None

        try:
            # Video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0.0 or np.isnan(fps):
                fps = float(self.output_fps)

            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if src_w <= 0 or src_h <= 0:
                raise RuntimeError(
                    f"Invalid video stream dimensions: {src_w}x{src_h} for {input_path}"
                )

            # Calculate target resolution (respecting max_resolution config)
            out_w = src_w
            out_h = src_h
            if self.max_resolution is not None:
                if (
                    isinstance(self.max_resolution, (list, tuple))
                    and len(self.max_resolution) == 2
                ):
                    max_w, max_h = self.max_resolution
                    scale = min(max_w / src_w, max_h / src_h)
                elif isinstance(self.max_resolution, (int, float)) and self.max_resolution > 0:
                    max_dim = max(src_w, src_h)
                    scale = min(1.0, float(self.max_resolution) / float(max_dim))
                else:
                    scale = 1.0

                if scale < 1.0:
                    out_w = int(round(src_w * scale))
                    out_h = int(round(src_h * scale))
                    # Ensure width and height are even integers for codec compatibility
                    out_w = out_w if out_w % 2 == 0 else out_w - 1
                    out_h = out_h if out_h % 2 == 0 else out_h - 1

            # Prepare output directory
            out_dir = Path(output_path).parent
            out_dir.mkdir(parents=True, exist_ok=True)

            # Initialize VideoWriter with primary codec and robust fallbacks
            codecs_to_try = [self.output_codec, "mp4v", "XVID", "MJPG"]
            for codec in codecs_to_try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
                if writer.isOpened():
                    logger.info(f"VideoWriter initialized with codec '{codec}'.")
                    break

            if writer is None or not writer.isOpened():
                raise RuntimeError(
                    f"Failed to create VideoWriter for '{output_path}' with codecs: {codecs_to_try}"
                )

            # Process frames
            processed_frames = 0
            frame_times: List[float] = []
            per_frame_metrics: List[Dict[str, Any]] = []

            logger.info(
                f"Starting video processing: {input_path} "
                f"({total_frames} frames, {src_w}x{src_h} -> {out_w}x{out_h} @ {fps:.1f} FPS)"
            )

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Video stream completed or encountered end-of-file
                    break

                # Resize if target resolution differs from source
                if (out_w, out_h) != (src_w, src_h):
                    frame = cv2.resize(
                        frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR
                    )

                # Process single frame
                annotated_frame, metrics = self.process_frame(frame)
                per_frame_metrics.append(metrics)
                frame_times.append(metrics["frame_time"])

                # Write annotated frame
                writer.write(annotated_frame)

                # Invoke progress callback if provided
                if progress_callback is not None:
                    try:
                        active_tracks = self.tracker.track_manager.get_active_tracks()
                        progress_callback(
                            processed_frames,
                            total_frames,
                            active_tracks,
                            metrics["frame_time"],
                        )
                    except Exception as cb_exc:
                        logger.debug(f"Progress callback exception: {cb_exc}")

                processed_frames += 1

            avg_time = float(np.mean(frame_times)) if frame_times else 0.0
            avg_fps = round(1.0 / avg_time, 2) if avg_time > 0 else 0.0

            logger.info(
                f"Video processing finished. Processed {processed_frames}/{total_frames} "
                f"frames @ {avg_fps:.1f} avg FPS."
            )

            return {
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "output_path": output_path,
                "avg_fps": avg_fps,
                "frame_metrics": per_frame_metrics,
                "tracker_stats": self.tracker.get_statistics(),
                "video_properties": {
                    "fps": fps,
                    "width": src_w,
                    "height": src_h,
                    "total_frames": total_frames,
                    "out_width": out_w,
                    "out_height": out_h,
                },
            }

        finally:
            cap.release()
            if writer is not None:
                writer.release()

    def export_tracks_csv(self, frame_metrics: list, output_path: str) -> None:
        """Export tracking results as CSV with columns:
        frame_id, track_id, x1, y1, x2, y2, confidence, state.

        Args:
            frame_metrics: Collected list of per-frame tracking metrics.
            output_path: Destination path for the CSV file.
        """
        export_tracking_csv(frame_metrics, output_path)

    def export_tracks_mot(self, frame_metrics: list, output_path: str) -> None:
        """Export tracking results in standard MOT challenge format:
        frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z.

        Args:
            frame_metrics: Collected list of per-frame tracking metrics.
            output_path: Destination path for the MOT text file.
        """
        export_mot_format(frame_metrics, output_path)

    def reset(self) -> None:
        """Reset the tracker and internal state completely."""
        self.tracker.reset()
        self._trajectories.clear()
        self.frame_count = 0
        self.frame_metrics.clear()
        logger.info("VideoProcessor reset completed.")
