"""Utility functions and helper classes for SmartTrack-AI."""

from __future__ import annotations

import colorsys
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "detector": {
        "model_name": "yolov8n",
        "confidence_threshold": 0.5,
        "person_class_id": 0,
        "input_size": 640,
    },
    "tracker": {
        "mode": "enhanced",
        "max_age": 30,
        "min_hits": 3,
        "iou_threshold": 0.3,
    },
    "motion": {
        "process_noise": 1.0,
        "measurement_noise": 1.0,
        "dt": 1.0 / 30.0,
    },
    "appearance": {
        "feature_dim": 512,
        "weight": 0.4,
        "use_deep_reid": False,
    },
    "association": {
        "motion_weight": 0.6,
        "appearance_weight": 0.4,
        "gate_threshold": 0.7,
        "lambda_iou": 0.5,
        "lambda_appearance": 0.5,
        "max_cosine_distance": 0.4,
    },
    "track_memory": {
        "max_lost_age": 60,
        "min_recovery_confidence": 0.5,
        "appearance_decay": 0.95,
    },
    "video": {
        "output_fps": 25,
        "output_codec": "mp4v",
        "max_resolution": 1280,
    },
    "evaluation": {
        "dataset_root": "data/datasets",
        "output_dir": "outputs",
    },
}


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = dict(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load YAML configuration with defaults fallback.

    If path is None, loads config/default.yaml from the project root.
    If a custom path is specified, it loads the default configuration first
    and deep merges the custom settings on top.

    Args:
        path: Optional path to a custom YAML configuration file.

    Returns:
        Consolidated configuration dictionary.
    """
    # Locate project default config relative to this file
    project_root = Path(__file__).resolve().parent.parent
    default_config_path = project_root / "config" / "default.yaml"

    config: Dict[str, Any] = dict(DEFAULT_CONFIG)

    if default_config_path.exists():
        try:
            with open(default_config_path, "r", encoding="utf-8") as f:
                loaded_default = yaml.safe_load(f)
                if isinstance(loaded_default, dict):
                    config = _deep_merge(config, loaded_default)
        except Exception as exc:
            logging.getLogger("smarttrack").warning(
                f"Failed to load default config at {default_config_path}: {exc}"
            )

    if path is not None:
        custom_path = Path(path)
        if not custom_path.is_absolute():
            custom_path = (project_root / custom_path).resolve()

        if custom_path.exists():
            try:
                with open(custom_path, "r", encoding="utf-8") as f:
                    custom_config = yaml.safe_load(f)
                    if isinstance(custom_config, dict):
                        config = _deep_merge(config, custom_config)
            except Exception as exc:
                logging.getLogger("smarttrack").error(
                    f"Failed to load custom config at {custom_path}: {exc}"
                )
        else:
            logging.getLogger("smarttrack").warning(
                f"Configuration path {custom_path} does not exist. Using defaults."
            )

    return config


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the project logger.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("smarttrack")
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Avoid adding multiple duplicate handlers if called repeatedly
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def generate_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a deterministic bright color for a given track ID using HSV.

    Args:
        track_id: Unique integer identifier of the track.

    Returns:
        RGB color tuple with values in range [0, 255].
    """
    # Use golden ratio conjugate to distribute hues evenly across track IDs
    golden_ratio_conjugate = 0.618033988749895
    hue = float((track_id * golden_ratio_conjugate) % 1.0)
    saturation = 0.85
    value = 0.95

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(r * 255), int(g * 255), int(b * 255)


def compute_iou(
    box1: Union[np.ndarray, Sequence[float]],
    box2: Union[np.ndarray, Sequence[float]],
) -> float:
    """Compute Intersection-over-Union (IoU) between two [x1, y1, x2, y2] boxes.

    Args:
        box1: First bounding box [x1, y1, x2, y2].
        box2: Second bounding box [x1, y1, x2, y2].

    Returns:
        IoU value in range [0.0, 1.0].
    """
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(
        0.0, float(box1[3]) - float(box1[1])
    )
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(
        0.0, float(box2[3]) - float(box2[1])
    )

    union = area1 + area2 - intersection
    if union <= 0.0:
        return 0.0

    return float(intersection / union)


def compute_iou_matrix(
    boxes1: Union[np.ndarray, Sequence[Sequence[float]]],
    boxes2: Union[np.ndarray, Sequence[Sequence[float]]],
) -> np.ndarray:
    """Compute pairwise IoU matrix between two sets of [x1, y1, x2, y2] bounding boxes.

    Args:
        boxes1: Array or list of shape [N, 4].
        boxes2: Array or list of shape [M, 4].

    Returns:
        Numpy array of shape [N, M] with pairwise IoU scores in [0.0, 1.0].
    """
    b1 = np.asarray(boxes1, dtype=np.float32)
    b2 = np.asarray(boxes2, dtype=np.float32)

    n = len(b1)
    m = len(b2)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    if b1.ndim == 1:
        b1 = b1.reshape(1, 4)
    if b2.ndim == 1:
        b2 = b2.reshape(1, 4)

    # Intersection coordinates
    inter_x1 = np.maximum(b1[:, 0:1], b2[None, :, 0])
    inter_y1 = np.maximum(b1[:, 1:2], b2[None, :, 1])
    inter_x2 = np.minimum(b1[:, 2:3], b2[None, :, 2])
    inter_y2 = np.minimum(b1[:, 3:4], b2[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Areas of individual boxes
    area1 = np.maximum(0.0, b1[:, 2] - b1[:, 0]) * np.maximum(0.0, b1[:, 3] - b1[:, 1])
    area2 = np.maximum(0.0, b2[:, 2] - b2[:, 0]) * np.maximum(0.0, b2[:, 3] - b2[:, 1])

    union = area1[:, None] + area2[None, :] - inter_area
    iou = np.where(union > 0, inter_area / np.maximum(union, 1e-7), 0.0)

    return iou.astype(np.float32)


def xyxy_to_xywh(box: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """Convert bounding box format from [x1, y1, x2, y2] to [x1, y1, width, height].

    Args:
        box: Bounding box(es) of shape [4] or [N, 4].

    Returns:
        Numpy array of same shape in [x1, y1, width, height] format.
    """
    b = np.asarray(box, dtype=np.float32).copy()
    if b.ndim == 1:
        w = max(0.0, float(b[2] - b[0]))
        h = max(0.0, float(b[3] - b[1]))
        return np.array([b[0], b[1], w, h], dtype=np.float32)

    b[:, 2] = np.maximum(0.0, b[:, 2] - b[:, 0])
    b[:, 3] = np.maximum(0.0, b[:, 3] - b[:, 1])
    return b


def xywh_to_xyxy(box: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """Convert bounding box format from [x1, y1, width, height] to [x1, y1, x2, y2].

    Args:
        box: Bounding box(es) of shape [4] or [N, 4].

    Returns:
        Numpy array of same shape in [x1, y1, x2, y2] format.
    """
    b = np.asarray(box, dtype=np.float32).copy()
    if b.ndim == 1:
        x2 = float(b[0] + b[2])
        y2 = float(b[1] + b[3])
        return np.array([b[0], b[1], x2, y2], dtype=np.float32)

    b[:, 2] = b[:, 0] + b[:, 2]
    b[:, 3] = b[:, 1] + b[:, 3]
    return b


def clip_box(
    box: Union[np.ndarray, Sequence[float]],
    img_shape: Tuple[int, ...],
) -> np.ndarray:
    """Clip [x1, y1, x2, y2] bounding box coordinates to image boundaries.

    Args:
        box: Bounding box of shape [4] or [N, 4].
        img_shape: Image dimensions (height, width, ...).

    Returns:
        Clipped bounding box array with coordinates within [0, width] and [0, height].
    """
    b = np.asarray(box, dtype=np.float32).copy()
    height, width = img_shape[0], img_shape[1]

    if b.ndim == 1:
        b[0] = np.clip(b[0], 0, width)
        b[1] = np.clip(b[1], 0, height)
        b[2] = np.clip(b[2], 0, width)
        b[3] = np.clip(b[3], 0, height)
    else:
        b[:, 0] = np.clip(b[:, 0], 0, width)
        b[:, 1] = np.clip(b[:, 1], 0, height)
        b[:, 2] = np.clip(b[:, 2], 0, width)
        b[:, 3] = np.clip(b[:, 3], 0, height)

    return b


class Timer:
    """Context manager and stopwatch utility measuring elapsed execution time."""

    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()
        self.end_time = self.start_time
        self._elapsed = 0.0

    def stop(self) -> float:
        """Stop the timer and record duration."""
        self.end_time = time.perf_counter()
        self._elapsed = self.end_time - self.start_time
        return self._elapsed

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        if self.start_time > 0 and self.end_time == self.start_time:
            return time.perf_counter() - self.start_time
        return self._elapsed

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed * 1000.0

    def __repr__(self) -> str:
        label = f"Timer({self.name}): " if self.name else "Timer: "
        return f"{label}{self.elapsed_ms:.2f} ms"
