"""Kalman filter motion model for bounding box tracking in SmartTrack-AI."""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Union

import numpy as np

# Robust import for FilterPy with custom KalmanFilter fallback if filterpy is not installed
try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    logging.getLogger("smarttrack").info(
        "filterpy not installed; using built-in KalmanFilter fallback implementation."
    )

    class KalmanFilter:  # type: ignore[no-redef]
        """Fallback linear Kalman Filter implementing standard filterpy interface."""

        def __init__(self, dim_x: int, dim_z: int):
            self.dim_x = dim_x
            self.dim_z = dim_z
            self.x = np.zeros((dim_x, 1), dtype=np.float64)
            self.P = np.eye(dim_x, dtype=np.float64)
            self.F = np.eye(dim_x, dtype=np.float64)
            self.H = np.zeros((dim_z, dim_x), dtype=np.float64)
            self.R = np.eye(dim_z, dtype=np.float64)
            self.Q = np.eye(dim_x, dtype=np.float64)
            self.B = None
            self._I = np.eye(dim_x, dtype=np.float64)

        def predict(self, u: Optional[np.ndarray] = None) -> None:
            """Predict next state ahead."""
            if u is not None and self.B is not None:
                self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
            else:
                self.x = np.dot(self.F, self.x)
            self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        def update(self, z: np.ndarray) -> None:
            """Update state from measurement z."""
            z = np.asarray(z, dtype=np.float64).reshape((self.dim_z, 1))
            # Innovation
            y = z - np.dot(self.H, self.x)
            # Innovation covariance
            s = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
            # Kalman gain
            k = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(s))
            # State update
            self.x = self.x + np.dot(k, y)
            # Covariance update (Joseph form for numerical stability)
            i_kh = self._I - np.dot(k, self.H)
            self.P = np.dot(np.dot(i_kh, self.P), i_kh.T) + np.dot(np.dot(k, self.R), k.T)


def bbox_to_z(bbox: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """Convert [x1, y1, x2, y2] bounding box to measurement state [x_center, y_center, area, aspect_ratio].

    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format.

    Returns:
        Measurement vector [x_center, y_center, area, aspect_ratio] as shape (4, 1).
    """
    b = np.asarray(bbox, dtype=np.float64).ravel()
    w = max(1e-4, float(b[2] - b[0]))
    h = max(1e-4, float(b[3] - b[1]))
    x_center = float(b[0]) + w / 2.0
    y_center = float(b[1]) + h / 2.0
    area = w * h
    aspect_ratio = w / h
    return np.array([[x_center], [y_center], [area], [aspect_ratio]], dtype=np.float64)


def x_to_bbox(x: np.ndarray) -> np.ndarray:
    """Convert Kalman state vector [x_center, y_center, area, aspect_ratio, ...] to [x1, y1, x2, y2].

    Handles negative area and negative aspect ratio edge cases.

    Args:
        x: State vector array of shape (8, 1) or (8,).

    Returns:
        Bounding box array [x1, y1, x2, y2] of shape (4,).
    """
    state = np.asarray(x, dtype=np.float64).ravel()
    x_center = state[0]
    y_center = state[1]
    area = state[2]
    aspect_ratio = state[3]

    # Robust handling for non-positive area or aspect ratio
    area = max(1e-4, float(area))
    aspect_ratio = max(1e-4, float(aspect_ratio))

    w = np.sqrt(area * aspect_ratio)
    h = area / w

    x1 = x_center - w / 2.0
    y1 = y_center - h / 2.0
    x2 = x_center + w / 2.0
    y2 = y_center + h / 2.0

    return np.array([x1, y1, x2, y2], dtype=np.float32)


class KalmanBoxTracker:
    """Kalman filter for tracking single 2D bounding boxes with constant velocity model.

    The 8-dimensional state vector is:
        [x_center, y_center, area, aspect_ratio, vx, vy, va, 0]
    The 4-dimensional measurement vector is:
        [x_center, y_center, area, aspect_ratio]
    """

    count: int = 0

    def __init__(
        self,
        bbox: Union[np.ndarray, Sequence[float]],
        process_noise: float = 1.0,
        measurement_noise: float = 1.0,
        dt: float = 1.0,
    ) -> None:
        """Initialize the tracker from an initial [x1, y1, x2, y2] detection.

        Args:
            bbox: Initial bounding box [x1, y1, x2, y2].
            process_noise: Process noise scaling factor.
            measurement_noise: Measurement noise scaling factor.
            dt: Time step between consecutive frames.
        """
        # Define 8 state dimensions and 4 measurement dimensions
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # State Transition Matrix F (constant velocity model)
        # x_center(k+1) = x_center(k) + vx * dt
        # y_center(k+1) = y_center(k) + vy * dt
        # area(k+1)     = area(k) + va * dt
        # aspect_ratio(k+1) = aspect_ratio(k)  (constant aspect ratio assumption)
        self.kf.F = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, dt, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, dt, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        # Measurement Matrix H (maps state to measurement space)
        self.kf.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        # Measurement Covariance Matrix R
        self.kf.R[2:, 2:] *= 10.0
        self.kf.R *= float(measurement_noise)

        # State Covariance Matrix P
        # Give high initial uncertainty to unobserved velocities
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        # Process Covariance Matrix Q
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.Q *= float(process_noise)

        # Initialize state with observation
        z = bbox_to_z(bbox)
        self.kf.x[:4] = z
        self.kf.x[4:] = 0.0

        self.time_since_update: int = 0
        self.id: int = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history: list[np.ndarray] = []
        self.hits: int = 0
        self.hit_streak: int = 0
        self.age: int = 0

    def predict(self) -> np.ndarray:
        """Advance the state vector and return the predicted bounding box estimate.

        Returns:
            Predicted bounding box [x1, y1, x2, y2].
        """
        # Handle edge case where area velocity would drive area below zero
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        # Safeguard state variables against negative values
        self.kf.x[2] = max(1e-4, float(np.asarray(self.kf.x[2]).reshape(-1)[0]))
        self.kf.x[3] = max(1e-4, float(np.asarray(self.kf.x[3]).reshape(-1)[0]))

        predicted_box = x_to_bbox(self.kf.x)
        self.history.append(predicted_box)
        return predicted_box

    def update(self, bbox: Union[np.ndarray, Sequence[float]]) -> None:
        """Update the state vector with an observed bounding box measurement.

        Args:
            bbox: Measured bounding box [x1, y1, x2, y2].
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        z = bbox_to_z(bbox)
        self.kf.update(z)

        # Enforce positive area and aspect ratio
        self.kf.x[2] = max(1e-4, float(np.asarray(self.kf.x[2]).reshape(-1)[0]))
        self.kf.x[3] = max(1e-4, float(np.asarray(self.kf.x[3]).reshape(-1)[0]))

    def get_state(self) -> np.ndarray:
        """Return the current bounding box estimate [x1, y1, x2, y2].

        Returns:
            Current bounding box array of shape (4,).
        """
        return x_to_bbox(self.kf.x)
