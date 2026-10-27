"""Track lifecycle and state management engine for SmartTrack-AI.

Manages individual tracks through tentative, confirmed, lost, and removed states,
governing Kalman filter predictions, appearance EMA updates, and recovery.
"""

from __future__ import annotations

import collections
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .motion_model import KalmanBoxTracker

logger = logging.getLogger("smarttrack")


class TrackState(Enum):
    """Enumeration of track lifecycle states."""

    TENTATIVE = 1
    CONFIRMED = 2
    LOST = 3
    REMOVED = 4


class Track:
    """Represents a single tracked object throughout its lifecycle."""

    def __init__(
        self,
        track_id: int,
        bbox: Union[np.ndarray, Sequence[float]],
        feature: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
    ) -> None:
        """Initialize a new track.

        Args:
            track_id: Unique integer identifier for this track.
            bbox: Initial detection bounding box [x1, y1, x2, y2].
            feature: Initial appearance feature vector.
            config: Optional configuration dictionary.
            confidence: Initial detection confidence.
        """
        self.track_id: int = int(track_id)
        self.state: TrackState = TrackState.TENTATIVE
        self.config: Dict[str, Any] = config or {}

        motion_cfg = self.config.get("motion", {})
        process_noise = float(motion_cfg.get("process_noise", 1.0))
        measurement_noise = float(motion_cfg.get("measurement_noise", 1.0))
        dt = float(motion_cfg.get("dt", 1.0 / 30.0))

        # Initialize Kalman filter motion model
        self.kalman: KalmanBoxTracker = KalmanBoxTracker(
            bbox=bbox,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            dt=dt,
        )

        self.bbox: np.ndarray = np.asarray(bbox, dtype=np.float32).copy()
        self.predicted_bbox: np.ndarray = self.bbox.copy()

        # Appearance feature with optional L2 normalization
        self.appearance_feature: Optional[np.ndarray] = None
        if feature is not None:
            feat_arr = np.asarray(feature, dtype=np.float32).copy()
            norm = np.linalg.norm(feat_arr)
            self.appearance_feature = feat_arr / norm if norm > 1e-7 else feat_arr

        # Lifetime counters and metrics
        self.age: int = 0
        self.hits: int = 1
        self.time_since_update: int = 0
        self.hit_streak: int = 1
        self.confidence: float = float(confidence)

        # Bounding box trajectory history (max 60 frames)
        self.history: collections.deque[np.ndarray] = collections.deque(maxlen=60)
        self.history.append(self.bbox.copy())

        memory_cfg = self.config.get("track_memory", {})
        self.appearance_decay: float = float(memory_cfg.get("appearance_decay", 0.95))

    def predict(self) -> np.ndarray:
        """Predict the next bounding box position using the Kalman filter.

        Returns:
            Predicted bounding box [x1, y1, x2, y2].
        """
        self.predicted_bbox = self.kalman.predict()
        self.age += 1
        return self.predicted_bbox

    def update(
        self,
        bbox: Union[np.ndarray, Sequence[float]],
        feature: Optional[np.ndarray] = None,
        confidence: float = 1.0,
    ) -> None:
        """Update track state with a new matched observation.

        Args:
            bbox: Measured bounding box [x1, y1, x2, y2].
            feature: Measured appearance feature vector.
            confidence: Detection confidence score.
        """
        self.kalman.update(bbox)
        self.bbox = np.asarray(bbox, dtype=np.float32).copy()
        self.history.append(self.bbox.copy())

        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.confidence = float(confidence)

        # Exponential Moving Average (EMA) update of appearance feature
        if feature is not None:
            new_feat = np.asarray(feature, dtype=np.float32)
            norm = np.linalg.norm(new_feat)
            if norm > 1e-7:
                new_feat = new_feat / norm

            if self.appearance_feature is not None:
                updated_feat = (
                    self.appearance_decay * self.appearance_feature
                    + (1.0 - self.appearance_decay) * new_feat
                )
                u_norm = np.linalg.norm(updated_feat)
                if u_norm > 1e-7:
                    updated_feat /= u_norm
                self.appearance_feature = updated_feat
            else:
                self.appearance_feature = new_feat

    def mark_missed(self) -> None:
        """Increment time since last update, reset hit streak, and transition state."""
        self.time_since_update += 1
        self.hit_streak = 0

        if self.state == TrackState.CONFIRMED:
            if self.time_since_update > 1:
                self.state = TrackState.LOST
        elif self.state == TrackState.TENTATIVE:
            if self.time_since_update > 1:
                self.state = TrackState.REMOVED

    def is_confirmed(self) -> bool:
        """Check if track is currently CONFIRMED."""
        return self.state == TrackState.CONFIRMED

    def is_lost(self) -> bool:
        """Check if track is currently LOST."""
        return self.state == TrackState.LOST

    def is_removed(self) -> bool:
        """Check if track is marked REMOVED."""
        return self.state == TrackState.REMOVED

    def should_remove(self, max_age: int) -> bool:
        """Determine whether the track should be pruned and removed.

        Args:
            max_age: Maximum allowed consecutive missed frames.

        Returns:
            True if track exceeded allowable age or is already marked REMOVED.
        """
        if self.state == TrackState.REMOVED:
            return True
        if self.state == TrackState.TENTATIVE and self.time_since_update > 1:
            return True
        if self.time_since_update > max_age:
            return True
        return False

    def get_predicted_bbox(self) -> np.ndarray:
        """Get the latest predicted bounding box coordinates.

        Returns:
            Array [x1, y1, x2, y2].
        """
        if self.predicted_bbox is not None:
            return self.predicted_bbox
        return self.kalman.get_state()


class TrackManager:
    """Manages collection of active, lost, and tentative tracks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the track manager.

        Args:
            config: Optional configuration dictionary.
        """
        self.config: Dict[str, Any] = config or {}
        self._next_id: int = 1
        self.tracks: List[Track] = []

        tracker_cfg = self.config.get("tracker", {})
        memory_cfg = self.config.get("track_memory", {})

        self.max_age: int = int(tracker_cfg.get("max_age", 30))
        self.min_hits: int = int(tracker_cfg.get("min_hits", 3))
        self.max_lost_age: int = int(memory_cfg.get("max_lost_age", 60))
        self.min_recovery_confidence: float = float(
            memory_cfg.get("min_recovery_confidence", 0.5)
        )

    def create_track(
        self,
        bbox: Union[np.ndarray, Sequence[float]],
        feature: Optional[np.ndarray] = None,
        confidence: float = 1.0,
    ) -> Track:
        """Instantiate and register a new tentative track.

        Args:
            bbox: Detection bounding box [x1, y1, x2, y2].
            feature: Extracted appearance feature.
            confidence: Detection confidence.

        Returns:
            Newly created Track instance.
        """
        track = Track(
            track_id=self._next_id,
            bbox=bbox,
            feature=feature,
            config=self.config,
            confidence=confidence,
        )
        self._next_id += 1
        self.tracks.append(track)
        return track

    def predict_all(self) -> List[np.ndarray]:
        """Advance Kalman filters for all non-removed tracks.

        Returns:
            List of predicted bounding boxes corresponding to self.tracks.
        """
        predictions: List[np.ndarray] = []
        for track in self.tracks:
            if not track.is_removed():
                pred = track.predict()
                predictions.append(pred)
            else:
                predictions.append(track.bbox)
        return predictions

    def update_tracks(
        self,
        matches: Sequence[Tuple[int, int]],
        unmatched_tracks: Sequence[int],
        unmatched_detections: Sequence[int],
        detections: Union[np.ndarray, Sequence[Sequence[float]]],
        features: Optional[Sequence[Optional[np.ndarray]]] = None,
        confidences: Optional[Sequence[float]] = None,
    ) -> None:
        """Update track lifecycle based on association results.

        Args:
            matches: Sequence of (track_idx, detection_idx) pairs.
            unmatched_tracks: Indices of unmatched tracks.
            unmatched_detections: Indices of unmatched detections.
            detections: Array of detections [M, 4].
            features: Optional list of appearance features for detections.
            confidences: Optional list of detection confidences.
        """
        dets_arr = np.asarray(detections, dtype=np.float32)
        n_dets = len(dets_arr)

        if confidences is None:
            det_confs = [1.0] * n_dets
        else:
            det_confs = list(confidences)

        # 1. Update matched tracks
        for t_idx, d_idx in matches:
            if t_idx < len(self.tracks):
                track = self.tracks[t_idx]
                det_box = dets_arr[d_idx]
                feat = features[d_idx] if features is not None else None
                conf = det_confs[d_idx]

                track.update(bbox=det_box, feature=feat, confidence=conf)

                # Promote tentative track to confirmed after reaching min_hits
                if track.state == TrackState.TENTATIVE:
                    if track.hit_streak >= self.min_hits or track.hits >= self.min_hits:
                        track.state = TrackState.CONFIRMED
                elif track.state == TrackState.LOST:
                    # Recover lost track if confidence passes minimum threshold
                    if conf >= self.min_recovery_confidence:
                        track.state = TrackState.CONFIRMED

        # 2. Process unmatched tracks
        for t_idx in unmatched_tracks:
            if t_idx < len(self.tracks):
                track = self.tracks[t_idx]
                track.mark_missed()

                # Transition to REMOVED if lost age or max age exceeded
                if track.time_since_update > self.max_lost_age or track.should_remove(self.max_age):
                    track.state = TrackState.REMOVED

        # 3. Initialize new tentative tracks for unmatched detections
        for d_idx in unmatched_detections:
            if d_idx < n_dets:
                det_box = dets_arr[d_idx]
                feat = features[d_idx] if features is not None else None
                conf = det_confs[d_idx]
                self.create_track(bbox=det_box, feature=feat, confidence=conf)

        # 4. Prune removed tracks
        self.remove_dead_tracks()

    def remove_dead_tracks(self) -> None:
        """Remove all REMOVED tracks from the internal tracking list."""
        self.tracks = [t for t in self.tracks if not t.is_removed()]

    def get_active_tracks(self) -> List[Track]:
        """Return list of active tracks (CONFIRMED and TENTATIVE).

        Returns:
            List of active Track instances.
        """
        return [
            t
            for t in self.tracks
            if t.state in (TrackState.CONFIRMED, TrackState.TENTATIVE)
        ]

    def get_lost_tracks(self) -> List[Track]:
        """Return list of LOST tracks currently retained in memory.

        Returns:
            List of lost Track instances.
        """
        return [t for t in self.tracks if t.is_lost()]

    def get_all_tracks(self) -> List[Track]:
        """Return all managed non-removed tracks (CONFIRMED, TENTATIVE, and LOST).

        Returns:
            List of all non-removed Track instances.
        """
        return [t for t in self.tracks if not t.is_removed()]

    def get_track_statistics(self) -> Dict[str, int]:
        """Compute summary statistics of managed tracks by lifecycle state.

        Returns:
            Dictionary with counts for tentative, confirmed, lost, removed, and total tracks.
        """
        tentative = sum(1 for t in self.tracks if t.state == TrackState.TENTATIVE)
        confirmed = sum(1 for t in self.tracks if t.state == TrackState.CONFIRMED)
        lost = sum(1 for t in self.tracks if t.state == TrackState.LOST)
        removed = sum(1 for t in self.tracks if t.state == TrackState.REMOVED)

        return {
            "tentative": tentative,
            "confirmed": confirmed,
            "lost": lost,
            "removed": removed,
            "total_active": tentative + confirmed,
            "total_managed": len(self.tracks),
        }
