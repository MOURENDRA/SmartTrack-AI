"""Data association modules using Hungarian assignment for SmartTrack-AI.

Implements baseline motion-only (IoU) association and enhanced multi-cue
(appearance + motion + confidence gating) two-stage association.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .utils import compute_iou_matrix

logger = logging.getLogger("smarttrack")

# Robust import for linear_sum_assignment with greedy fallback
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    logger.warning("scipy not installed; using greedy matching fallback for Hungarian assignment.")

    def linear_sum_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore[no-redef]
        """Greedy approximation fallback for linear sum assignment."""
        costs = np.array(cost_matrix, copy=True)
        rows, cols = costs.shape
        matched_rows: List[int] = []
        matched_cols: List[int] = []

        for _ in range(min(rows, cols)):
            min_val = np.min(costs)
            if min_val >= 1e4:
                break
            min_idx = np.unravel_index(np.argmin(costs), costs.shape)
            r, c = int(min_idx[0]), int(min_idx[1])
            matched_rows.append(r)
            matched_cols.append(c)
            costs[r, :] = 1e5
            costs[:, c] = 1e5

        return np.array(matched_rows, dtype=np.int64), np.array(matched_cols, dtype=np.int64)


def compute_motion_cost(
    tracks: Sequence[Any],
    detections: Union[np.ndarray, Sequence[Sequence[float]]],
    track_predictions: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
) -> np.ndarray:
    """Compute motion cost matrix (1.0 - IoU) between track predictions and detections.

    Args:
        tracks: List of Track objects.
        detections: Bounding boxes of detections [M, 4].
        track_predictions: Optional precomputed predicted bounding boxes [N, 4].

    Returns:
        Cost matrix of shape [N, M] with values in [0.0, 1.0].
    """
    # Predictions are sufficient to calculate the cost matrix; callers may pass
    # tracks=None when they only have precomputed predicted boxes available.
    if tracks is None:
        n_tracks = len(track_predictions) if track_predictions is not None else 0
    else:
        n_tracks = len(tracks)
    n_dets = len(detections) if detections is not None else 0
    if n_tracks == 0 or n_dets == 0:
        return np.zeros((n_tracks, n_dets), dtype=np.float32)

    if track_predictions is not None and len(track_predictions) == n_tracks:
        boxes_tracks = np.asarray(track_predictions, dtype=np.float32)
    elif tracks is not None:
        boxes_list = []
        for t in tracks:
            if hasattr(t, "get_predicted_bbox"):
                boxes_list.append(t.get_predicted_bbox())
            elif hasattr(t, "bbox"):
                boxes_list.append(t.bbox)
            else:
                boxes_list.append(t)
        boxes_tracks = np.asarray(boxes_list, dtype=np.float32)
    else:
        # No tracks or matching predictions were supplied; return a correctly
        # shaped empty matrix rather than raising while inspecting tracks.
        return np.zeros((n_tracks, n_dets), dtype=np.float32)

    boxes_dets = np.asarray(detections, dtype=np.float32)
    iou_matrix = compute_iou_matrix(boxes_tracks, boxes_dets)
    cost_matrix = 1.0 - iou_matrix
    return cost_matrix.astype(np.float32)


def compute_appearance_cost(
    track_features: Optional[Sequence[Optional[np.ndarray]]],
    detection_features: Optional[Sequence[Optional[np.ndarray]]],
) -> np.ndarray:
    """Compute appearance cost matrix (cosine distance) between track and detection features.

    Args:
        track_features: Sequence of N track feature vectors.
        detection_features: Sequence of M detection feature vectors.

    Returns:
        Cost matrix of shape [N, M] with cosine distances in [0.0, 2.0].
    """
    n_tracks = len(track_features) if track_features is not None else 0
    n_dets = len(detection_features) if detection_features is not None else 0
    if n_tracks == 0 or n_dets == 0:
        return np.zeros((n_tracks, n_dets), dtype=np.float32)

    cost_matrix = np.ones((n_tracks, n_dets), dtype=np.float32)

    # Filter out empty or None features
    valid_t_indices = [
        i for i, f in enumerate(track_features) if f is not None and len(f) > 0
    ]
    valid_d_indices = [
        j for j, f in enumerate(detection_features) if f is not None and len(f) > 0
    ]

    if not valid_t_indices or not valid_d_indices:
        return cost_matrix

    tf_sub = np.asarray([track_features[i] for i in valid_t_indices], dtype=np.float32)
    df_sub = np.asarray([detection_features[j] for j in valid_d_indices], dtype=np.float32)

    # Normalize vectors
    norm_t = np.linalg.norm(tf_sub, axis=1, keepdims=True)
    norm_d = np.linalg.norm(df_sub, axis=1, keepdims=True)
    tf_norm = np.where(norm_t > 1e-7, tf_sub / np.maximum(norm_t, 1e-7), 0.0)
    df_norm = np.where(norm_d > 1e-7, df_sub / np.maximum(norm_d, 1e-7), 0.0)

    sim = np.dot(tf_norm, df_norm.T)
    dist = np.clip(1.0 - sim, 0.0, 2.0)

    for sub_r, r in enumerate(valid_t_indices):
        for sub_c, c in enumerate(valid_d_indices):
            cost_matrix[r, c] = dist[sub_r, sub_c]

    return cost_matrix


def gate_cost_matrix(
    cost_matrix: np.ndarray,
    tracks: Sequence[Any],
    detections: Union[np.ndarray, Sequence[Sequence[float]]],
    gate_threshold: float = 0.7,
    track_predictions: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
) -> np.ndarray:
    """Apply spatial and threshold gating to the cost matrix.

    Entries exceeding the threshold or failing Mahalanobis-like spatial bounds
    are set to a large penalizing value (1e5).

    Args:
        cost_matrix: Cost matrix of shape [N, M].
        tracks: Sequence of N Track objects.
        detections: Sequence of M bounding box detections [M, 4].
        gate_threshold: Maximum allowed cost value.
        track_predictions: Optional precomputed predicted bounding boxes [N, 4].

    Returns:
        Gated cost matrix copy of shape [N, M].
    """
    gated = np.array(cost_matrix, copy=True, dtype=np.float32)
    n_tracks, n_dets = gated.shape
    if n_tracks == 0 or n_dets == 0:
        return gated

    # Extract boxes for spatial distance check
    if track_predictions is not None and len(track_predictions) == n_tracks:
        t_boxes = np.asarray(track_predictions, dtype=np.float32)
    else:
        t_boxes = np.asarray(
            [
                t.get_predicted_bbox() if hasattr(t, "get_predicted_bbox")
                else (t.bbox if hasattr(t, "bbox") else t)
                for t in tracks
            ],
            dtype=np.float32,
        )

    d_boxes = np.asarray(detections, dtype=np.float32)

    # 1. Cost threshold gating
    gated[gated > gate_threshold] = 1e5

    # 2. Mahalanobis-like spatial gating based on normalized center distance
    t_w = np.maximum(1e-4, t_boxes[:, 2] - t_boxes[:, 0])
    t_h = np.maximum(1e-4, t_boxes[:, 3] - t_boxes[:, 1])
    t_cx = t_boxes[:, 0] + t_w / 2.0
    t_cy = t_boxes[:, 1] + t_h / 2.0

    d_w = np.maximum(1e-4, d_boxes[:, 2] - d_boxes[:, 0])
    d_h = np.maximum(1e-4, d_boxes[:, 3] - d_boxes[:, 1])
    d_cx = d_boxes[:, 0] + d_w / 2.0
    d_cy = d_boxes[:, 1] + d_h / 2.0

    for i in range(n_tracks):
        # Normalized squared center distance relative to track scale
        dx = (d_cx - t_cx[i]) / t_w[i]
        dy = (d_cy - t_cy[i]) / t_h[i]
        dist_sq = dx * dx + dy * dy

        # Discard pairings if centers are separated by more than 4 box dimensions
        gated[i, dist_sq > 16.0] = 1e5

    return gated


class BaselineAssociator:
    """Baseline tracker association engine utilizing IoU motion cost only."""

    def __init__(self, iou_threshold: float = 0.3) -> None:
        """Initialize BaselineAssociator.

        Args:
            iou_threshold: Minimum IoU required to validate a match.
        """
        self.iou_threshold = iou_threshold

    def associate(
        self,
        tracks: Sequence[Any],
        detections: Union[np.ndarray, Sequence[Sequence[float]]],
        track_predictions: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Perform Hungarian matching based on IoU overlap.

        Args:
            tracks: List of active Track instances.
            detections: Bounding boxes [M, 4].
            track_predictions: Optional predicted track boxes [N, 4].
            config: Optional configuration dictionary.

        Returns:
            Tuple of:
                - matches: List of (track_idx, detection_idx)
                - unmatched_tracks: List of track indices
                - unmatched_detections: List of detection indices
        """
        n_tracks = len(tracks)
        n_dets = len(detections)

        if n_tracks == 0:
            return [], [], list(range(n_dets))
        if n_dets == 0:
            return [], list(range(n_tracks)), []

        iou_threshold = self.iou_threshold
        if config:
            iou_threshold = float(
                config.get("tracker", {}).get("iou_threshold", self.iou_threshold)
            )

        cost_matrix = compute_motion_cost(tracks, detections, track_predictions)

        # Apply simple gating: cost > (1 - iou_threshold) is invalid
        max_cost = 1.0 - iou_threshold
        cost_matrix_gated = np.array(cost_matrix, copy=True)
        cost_matrix_gated[cost_matrix_gated > max_cost] = 1e5

        row_indices, col_indices = linear_sum_assignment(cost_matrix_gated)

        matches: List[Tuple[int, int]] = []
        matched_track_set = set()
        matched_det_set = set()

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] <= max_cost:
                matches.append((int(r), int(c)))
                matched_track_set.add(int(r))
                matched_det_set.add(int(c))

        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_track_set]
        unmatched_detections = [j for j in range(n_dets) if j not in matched_det_set]

        return matches, unmatched_tracks, unmatched_detections


class EnhancedAssociator:
    """Enhanced multi-cue association combining motion, appearance, and confidence gating."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EnhancedAssociator.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        assoc_cfg = self.config.get("association", {})
        self.lambda_iou: float = float(assoc_cfg.get("lambda_iou", 0.5))
        self.lambda_appearance: float = float(assoc_cfg.get("lambda_appearance", 0.5))
        self.gate_threshold: float = float(assoc_cfg.get("gate_threshold", 0.7))
        self.max_cosine_distance: float = float(assoc_cfg.get("max_cosine_distance", 0.4))

    def _match_subset(
        self,
        track_indices: List[int],
        detection_indices: List[int],
        tracks: Sequence[Any],
        detections: np.ndarray,
        track_predictions: Optional[np.ndarray],
        track_features: Optional[Sequence[Optional[np.ndarray]]],
        detection_features: Optional[Sequence[Optional[np.ndarray]]],
        gate_thresh: float,
        detection_confidences: Optional[Sequence[float]] = None,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match a sub-collection of tracks and detections."""
        if not track_indices or not detection_indices:
            return [], list(track_indices), list(detection_indices)

        sub_tracks = [tracks[i] for i in track_indices]
        sub_dets = detections[detection_indices]
        sub_preds = (
            track_predictions[track_indices] if track_predictions is not None else None
        )

        sub_t_feats = (
            [track_features[i] for i in track_indices] if track_features is not None else None
        )
        sub_d_feats = (
            [detection_features[j] for j in detection_indices]
            if detection_features is not None
            else None
        )

        # 1. Motion cost (1 - IoU)
        motion_cost = compute_motion_cost(sub_tracks, sub_dets, sub_preds)

        # 2. Appearance cost (cosine distance)
        appearance_cost = compute_appearance_cost(sub_t_feats, sub_d_feats)

        # 3. Combined cost
        combined_cost = (
            self.lambda_iou * motion_cost + self.lambda_appearance * appearance_cost
        )

        # 4. Gating with spatial & confidence-aware threshold
        gated_cost = gate_cost_matrix(
            combined_cost, sub_tracks, sub_dets, gate_threshold=gate_thresh, track_predictions=sub_preds
        )

        # Disallow pairing if appearance distance alone exceeds max_cosine_distance
        if sub_t_feats is not None and sub_d_feats is not None:
            appearance_mask = appearance_cost > self.max_cosine_distance
            # Only enforce if appearance is valid and non-zero
            valid_appearance_mask = (appearance_cost > 0.0) & (appearance_cost < 1.99)
            gated_cost[appearance_mask & valid_appearance_mask] = 1e5

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(gated_cost)

        matches: List[Tuple[int, int]] = []
        matched_t_set = set()
        matched_d_set = set()

        for r, c in zip(row_ind, col_ind):
            if gated_cost[r, c] < 1e4:
                orig_t = track_indices[r]
                orig_d = detection_indices[c]
                matches.append((orig_t, orig_d))
                matched_t_set.add(orig_t)
                matched_d_set.add(orig_d)

        unmatched_t = [i for i in track_indices if i not in matched_t_set]
        unmatched_d = [j for j in detection_indices if j not in matched_d_set]

        return matches, unmatched_t, unmatched_d

    def associate(
        self,
        tracks: Sequence[Any],
        detections: Union[np.ndarray, Sequence[Sequence[float]]],
        track_predictions: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
        track_features: Optional[Sequence[Optional[np.ndarray]]] = None,
        detection_features: Optional[Sequence[Optional[np.ndarray]]] = None,
        config: Optional[Dict[str, Any]] = None,
        detection_confidences: Optional[Sequence[float]] = None,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Perform two-stage confidence-aware association.

        Stage 1: Match CONFIRMED tracks with high-confidence detections.
        Stage 2: Match remaining tracks (TENTATIVE + LOST + unmatched CONFIRMED)
                 with remaining detections using relaxed recovery thresholds.

        Args:
            tracks: Sequence of active Track instances.
            detections: Bounding box detections [M, 4].
            track_predictions: Optional predicted track boxes [N, 4].
            track_features: Optional appearance features for tracks.
            detection_features: Optional appearance features for detections.
            config: Optional configuration dictionary.
            detection_confidences: Optional confidence scores for detections [M].

        Returns:
            Tuple of:
                - matches: List of (track_idx, detection_idx) pairs.
                - unmatched_tracks: List of unmatched track indices.
                - unmatched_detections: List of unmatched detection indices.
        """
        n_tracks = len(tracks)
        n_dets = len(detections)

        if n_tracks == 0:
            return [], [], list(range(n_dets))
        if n_dets == 0:
            return [], list(range(n_tracks)), []

        dets_arr = np.asarray(detections, dtype=np.float32)
        preds_arr = (
            np.asarray(track_predictions, dtype=np.float32)
            if track_predictions is not None
            else None
        )

        conf_threshold = 0.5
        if config:
            conf_threshold = float(
                config.get("detector", {}).get("confidence_threshold", 0.5)
            )

        if detection_confidences is None:
            detection_confidences = [1.0] * n_dets

        # Partition detections: high confidence vs low confidence
        high_det_indices = [
            j for j, c in enumerate(detection_confidences) if c >= conf_threshold
        ]
        low_det_indices = [
            j for j, c in enumerate(detection_confidences) if c < conf_threshold
        ]

        # Partition tracks: confirmed vs tentative/lost
        confirmed_track_indices = [
            i for i, t in enumerate(tracks)
            if getattr(t, "is_confirmed", lambda: False)()
        ]
        other_track_indices = [
            i for i in range(n_tracks) if i not in confirmed_track_indices
        ]

        # Stage 1: Associate confirmed tracks with high-confidence detections
        stage1_matches, s1_unmatched_t, s1_unmatched_d = self._match_subset(
            track_indices=confirmed_track_indices,
            detection_indices=high_det_indices,
            tracks=tracks,
            detections=dets_arr,
            track_predictions=preds_arr,
            track_features=track_features,
            detection_features=detection_features,
            gate_thresh=self.gate_threshold,
            detection_confidences=detection_confidences,
        )

        # Stage 2: Match remaining tracks with remaining detections using relaxed gate
        remaining_t_indices = list(s1_unmatched_t) + other_track_indices
        remaining_d_indices = list(s1_unmatched_d) + low_det_indices

        # Relax gate threshold slightly for track recovery
        relaxed_gate = min(0.9, self.gate_threshold + 0.15)

        stage2_matches, final_unmatched_t, final_unmatched_d = self._match_subset(
            track_indices=remaining_t_indices,
            detection_indices=remaining_d_indices,
            tracks=tracks,
            detections=dets_arr,
            track_predictions=preds_arr,
            track_features=track_features,
            detection_features=detection_features,
            gate_thresh=relaxed_gate,
            detection_confidences=detection_confidences,
        )

        all_matches = stage1_matches + stage2_matches
        return all_matches, final_unmatched_t, final_unmatched_d


def associate_detections_to_tracks(
    tracks: Sequence[Any],
    detections: Union[np.ndarray, Sequence[Sequence[float]]],
    track_predictions: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
    track_features: Optional[Sequence[Optional[np.ndarray]]] = None,
    detection_features: Optional[Sequence[Optional[np.ndarray]]] = None,
    config: Optional[Dict[str, Any]] = None,
    mode: str = "enhanced",
    detection_confidences: Optional[Sequence[float]] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Dispatcher function for associating detections to tracks.

    Args:
        tracks: Sequence of active Track instances.
        detections: Bounding box detections [M, 4].
        track_predictions: Optional predicted track boxes [N, 4].
        track_features: Optional appearance features for tracks.
        detection_features: Optional appearance features for detections.
        config: Configuration dictionary.
        mode: Matching mode, either 'baseline' or 'enhanced'.
        detection_confidences: Optional detection confidence scores [M].

    Returns:
        Tuple of (matches, unmatched_tracks, unmatched_detections).
    """
    n_tracks = len(tracks)
    n_dets = len(detections)

    if n_tracks == 0:
        return [], [], list(range(n_dets))
    if n_dets == 0:
        return [], list(range(n_tracks)), []

    selected_mode = mode
    if config:
        selected_mode = config.get("tracker", {}).get("mode", mode)

    if selected_mode.lower() == "baseline":
        associator = BaselineAssociator()
        return associator.associate(
            tracks=tracks,
            detections=detections,
            track_predictions=track_predictions,
            config=config,
        )
    else:
        enhanced_associator = EnhancedAssociator(config=config)
        return enhanced_associator.associate(
            tracks=tracks,
            detections=detections,
            track_predictions=track_predictions,
            track_features=track_features,
            detection_features=detection_features,
            config=config,
            detection_confidences=detection_confidences,
        )
