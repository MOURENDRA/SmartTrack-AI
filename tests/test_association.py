import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.association import (
    compute_motion_cost, compute_appearance_cost, gate_cost_matrix,
    BaselineAssociator, EnhancedAssociator, associate_detections_to_tracks
)
from src.track_manager import TrackManager, Track, TrackState
from src.motion_model import KalmanBoxTracker
from src.utils import load_config, compute_iou


class TestIoU:
    def test_perfect_overlap(self):
        box = np.array([0, 0, 100, 100])
        assert abs(compute_iou(box, box) - 1.0) < 1e-6
    
    def test_no_overlap(self):
        box1 = np.array([0, 0, 50, 50])
        box2 = np.array([100, 100, 200, 200])
        assert compute_iou(box1, box2) == 0.0
    
    def test_partial_overlap(self):
        box1 = np.array([0, 0, 100, 100])
        box2 = np.array([50, 50, 150, 150])
        iou = compute_iou(box1, box2)
        assert 0 < iou < 1
        # Intersection: 50x50=2500, Union: 10000+10000-2500=17500
        expected = 2500 / 17500
        assert abs(iou - expected) < 1e-6


class TestMotionCost:
    def test_identical_positions(self):
        """Tracks at same position as detections should have zero cost."""
        predictions = [np.array([10, 10, 50, 50])]
        detections = np.array([[10, 10, 50, 50]])
        cost = compute_motion_cost(None, detections, predictions)
        assert cost.shape == (1, 1)
        assert cost[0, 0] < 0.01  # IoU ~1, cost ~0
    
    def test_no_overlap_high_cost(self):
        predictions = [np.array([0, 0, 50, 50])]
        detections = np.array([[200, 200, 300, 300]])
        cost = compute_motion_cost(None, detections, predictions)
        assert cost[0, 0] >= 0.99  # IoU ~0, cost ~1


class TestBaselineAssociator:
    def setup_method(self):
        self.config = load_config()
        self.associator = BaselineAssociator()
    
    def test_empty_detections(self):
        """No detections: all tracks unmatched."""
        # Create a mock track
        track = Track(1, np.array([10, 10, 50, 50]))
        tracks = [track]
        detections = np.empty((0, 4))
        predictions = [track.kalman.predict()]
        
        matches, unmatched_t, unmatched_d = self.associator.associate(
            tracks, detections, predictions
        )
        assert len(matches) == 0
        assert len(unmatched_t) == 1
        assert len(unmatched_d) == 0
    
    def test_empty_tracks(self):
        """No tracks: all detections unmatched."""
        tracks = []
        detections = np.array([[10, 10, 50, 50]])
        predictions = []
        
        matches, unmatched_t, unmatched_d = self.associator.associate(
            tracks, detections, predictions
        )
        assert len(matches) == 0
        assert len(unmatched_t) == 0
        assert len(unmatched_d) == 1
    
    def test_simple_match(self):
        """One track, one nearby detection -> match."""
        track = Track(1, np.array([10, 10, 50, 50]))
        tracks = [track]
        detections = np.array([[12, 12, 52, 52]])
        predictions = [track.kalman.predict()]
        
        matches, unmatched_t, unmatched_d = self.associator.associate(
            tracks, detections, predictions
        )
        assert len(matches) == 1
        assert matches[0] == (0, 0)
    
    def test_reject_distant_match(self):
        """Track far from detection -> no match."""
        track = Track(1, np.array([10, 10, 50, 50]))
        tracks = [track]
        detections = np.array([[500, 500, 600, 600]])
        predictions = [track.kalman.predict()]
        
        matches, unmatched_t, unmatched_d = self.associator.associate(
            tracks, detections, predictions
        )
        assert len(matches) == 0
        assert len(unmatched_t) == 1
        assert len(unmatched_d) == 1
    
    def test_multiple_tracks_and_detections(self):
        """Multiple tracks and detections -> correct assignments."""
        track1 = Track(1, np.array([10, 10, 50, 50]))
        track2 = Track(2, np.array([200, 200, 250, 250]))
        tracks = [track1, track2]
        detections = np.array([
            [202, 202, 252, 252],  # Close to track2
            [12, 12, 52, 52],     # Close to track1
        ])
        predictions = [t.kalman.predict() for t in tracks]
        
        matches, unmatched_t, unmatched_d = self.associator.associate(
            tracks, detections, predictions
        )
        assert len(matches) == 2
        # Track 0 should match detection 1, track 1 should match detection 0
        match_dict = {m[0]: m[1] for m in matches}
        assert match_dict[0] == 1  # track1 -> det near track1
        assert match_dict[1] == 0  # track2 -> det near track2


class TestEnhancedAssociator:
    def setup_method(self):
        self.config = load_config()
    
    def test_uses_appearance_when_available(self):
        """Enhanced associator should work with appearance features."""
        track = Track(1, np.array([10, 10, 50, 50]))
        track.appearance_feature = np.random.randn(80)
        track.appearance_feature /= np.linalg.norm(track.appearance_feature)
        tracks = [track]
        detections = np.array([[12, 12, 52, 52]])
        predictions = [track.kalman.predict()]
        track_features = [track.appearance_feature]
        detection_features = [track.appearance_feature.copy()]  # Same feature
        
        associator = EnhancedAssociator()
        matches, _, _ = associator.associate(
            tracks, detections, predictions,
            track_features, detection_features, self.config
        )
        assert len(matches) == 1


class TestDispatcher:
    def test_baseline_mode(self):
        config = load_config()
        track = Track(1, np.array([10, 10, 50, 50]))
        tracks = [track]
        detections = np.array([[12, 12, 52, 52]])
        predictions = [track.kalman.predict()]
        
        matches, _, _ = associate_detections_to_tracks(
            tracks, detections, predictions, mode='baseline'
        )
        assert len(matches) == 1
    
    def test_enhanced_mode(self):
        config = load_config()
        track = Track(1, np.array([10, 10, 50, 50]))
        track.appearance_feature = np.random.randn(80)
        track.appearance_feature /= np.linalg.norm(track.appearance_feature)
        tracks = [track]
        detections = np.array([[12, 12, 52, 52]])
        predictions = [track.kalman.predict()]
        
        matches, _, _ = associate_detections_to_tracks(
            tracks, detections, predictions, 
            track_features=[track.appearance_feature],
            detection_features=[track.appearance_feature.copy()],
            config=config, mode='enhanced'
        )
        assert len(matches) == 1
