import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tracker import SmartTracker
from src.track_manager import TrackManager, Track, TrackState
from src.association import EnhancedAssociator, associate_detections_to_tracks
from src.utils import load_config


class TestReassociation:
    def setup_method(self):
        self.config = load_config()
        self.config['tracker']['mode'] = 'enhanced'
        self.config['tracker']['min_hits'] = 2
        self.config['tracker']['max_age'] = 30
        self.config['track_memory']['max_lost_age'] = 30
        self.config['track_memory']['min_recovery_confidence'] = 0.5
        self.frame_shape = (480, 640, 3)
        self.dummy_frame = np.zeros(self.frame_shape, dtype=np.uint8)

    def test_lost_track_recovery(self):
        """A track goes lost, same detection returns -> re-associated with same ID."""
        tracker = SmartTracker(self.config)

        box = np.array([[50.0, 50.0, 100.0, 100.0]])
        conf = np.array([0.9])

        # Frame 1: Create tentative track
        tracks = tracker.update(self.dummy_frame, box, conf)
        initial_id = tracker.track_manager.tracks[0].track_id

        # Frame 2: Confirm track (min_hits=2)
        box2 = np.array([[52.0, 50.0, 102.0, 100.0]])
        tracks = tracker.update(self.dummy_frame, box2, conf)
        assert len(tracks) == 1
        assert tracks[0].track_id == initial_id
        assert tracks[0].is_confirmed()

        # Frames 3 & 4: Occlusion / missed detections -> track becomes LOST
        empty_boxes = np.empty((0, 4))
        empty_confs = np.empty((0,))
        tracker.update(self.dummy_frame, empty_boxes, empty_confs)
        tracker.update(self.dummy_frame, empty_boxes, empty_confs)

        lost_tracks = tracker.track_manager.get_lost_tracks()
        assert len(lost_tracks) == 1
        assert lost_tracks[0].track_id == initial_id
        assert lost_tracks[0].is_lost()

        # Frame 5: Target reappears near last location
        reappearing_box = np.array([[54.0, 50.0, 104.0, 100.0]])
        tracks = tracker.update(self.dummy_frame, reappearing_box, conf)

        # Track should be recovered with the same track ID
        assert len(tracks) == 1
        assert tracks[0].track_id == initial_id
        assert tracks[0].is_confirmed()

    def test_lost_track_appearance_match(self):
        """Use appearance features to recover lost track."""
        config = load_config()
        config['tracker']['mode'] = 'enhanced'
        associator = EnhancedAssociator(config)

        # Create a track with a distinct 80-dim feature vector
        feat_target = np.zeros(80, dtype=np.float32)
        feat_target[:40] = 1.0
        feat_target /= np.linalg.norm(feat_target)

        track = Track(1, np.array([50.0, 50.0, 100.0, 100.0]), feature=feat_target, config=config)
        track.state = TrackState.LOST
        tracks = [track]
        predictions = [track.kalman.predict()]

        # Detection with identical appearance feature
        det_matching = np.array([[55.0, 50.0, 105.0, 100.0]])
        det_feat_matching = [feat_target.copy()]

        matches, unmatched_t, unmatched_d = associator.associate(
            tracks=tracks,
            detections=det_matching,
            track_predictions=predictions,
            track_features=[track.appearance_feature],
            detection_features=det_feat_matching,
            config=config,
            detection_confidences=[0.9]
        )
        assert len(matches) == 1
        assert matches[0] == (0, 0)
        assert len(unmatched_t) == 0

        # Detection with very different appearance feature (orthogonal)
        feat_different = np.zeros(80, dtype=np.float32)
        feat_different[40:] = 1.0
        feat_different /= np.linalg.norm(feat_different)
        det_feat_diff = [feat_different]

        matches_diff, unmatched_t_diff, unmatched_d_diff = associator.associate(
            tracks=tracks,
            detections=det_matching,
            track_predictions=predictions,
            track_features=[track.appearance_feature],
            detection_features=det_feat_diff,
            config=config,
            detection_confidences=[0.9]
        )
        # Cosine distance = 1.0 > max_cosine_distance (0.4) -> rejected
        assert len(matches_diff) == 0
        assert len(unmatched_t_diff) == 1

    def test_no_false_recovery(self):
        """A very different detection should NOT recover a lost track."""
        tracker = SmartTracker(self.config)

        box = np.array([[50.0, 50.0, 100.0, 100.0]])
        conf = np.array([0.9])

        # Initialize and confirm track
        tracker.update(self.dummy_frame, box, conf)
        tracker.update(self.dummy_frame, box, conf)
        initial_id = tracker.track_manager.tracks[0].track_id

        # Make track go lost
        empty_boxes = np.empty((0, 4))
        empty_confs = np.empty((0,))
        tracker.update(self.dummy_frame, empty_boxes, empty_confs)
        tracker.update(self.dummy_frame, empty_boxes, empty_confs)
        assert len(tracker.track_manager.get_lost_tracks()) == 1

        # Completely distant detection arrives
        distant_box = np.array([[400.0, 400.0, 460.0, 460.0]])
        active_tracks = tracker.update(self.dummy_frame, distant_box, conf)

        # Distant detection must NOT recover the initial lost track
        # It creates a new track with a different ID
        assert len(active_tracks) == 1
        assert active_tracks[0].track_id != initial_id

    def test_track_memory_expiry(self):
        """After max_lost_age, track is removed and cannot be recovered."""
        config = load_config()
        config['tracker']['mode'] = 'enhanced'
        config['tracker']['min_hits'] = 1
        config['track_memory']['max_lost_age'] = 3
        config['tracker']['max_age'] = 3
        tracker = SmartTracker(config)

        box = np.array([[50.0, 50.0, 100.0, 100.0]])
        conf = np.array([0.9])

        # Frame 1: Create and confirm track
        tracker.update(self.dummy_frame, box, conf)
        tracker.track_manager.tracks[0].state = TrackState.CONFIRMED
        initial_id = tracker.track_manager.tracks[0].track_id

        # Miss frames for max_lost_age + 1 frames
        empty_boxes = np.empty((0, 4))
        empty_confs = np.empty((0,))
        for _ in range(4):
            tracker.update(self.dummy_frame, empty_boxes, empty_confs)

        # Memory expired; track should be pruned
        assert len(tracker.track_manager.tracks) == 0

        # Now target reappears at the exact same location
        tracks = tracker.update(self.dummy_frame, box, conf)
        # Because track memory expired, a brand new track ID must be assigned
        assert len(tracks) == 1
        assert tracks[0].track_id != initial_id

    def test_multiple_tracks_recover_correctly(self):
        """Two tracks go lost, both return -> correct ID assignment."""
        tracker = SmartTracker(self.config)

        box1 = np.array([50.0, 50.0, 100.0, 100.0])
        box2 = np.array([300.0, 300.0, 360.0, 360.0])
        boxes = np.array([box1, box2])
        confs = np.array([0.9, 0.9])

        # Step 1: Initialize and confirm both tracks
        tracker.update(self.dummy_frame, boxes, confs)
        tracks = tracker.update(self.dummy_frame, boxes, confs)
        assert len(tracks) == 2

        track_id_map = {}
        for t in tracks:
            if np.linalg.norm(t.bbox[:2] - box1[:2]) < 20:
                track_id_map['t1'] = t.track_id
            else:
                track_id_map['t2'] = t.track_id

        assert track_id_map['t1'] != track_id_map['t2']

        # Step 2: Both tracks go lost during occlusion
        empty_boxes = np.empty((0, 4))
        empty_confs = np.empty((0,))
        tracker.update(self.dummy_frame, empty_boxes, empty_confs)
        tracker.update(self.dummy_frame, empty_boxes, empty_confs)

        assert len(tracker.track_manager.get_lost_tracks()) == 2

        # Step 3: Both tracks reappear near their respective locations
        reappear_b1 = np.array([54.0, 52.0, 104.0, 102.0])
        reappear_b2 = np.array([304.0, 302.0, 364.0, 362.0])
        reappear_boxes = np.array([reappear_b1, reappear_b2])

        active_tracks = tracker.update(self.dummy_frame, reappear_boxes, confs)
        assert len(active_tracks) == 2

        # Verify correct ID assignment without ID switch
        recovered_ids = {}
        for t in active_tracks:
            if np.linalg.norm(t.bbox[:2] - box1[:2]) < 20:
                recovered_ids['t1'] = t.track_id
            else:
                recovered_ids['t2'] = t.track_id

        assert recovered_ids['t1'] == track_id_map['t1']
        assert recovered_ids['t2'] == track_id_map['t2']

    def test_confidence_gating_prevents_bad_match(self):
        """Low-confidence match is rejected or does not improperly recover lost track."""
        config = load_config()
        config['tracker']['mode'] = 'enhanced'
        config['track_memory']['min_recovery_confidence'] = 0.6
        tm = TrackManager(config)

        # Create a track and put it in LOST state
        track = tm.create_track(bbox=[50.0, 50.0, 100.0, 100.0])
        track.state = TrackState.LOST

        # Attempt to recover with low confidence detection (0.3 < min_recovery_confidence 0.6)
        low_conf_det = np.array([[52.0, 50.0, 102.0, 100.0]])
        tm.update_tracks(
            matches=[(0, 0)],
            unmatched_tracks=[],
            unmatched_detections=[],
            detections=low_conf_det,
            confidences=[0.3]
        )

        # Track should NOT be restored to CONFIRMED because confidence is too low
        assert track.state != TrackState.CONFIRMED
