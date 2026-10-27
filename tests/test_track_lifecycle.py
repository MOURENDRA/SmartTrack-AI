import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.track_manager import TrackManager, Track, TrackState
from src.motion_model import KalmanBoxTracker
from src.utils import load_config


class TestTrackLifecycle:
    def setup_method(self):
        self.config = load_config()

    def test_detection_creates_track(self):
        """A detection creates a tentative track."""
        tm = TrackManager(self.config)
        det = np.array([10.0, 10.0, 50.0, 50.0])
        track = tm.create_track(bbox=det, confidence=0.9)

        assert track.track_id == 1
        assert track.state == TrackState.TENTATIVE
        assert len(tm.tracks) == 1
        assert tm.tracks[0].state == TrackState.TENTATIVE

        # Also test creation via update_tracks with an unmatched detection
        tm2 = TrackManager(self.config)
        tm2.update_tracks(
            matches=[],
            unmatched_tracks=[],
            unmatched_detections=[0],
            detections=np.array([[20.0, 20.0, 60.0, 60.0]]),
            confidences=[0.85]
        )
        assert len(tm2.tracks) == 1
        assert tm2.tracks[0].state == TrackState.TENTATIVE
        assert tm2.tracks[0].track_id == 1

    def test_consecutive_hits_confirm_track(self):
        """After min_hits, track becomes confirmed."""
        config = load_config()
        config['tracker']['min_hits'] = 3
        tm = TrackManager(config)

        det = np.array([[10.0, 10.0, 50.0, 50.0]])

        # Frame 1: creates tentative track (hit 1)
        tm.update_tracks(
            matches=[],
            unmatched_tracks=[],
            unmatched_detections=[0],
            detections=det
        )
        assert len(tm.tracks) == 1
        track = tm.tracks[0]
        assert track.state == TrackState.TENTATIVE
        assert track.hits == 1

        # Frame 2: 1st consecutive match (hit 2)
        tm.update_tracks(
            matches=[(0, 0)],
            unmatched_tracks=[],
            unmatched_detections=[],
            detections=det
        )
        assert track.state == TrackState.TENTATIVE
        assert track.hits == 2
        assert track.hit_streak == 2

        # Frame 3: 2nd consecutive match (hit 3 == min_hits)
        tm.update_tracks(
            matches=[(0, 0)],
            unmatched_tracks=[],
            unmatched_detections=[],
            detections=det
        )
        assert track.state == TrackState.CONFIRMED
        assert track.is_confirmed()
        assert track.hits == 3

    def test_missed_detection_track_survives(self):
        """Missing 1 frame doesn't kill track."""
        config = load_config()
        config['tracker']['min_hits'] = 1
        tm = TrackManager(config)

        det = np.array([[10.0, 10.0, 50.0, 50.0]])
        # Create and confirm track
        tm.update_tracks(matches=[], unmatched_tracks=[], unmatched_detections=[0], detections=det)
        track = tm.tracks[0]
        track.state = TrackState.CONFIRMED

        # Frame with missed detection (track 0 unmatched)
        tm.predict_all()
        tm.update_tracks(
            matches=[],
            unmatched_tracks=[0],
            unmatched_detections=[],
            detections=np.empty((0, 4))
        )

        assert track.time_since_update == 1
        assert track.state == TrackState.CONFIRMED
        assert not track.is_removed()
        assert not track.is_lost()
        assert len(tm.tracks) == 1

    def test_track_goes_lost(self):
        """After several misses, track transitions to LOST."""
        config = load_config()
        config['tracker']['min_hits'] = 1
        tm = TrackManager(config)

        det = np.array([[10.0, 10.0, 50.0, 50.0]])
        tm.update_tracks(matches=[], unmatched_tracks=[], unmatched_detections=[0], detections=det)
        track = tm.tracks[0]
        track.state = TrackState.CONFIRMED

        # Miss 1 frame -> still CONFIRMED
        tm.predict_all()
        tm.update_tracks(matches=[], unmatched_tracks=[0], unmatched_detections=[], detections=np.empty((0, 4)))
        assert track.state == TrackState.CONFIRMED
        assert track.time_since_update == 1

        # Miss 2nd frame -> transitions to LOST (time_since_update > 1)
        tm.predict_all()
        tm.update_tracks(matches=[], unmatched_tracks=[0], unmatched_detections=[], detections=np.empty((0, 4)))
        assert track.state == TrackState.LOST
        assert track.is_lost()
        assert track.time_since_update == 2
        assert len(tm.get_lost_tracks()) == 1

    def test_track_removed_after_max_age(self):
        """After max_age misses, track is removed."""
        config = load_config()
        config['tracker']['max_age'] = 4
        config['track_memory']['max_lost_age'] = 4
        tm = TrackManager(config)

        det = np.array([[10.0, 10.0, 50.0, 50.0]])
        tm.update_tracks(matches=[], unmatched_tracks=[], unmatched_detections=[0], detections=det)
        track = tm.tracks[0]
        track.state = TrackState.CONFIRMED

        # Miss frames up to max_age
        for _ in range(4):
            tm.predict_all()
            tm.update_tracks(matches=[], unmatched_tracks=[0], unmatched_detections=[], detections=np.empty((0, 4)))
            assert not track.is_removed()

        # Miss 5th frame (> max_age of 4)
        tm.predict_all()
        tm.update_tracks(matches=[], unmatched_tracks=[0], unmatched_detections=[], detections=np.empty((0, 4)))
        assert track.is_removed()
        assert len(tm.tracks) == 0

    def test_expired_tracks_cleaned_up(self):
        """Removed tracks are deleted from list."""
        config = load_config()
        tm = TrackManager(config)

        track1 = tm.create_track(bbox=[10.0, 10.0, 50.0, 50.0])
        track2 = tm.create_track(bbox=[100.0, 100.0, 150.0, 150.0])
        assert len(tm.tracks) == 2

        track1.state = TrackState.REMOVED
        assert track1.is_removed()

        tm.remove_dead_tracks()
        assert len(tm.tracks) == 1
        assert tm.tracks[0].track_id == track2.track_id

    def test_multiple_independent_tracks(self):
        """Two tracks don't interfere."""
        config = load_config()
        tm = TrackManager(config)

        t1 = tm.create_track(bbox=[10.0, 10.0, 50.0, 50.0])
        t2 = tm.create_track(bbox=[200.0, 200.0, 250.0, 250.0])

        assert t1.track_id != t2.track_id

        # Update only track 1
        t1.update(bbox=[12.0, 12.0, 52.0, 52.0])

        assert t1.hits == 2
        assert t1.hit_streak == 2
        assert t2.hits == 1
        assert t2.hit_streak == 1
        assert not np.array_equal(t1.bbox, t2.bbox)
        assert len(t1.history) == 2
        assert len(t2.history) == 1

        # Mark only track 2 as missed
        t2.mark_missed()
        assert t2.time_since_update == 1
        assert t1.time_since_update == 0

    def test_kalman_prediction(self):
        """Predicted position is reasonable."""
        initial_bbox = np.array([10.0, 10.0, 50.0, 50.0], dtype=np.float32)
        track = Track(1, initial_bbox)

        predicted_box = track.predict()
        assert predicted_box.shape == (4,)

        # Expected initial width = 40, height = 40
        w = predicted_box[2] - predicted_box[0]
        h = predicted_box[3] - predicted_box[1]
        assert abs(w - 40.0) < 5.0
        assert abs(h - 40.0) < 5.0

        # Now simulate constant velocity
        track.update(np.array([15.0, 15.0, 55.0, 55.0]))
        pred2 = track.predict()
        # Position should move forward
        assert pred2[0] > 10.0
        assert pred2[1] > 10.0

    def test_track_id_uniqueness(self):
        """Each new track gets unique ID."""
        config = load_config()
        tm = TrackManager(config)

        tracks = [
            tm.create_track(bbox=[i * 20.0, i * 20.0, i * 20.0 + 30.0, i * 20.0 + 30.0])
            for i in range(10)
        ]
        ids = [t.track_id for t in tracks]
        assert len(ids) == 10
        assert len(set(ids)) == 10

    def test_track_history_recording(self):
        """Track stores bbox history."""
        track = Track(1, np.array([10.0, 10.0, 50.0, 50.0]))
        assert len(track.history) == 1
        assert np.allclose(track.history[0], [10.0, 10.0, 50.0, 50.0])

        box2 = np.array([15.0, 15.0, 55.0, 55.0])
        box3 = np.array([20.0, 20.0, 60.0, 60.0])
        track.update(box2)
        track.update(box3)

        assert len(track.history) == 3
        assert np.allclose(track.history[1], box2)
        assert np.allclose(track.history[2], box3)
