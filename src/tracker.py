import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

from .track_manager import TrackManager, Track, TrackState
from .appearance import AppearanceExtractor
from .association import associate_detections_to_tracks
from .utils import Timer

logger = logging.getLogger(__name__)


class SmartTracker:
    """SmartTrack AI: Identity-Preserving Multi-Object Tracker.
    
    Supports two modes:
    - 'baseline': IoU-only association (standard SORT-like tracker)
    - 'enhanced': Combined motion + appearance + confidence-aware association
      with track memory and cautious re-association
    """
    
    def __init__(self, config: dict):
        self.config = config
        tracker_cfg = config.get('tracker', {})
        self.mode = tracker_cfg.get('association_method') or tracker_cfg.get('mode', 'enhanced')
        if self.mode not in {'baseline', 'enhanced'}:
            logger.warning("Unknown tracker mode %r; defaulting to 'enhanced'.", self.mode)
            self.mode = 'enhanced'
        self.track_manager = TrackManager(config)
        
        # Initialize appearance extractor for enhanced mode
        self.appearance_extractor = None
        if self.mode == 'enhanced':
            try:
                self.appearance_extractor = AppearanceExtractor(config.get('appearance', {}))
                logger.info("Appearance extractor initialized for enhanced tracking")
            except Exception as e:
                logger.warning(f"Could not initialize appearance extractor: {e}")
                logger.warning("Enhanced mode will use motion-only association")
        
        # Stats
        self.frame_count = 0
        self.total_id_switches = 0  # Proxy count based on re-associations
        self._frame_times = []
        
        logger.info(f"SmartTracker initialized in '{self.mode}' mode")
    
    def update(self, frame: np.ndarray, detections: np.ndarray, 
               confidences: np.ndarray) -> List[Track]:
        """Process one frame of detections and return active tracks.
        
        Args:
            frame: BGR image (H, W, 3)
            detections: (N, 4) array of [x1, y1, x2, y2] boxes
            confidences: (N,) array of detection confidences
            
        Returns:
            List of active Track objects with updated states
        """
        with Timer() as t:
            self.frame_count += 1
            
            # Step 1: Predict all existing tracks
            predictions = self.track_manager.predict_all()
            
            # Step 2: Extract appearance features if in enhanced mode
            detection_features = None
            track_features = None
            
            if self.mode == 'enhanced' and self.appearance_extractor is not None:
                if len(detections) > 0:
                    detection_features = []
                    for det in detections:
                        feat = self.appearance_extractor.extract(frame, det.astype(int))
                        detection_features.append(feat)
                
                tracks = self.track_manager.get_all_tracks()
                if tracks:
                    track_features = [t.appearance_feature for t in tracks]
            
            # Step 3: Associate detections to tracks
            all_tracks = self.track_manager.get_all_tracks()
            
            if len(all_tracks) == 0 and len(detections) == 0:
                matches = []
                unmatched_tracks = []
                unmatched_detections = []
            elif len(all_tracks) == 0:
                matches = []
                unmatched_tracks = []
                unmatched_detections = list(range(len(detections)))
            elif len(detections) == 0:
                matches = []
                unmatched_tracks = list(range(len(all_tracks)))
                unmatched_detections = []
            else:
                matches, unmatched_tracks, unmatched_detections = \
                    associate_detections_to_tracks(
                        tracks=all_tracks,
                        detections=detections,
                        track_predictions=predictions,
                        track_features=track_features,
                        detection_features=detection_features,
                        config=self.config,
                        mode=self.mode
                    )
            
            # Step 4: Update track manager
            self.track_manager.update_tracks(
                matches=matches,
                unmatched_tracks=unmatched_tracks,
                unmatched_detections=unmatched_detections,
                detections=detections,
                features=detection_features,
                confidences=confidences
            )
            
            # Step 5: Remove dead tracks
            self.track_manager.remove_dead_tracks()
        
        self._frame_times.append(t.elapsed)
        
        return self.track_manager.get_active_tracks()
    
    @property
    def association_method(self) -> str:
        """Alias for tracking mode ('baseline' or 'enhanced')."""
        return self.mode

    def reset(self):
        """Reset the tracker state."""
        self.track_manager = TrackManager(self.config)
        self.frame_count = 0
        self.total_id_switches = 0
        self._frame_times = []
        logger.info("Tracker reset")
    
    def get_statistics(self) -> dict:
        """Get current tracking statistics."""
        track_stats = self.track_manager.get_track_statistics()
        avg_time = np.mean(self._frame_times) if self._frame_times else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'mode': self.mode,
            'frame_count': self.frame_count,
            'avg_frame_time_ms': avg_time * 1000,
            'processing_fps': round(fps, 1),
            **track_stats
        }
