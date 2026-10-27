import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PersonDetector:
    """Person detector using YOLOv8 (ultralytics).
    
    Falls back to a simple background subtraction detector if ultralytics is not installed.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.person_class_id = config.get('person_class_id', 0)  # COCO person class
        self.input_size = config.get('input_size', 640)
        self.model_name = config.get('model_name', 'yolov8n.pt')
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model. Handles missing weights gracefully."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            self._backend = 'yolov8'
            logger.info(f"Loaded YOLOv8 model: {self.model_name}")
        except ImportError:
            logger.warning(
                "ultralytics not installed. Install with: pip install ultralytics\n"
                "Falling back to mock detector for testing only."
            )
            self._backend = 'mock'
        except Exception as e:
            logger.error(f"Failed to load model '{self.model_name}': {e}")
            logger.warning(
                "Could not load YOLO model. Check that the model file exists or "
                "that you have internet access for automatic download.\n"
                "Falling back to mock detector for testing only."
            )
            self._backend = 'mock'
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect people in a frame.
        
        Args:
            frame: BGR image as numpy array (H, W, 3)
            
        Returns:
            boxes: np.ndarray of shape (N, 4) with [x1, y1, x2, y2] format
            confidences: np.ndarray of shape (N,) with detection confidences
        """
        if self._backend == 'yolov8':
            return self._detect_yolo(frame)
        else:
            return self._detect_mock(frame)
    
    def _detect_yolo(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run YOLOv8 detection."""
        results = self.model(frame, imgsz=self.input_size, verbose=False)
        
        boxes = []
        confidences = []
        
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id == self.person_class_id and conf >= self.confidence_threshold:
                    xyxy = box.xyxy[0].cpu().numpy()
                    boxes.append(xyxy)
                    confidences.append(conf)
        
        if len(boxes) == 0:
            return np.empty((0, 4)), np.empty((0,))
        
        return np.array(boxes), np.array(confidences)
    
    def _detect_mock(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Mock detector that returns empty detections. For testing pipeline only."""
        logger.debug("Mock detector called - returning empty detections")
        return np.empty((0, 4)), np.empty((0,))
    
    @property
    def is_ready(self) -> bool:
        """Check if the detector is properly loaded."""
        return self._backend == 'yolov8'
    
    @property
    def backend_name(self) -> str:
        """Return the name of the active detection backend."""
        return self._backend
