"""SmartTrack-AI: Advanced Computer Vision Multi-Object Tracking Engine."""

from .detector import PersonDetector
from .exporters import (
    export_json_summary,
    export_mot_format,
    export_tracking_csv,
    generate_results_report,
)
from .tracker import SmartTracker
from .video_processor import VideoProcessor

__version__ = "1.0.0"

__all__ = [
    "PersonDetector",
    "SmartTracker",
    "VideoProcessor",
    "export_tracking_csv",
    "export_mot_format",
    "export_json_summary",
    "generate_results_report",
]
