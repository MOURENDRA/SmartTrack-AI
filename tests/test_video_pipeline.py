"""Integration and end-to-end tests for VideoProcessor pipeline."""

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import load_config
from src.video_processor import VideoProcessor


def create_synthetic_video(
    output_path: str,
    num_frames: int = 15,
    width: int = 320,
    height: int = 240,
    fps: float = 20.0,
) -> str:
    """Helper to generate a short synthetic video with a moving rectangular object."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        # Fallback codec
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx in range(num_frames):
        # Background: dark gray
        frame = np.full((height, width, 3), 40, dtype=np.uint8)

        # Draw a moving human-like silhouette / rectangle
        box_w = 40
        box_h = 80
        x1 = int(20 + frame_idx * 8)
        y1 = 60
        x2 = min(width - 5, x1 + box_w)
        y2 = min(height - 5, y1 + box_h)

        if x1 < width - 10:
            # Body
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), -1)
            # Head
            head_center = (int((x1 + x2) / 2), max(10, y1 - 15))
            cv2.circle(frame, head_center, 12, (220, 220, 220), -1)

        writer.write(frame)

    writer.release()
    return output_path


class TestVideoProcessorInit:
    """Tests initialization and configuration of VideoProcessor."""

    def test_default_init(self):
        config = load_config()
        processor = VideoProcessor(config)
        assert processor.tracker is not None
        assert processor.detector is not None
        assert processor.frame_count == 0
        assert len(processor.frame_metrics) == 0

    def test_baseline_mode_init(self):
        config = load_config()
        config["tracker"]["association_method"] = "baseline"
        processor = VideoProcessor(config)
        assert processor.tracker.association_method == "baseline"

    def test_enhanced_mode_init(self):
        config = load_config()
        config["tracker"]["association_method"] = "enhanced"
        processor = VideoProcessor(config)
        assert processor.tracker.association_method == "enhanced"


class TestSingleFrameProcessing:
    """Tests single-frame processing step."""

    def test_process_empty_frame(self):
        config = load_config()
        processor = VideoProcessor(config)

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        annotated, metric = processor.process_frame(frame)

        assert annotated.shape == frame.shape
        assert "frame_id" in metric
        assert "frame_time" in metric
        assert "active_tracks" in metric
        assert "lost_tracks" in metric
        assert metric["frame_id"] == 0
        assert processor.frame_count == 1

    def test_process_consecutive_frames(self):
        config = load_config()
        processor = VideoProcessor(config)

        for i in range(5):
            frame = np.full((200, 300, 3), 50, dtype=np.uint8)
            annotated, metric = processor.process_frame(frame)
            assert metric["frame_id"] == i
            assert annotated.shape == (200, 300, 3)

        assert processor.frame_count == 5
        assert len(processor.frame_metrics) == 5


class TestSyntheticVideoPipeline:
    """Tests full video processing on a generated video file."""

    def test_full_video_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_video = os.path.join(tmpdir, "test_input.mp4")
            output_video = os.path.join(tmpdir, "test_output.mp4")

            create_synthetic_video(input_video, num_frames=10, width=320, height=240)
            assert os.path.exists(input_video)

            config = load_config()
            processor = VideoProcessor(config)

            callback_invocations = []

            def mock_callback(frame_idx, total_frames, active_tracks, frame_time):
                callback_invocations.append((frame_idx, total_frames))

            summary = processor.process_video(
                input_video, output_video, progress_callback=mock_callback
            )

            assert os.path.exists(output_video)
            assert os.path.getsize(output_video) > 0
            assert summary["processed_frames"] == 10
            assert summary["total_frames"] == 10
            assert summary["avg_fps"] >= 0.0
            assert len(summary["frame_metrics"]) == 10
            assert len(callback_invocations) == 10

    def test_invalid_video_path_raises(self):
        config = load_config()
        processor = VideoProcessor(config)
        with pytest.raises(FileNotFoundError):
            processor.process_video("non_existent_video_path_12345.mp4", "out.mp4")


class TestExportersIntegration:
    """Tests track export to CSV and MOT format."""

    def test_export_csv_and_mot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config()
            processor = VideoProcessor(config)

            # Process 3 frames
            for _ in range(3):
                frame = np.zeros((200, 200, 3), dtype=np.uint8)
                processor.process_frame(frame)

            csv_path = os.path.join(tmpdir, "tracks.csv")
            mot_path = os.path.join(tmpdir, "tracks_mot.txt")

            processor.export_tracks_csv(processor.frame_metrics, csv_path)
            processor.export_tracks_mot(processor.frame_metrics, mot_path)

            assert os.path.exists(csv_path)
            assert os.path.exists(mot_path)

            with open(csv_path, "r", encoding="utf-8") as f:
                header = f.readline()
                assert "frame_id" in header
                assert "track_id" in header


class TestReset:
    """Tests reset of video processor state."""

    def test_reset_clears_metrics_and_tracks(self):
        config = load_config()
        processor = VideoProcessor(config)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        processor.process_frame(frame)
        processor.process_frame(frame)

        assert processor.frame_count == 2
        assert len(processor.frame_metrics) == 2

        processor.reset()

        assert processor.frame_count == 0
        assert len(processor.frame_metrics) == 0
        assert len(processor._trajectories) == 0
