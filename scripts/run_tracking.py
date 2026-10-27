"""
SmartTrack AI - Standalone Tracking Script

Usage:
    python scripts/run_tracking.py --input path/to/video.mp4
    python scripts/run_tracking.py --input path/to/video.mp4 --mode baseline
    python scripts/run_tracking.py --input path/to/video.mp4 --output outputs/result.mp4 --csv outputs/tracks.csv
"""

import argparse
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, setup_logging
from src.video_processor import VideoProcessor


def main():
    parser = argparse.ArgumentParser(description='SmartTrack AI - Run Tracking')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output video file (default: outputs/<input_name>_tracked.mp4)')
    parser.add_argument('--csv', type=str, default=None,
                       help='Path to export tracking CSV')
    parser.add_argument('--mot', type=str, default=None,
                       help='Path to export MOT-format results')
    parser.add_argument('--mode', type=str, choices=['baseline', 'enhanced'],
                       default='enhanced', help='Tracker mode')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--confidence', type=float, default=None,
                       help='Detection confidence threshold')
    parser.add_argument('--show', action='store_true',
                       help='Display frames while processing (requires display)')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with CLI args
    if 'tracker' not in config:
        config['tracker'] = {}
    config['tracker']['mode'] = args.mode
    
    if args.confidence is not None:
        if 'detector' not in config:
            config['detector'] = {}
        config['detector']['confidence_threshold'] = args.confidence
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Set output path
    if args.output is None:
        os.makedirs('outputs', exist_ok=True)
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"outputs/{input_name}_tracked_{args.mode}.mp4"
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    print(f"SmartTrack AI - {args.mode.capitalize()} Tracker")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Mode:   {args.mode}")
    print()
    
    # Initialize processor
    processor = VideoProcessor(config)
    
    if not processor.detector.is_ready:
        print("WARNING: YOLO model not loaded. Detections will be empty.")
        print("Install ultralytics and ensure model weights are available.")
        print()
    
    # Process video
    def progress(frame_idx, total_frames, tracks, frame_time):
        n_tracks = len(tracks) if tracks else 0
        fps = 1.0 / frame_time if frame_time > 0 else 0
        print(f"\rFrame {frame_idx}/{total_frames} | "
              f"Tracks: {n_tracks} | "
              f"FPS: {fps:.1f}", end='')
    
    start = time.time()
    results = processor.process_video(args.input, args.output, 
                                       progress_callback=progress)
    total_time = time.time() - start
    
    print(f"\n\nProcessing complete!")
    print(f"Total time: {total_time:.1f}s")
    print(f"Frames processed: {results['processed_frames']}/{results['total_frames']}")
    print(f"Average FPS: {results['avg_fps']:.1f}")
    print(f"Output: {results['output_path']}")
    
    # Export CSV
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
        processor.export_tracks_csv(results['frame_metrics'], args.csv)
        print(f"CSV exported: {args.csv}")
    
    # Export MOT format
    if args.mot:
        os.makedirs(os.path.dirname(args.mot) or '.', exist_ok=True)
        processor.export_tracks_mot(results['frame_metrics'], args.mot)
        print(f"MOT format exported: {args.mot}")
    
    # Print statistics
    stats = results.get('tracker_stats', {})
    print(f"\nTracker Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
