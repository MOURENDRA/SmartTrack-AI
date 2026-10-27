"""
SmartTrack AI - Benchmark Evaluation Pipeline

Supports MOT17/MOT20 benchmark evaluation with configurable sequences.
"""

import os
import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .metrics import evaluate_sequence, try_motmetrics_evaluation, load_mot_gt
from .detector import PersonDetector
from .tracker import SmartTracker
from .utils import load_config

logger = logging.getLogger(__name__)


# Expected MOT dataset folder structure:
# data/datasets/MOT17/
#   train/
#     MOT17-02-DPM/
#       img1/
#         000001.jpg
#         000002.jpg
#         ...
#       gt/
#         gt.txt
#       seqinfo.ini
#     MOT17-04-DPM/
#       ...

MOT17_SEQUENCES = [
    'MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09',
    'MOT17-10', 'MOT17-11', 'MOT17-13'
]

MOT20_SEQUENCES = [
    'MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05'
]

# Occlusion/density grouping for stress analysis
# Based on average pedestrian count and known scene characteristics
DENSITY_GROUPS = {
    'low': {
        'description': 'Low density (avg <20 pedestrians/frame)',
        'mot17': ['MOT17-09', 'MOT17-05'],
    },
    'medium': {
        'description': 'Medium density (avg 20-50 pedestrians/frame)',
        'mot17': ['MOT17-02', 'MOT17-10', 'MOT17-11', 'MOT17-13'],
    },
    'high': {
        'description': 'High density (avg >50 pedestrians/frame)',
        'mot17': ['MOT17-04'],
        'mot20': ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05'],
    },
}


def find_sequence_path(dataset_root: str, seq_name: str) -> Optional[str]:
    """Find the path to a MOT sequence, handling different detector suffixes.
    
    MOT17 sequences have suffixes like -DPM, -FRCNN, -SDP.
    We prefer -SDP or use the first available.
    """
    dataset_root = Path(dataset_root)
    
    # Check train folder
    for split in ['train', 'test']:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        
        # Try exact name first
        if (split_dir / seq_name).exists():
            return str(split_dir / seq_name)
        
        # Try with detector suffixes (MOT17)
        for suffix in ['-SDP', '-FRCNN', '-DPM', '']:
            candidate = split_dir / f"{seq_name}{suffix}"
            if candidate.exists():
                return str(candidate)
    
    # Try direct path
    if (dataset_root / seq_name).exists():
        return str(dataset_root / seq_name)
    
    return None


def load_sequence_info(seq_path: str) -> Dict:
    """Load sequence information from seqinfo.ini."""
    info = {
        'name': os.path.basename(seq_path),
        'imDir': 'img1',
        'frameRate': 30,
        'seqLength': 0,
        'imWidth': 0,
        'imHeight': 0,
        'imExt': '.jpg'
    }
    
    ini_path = os.path.join(seq_path, 'seqinfo.ini')
    if os.path.exists(ini_path):
        with open(ini_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if key in info:
                        try:
                            info[key] = int(value)
                        except ValueError:
                            info[key] = value
    
    # Count images if seqLength not in ini
    if info['seqLength'] == 0:
        img_dir = os.path.join(seq_path, info['imDir'])
        if os.path.exists(img_dir):
            info['seqLength'] = len([f for f in os.listdir(img_dir) 
                                     if f.endswith(('.jpg', '.png'))])
    
    return info


def run_tracker_on_sequence(seq_path: str, config: dict,
                           progress_callback=None) -> Tuple[str, Dict]:
    """Run the tracker on a MOT sequence and save results.
    
    Args:
        seq_path: Path to sequence directory
        config: Tracker configuration
        progress_callback: Optional callback(frame_idx, total_frames)
        
    Returns:
        Tuple of (results_file_path, runtime_stats)
    """
    import cv2
    
    seq_info = load_sequence_info(seq_path)
    seq_name = seq_info['name']
    img_dir = os.path.join(seq_path, seq_info['imDir'])
    
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    # Get sorted image list
    img_files = sorted([f for f in os.listdir(img_dir) 
                       if f.endswith(('.jpg', '.png'))])
    
    if len(img_files) == 0:
        raise ValueError(f"No images found in {img_dir}")
    
    # Initialize detector and tracker
    detector = PersonDetector(config.get('detector', {}))
    tracker = SmartTracker(config)
    
    # Prepare output
    output_dir = config.get('evaluation', {}).get('output_dir', 'outputs')
    mode = config.get('tracker', {}).get('mode', 'enhanced')
    results_dir = os.path.join(output_dir, 'eval_results', f"{seq_name}_{mode}")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{seq_name}.txt")
    
    # Track
    frame_times = []
    results_lines = []
    
    logger.info(f"Processing sequence {seq_name}: {len(img_files)} frames")
    
    for idx, img_file in enumerate(img_files):
        frame_id = idx + 1  # MOT format is 1-indexed
        img_path = os.path.join(img_dir, img_file)
        
        frame = cv2.imread(img_path)
        if frame is None:
            logger.warning(f"Could not read frame: {img_path}")
            continue
        
        start_time = time.time()
        
        # Detect
        boxes, confidences = detector.detect(frame)
        
        # Track
        active_tracks = tracker.update(frame, boxes, confidences)
        
        elapsed = time.time() - start_time
        frame_times.append(elapsed)
        
        # Write results in MOT format
        for track in active_tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            conf = track.confidence
            # MOT format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
            results_lines.append(
                f"{frame_id},{track.track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n"
            )
        
        if progress_callback:
            progress_callback(idx + 1, len(img_files))
    
    # Save results
    with open(results_file, 'w') as f:
        f.writelines(results_lines)
    
    avg_time = np.mean(frame_times) if frame_times else 0
    runtime_stats = {
        'sequence': seq_name,
        'mode': mode,
        'total_frames': len(img_files),
        'processed_frames': len(frame_times),
        'avg_frame_time_ms': round(avg_time * 1000, 2),
        'processing_fps': round(1.0 / avg_time, 1) if avg_time > 0 else 0,
        'total_time_s': round(sum(frame_times), 2),
        'results_file': results_file,
    }
    
    logger.info(f"Sequence {seq_name} complete: {runtime_stats['processing_fps']} FPS")
    
    return results_file, runtime_stats


def evaluate_mot_sequence(dataset_root: str, seq_name: str, config: dict,
                         progress_callback=None) -> Optional[Dict]:
    """Run tracker on a sequence and evaluate against ground truth.
    
    Args:
        dataset_root: Root directory of MOT dataset
        seq_name: Sequence name (e.g., 'MOT17-02')
        config: Tracker configuration
        progress_callback: Optional progress callback
        
    Returns:
        Dictionary with metrics and runtime stats, or None if sequence not found
    """
    seq_path = find_sequence_path(dataset_root, seq_name)
    if seq_path is None:
        logger.warning(f"Sequence {seq_name} not found in {dataset_root}")
        return None
    
    # Check for ground truth
    gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
    has_gt = os.path.exists(gt_path)
    
    if not has_gt:
        logger.warning(f"No ground truth found for {seq_name} at {gt_path}")
    
    # Run tracker
    results_file, runtime_stats = run_tracker_on_sequence(
        seq_path, config, progress_callback
    )
    
    result = {
        'sequence': seq_name,
        'runtime': runtime_stats,
        'metrics': None,
        'metrics_source': None,
    }
    
    if has_gt:
        # Try motmetrics first
        mm_result = try_motmetrics_evaluation(gt_path, results_file)
        if mm_result is not None:
            result['metrics'] = mm_result
            result['metrics_source'] = 'motmetrics'
        else:
            # Use built-in evaluation
            metrics = evaluate_sequence(gt_path, results_file)
            result['metrics'] = metrics
            result['metrics_source'] = 'built-in (approximate)'
    
    # Save evaluation result
    output_dir = config.get('evaluation', {}).get('output_dir', 'outputs')
    mode = config.get('tracker', {}).get('mode', 'enhanced')
    eval_json = os.path.join(output_dir, 'eval_results', 
                             f"{seq_name}_{mode}_eval.json")
    os.makedirs(os.path.dirname(eval_json), exist_ok=True)
    with open(eval_json, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    return result


def run_comparison(dataset_root: str, seq_names: List[str], 
                   base_config: dict, progress_callback=None) -> Dict:
    """Run baseline vs enhanced comparison on specified sequences.
    
    Args:
        dataset_root: Root directory of MOT dataset
        seq_names: List of sequence names to evaluate
        base_config: Base configuration (mode will be overridden)
        progress_callback: Optional progress callback(seq_name, mode, progress)
        
    Returns:
        Comparison results dictionary
    """
    results = {
        'baseline': {},
        'enhanced': {},
        'comparison': {},
    }
    
    for mode in ['baseline', 'enhanced']:
        config = base_config.copy()
        if 'tracker' not in config:
            config['tracker'] = {}
        config['tracker']['mode'] = mode
        
        for seq_name in seq_names:
            logger.info(f"Evaluating {seq_name} with {mode} tracker")
            
            def cb(idx, total):
                if progress_callback:
                    progress_callback(seq_name, mode, idx, total)
            
            eval_result = evaluate_mot_sequence(
                dataset_root, seq_name, config, cb
            )
            
            if eval_result is not None:
                results[mode][seq_name] = eval_result
    
    # Compute comparison
    for seq_name in seq_names:
        if seq_name in results['baseline'] and seq_name in results['enhanced']:
            bl = results['baseline'][seq_name].get('metrics')
            en = results['enhanced'][seq_name].get('metrics')
            
            if bl and en:
                comparison = {}
                for metric in ['MOTA', 'IDF1', 'HOTA', 'num_id_switches']:
                    if metric in bl and metric in en:
                        bl_val = bl[metric]
                        en_val = en[metric]
                        if bl_val is not None and en_val is not None:
                            if metric == 'num_id_switches':
                                # Lower is better for ID switches
                                diff = bl_val - en_val
                                better = 'enhanced' if diff > 0 else 'baseline' if diff < 0 else 'tied'
                            else:
                                # Higher is better for accuracy metrics
                                diff = en_val - bl_val
                                better = 'enhanced' if diff > 0 else 'baseline' if diff < 0 else 'tied'
                            comparison[metric] = {
                                'baseline': bl_val,
                                'enhanced': en_val,
                                'difference': round(diff, 2),
                                'better': better,
                            }
                
                results['comparison'][seq_name] = comparison
    
    # Save comparison
    output_dir = base_config.get('evaluation', {}).get('output_dir', 'outputs')
    comp_json = os.path.join(output_dir, 'eval_results', 'comparison.json')
    os.makedirs(os.path.dirname(comp_json), exist_ok=True)
    with open(comp_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def run_stress_analysis(dataset_root: str, base_config: dict,
                       progress_callback=None) -> Dict:
    """Run occlusion/density stress analysis.
    
    Groups sequences by density level and compares tracker performance.
    
    Args:
        dataset_root: Root directory of MOT dataset(s)
        base_config: Base tracker configuration
        progress_callback: Optional progress callback
        
    Returns:
        Stress analysis results organized by density group
    """
    stress_results = {}
    
    for density_level, group_info in DENSITY_GROUPS.items():
        logger.info(f"Stress analysis: {density_level} density")
        
        # Collect available sequences for this density level
        available_seqs = []
        for dataset_key in ['mot17', 'mot20']:
            if dataset_key in group_info:
                for seq in group_info[dataset_key]:
                    seq_path = find_sequence_path(dataset_root, seq)
                    if seq_path is not None:
                        available_seqs.append(seq)
        
        if not available_seqs:
            logger.info(f"No sequences available for {density_level} density group")
            stress_results[density_level] = {
                'description': group_info['description'],
                'sequences': [],
                'note': 'No sequences found. Download the MOT dataset first.',
            }
            continue
        
        # Run comparison on available sequences
        group_results = run_comparison(
            dataset_root, available_seqs, base_config, progress_callback
        )
        
        # Aggregate metrics per mode
        aggregated = {}
        for mode in ['baseline', 'enhanced']:
            mode_metrics = []
            for seq in available_seqs:
                if seq in group_results[mode]:
                    m = group_results[mode][seq].get('metrics')
                    if m:
                        mode_metrics.append(m)
            
            if mode_metrics:
                avg_metrics = {}
                for key in ['MOTA', 'IDF1', 'HOTA']:
                    values = [m[key] for m in mode_metrics if m.get(key) is not None]
                    if values:
                        avg_metrics[key] = round(np.mean(values), 2)
                
                id_sw = [m.get('num_id_switches', 0) for m in mode_metrics]
                avg_metrics['total_id_switches'] = sum(id_sw)
                avg_metrics['avg_id_switches'] = round(np.mean(id_sw), 1) if id_sw else 0
                
                aggregated[mode] = avg_metrics
        
        stress_results[density_level] = {
            'description': group_info['description'],
            'sequences': available_seqs,
            'results': group_results,
            'aggregated': aggregated,
        }
    
    # Save stress analysis
    output_dir = base_config.get('evaluation', {}).get('output_dir', 'outputs')
    stress_json = os.path.join(output_dir, 'eval_results', 'stress_analysis.json')
    os.makedirs(os.path.dirname(stress_json), exist_ok=True)
    with open(stress_json, 'w') as f:
        json.dump(stress_results, f, indent=2, default=str)
    
    return stress_results


def get_dataset_status(dataset_root: str) -> Dict:
    """Check which MOT sequences are available.
    
    Returns:
        Dictionary with dataset availability information
    """
    status = {
        'dataset_root': dataset_root,
        'exists': os.path.exists(dataset_root),
        'mot17': {},
        'mot20': {},
    }
    
    for seq in MOT17_SEQUENCES:
        path = find_sequence_path(dataset_root, seq)
        has_gt = False
        if path:
            has_gt = os.path.exists(os.path.join(path, 'gt', 'gt.txt'))
        status['mot17'][seq] = {
            'available': path is not None,
            'path': path,
            'has_ground_truth': has_gt,
        }
    
    for seq in MOT20_SEQUENCES:
        path = find_sequence_path(dataset_root, seq)
        has_gt = False
        if path:
            has_gt = os.path.exists(os.path.join(path, 'gt', 'gt.txt'))
        status['mot20'][seq] = {
            'available': path is not None,
            'path': path,
            'has_ground_truth': has_gt,
        }
    
    status['mot17_available'] = sum(1 for s in status['mot17'].values() if s['available'])
    status['mot20_available'] = sum(1 for s in status['mot20'].values() if s['available'])
    
    return status
