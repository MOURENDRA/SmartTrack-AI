"""
SmartTrack AI - MOT Benchmark Evaluation Script

Usage:
    python scripts/evaluate_mot.py --dataset data/datasets/MOT17 --sequence MOT17-02
    python scripts/evaluate_mot.py --dataset data/datasets/MOT17 --all
    python scripts/evaluate_mot.py --dataset data/datasets/MOT17 --compare
    python scripts/evaluate_mot.py --dataset data/datasets/MOT17 --stress
"""

import argparse
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, setup_logging
from src.evaluation import (
    evaluate_mot_sequence, run_comparison, run_stress_analysis,
    get_dataset_status, MOT17_SEQUENCES
)


def main():
    parser = argparse.ArgumentParser(description='SmartTrack AI - MOT Evaluation')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to MOT dataset root (e.g., data/datasets/MOT17)')
    parser.add_argument('--sequence', type=str, default=None,
                       help='Specific sequence to evaluate (e.g., MOT17-02)')
    parser.add_argument('--all', action='store_true',
                       help='Evaluate all available sequences')
    parser.add_argument('--compare', action='store_true',
                       help='Run baseline vs enhanced comparison')
    parser.add_argument('--stress', action='store_true',
                       help='Run occlusion stress analysis')
    parser.add_argument('--mode', type=str, choices=['baseline', 'enhanced'],
                       default='enhanced', help='Tracker mode')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--status', action='store_true',
                       help='Show dataset availability status')
    
    args = parser.parse_args()
    logger = setup_logging()
    config = load_config(args.config)
    
    if args.status:
        status = get_dataset_status(args.dataset)
        print("\n=== Dataset Status ===")
        print(f"Dataset root: {args.dataset}")
        print(f"Exists: {status['exists']}")
        print(f"\nMOT17 sequences available: {status['mot17_available']}/{len(status['mot17'])}")
        for seq, info in status['mot17'].items():
            marker = "✓" if info['available'] else "✗"
            gt = " (with GT)" if info['has_ground_truth'] else ""
            print(f"  {marker} {seq}{gt}")
        print(f"\nMOT20 sequences available: {status['mot20_available']}/{len(status['mot20'])}")
        for seq, info in status['mot20'].items():
            marker = "✓" if info['available'] else "✗"
            gt = " (with GT)" if info['has_ground_truth'] else ""
            print(f"  {marker} {seq}{gt}")
        return
    
    # Set tracker mode
    if 'tracker' not in config:
        config['tracker'] = {}
    config['tracker']['mode'] = args.mode
    
    if args.stress:
        print("\n=== Running Occlusion Stress Analysis ===")
        results = run_stress_analysis(args.dataset, config, 
                                      progress_callback=_progress_cb)
        _print_stress_results(results)
        return
    
    if args.compare:
        print("\n=== Running Baseline vs Enhanced Comparison ===")
        # Find available sequences
        status = get_dataset_status(args.dataset)
        available = [s for s, info in status['mot17'].items() 
                    if info['available'] and info['has_ground_truth']]
        if not available:
            print("No sequences with ground truth found. Download MOT17 first.")
            return
        
        print(f"Sequences: {', '.join(available)}")
        results = run_comparison(args.dataset, available, config,
                                progress_callback=_progress_cb_compare)
        _print_comparison(results)
        return
    
    if args.sequence:
        sequences = [args.sequence]
    elif args.all:
        status = get_dataset_status(args.dataset)
        sequences = [s for s, info in status['mot17'].items() if info['available']]
        if not sequences:
            print("No sequences found. Download MOT17 first.")
            return
    else:
        parser.print_help()
        return
    
    for seq_name in sequences:
        print(f"\n=== Evaluating {seq_name} ({args.mode}) ===")
        result = evaluate_mot_sequence(args.dataset, seq_name, config,
                                       progress_callback=_progress_single)
        if result is None:
            print(f"Sequence {seq_name} not found.")
            continue
        
        _print_result(result)


def _progress_single(idx, total):
    if idx % 50 == 0 or idx == total:
        print(f"  Frame {idx}/{total}", end='\r')


def _progress_cb(seq_name, mode, idx, total):
    if idx % 50 == 0 or idx == total:
        print(f"  [{mode}] {seq_name}: Frame {idx}/{total}", end='\r')


def _progress_cb_compare(seq_name, mode, idx, total):
    _progress_cb(seq_name, mode, idx, total)


def _print_result(result):
    """Print evaluation result."""
    rt = result['runtime']
    print(f"\n  Sequence: {result['sequence']}")
    print(f"  Frames: {rt['processed_frames']}")
    print(f"  Processing FPS: {rt['processing_fps']}")
    print(f"  Total time: {rt['total_time_s']}s")
    
    metrics = result.get('metrics')
    if metrics:
        source = result.get('metrics_source', 'unknown')
        print(f"\n  Metrics (source: {source}):")
        for key in ['MOTA', 'IDF1', 'HOTA', 'num_id_switches', 
                     'num_false_positives', 'num_false_negatives',
                     'precision', 'recall']:
            if key in metrics and metrics[key] is not None:
                val = metrics[key]
                unit = '%' if key in ['MOTA', 'IDF1', 'HOTA', 'precision', 'recall'] else ''
                print(f"    {key}: {val}{unit}")
    else:
        print("\n  No ground truth available for metric computation.")


def _print_comparison(results):
    """Print baseline vs enhanced comparison."""
    print("\n\n=== COMPARISON RESULTS ===")
    for seq_name, comp in results.get('comparison', {}).items():
        print(f"\n  {seq_name}:")
        for metric, info in comp.items():
            print(f"    {metric}: baseline={info['baseline']}, "
                  f"enhanced={info['enhanced']}, "
                  f"diff={info['difference']:+.2f} ({info['better']})")


def _print_stress_results(results):
    """Print stress analysis results."""
    print("\n\n=== STRESS ANALYSIS RESULTS ===")
    for level, data in results.items():
        print(f"\n  {level.upper()}: {data['description']}")
        print(f"    Sequences: {', '.join(data.get('sequences', []))}")
        
        agg = data.get('aggregated', {})
        for mode in ['baseline', 'enhanced']:
            if mode in agg:
                print(f"    {mode}:")
                for k, v in agg[mode].items():
                    print(f"      {k}: {v}")


if __name__ == '__main__':
    main()
