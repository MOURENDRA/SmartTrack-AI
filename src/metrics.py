"""
SmartTrack AI - Evaluation Metrics

Implements MOT evaluation metrics including IDF1, MOTA, ID switches.
Uses motmetrics library when available, with fallback basic implementations.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


class MOTAccumulator:
    """Accumulates frame-by-frame tracking results for MOT metric computation.
    
    This is a lightweight accumulator that computes basic MOT metrics.
    For full benchmark evaluation, use motmetrics or TrackEval libraries.
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset all accumulated data."""
        self.frame_events = []  # List of per-frame event dicts
        self._gt_id_map = {}   # Maps gt_id -> last matched hypothesis id
        self._prev_matches = {}  # Previous frame matches: gt_id -> hyp_id
    
    def update(self, gt_boxes: np.ndarray, gt_ids: np.ndarray,
               hyp_boxes: np.ndarray, hyp_ids: np.ndarray):
        """Update accumulator with one frame of data.
        
        Args:
            gt_boxes: (M, 4) ground truth boxes [x1, y1, x2, y2]
            gt_ids: (M,) ground truth identity labels
            hyp_boxes: (N, 4) hypothesis (tracker output) boxes
            hyp_ids: (N,) hypothesis identity labels
        """
        n_gt = len(gt_boxes)
        n_hyp = len(hyp_boxes)
        
        event = {
            'n_gt': n_gt,
            'n_hyp': n_hyp,
            'matches': [],      # (gt_id, hyp_id) pairs
            'false_positives': 0,
            'false_negatives': 0,
            'id_switches': 0,
            'gt_ids': gt_ids.copy() if len(gt_ids) > 0 else np.array([]),
            'hyp_ids': hyp_ids.copy() if len(hyp_ids) > 0 else np.array([]),
        }
        
        if n_gt == 0 and n_hyp == 0:
            self.frame_events.append(event)
            self._prev_matches = {}
            return
        
        if n_gt == 0:
            event['false_positives'] = n_hyp
            self.frame_events.append(event)
            self._prev_matches = {}
            return
        
        if n_hyp == 0:
            event['false_negatives'] = n_gt
            self.frame_events.append(event)
            self._prev_matches = {}
            return
        
        # Compute IoU matrix
        iou_matrix = np.zeros((n_gt, n_hyp))
        for i in range(n_gt):
            for j in range(n_hyp):
                iou_matrix[i, j] = compute_iou(gt_boxes[i], hyp_boxes[j])
        
        # Hungarian matching
        from scipy.optimize import linear_sum_assignment
        cost_matrix = 1 - iou_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches by IoU threshold
        matched_gt = set()
        matched_hyp = set()
        current_matches = {}
        
        for r, c in zip(row_indices, col_indices):
            if iou_matrix[r, c] >= self.iou_threshold:
                gt_id = int(gt_ids[r])
                hyp_id = int(hyp_ids[c])
                event['matches'].append((gt_id, hyp_id))
                matched_gt.add(r)
                matched_hyp.add(c)
                current_matches[gt_id] = hyp_id
                
                # Check for ID switch
                if gt_id in self._prev_matches:
                    if self._prev_matches[gt_id] != hyp_id:
                        event['id_switches'] += 1
        
        event['false_negatives'] = n_gt - len(matched_gt)
        event['false_positives'] = n_hyp - len(matched_hyp)
        
        self.frame_events.append(event)
        self._prev_matches = current_matches
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute MOT metrics from accumulated events.
        
        Returns:
            Dictionary with metrics:
            - MOTA: Multi-Object Tracking Accuracy
            - IDF1: ID F1 Score (approximate)
            - num_id_switches: Total ID switches
            - num_false_positives: Total false positives
            - num_false_negatives: Total false negatives (misses)
            - num_matches: Total matches
            - precision: Detection precision
            - recall: Detection recall
            - num_frames: Number of frames processed
        """
        total_gt = 0
        total_matches = 0
        total_fp = 0
        total_fn = 0
        total_id_switches = 0
        
        # For IDF1 computation
        gt_id_matched_frames = {}  # gt_id -> count of frames matched
        hyp_id_matched_frames = {}  # hyp_id -> count of frames matched
        gt_id_total_frames = {}  # gt_id -> count of frames present
        hyp_id_total_frames = {}  # hyp_id -> count of frames present
        
        for event in self.frame_events:
            total_gt += event['n_gt']
            total_matches += len(event['matches'])
            total_fp += event['false_positives']
            total_fn += event['false_negatives']
            total_id_switches += event['id_switches']
            
            # Track per-ID statistics
            for gt_id in event['gt_ids']:
                gt_id = int(gt_id)
                gt_id_total_frames[gt_id] = gt_id_total_frames.get(gt_id, 0) + 1
            
            for hyp_id in event['hyp_ids']:
                hyp_id = int(hyp_id)
                hyp_id_total_frames[hyp_id] = hyp_id_total_frames.get(hyp_id, 0) + 1
            
            for gt_id, hyp_id in event['matches']:
                gt_id_matched_frames[gt_id] = gt_id_matched_frames.get(gt_id, 0) + 1
                hyp_id_matched_frames[hyp_id] = hyp_id_matched_frames.get(hyp_id, 0) + 1
        
        # MOTA = 1 - (FN + FP + ID_SW) / total_gt
        mota = 1.0 - (total_fn + total_fp + total_id_switches) / max(total_gt, 1)
        
        # Precision and recall
        precision = total_matches / max(total_matches + total_fp, 1)
        recall = total_matches / max(total_matches + total_fn, 1)
        
        # IDF1 (simplified): harmonic mean of ID precision and ID recall
        # ID Precision = IDTP / (IDTP + IDFP)
        # ID Recall = IDTP / (IDTP + IDFN)
        # Here we approximate IDTP as sum of matched frames per best gt-hyp pair
        idtp = sum(gt_id_matched_frames.values())
        idfp = sum(hyp_id_total_frames.values()) - sum(hyp_id_matched_frames.values())
        idfn = sum(gt_id_total_frames.values()) - sum(gt_id_matched_frames.values())
        
        id_precision = idtp / max(idtp + idfp, 1)
        id_recall = idtp / max(idtp + idfn, 1)
        idf1 = 2 * id_precision * id_recall / max(id_precision + id_recall, 1e-6)
        
        return {
            'MOTA': round(mota * 100, 2),
            'IDF1': round(idf1 * 100, 2),
            'num_id_switches': total_id_switches,
            'num_false_positives': total_fp,
            'num_false_negatives': total_fn,
            'num_matches': total_matches,
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'num_frames': len(self.frame_events),
        }


def compute_hota(gt_data: Dict, hyp_data: Dict, iou_thresholds: Optional[List[float]] = None) -> Dict[str, float]:
    """Compute HOTA (Higher Order Tracking Accuracy) metric.
    
    This is a simplified HOTA implementation. For official benchmark results,
    use the TrackEval library: https://github.com/JonathonLuiten/TrackEval
    
    Args:
        gt_data: {frame_id: [(gt_id, x1, y1, x2, y2), ...]}
        hyp_data: {frame_id: [(hyp_id, x1, y1, x2, y2), ...]}
        iou_thresholds: List of IoU thresholds to average over (default: 0.05 to 0.95)
    
    Returns:
        Dictionary with HOTA, DetA, AssA metrics
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.05, 1.0, 0.05).tolist()
    
    hota_per_threshold = []
    deta_per_threshold = []
    assa_per_threshold = []
    
    for alpha in iou_thresholds:
        tp = 0
        fp = 0
        fn = 0
        ass_sum = 0.0
        
        all_frames = sorted(set(list(gt_data.keys()) + list(hyp_data.keys())))
        
        for frame_id in all_frames:
            gt_objs = gt_data.get(frame_id, [])
            hyp_objs = hyp_data.get(frame_id, [])
            
            n_gt = len(gt_objs)
            n_hyp = len(hyp_objs)
            
            if n_gt == 0:
                fp += n_hyp
                continue
            if n_hyp == 0:
                fn += n_gt
                continue
            
            # Compute IoU matrix
            iou_matrix = np.zeros((n_gt, n_hyp))
            for i, gt_obj in enumerate(gt_objs):
                for j, hyp_obj in enumerate(hyp_objs):
                    iou_matrix[i, j] = compute_iou(
                        np.array(gt_obj[1:5]), np.array(hyp_obj[1:5])
                    )
            
            # Match
            from scipy.optimize import linear_sum_assignment
            cost = 1 - iou_matrix
            rows, cols = linear_sum_assignment(cost)
            
            matched = 0
            for r, c in zip(rows, cols):
                if iou_matrix[r, c] >= alpha:
                    matched += 1
                    # Simple association accuracy: 1 if same ID, 0 otherwise
                    if gt_objs[r][0] == hyp_objs[c][0]:
                        ass_sum += 1.0
                    else:
                        ass_sum += 0.0
            
            tp += matched
            fp += n_hyp - matched
            fn += n_gt - matched
        
        deta = tp / max(tp + fp + fn, 1)
        assa = ass_sum / max(tp, 1)
        hota = np.sqrt(deta * assa)
        
        deta_per_threshold.append(deta)
        assa_per_threshold.append(assa)
        hota_per_threshold.append(hota)
    
    return {
        'HOTA': round(np.mean(hota_per_threshold) * 100, 2),
        'DetA': round(np.mean(deta_per_threshold) * 100, 2),
        'AssA': round(np.mean(assa_per_threshold) * 100, 2),
    }


def load_mot_gt(gt_path: str) -> Tuple[Dict, Dict]:
    """Load MOT ground truth annotation file.
    
    MOT format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    
    Args:
        gt_path: Path to gt.txt file
        
    Returns:
        gt_boxes: {frame_id: np.ndarray (N, 4) in [x1,y1,x2,y2]}
        gt_ids: {frame_id: np.ndarray (N,)}
    """
    gt_boxes = {}
    gt_ids = {}
    
    path = Path(gt_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split(',')
            if len(parts) < 6:
                continue
            
            frame_id = int(parts[0])
            obj_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            
            # Filter: conf > 0 and class == 1 (pedestrian) if available
            if len(parts) >= 7:
                conf = float(parts[6])
                if conf == 0:
                    continue
            if len(parts) >= 8:
                cls = int(parts[7])
                if cls not in [1, -1]:  # 1=pedestrian, -1=don't care filter
                    continue
            # Visibility filter
            if len(parts) >= 9:
                visibility = float(parts[8])
                if visibility < 0.1:
                    continue
            
            # Convert to [x1, y1, x2, y2]
            box = np.array([x, y, x + w, y + h])
            
            if frame_id not in gt_boxes:
                gt_boxes[frame_id] = []
                gt_ids[frame_id] = []
            
            gt_boxes[frame_id].append(box)
            gt_ids[frame_id].append(obj_id)
    
    # Convert to numpy arrays
    for frame_id in gt_boxes:
        gt_boxes[frame_id] = np.array(gt_boxes[frame_id])
        gt_ids[frame_id] = np.array(gt_ids[frame_id])
    
    return gt_boxes, gt_ids


def load_mot_results(results_path: str) -> Tuple[Dict, Dict]:
    """Load MOT-format tracking results.
    
    Same format as ground truth: frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
    
    Returns:
        hyp_boxes: {frame_id: np.ndarray (N, 4) in [x1,y1,x2,y2]}
        hyp_ids: {frame_id: np.ndarray (N,)}
    """
    hyp_boxes = {}
    hyp_ids = {}
    
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split(',')
            if len(parts) < 6:
                continue
            
            frame_id = int(parts[0])
            obj_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            
            box = np.array([x, y, x + w, y + h])
            
            if frame_id not in hyp_boxes:
                hyp_boxes[frame_id] = []
                hyp_ids[frame_id] = []
            
            hyp_boxes[frame_id].append(box)
            hyp_ids[frame_id].append(obj_id)
    
    for frame_id in hyp_boxes:
        hyp_boxes[frame_id] = np.array(hyp_boxes[frame_id])
        hyp_ids[frame_id] = np.array(hyp_ids[frame_id])
    
    return hyp_boxes, hyp_ids


def evaluate_sequence(gt_path: str, results_path: str, 
                      iou_threshold: float = 0.5) -> Dict[str, float]:
    """Evaluate a tracking result against ground truth.
    
    Args:
        gt_path: Path to ground truth file (MOT format)
        results_path: Path to tracking results file (MOT format)
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dictionary with all computed metrics
    """
    gt_boxes, gt_ids = load_mot_gt(gt_path)
    hyp_boxes, hyp_ids = load_mot_results(results_path)
    
    # Compute basic MOT metrics
    accumulator = MOTAccumulator(iou_threshold=iou_threshold)
    
    all_frames = sorted(set(list(gt_boxes.keys()) + list(hyp_boxes.keys())))
    
    for frame_id in all_frames:
        gt_b = gt_boxes.get(frame_id, np.empty((0, 4)))
        gt_i = gt_ids.get(frame_id, np.array([]))
        hyp_b = hyp_boxes.get(frame_id, np.empty((0, 4)))
        hyp_i = hyp_ids.get(frame_id, np.array([]))
        
        accumulator.update(gt_b, gt_i, hyp_b, hyp_i)
    
    mot_metrics = accumulator.compute_metrics()
    
    # Compute HOTA
    gt_data = {}
    for fid in gt_boxes:
        gt_data[fid] = [(int(gt_ids[fid][i]), *gt_boxes[fid][i].tolist()) 
                        for i in range(len(gt_ids[fid]))]
    
    hyp_data = {}
    for fid in hyp_boxes:
        hyp_data[fid] = [(int(hyp_ids[fid][i]), *hyp_boxes[fid][i].tolist()) 
                         for i in range(len(hyp_ids[fid]))]
    
    try:
        hota_metrics = compute_hota(gt_data, hyp_data)
        mot_metrics.update(hota_metrics)
    except Exception as e:
        logger.warning(f"Could not compute HOTA: {e}")
        mot_metrics['HOTA'] = None
        mot_metrics['DetA'] = None
        mot_metrics['AssA'] = None
    
    return mot_metrics


def try_motmetrics_evaluation(gt_path: str, results_path: str) -> Optional[Dict]:
    """Try to use the motmetrics library for official metric computation.
    
    Returns None if motmetrics is not installed.
    """
    try:
        import motmetrics as mm
    except ImportError:
        logger.info(
            "motmetrics not installed. Using built-in metric computation. "
            "For official benchmark results, install: pip install motmetrics"
        )
        return None
    
    try:
        gt_boxes, gt_ids = load_mot_gt(gt_path)
        hyp_boxes, hyp_ids = load_mot_results(results_path)
        
        acc = mm.MOTAccumulator(auto_id=True)
        
        all_frames = sorted(set(list(gt_boxes.keys()) + list(hyp_boxes.keys())))
        
        for frame_id in all_frames:
            gt_b = gt_boxes.get(frame_id, np.empty((0, 4)))
            gt_i = gt_ids.get(frame_id, np.array([]))
            hyp_b = hyp_boxes.get(frame_id, np.empty((0, 4)))
            hyp_i = hyp_ids.get(frame_id, np.array([]))
            
            if len(gt_b) == 0 and len(hyp_b) == 0:
                continue
            
            # Compute distance matrix (1 - IoU)
            if len(gt_b) > 0 and len(hyp_b) > 0:
                dist = np.zeros((len(gt_b), len(hyp_b)))
                for i in range(len(gt_b)):
                    for j in range(len(hyp_b)):
                        iou = compute_iou(gt_b[i], hyp_b[j])
                        dist[i, j] = 1 - iou
                # Gate: set large IoU distance to NaN
                dist[dist > 0.5] = np.nan
            else:
                dist = np.empty((len(gt_b), len(hyp_b)))
            
            acc.update(
                gt_i.tolist() if len(gt_i) > 0 else [],
                hyp_i.tolist() if len(hyp_i) > 0 else [],
                dist
            )
        
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=[
            'mota', 'idf1', 'num_switches', 'num_false_positives',
            'num_misses', 'precision', 'recall'
        ], name='SmartTrack')
        
        return {
            'MOTA': round(float(summary['mota'].iloc[0]) * 100, 2),
            'IDF1': round(float(summary['idf1'].iloc[0]) * 100, 2),
            'num_id_switches': int(summary['num_switches'].iloc[0]),
            'num_false_positives': int(summary['num_false_positives'].iloc[0]),
            'num_false_negatives': int(summary['num_misses'].iloc[0]),
            'precision': round(float(summary['precision'].iloc[0]) * 100, 2),
            'recall': round(float(summary['recall'].iloc[0]) * 100, 2),
            'source': 'motmetrics'
        }
    except Exception as e:
        logger.error(f"motmetrics evaluation failed: {e}")
        return None
