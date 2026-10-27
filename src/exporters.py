"""Export utilities for SmartTrack-AI tracking results and performance metrics."""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def _normalize_tracking_records(
    results: Sequence[Any],
) -> List[Tuple[int, int, float, float, float, float, float, str]]:
    """Flatten and normalize diverse tracking results structures into standard tuples:
    (frame_id, track_id, x1, y1, x2, y2, confidence, state).

    Supports:
    - List of per-frame metric dicts containing 'tracks' list
    - Flat list of track dicts
    - List of Track objects
    - List of tuple/list rows
    """
    records: List[Tuple[int, int, float, float, float, float, float, str]] = []

    for item in results:
        # Case 1: Per-frame metric dictionary
        if isinstance(item, dict) and ("tracks" in item or "current_tracks" in item):
            fid = item.get("frame_id", item.get("frame_idx", 0))
            tracks = item.get("tracks", item.get("current_tracks", []))
            for trk in tracks:
                if isinstance(trk, dict):
                    tid = trk.get("track_id", trk.get("id", 0))
                    if all(k in trk for k in ("x1", "y1", "x2", "y2")):
                        x1, y1, x2, y2 = trk["x1"], trk["y1"], trk["x2"], trk["y2"]
                    elif "bbox" in trk:
                        b = trk["bbox"]
                        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                    elif "box" in trk:
                        b = trk["box"]
                        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                    else:
                        x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
                    conf = trk.get("confidence", trk.get("conf", trk.get("score", 1.0)))
                    state = str(trk.get("state", "confirmed"))
                    records.append(
                        (
                            int(fid),
                            int(tid),
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                            float(conf),
                            state,
                        )
                    )
                else:
                    # Object instance with attributes
                    tid = getattr(trk, "track_id", getattr(trk, "id", 0))
                    box = getattr(
                        trk, "box", getattr(trk, "bbox", getattr(trk, "tlbr", None))
                    )
                    if box is None and hasattr(trk, "to_tlbr") and callable(trk.to_tlbr):
                        box = trk.to_tlbr()
                    if box is not None:
                        b = np.asarray(box, dtype=float).ravel()
                        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                    else:
                        x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
                    conf = getattr(trk, "confidence", getattr(trk, "score", 1.0))
                    st = getattr(trk, "state", "confirmed")
                    state_str = st.name.lower() if hasattr(st, "name") else str(st).lower()
                    records.append(
                        (
                            int(fid),
                            int(tid),
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                            float(conf),
                            state_str,
                        )
                    )

        # Case 2: Flat dictionary per track detection
        elif isinstance(item, dict):
            fid = item.get("frame_id", item.get("frame_idx", item.get("frame", 0)))
            tid = item.get("track_id", item.get("id", 0))
            if all(k in item for k in ("x1", "y1", "x2", "y2")):
                x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
            elif "bbox" in item:
                b = item["bbox"]
                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            elif "box" in item:
                b = item["box"]
                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            elif all(k in item for k in ("bb_left", "bb_top", "bb_width", "bb_height")):
                x1 = item["bb_left"]
                y1 = item["bb_top"]
                x2 = x1 + item["bb_width"]
                y2 = y1 + item["bb_height"]
            else:
                x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
            conf = item.get("confidence", item.get("conf", item.get("score", 1.0)))
            state = str(item.get("state", "confirmed"))
            records.append(
                (
                    int(fid),
                    int(tid),
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(conf),
                    state,
                )
            )

        # Case 3: Raw tuple / list
        elif isinstance(item, (list, tuple)):
            if len(item) >= 8:
                fid, tid, x1, y1, x2, y2, conf, state = item[:8]
                records.append(
                    (
                        int(fid),
                        int(tid),
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                        float(conf),
                        str(state),
                    )
                )
            elif len(item) == 7:
                fid, tid, x1, y1, x2, y2, conf = item
                records.append(
                    (
                        int(fid),
                        int(tid),
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                        float(conf),
                        "confirmed",
                    )
                )
            elif len(item) == 6:
                fid, tid, x1, y1, x2, y2 = item
                records.append(
                    (
                        int(fid),
                        int(tid),
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                        1.0,
                        "confirmed",
                    )
                )

    return records


def export_tracking_csv(results: list, path: str) -> None:
    """Export tracking results as a CSV file.

    Columns: frame_id, track_id, x1, y1, x2, y2, confidence, state

    Args:
        results: Tracking results or per-frame metrics.
        path: Destination file path for CSV.
    """
    records = _normalize_tracking_records(results)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["frame_id", "track_id", "x1", "y1", "x2", "y2", "confidence", "state"]

    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in records:
            writer.writerow(
                [
                    r[0],  # frame_id
                    r[1],  # track_id
                    f"{r[2]:.2f}",  # x1
                    f"{r[3]:.2f}",  # y1
                    f"{r[4]:.2f}",  # x2
                    f"{r[5]:.2f}",  # y2
                    f"{r[6]:.4f}",  # confidence
                    r[7],  # state
                ]
            )

    logger.info(f"Exported {len(records)} tracking records to CSV: {path}")


def export_mot_format(results: list, path: str) -> None:
    """Export tracking results in MOT Challenge format.

    MOT Challenge 2D format:
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
    Where frame is 1-indexed, and x, y, z are set to -1 for 2D MOT.

    Args:
        results: Tracking results or per-frame metrics.
        path: Destination file path for MOT format text file.
    """
    records = _normalize_tracking_records(results)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Check whether frames are 0-indexed; MOT challenge expects 1-indexed frames
    min_fid = min([r[0] for r in records]) if records else 1
    offset = 1 if min_fid == 0 else 0

    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for r in records:
            frame_idx = r[0] + offset
            track_id = r[1]
            x1, y1, x2, y2 = r[2], r[3], r[4], r[5]
            bb_left = x1
            bb_top = y1
            bb_width = max(0.0, x2 - x1)
            bb_height = max(0.0, y2 - y1)
            conf = r[6]
            x, y, z = -1, -1, -1

            writer.writerow(
                [
                    frame_idx,
                    track_id,
                    f"{bb_left:.2f}",
                    f"{bb_top:.2f}",
                    f"{bb_width:.2f}",
                    f"{bb_height:.2f}",
                    f"{conf:.4f}",
                    x,
                    y,
                    z,
                ]
            )

    logger.info(f"Exported {len(records)} tracking records to MOT format: {path}")


def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy types and non-serializable objects for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, set):
        return [_to_serializable(x) for x in sorted(obj)]
    if hasattr(obj, "__dict__"):
        return _to_serializable(vars(obj))
    return obj


def export_json_summary(config: dict, stats: dict, path: str) -> None:
    """Write JSON summary of the tracking run including configuration and statistics.

    Args:
        config: Configuration dictionary used during tracking.
        stats: Tracking performance statistics dictionary.
        path: Destination file path for JSON.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "config": _to_serializable(config),
        "statistics": _to_serializable(stats),
    }

    with open(out_path, mode="w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, default=str)

    logger.info(f"Exported run summary JSON to: {path}")


def generate_results_report(stats: dict, comparison: Optional[dict] = None) -> str:
    """Generate a clean, structured Markdown tracking performance report.

    Args:
        stats: Primary tracking statistics dictionary.
        comparison: Optional secondary statistics dictionary for comparative benchmarking.

    Returns:
        Formatted Markdown report string.
    """
    mode = stats.get("mode", "enhanced").upper()
    total_frames = stats.get("frame_count", stats.get("total_frames", 0))
    fps = stats.get("processing_fps", stats.get("avg_fps", 0.0))
    avg_latency = stats.get("avg_frame_time_ms", 0.0)
    total_tracks = stats.get("total_tracks", 0)
    confirmed_tracks = stats.get("confirmed_tracks", 0)
    id_switches = stats.get("total_id_switches", stats.get("id_switches", 0))
    tentative_tracks = stats.get("tentative_tracks", 0)
    lost_tracks = stats.get("lost_tracks", 0)

    report_lines = [
        "# SmartTrack-AI Tracking Performance Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Tracker Mode:** `{mode}`",
        "",
        "## 1. Executive Summary",
        "",
        f"The tracker processed **{total_frames} frames** at an average throughput of "
        f"**{fps:.1f} FPS** ({avg_latency:.2f} ms/frame). Across the sequence, a total of "
        f"**{total_tracks} unique tracks** were initiated with **{id_switches} total ID switches**.",
        "",
        "## 2. Key Performance Metrics",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Tracking Mode** | `{mode}` |",
        f"| **Total Processed Frames** | {total_frames:,} |",
        f"| **Processing Speed** | {fps:.1f} FPS |",
        f"| **Average Frame Latency** | {avg_latency:.2f} ms |",
        f"| **Total Unique Tracks** | {total_tracks} |",
        f"| **Confirmed Tracks** | {confirmed_tracks} |",
        f"| **Tentative Tracks** | {tentative_tracks} |",
        f"| **Lost / Terminated Tracks** | {lost_tracks} |",
        f"| **Total Identity Switches** | {id_switches} |",
    ]

    if comparison is not None:
        comp_mode = comparison.get("mode", "baseline").upper()
        comp_fps = comparison.get("processing_fps", comparison.get("avg_fps", 0.0))
        comp_switches = comparison.get("total_id_switches", comparison.get("id_switches", 0))
        comp_tracks = comparison.get("total_tracks", 0)
        comp_latency = comparison.get("avg_frame_time_ms", 0.0)

        # ID switch reduction calculation
        if comp_switches > 0:
            switch_diff = comp_switches - id_switches
            reduction_pct = (switch_diff / comp_switches) * 100.0
            switch_note = f"{reduction_pct:+.1f}% ({switch_diff:+d})"
        else:
            switch_note = "N/A"

        # FPS difference
        fps_diff = fps - comp_fps

        report_lines.extend(
            [
                "",
                "## 3. Comparative Analysis",
                "",
                f"Comparing primary run (`{mode}`) against baseline (`{comp_mode}`):",
                "",
                f"| Metric | {comp_mode} (Baseline) | {mode} (Enhanced) | Delta / Improvement |",
                "| :--- | :--- | :--- | :--- |",
                f"| **Identity Switches** | {comp_switches} | {id_switches} | **{switch_note}** |",
                f"| **Unique Tracks** | {comp_tracks} | {total_tracks} | {total_tracks - comp_tracks:+d} |",
                f"| **Throughput (FPS)** | {comp_fps:.1f} FPS | {fps:.1f} FPS | {fps_diff:+.1f} FPS |",
                f"| **Frame Latency (ms)** | {comp_latency:.2f} ms | {avg_latency:.2f} ms | {avg_latency - comp_latency:+.2f} ms |",
                "",
                "### Analysis & Key Insights",
                "",
                f"- **Identity Preservation**: `{mode}` achieved an ID switch reduction of **{switch_note}** "
                f"relative to `{comp_mode}`, demonstrating effective re-association and occlusion handling.",
                f"- **Runtime Efficiency**: Tracking operated at **{fps:.1f} FPS**, "
                "demonstrating real-time capability while retaining robust identity management.",
            ]
        )

    report_lines.extend(
        [
            "",
            "---",
            "*Report auto-generated by SmartTrack-AI Video Processing Engine.*",
            "",
        ]
    )

    return "\n".join(report_lines)
