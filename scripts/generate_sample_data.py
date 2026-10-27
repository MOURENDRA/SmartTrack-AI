"""Utility script to generate synthetic test videos and MOT evaluation sequences.

Allows full testing of tracking, association, re-association, and benchmark
evaluation even in offline environments without downloading external gigabyte datasets.
"""

import os
from pathlib import Path
import numpy as np


def generate_synthetic_video(
    output_path: str,
    num_frames: int = 60,
    width: int = 640,
    height: int = 480,
    fps: float = 20.0,
) -> str:
    """Generate a realistic synthetic video with 2 pedestrians walking towards each other,
    undergoing dense mutual occlusion in the middle frames, and emerging on opposite sides.
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed yet. Skipping video generation.")
        return output_path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Pedestrian dimensions
    pw, ph = 60, 140
    y_pos = int(height * 0.45)

    # Person 1: walks from left (x=50) to right (x=550) - blue tint
    # Person 2: walks from right (x=550) to left (x=50) - red tint
    for f in range(num_frames):
        alpha = f / float(num_frames - 1)
        p1_x = int(50 + alpha * 500)
        p2_x = int(550 - alpha * 500)

        # Background: office corridor floor
        frame = np.full((height, width, 3), 35, dtype=np.uint8)
        # Add floor grid lines
        for y_line in range(y_pos + ph, height, 40):
            cv2.line(frame, (0, y_line), (width, y_line), (45, 45, 45), 1)

        # Determine draw order: person 1 is closer during first half, person 2 during second half
        order = [1, 2] if f < num_frames // 2 else [2, 1]

        for p in order:
            if p == 1:
                px = p1_x
                color_body = (180, 80, 50)  # Blue-ish BGR
                color_head = (200, 180, 160)
            else:
                px = p2_x
                color_body = (60, 60, 200)  # Red-ish BGR
                color_head = (210, 190, 170)

            # Draw person body (torso and legs)
            cv2.rectangle(
                frame, (px, y_pos + 30), (px + pw, y_pos + ph), color_body, -1
            )
            cv2.rectangle(
                frame, (px, y_pos + 30), (px + pw, y_pos + ph), (255, 255, 255), 1
            )
            # Draw head
            center_x = px + pw // 2
            cv2.circle(frame, (center_x, y_pos + 15), 16, color_head, -1)
            cv2.circle(frame, (center_x, y_pos + 15), 16, (255, 255, 255), 1)

        # Add visual timestamp/frame counter
        cv2.putText(
            frame,
            f"Frame: {f+1}/{num_frames} | Synthetic Occlusion Test",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()
    print(f"Synthetic video successfully saved to: {output_path}")
    return output_path


def generate_synthetic_mot_dataset(output_dir: str, num_frames: int = 50) -> str:
    """Create a valid MOT-compliant dataset directory (SYNTH-OCCLUSION-01)
    with seqinfo.ini and ground truth gt/gt.txt.
    """
    base_dir = Path(output_dir)
    gt_dir = base_dir / "gt"
    img_dir = base_dir / "img1"
    gt_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1. seqinfo.ini
    seqinfo_content = f"""[Sequence]
name=SYNTH-OCCLUSION-01
imDir=img1
frameRate=20
seqLength={num_frames}
imWidth=640
imHeight=480
imExt=.jpg
"""
    with open(base_dir / "seqinfo.ini", "w", encoding="utf-8") as f:
        f.write(seqinfo_content)

    # 2. gt/gt.txt
    # Format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class_id, visibility
    gt_lines = []
    pw, ph = 60, 140
    y_pos = int(480 * 0.45)

    for f in range(1, num_frames + 1):
        alpha = (f - 1) / float(num_frames - 1)
        p1_x = int(50 + alpha * 500)
        p2_x = int(550 - alpha * 500)

        # Visibility drops during middle frames when crossing
        overlap_dist = abs(p1_x - p2_x)
        vis1 = max(0.2, min(1.0, overlap_dist / 60.0))
        vis2 = max(0.2, min(1.0, overlap_dist / 60.0))

        # Person 1
        gt_lines.append(f"{f},1,{p1_x},{y_pos},{pw},{ph},1,1,{vis1:.2f}")
        # Person 2
        gt_lines.append(f"{f},2,{p2_x},{y_pos},{pw},{ph},1,1,{vis2:.2f}")

    with open(gt_dir / "gt.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(gt_lines) + "\n")

    print(f"Synthetic MOT sequence generated at: {output_dir}")
    return str(base_dir)


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    sample_vid = data_dir / "sample_videos" / "crossing_pedestrians.mp4"
    generate_synthetic_video(str(sample_vid))

    synth_dataset = data_dir / "datasets" / "SYNTH-OCCLUSION-01"
    generate_synthetic_mot_dataset(str(synth_dataset))
