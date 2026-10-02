"""
ASTRA — Video Data Collection & Extraction Tools

Extracts frames, clips, thumbnails from experiment videos.
Generates annotation templates and session manifests.

Usage:
    python -m ml.data.collect --input video.mp4 --output data/raw/session_001/
    python -m ml.data.collect --input video.mp4 --output data/raw/session_001/ --fps 5 --clips
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 5.0,
    max_frames: int = -1,
    resize: tuple = None,
) -> dict:
    """
    Extract frames from video at specified FPS.

    Returns manifest dict.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps
    frame_interval = max(1, int(src_fps / fps))

    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    manifest = {
        "video_path": os.path.abspath(video_path),
        "source_fps": src_fps,
        "extract_fps": fps,
        "total_source_frames": total_frames,
        "duration_seconds": round(duration, 2),
        "resize": list(resize) if resize else None,
        "extraction_date": datetime.now().isoformat(),
        "frames": [],
    }

    frame_idx = 0
    extracted = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % frame_interval != 0:
            continue
        if max_frames > 0 and extracted >= max_frames:
            break

        if resize:
            frame = cv2.resize(frame, resize)

        fname = f"frame_{extracted:06d}.jpg"
        fpath = os.path.join(frames_dir, fname)
        cv2.imwrite(fpath, frame)

        manifest["frames"].append({
            "frame_id": extracted,
            "source_frame": frame_idx,
            "timestamp": round(frame_idx / src_fps, 4),
            "filename": fname,
        })
        extracted += 1

    cap.release()
    manifest["extracted_frames"] = extracted
    logger.info(f"Extracted {extracted} frames from {video_path}")
    return manifest


def extract_clips(
    video_path: str,
    output_dir: str,
    clip_duration: float = 5.0,
    overlap: float = 0.0,
) -> list:
    """Extract temporal clips from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps

    clips_dir = os.path.join(output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    step = clip_duration - overlap
    clips = []
    clip_idx = 0
    t = 0.0

    while t < duration:
        start_frame = int(t * src_fps)
        end_frame = min(int((t + clip_duration) * src_fps), total_frames)

        clip_name = f"clip_{clip_idx:04d}.mp4"
        clip_path = os.path.join(clips_dir, clip_name)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_path, fourcc, src_fps, (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()

        clips.append({
            "clip_id": clip_idx,
            "filename": clip_name,
            "start_time": round(t, 3),
            "end_time": round(min(t + clip_duration, duration), 3),
            "start_frame": start_frame,
            "end_frame": end_frame,
        })

        clip_idx += 1
        t += step

    cap.release()
    logger.info(f"Extracted {len(clips)} clips from {video_path}")
    return clips


def generate_thumbnails(
    video_path: str,
    output_dir: str,
    n_thumbnails: int = 10,
    size: tuple = (160, 120),
) -> list:
    """Generate evenly-spaced thumbnails for preview."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    thumb_dir = os.path.join(output_dir, "thumbnails")
    os.makedirs(thumb_dir, exist_ok=True)

    interval = max(1, total // n_thumbnails)
    thumbs = []

    for i in range(n_thumbnails):
        frame_pos = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            break
        thumb = cv2.resize(frame, size)
        fname = f"thumb_{i:03d}.jpg"
        cv2.imwrite(os.path.join(thumb_dir, fname), thumb)
        thumbs.append({
            "index": i, "filename": fname,
            "frame": frame_pos, "timestamp": round(frame_pos / fps, 2),
        })

    cap.release()
    return thumbs


def generate_annotation_template(
    manifest: dict,
    experiment_id: str,
    session_id: str,
    performer_id: str,
    output_dir: str,
) -> str:
    """Generate empty annotation template files for a session."""
    ann_dir = os.path.join(output_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    # Detection template
    det_template = {
        "info": {"experiment_id": experiment_id, "session_id": session_id,
                 "performer_id": performer_id, "date": datetime.now().isoformat()},
        "images": [{"id": f["frame_id"], "file_name": f["filename"],
                     "frame_id": f["frame_id"], "timestamp": f["timestamp"]}
                    for f in manifest.get("frames", [])],
        "annotations": [],
        "categories": [],
    }
    with open(os.path.join(ann_dir, "detections.json"), "w") as f:
        json.dump(det_template, f, indent=2)

    # JSONL templates
    for name in ["poses", "actions", "hoi", "object_states", "steps", "anomalies"]:
        with open(os.path.join(ann_dir, f"{name}.jsonl"), "w") as f:
            f.write("")  # Empty JSONL

    # Step sequence template
    seq_template = {"experiment_id": experiment_id, "session_id": session_id, "sequence": []}
    with open(os.path.join(ann_dir, "step_sequence.json"), "w") as f:
        json.dump(seq_template, f, indent=2)

    # Session manifest
    session_manifest = {
        "experiment_id": experiment_id,
        "session_id": session_id,
        "performer_id": performer_id,
        "video_path": manifest.get("video_path", ""),
        "duration": manifest.get("duration_seconds", 0),
        "extracted_frames": manifest.get("extracted_frames", 0),
        "extract_fps": manifest.get("extract_fps", 5),
        "annotation_status": "empty",
        "created": datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "session_manifest.json"), "w") as f:
        json.dump(session_manifest, f, indent=2)

    logger.info(f"Annotation templates created in: {ann_dir}")
    return ann_dir


def main():
    parser = argparse.ArgumentParser(description="ASTRA Data Collection")
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--fps", type=float, default=5.0, help="Extraction FPS")
    parser.add_argument("--max_frames", type=int, default=-1, help="Max frames")
    parser.add_argument("--clips", action="store_true", help="Also extract clips")
    parser.add_argument("--clip_duration", type=float, default=5.0, help="Clip length (s)")
    parser.add_argument("--thumbnails", type=int, default=10, help="Number of thumbnails")
    parser.add_argument("--experiment_id", default="exp_001")
    parser.add_argument("--session_id", default=None)
    parser.add_argument("--performer_id", default="performer_001")
    parser.add_argument("--resize", nargs=2, type=int, default=None, help="Resize W H")

    args = parser.parse_args()

    if args.session_id is None:
        args.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(args.output, exist_ok=True)

    # Extract frames
    resize = tuple(args.resize) if args.resize else None
    manifest = extract_frames(args.input, args.output, fps=args.fps,
                               max_frames=args.max_frames, resize=resize)

    # Save manifest
    with open(os.path.join(args.output, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Extract clips
    if args.clips:
        clips = extract_clips(args.input, args.output, clip_duration=args.clip_duration)
        with open(os.path.join(args.output, "clips_manifest.json"), "w") as f:
            json.dump(clips, f, indent=2)

    # Thumbnails
    thumbs = generate_thumbnails(args.input, args.output, n_thumbnails=args.thumbnails)
    with open(os.path.join(args.output, "thumbnails.json"), "w") as f:
        json.dump(thumbs, f, indent=2)

    # Annotation templates
    generate_annotation_template(manifest, args.experiment_id, args.session_id,
                                  args.performer_id, args.output)

    print(f"\nData collection complete:")
    print(f"  Frames:     {manifest['extracted_frames']}")
    print(f"  Session:    {args.session_id}")
    print(f"  Output:     {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
