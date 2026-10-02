"""
ASTRA — Video Frame & Clip Extraction Tool

Extract frames and clips from experiment videos.

Usage:
    python -m ml.data.extract.extract_video --input video.mp4 --output data/frames/session_01/
    python -m ml.data.extract.extract_video --input video.mp4 --output data/clips/ --mode clips --clip_length 5.0
    python -m ml.data.extract.extract_video --input video.mp4 --output data/thumbs/ --mode thumbnails --interval 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_video_info(video_path: str) -> Dict:
    """Extract metadata from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    info = {
        "path": video_path,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_seconds": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    cap.release()
    return info


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval: int = 1,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None,
    image_format: str = "jpg",
    quality: int = 95,
    session_id: str = "session_00",
    camera_id: str = "cam_00",
    video_id: str = "video_00",
    experiment_id: str = "unknown",
) -> Dict:
    """
    Extract frames from a video file.

    Args:
        video_path: Path to input video.
        output_dir: Directory to save extracted frames.
        frame_interval: Extract every Nth frame (1 = every frame).
        start_frame: First frame to extract.
        end_frame: Last frame to extract (None = end of video).
        resize: Optional (width, height) to resize frames.
        image_format: Output format ('jpg', 'png').
        quality: JPEG quality (1-100).
        session_id: Session identifier for metadata.
        camera_id: Camera identifier.
        video_id: Video identifier.
        experiment_id: Experiment identifier.

    Returns:
        Dict with extraction statistics and frame manifest.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if end_frame is None:
        end_frame = total_frames

    end_frame = min(end_frame, total_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_extracted = []
    frame_idx = start_frame
    extracted_count = 0

    logger.info(
        f"Extracting frames from {video_path}: "
        f"frames {start_frame}-{end_frame}, interval={frame_interval}"
    )

    # Encoding params
    if image_format == "jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        ext = ".jpg"
    else:
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        ext = ".png"

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % frame_interval == 0:
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)

            frame_filename = f"frame_{frame_idx:08d}{ext}"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame, encode_params)

            timestamp = frame_idx / fps if fps > 0 else 0.0

            frames_extracted.append({
                "frame_id": frame_idx,
                "filename": frame_filename,
                "timestamp": round(timestamp, 4),
                "experiment_id": experiment_id,
                "session_id": session_id,
                "video_id": video_id,
                "camera_id": camera_id,
            })
            extracted_count += 1

            if extracted_count % 100 == 0:
                logger.info(f"  Extracted {extracted_count} frames...")

        frame_idx += 1

    cap.release()

    # Save frame manifest
    manifest = {
        "video_path": video_path,
        "output_dir": output_dir,
        "experiment_id": experiment_id,
        "session_id": session_id,
        "video_id": video_id,
        "camera_id": camera_id,
        "source_resolution": [width, height],
        "source_fps": fps,
        "source_total_frames": total_frames,
        "extraction_interval": frame_interval,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "resize": list(resize) if resize else None,
        "frames_extracted": extracted_count,
        "frames": frames_extracted,
    }

    manifest_path = os.path.join(output_dir, "frame_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        f"Extraction complete: {extracted_count} frames saved to {output_dir}"
    )

    return manifest


def extract_clips(
    video_path: str,
    output_dir: str,
    clip_length_seconds: float = 5.0,
    overlap_seconds: float = 0.0,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    resize: Optional[Tuple[int, int]] = None,
    session_id: str = "session_00",
    camera_id: str = "cam_00",
    video_id: str = "video_00",
) -> Dict:
    """
    Extract fixed-length clips from a video.

    Args:
        video_path: Path to input video.
        output_dir: Directory to save clips.
        clip_length_seconds: Duration of each clip.
        overlap_seconds: Overlap between consecutive clips.
        start_time: Start time in seconds.
        end_time: End time in seconds (None = end of video).
        resize: Optional (width, height) to resize.
        session_id: Session identifier.
        camera_id: Camera identifier.
        video_id: Video identifier.

    Returns:
        Dict with clip information.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if end_time is None:
        end_time = duration

    clip_length_frames = int(clip_length_seconds * fps)
    overlap_frames = int(overlap_seconds * fps)
    stride_frames = clip_length_frames - overlap_frames

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    clips = []
    clip_idx = 0
    current_start = start_frame

    # Get video codec info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if resize:
        out_w, out_h = resize
    else:
        out_w, out_h = width, height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    while current_start + clip_length_frames <= end_frame:
        clip_end = current_start + clip_length_frames
        clip_filename = f"clip_{clip_idx:06d}.mp4"
        clip_path = os.path.join(output_dir, clip_filename)

        writer = cv2.VideoWriter(clip_path, fourcc, fps, (out_w, out_h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_start)

        for _ in range(clip_length_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if resize:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            writer.write(frame)

        writer.release()

        clips.append({
            "clip_id": clip_idx,
            "filename": clip_filename,
            "start_frame": current_start,
            "end_frame": clip_end,
            "start_time": round(current_start / fps, 4),
            "end_time": round(clip_end / fps, 4),
            "duration": round(clip_length_seconds, 4),
            "session_id": session_id,
            "camera_id": camera_id,
            "video_id": video_id,
        })

        clip_idx += 1
        current_start += stride_frames

    cap.release()

    manifest = {
        "video_path": video_path,
        "output_dir": output_dir,
        "clip_length_seconds": clip_length_seconds,
        "overlap_seconds": overlap_seconds,
        "clips_extracted": len(clips),
        "clips": clips,
    }

    manifest_path = os.path.join(output_dir, "clip_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Extracted {len(clips)} clips to {output_dir}")
    return manifest


def extract_thumbnails(
    video_path: str,
    output_dir: str,
    interval_seconds: float = 10.0,
    thumbnail_size: Tuple[int, int] = (320, 180),
) -> Dict:
    """
    Extract thumbnail images at regular intervals.

    Args:
        video_path: Path to input video.
        output_dir: Directory to save thumbnails.
        interval_seconds: Time between thumbnails.
        thumbnail_size: (width, height) for thumbnails.

    Returns:
        Dict with thumbnail information.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = int(interval_seconds * fps)

    thumbnails = []
    frame_idx = 0
    thumb_idx = 0

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        thumb = cv2.resize(frame, thumbnail_size, interpolation=cv2.INTER_AREA)
        filename = f"thumb_{thumb_idx:06d}.jpg"
        cv2.imwrite(
            os.path.join(output_dir, filename),
            thumb,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )

        thumbnails.append({
            "thumbnail_id": thumb_idx,
            "filename": filename,
            "frame_id": frame_idx,
            "timestamp": round(frame_idx / fps, 4),
        })

        thumb_idx += 1
        frame_idx += interval_frames

    cap.release()

    manifest = {
        "video_path": video_path,
        "thumbnails_extracted": len(thumbnails),
        "interval_seconds": interval_seconds,
        "thumbnail_size": list(thumbnail_size),
        "thumbnails": thumbnails,
    }

    manifest_path = os.path.join(output_dir, "thumbnail_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Extracted {len(thumbnails)} thumbnails to {output_dir}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="ASTRA Video Extraction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract all frames
    python -m ml.data.extract.extract_video --input video.mp4 --output data/frames/

    # Extract every 5th frame
    python -m ml.data.extract.extract_video --input video.mp4 --output data/frames/ --interval 5

    # Extract 5-second clips
    python -m ml.data.extract.extract_video --input video.mp4 --output data/clips/ --mode clips --clip_length 5.0

    # Extract thumbnails every 10 seconds
    python -m ml.data.extract.extract_video --input video.mp4 --output data/thumbs/ --mode thumbnails --thumb_interval 10

    # Get video info only
    python -m ml.data.extract.extract_video --input video.mp4 --mode info
        """,
    )

    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument(
        "--mode",
        choices=["frames", "clips", "thumbnails", "info"],
        default="frames",
        help="Extraction mode",
    )

    # Frame extraction options
    parser.add_argument("--interval", type=int, default=1, help="Frame interval (extract every Nth frame)")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame")
    parser.add_argument("--resize_w", type=int, default=None, help="Resize width")
    parser.add_argument("--resize_h", type=int, default=None, help="Resize height")
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg", help="Image format")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality")

    # Clip extraction options
    parser.add_argument("--clip_length", type=float, default=5.0, help="Clip length in seconds")
    parser.add_argument("--overlap", type=float, default=0.0, help="Clip overlap in seconds")

    # Thumbnail options
    parser.add_argument("--thumb_interval", type=float, default=10.0, help="Thumbnail interval in seconds")
    parser.add_argument("--thumb_w", type=int, default=320, help="Thumbnail width")
    parser.add_argument("--thumb_h", type=int, default=180, help="Thumbnail height")

    # Metadata
    parser.add_argument("--experiment_id", default="unknown", help="Experiment ID")
    parser.add_argument("--session_id", default="session_00", help="Session ID")
    parser.add_argument("--video_id", default="video_00", help="Video ID")
    parser.add_argument("--camera_id", default="cam_00", help="Camera ID")

    args = parser.parse_args()

    if args.mode == "info":
        info = get_video_info(args.input)
        print(json.dumps(info, indent=2))
        return

    if not args.output:
        print("Error: --output is required for extraction modes", file=sys.stderr)
        sys.exit(1)

    resize = None
    if args.resize_w and args.resize_h:
        resize = (args.resize_w, args.resize_h)

    if args.mode == "frames":
        extract_frames(
            video_path=args.input,
            output_dir=args.output,
            frame_interval=args.interval,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            resize=resize,
            image_format=args.format,
            quality=args.quality,
            experiment_id=args.experiment_id,
            session_id=args.session_id,
            video_id=args.video_id,
            camera_id=args.camera_id,
        )
    elif args.mode == "clips":
        extract_clips(
            video_path=args.input,
            output_dir=args.output,
            clip_length_seconds=args.clip_length,
            overlap_seconds=args.overlap,
            resize=resize,
            session_id=args.session_id,
            camera_id=args.camera_id,
            video_id=args.video_id,
        )
    elif args.mode == "thumbnails":
        extract_thumbnails(
            video_path=args.input,
            output_dir=args.output,
            interval_seconds=args.thumb_interval,
            thumbnail_size=(args.thumb_w, args.thumb_h),
        )


if __name__ == "__main__":
    main()
