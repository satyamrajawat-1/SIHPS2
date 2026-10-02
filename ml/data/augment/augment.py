"""
ASTRA — Data Augmentation Pipeline

Rotation-robust augmentations for microgravity environments.
Transforms images, bounding boxes, keypoints, and skeleton data consistently.

Key design: arbitrary orientation augmentation is critical because
standard upright-human assumptions do NOT apply in microgravity.

Usage:
    python -m ml.data.augment.augment --config configs/augmentation.yaml --input data/frames/ --output data/augmented/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for augmentation pipeline."""
    # Rotation (CRITICAL for microgravity)
    rotation_enabled: bool = True
    rotation_range: Tuple[float, float] = (-180.0, 180.0)  # Full rotation
    rotation_prob: float = 0.8

    # Scale
    scale_enabled: bool = True
    scale_range: Tuple[float, float] = (0.8, 1.2)
    scale_prob: float = 0.5

    # Translation
    translate_enabled: bool = True
    translate_range: Tuple[float, float] = (-0.1, 0.1)  # fraction of image size
    translate_prob: float = 0.5

    # Perspective
    perspective_enabled: bool = True
    perspective_strength: float = 0.05
    perspective_prob: float = 0.3

    # Brightness
    brightness_enabled: bool = True
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    brightness_prob: float = 0.5

    # Contrast
    contrast_enabled: bool = True
    contrast_range: Tuple[float, float] = (0.7, 1.3)
    contrast_prob: float = 0.5

    # Blur
    blur_enabled: bool = True
    blur_kernel_range: Tuple[int, int] = (3, 7)
    blur_prob: float = 0.2

    # Noise
    noise_enabled: bool = True
    noise_std_range: Tuple[float, float] = (5.0, 25.0)
    noise_prob: float = 0.3

    # Partial occlusion
    occlusion_enabled: bool = True
    occlusion_count_range: Tuple[int, int] = (0, 3)
    occlusion_size_range: Tuple[float, float] = (0.02, 0.1)  # fraction of image
    occlusion_prob: float = 0.3

    # Random crop
    crop_enabled: bool = True
    crop_range: Tuple[float, float] = (0.8, 1.0)  # fraction of image
    crop_prob: float = 0.3

    # Horizontal flip (CAREFUL: only when left/right can be remapped)
    hflip_enabled: bool = False  # Disabled by default for safety
    hflip_prob: float = 0.5

    # Random seed
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "AugmentationConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        aug_data = data.get("augmentation", data)
        config = cls()
        for key, value in aug_data.items():
            if hasattr(config, key):
                if isinstance(value, list) and len(value) == 2:
                    setattr(config, key, tuple(value))
                else:
                    setattr(config, key, value)
        return config

    def to_dict(self) -> Dict[str, Any]:
        return {k: list(v) if isinstance(v, tuple) else v
                for k, v in self.__dict__.items()}


def rotate_image(
    image: np.ndarray,
    angle: float,
    border_value: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate image by arbitrary angle. Returns rotated image and the rotation matrix.

    Args:
        image: Input image (H, W, C).
        angle: Rotation angle in degrees (counter-clockwise).
        border_value: Fill color for borders.

    Returns:
        (rotated_image, rotation_matrix_2x3)
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding box to avoid cropping
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # Adjust rotation matrix for the new image size
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0

    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    return rotated, M


def transform_bbox(
    bbox: List[float],
    M: np.ndarray,
    img_w: int,
    img_h: int,
) -> List[float]:
    """
    Transform a bounding box [x, y, w, h] using an affine matrix.
    Returns the axis-aligned bounding box of the transformed corners.
    """
    x, y, w, h = bbox
    corners = np.array([
        [x, y, 1],
        [x + w, y, 1],
        [x + w, y + h, 1],
        [x, y + h, 1],
    ], dtype=np.float64)

    transformed = (M @ corners.T).T  # (4, 2)

    x_min = max(0, transformed[:, 0].min())
    y_min = max(0, transformed[:, 1].min())
    x_max = min(img_w, transformed[:, 0].max())
    y_max = min(img_h, transformed[:, 1].max())

    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def transform_keypoints(
    keypoints: List[List[float]],
    M: np.ndarray,
    img_w: int,
    img_h: int,
) -> List[List[float]]:
    """
    Transform 2D keypoints [[x, y, conf], ...] using an affine matrix.
    Keypoints outside the image boundary get confidence set to 0.
    """
    result = []
    for kp in keypoints:
        x, y = kp[0], kp[1]
        conf = kp[2] if len(kp) > 2 else 1.0

        point = np.array([x, y, 1.0])
        transformed = M @ point  # (2,)

        tx, ty = transformed[0], transformed[1]

        # Mark as invisible if outside image
        if tx < 0 or tx >= img_w or ty < 0 or ty >= img_h:
            conf = 0.0

        result.append([float(tx), float(ty), float(conf)])

    return result


def transform_skeleton_3d(
    skeleton: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    """
    Apply mathematically correct rotation to skeleton coordinates (x, y, [z]).
    This is for skeleton sequences used in action recognition.

    Args:
        skeleton: (J, C) array where C >= 2 (x, y, [z], [conf], ...)
        angle_deg: Rotation angle in degrees.

    Returns:
        Rotated skeleton array.
    """
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    result = skeleton.copy()
    # Rotate x, y
    x = skeleton[:, 0]
    y = skeleton[:, 1]
    result[:, 0] = cos_a * x - sin_a * y
    result[:, 1] = sin_a * x + cos_a * y

    return result


def apply_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Adjust brightness by a multiplicative factor."""
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """Adjust contrast by a multiplicative factor around the mean."""
    mean = image.mean()
    return np.clip(
        mean + (image.astype(np.float32) - mean) * factor, 0, 255
    ).astype(np.uint8)


def apply_gaussian_noise(image: np.ndarray, std: float) -> np.ndarray:
    """Add Gaussian noise to the image."""
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply Gaussian blur."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_random_occlusion(
    image: np.ndarray,
    count: int,
    size_fraction: float,
) -> np.ndarray:
    """Add random rectangular occlusions to simulate partial obstruction."""
    h, w = image.shape[:2]
    result = image.copy()

    for _ in range(count):
        occ_w = int(w * size_fraction * random.uniform(0.5, 1.5))
        occ_h = int(h * size_fraction * random.uniform(0.5, 1.5))
        x = random.randint(0, max(0, w - occ_w))
        y = random.randint(0, max(0, h - occ_h))

        # Use random gray color for occlusion
        color = random.randint(0, 128)
        result[y:y + occ_h, x:x + occ_w] = color

    return result


class AugmentationPipeline:
    """
    Augmentation pipeline that consistently transforms images, bboxes, and keypoints.
    """

    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        np.random.seed(config.seed)

    def augment_frame(
        self,
        image: np.ndarray,
        bboxes: Optional[List[List[float]]] = None,
        keypoints: Optional[List[List[List[float]]]] = None,
    ) -> Dict[str, Any]:
        """
        Apply augmentations to a frame and its annotations.

        Args:
            image: (H, W, C) image array.
            bboxes: List of [x, y, w, h] bounding boxes.
            keypoints: List of keypoint arrays [[x, y, conf], ...] per person.

        Returns:
            Dict with 'image', 'bboxes', 'keypoints', 'transforms_applied'.
        """
        transforms_applied = []
        h, w = image.shape[:2]

        # Cumulative affine matrix (identity)
        M_total = np.eye(2, 3, dtype=np.float64)

        # 1. Rotation (most important for microgravity)
        if self.config.rotation_enabled and self.rng.random() < self.config.rotation_prob:
            angle = self.rng.uniform(*self.config.rotation_range)
            image, M_rot = rotate_image(image, angle)
            h, w = image.shape[:2]
            M_total = M_rot
            transforms_applied.append(f"rotate({angle:.1f}°)")

        # 2. Scale
        if self.config.scale_enabled and self.rng.random() < self.config.scale_prob:
            scale = self.rng.uniform(*self.config.scale_range)
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w > 0 and new_h > 0:
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                M_scale = np.array([[scale, 0, 0], [0, scale, 0]], dtype=np.float64)
                M_total = M_scale @ np.vstack([M_total, [0, 0, 1]])
                M_total = M_total[:2]
                h, w = image.shape[:2]
                transforms_applied.append(f"scale({scale:.2f})")

        # 3. Brightness
        if self.config.brightness_enabled and self.rng.random() < self.config.brightness_prob:
            factor = self.rng.uniform(*self.config.brightness_range)
            image = apply_brightness(image, factor)
            transforms_applied.append(f"brightness({factor:.2f})")

        # 4. Contrast
        if self.config.contrast_enabled and self.rng.random() < self.config.contrast_prob:
            factor = self.rng.uniform(*self.config.contrast_range)
            image = apply_contrast(image, factor)
            transforms_applied.append(f"contrast({factor:.2f})")

        # 5. Blur
        if self.config.blur_enabled and self.rng.random() < self.config.blur_prob:
            kernel = self.rng.randint(*self.config.blur_kernel_range)
            image = apply_blur(image, kernel)
            transforms_applied.append(f"blur(k={kernel})")

        # 6. Noise
        if self.config.noise_enabled and self.rng.random() < self.config.noise_prob:
            std = self.rng.uniform(*self.config.noise_std_range)
            image = apply_gaussian_noise(image, std)
            transforms_applied.append(f"noise(σ={std:.1f})")

        # 7. Occlusion
        if self.config.occlusion_enabled and self.rng.random() < self.config.occlusion_prob:
            count = self.rng.randint(*self.config.occlusion_count_range)
            size = self.rng.uniform(*self.config.occlusion_size_range)
            if count > 0:
                image = apply_random_occlusion(image, count, size)
                transforms_applied.append(f"occlusion(n={count})")

        # Transform bboxes
        aug_bboxes = None
        if bboxes is not None:
            aug_bboxes = [transform_bbox(bb, M_total, w, h) for bb in bboxes]

        # Transform keypoints
        aug_keypoints = None
        if keypoints is not None:
            aug_keypoints = [
                transform_keypoints(kps, M_total, w, h) for kps in keypoints
            ]

        return {
            "image": image,
            "bboxes": aug_bboxes,
            "keypoints": aug_keypoints,
            "transforms_applied": transforms_applied,
            "affine_matrix": M_total.tolist(),
            "output_size": [w, h],
        }

    def augment_skeleton_sequence(
        self,
        skeleton_seq: np.ndarray,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Augment a skeleton sequence for action recognition training.
        Applies mathematically correct rotation to coordinates.

        Args:
            skeleton_seq: (T, J, C) skeleton sequence.

        Returns:
            (augmented_skeleton, transforms_applied)
        """
        transforms = []
        seq = skeleton_seq.copy()

        # Rotation
        if self.config.rotation_enabled and self.rng.random() < self.config.rotation_prob:
            angle = self.rng.uniform(*self.config.rotation_range)
            for t in range(seq.shape[0]):
                seq[t] = transform_skeleton_3d(seq[t], angle)
            transforms.append(f"rotate({angle:.1f}°)")

        # Scale
        if self.config.scale_enabled and self.rng.random() < self.config.scale_prob:
            scale = self.rng.uniform(*self.config.scale_range)
            seq[:, :, :2] *= scale
            transforms.append(f"scale({scale:.2f})")

        # Translation
        if self.config.translate_enabled and self.rng.random() < self.config.translate_prob:
            tx = self.rng.uniform(*self.config.translate_range)
            ty = self.rng.uniform(*self.config.translate_range)
            seq[:, :, 0] += tx
            seq[:, :, 1] += ty
            transforms.append(f"translate({tx:.3f}, {ty:.3f})")

        # Add noise to coordinates
        if self.config.noise_enabled and self.rng.random() < self.config.noise_prob:
            std = self.rng.uniform(0.001, 0.01)  # Small noise for normalized coords
            noise = np.random.normal(0, std, seq[:, :, :2].shape)
            seq[:, :, :2] += noise
            transforms.append(f"coord_noise(σ={std:.4f})")

        return seq, transforms


def create_default_augmentation_config(output_path: str):
    """Create a default augmentation configuration YAML file."""
    config = AugmentationConfig()
    data = {"augmentation": config.to_dict()}

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Default augmentation config saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ASTRA Data Augmentation Pipeline",
    )
    parser.add_argument("--config", help="Augmentation config YAML")
    parser.add_argument("--input", help="Input frames directory")
    parser.add_argument("--output", help="Output augmented directory")
    parser.add_argument("--create_config", help="Create default config at path")
    parser.add_argument("--num_augments", type=int, default=3,
                        help="Number of augmented versions per frame")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.create_config:
        create_default_augmentation_config(args.create_config)
        return

    if not args.input or not args.output:
        parser.print_help()
        return

    # Load config
    if args.config:
        config = AugmentationConfig.from_yaml(args.config)
    else:
        config = AugmentationConfig(seed=args.seed)

    pipeline = AugmentationPipeline(config)
    os.makedirs(args.output, exist_ok=True)

    # Process frames
    frame_files = sorted([
        f for f in os.listdir(args.input)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    logger.info(f"Augmenting {len(frame_files)} frames × {args.num_augments} = "
                f"{len(frame_files) * args.num_augments} augmented frames")

    total = 0
    for fname in frame_files:
        image = cv2.imread(os.path.join(args.input, fname))
        if image is None:
            continue

        for aug_idx in range(args.num_augments):
            result = pipeline.augment_frame(image)
            aug_name = f"{Path(fname).stem}_aug{aug_idx:02d}{Path(fname).suffix}"
            cv2.imwrite(os.path.join(args.output, aug_name), result["image"])
            total += 1

    logger.info(f"Augmentation complete: {total} augmented frames saved to {args.output}")


if __name__ == "__main__":
    main()
