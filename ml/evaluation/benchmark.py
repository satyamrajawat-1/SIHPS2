"""
ASTRA — Benchmark Module

Measures: inference latency, FPS, memory/VRAM usage.
Compares CPU and GPU paths when available.
"""
from __future__ import annotations

import gc
import logging
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / 1024 / 1024
    except ImportError:
        return float("nan")


def _get_gpu_memory_mb() -> Dict[str, float]:
    """Get GPU memory usage in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            total = torch.cuda.get_device_properties(0).total_mem / 1024 / 1024
            return {
                "allocated_mb": round(allocated, 1),
                "reserved_mb": round(reserved, 1),
                "total_mb": round(total, 1),
                "gpu_name": torch.cuda.get_device_name(0),
            }
    except Exception:
        pass
    return {"status": "no_gpu_available"}


def benchmark_component(
    fn,
    args: tuple = (),
    kwargs: dict = None,
    n_warmup: int = 3,
    n_runs: int = 20,
    label: str = "component",
) -> Dict[str, Any]:
    """
    Benchmark a single callable.

    Returns latency stats (mean, std, min, max, p50, p95, p99) in ms.
    """
    kwargs = kwargs or {}

    # Warmup
    for _ in range(n_warmup):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            return {"label": label, "error": str(e)}

    # Timed runs
    latencies = []
    for _ in range(n_runs):
        gc.collect()
        t0 = time.perf_counter()
        try:
            fn(*args, **kwargs)
        except Exception as e:
            return {"label": label, "error": str(e)}
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    latencies = np.array(latencies)

    return {
        "label": label,
        "n_runs": n_runs,
        "mean_ms": round(float(np.mean(latencies)), 2),
        "std_ms": round(float(np.std(latencies)), 2),
        "min_ms": round(float(np.min(latencies)), 2),
        "max_ms": round(float(np.max(latencies)), 2),
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "fps": round(1000.0 / float(np.mean(latencies)), 1) if np.mean(latencies) > 0 else 0,
    }


def benchmark_pipeline(
    pipeline,
    n_frames: int = 50,
    frame_size: tuple = (480, 640, 3),
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Benchmark the full ASTRA pipeline.

    Measures:
    - Full pipeline FPS
    - Per-component latency
    - Memory usage
    """
    rng = np.random.RandomState(seed)
    frames = [rng.randint(0, 255, frame_size, dtype=np.uint8) for _ in range(n_frames)]

    results = {
        "system_info": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu": platform.processor(),
            "n_frames": n_frames,
            "frame_size": list(frame_size),
        },
        "memory_before": {
            "rss_mb": round(_get_memory_mb(), 1),
            "gpu": _get_gpu_memory_mb(),
        },
    }

    # Full pipeline benchmark
    pipeline.reset()

    # Warmup
    for i in range(min(5, n_frames)):
        pipeline.process_frame(frames[i], timestamp=i / 30.0)

    pipeline.reset()

    latencies = []
    for i, frame in enumerate(frames):
        gc.collect()
        t0 = time.perf_counter()
        pipeline.process_frame(frame, timestamp=i / 30.0)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    lat = np.array(latencies)

    results["pipeline_latency"] = {
        "mean_ms": round(float(np.mean(lat)), 2),
        "std_ms": round(float(np.std(lat)), 2),
        "min_ms": round(float(np.min(lat)), 2),
        "max_ms": round(float(np.max(lat)), 2),
        "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
        "p99_ms": round(float(np.percentile(lat, 99)), 2),
        "fps": round(1000.0 / float(np.mean(lat)), 1) if np.mean(lat) > 0 else 0,
    }

    results["memory_after"] = {
        "rss_mb": round(_get_memory_mb(), 1),
        "gpu": _get_gpu_memory_mb(),
    }

    results["memory_delta_mb"] = round(
        results["memory_after"]["rss_mb"] - results["memory_before"]["rss_mb"], 1
    )

    # Per-component breakdown
    frame = frames[0]
    components = {}

    # Detection
    components["detection"] = benchmark_component(
        pipeline.detector.detect, (frame,), label="detection", n_runs=n_frames,
    )

    # Pose
    components["pose"] = benchmark_component(
        pipeline.pose_estimator.predict, (frame,), label="pose", n_runs=n_frames,
    )

    # Visual features
    components["visual_features"] = benchmark_component(
        pipeline.visual_extractor.extract, (frame,), label="visual_features", n_runs=n_frames,
    )

    results["per_component"] = components

    return results


def compare_cpu_gpu(
    pipeline_factory,
    experiment_config,
    n_frames: int = 30,
) -> Dict[str, Any]:
    """
    Compare CPU and GPU pipeline performance.

    pipeline_factory: callable(config, device) → pipeline
    """
    results = {}

    # CPU benchmark
    try:
        cpu_pipeline = pipeline_factory(experiment_config, "cpu")
        cpu_pipeline.load_models()
        results["cpu"] = benchmark_pipeline(cpu_pipeline, n_frames=n_frames)
        cpu_pipeline.close()
    except Exception as e:
        results["cpu"] = {"error": str(e)}

    # GPU benchmark
    try:
        import torch
        if torch.cuda.is_available():
            gpu_pipeline = pipeline_factory(experiment_config, "cuda")
            gpu_pipeline.load_models()
            results["gpu"] = benchmark_pipeline(gpu_pipeline, n_frames=n_frames)
            gpu_pipeline.close()
        else:
            results["gpu"] = {"status": "no_gpu_available"}
    except Exception as e:
        results["gpu"] = {"error": str(e)}

    # Speedup
    if "cpu" in results and "gpu" in results:
        cpu_fps = results.get("cpu", {}).get("pipeline_latency", {}).get("fps", 0)
        gpu_fps = results.get("gpu", {}).get("pipeline_latency", {}).get("fps", 0)
        if cpu_fps > 0 and gpu_fps > 0:
            results["gpu_speedup"] = round(gpu_fps / cpu_fps, 2)

    return results
