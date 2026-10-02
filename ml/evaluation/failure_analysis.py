"""
ASTRA — Failure Case Analysis & Visualization

For every incorrect decision, saves a detailed record with:
  video, timestamp, frame/clip, expected vs predicted state,
  model evidence, confidence, reason for state transition.

Generates visual overlays when frames are available.

Usage:
    python -m ml.evaluation.failure_analysis --results eval_results/evaluation_results.json --output eval_results/failures/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def analyze_failures(
    failure_cases: List[Dict],
    frames_dir: str = None,
    output_dir: str = "eval_results/failures",
) -> Dict[str, Any]:
    """
    Analyze failure cases and generate visual overlays.
    """
    os.makedirs(output_dir, exist_ok=True)

    analysis = {
        "total_failures": len(failure_cases),
        "by_category": {},
        "by_step": {},
        "details": [],
    }

    for i, fc in enumerate(failure_cases):
        category = fc.get("category", "unknown")
        step_id = fc.get("step_id", "?")

        analysis["by_category"][category] = analysis["by_category"].get(category, 0) + 1
        analysis["by_step"][step_id] = analysis["by_step"].get(step_id, 0) + 1

        detail = {
            "index": i,
            "step_id": step_id,
            "expected_status": fc.get("expected_status"),
            "predicted_status": fc.get("predicted_status"),
            "confidence": fc.get("confidence"),
            "reason": fc.get("reason"),
            "category": category,
            "evidence": fc.get("evidence", {}),
        }

        # Generate visual overlay if frame is available
        if frames_dir and fc.get("frame_path"):
            frame_path = os.path.join(frames_dir, fc["frame_path"])
            if os.path.exists(frame_path):
                overlay = _generate_overlay(frame_path, fc)
                overlay_path = os.path.join(output_dir, f"failure_{i:04d}.jpg")
                cv2.imwrite(overlay_path, overlay)
                detail["overlay_path"] = overlay_path

        analysis["details"].append(detail)

    # Generate report
    report_path = os.path.join(output_dir, "failure_report.md")
    _generate_failure_report(analysis, report_path)

    # Save JSON
    json_path = os.path.join(output_dir, "failure_analysis.json")
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    return analysis


def _generate_overlay(frame_path: str, failure: Dict) -> np.ndarray:
    """Generate a visual overlay showing the failure case."""
    frame = cv2.imread(frame_path)
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    h, w = frame.shape[:2]

    # Red border for failure
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)

    # Info overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 120), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    y = h - 100
    texts = [
        f"Step: {failure.get('step_id', '?')}",
        f"Expected: {failure.get('expected_status', '?')} | Got: {failure.get('predicted_status', '?')}",
        f"Confidence: {failure.get('confidence', '?')} | {failure.get('reason', '')}",
    ]
    for text in texts:
        cv2.putText(frame, text[:80], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        y += 25

    return frame


def _generate_failure_report(analysis: Dict, output_path: str):
    """Generate markdown failure report."""
    lines = [
        "# ASTRA Failure Case Report",
        f"\n**Generated**: {datetime.now().isoformat()}",
        f"\n**Total failures**: {analysis['total_failures']}",
        "\n## By Category\n",
    ]

    for cat, count in sorted(analysis["by_category"].items()):
        lines.append(f"- **{cat}**: {count}")

    lines.append("\n## By Step\n")
    for step, count in sorted(analysis["by_step"].items()):
        lines.append(f"- **{step}**: {count}")

    lines.append("\n## Detailed Failures\n")
    lines.append("| # | Step | Expected | Predicted | Confidence | Reason |")
    lines.append("| --- | --- | --- | --- | --- | --- |")

    for d in analysis["details"]:
        lines.append(
            f"| {d['index']} | {d['step_id']} | {d['expected_status']} | "
            f"{d['predicted_status']} | {d['confidence']} | {d['reason']} |"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Failure report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ASTRA Failure Analysis")
    parser.add_argument("--results", required=True, help="evaluation_results.json or failure_cases.json")
    parser.add_argument("--frames", help="Frames directory for overlays")
    parser.add_argument("--output", default="eval_results/failures")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    # Accept either full results or just failure cases
    if isinstance(data, list):
        failures = data
    else:
        failures = data.get("failure_cases", [])

    if not failures:
        print("No failure cases found.")
        return

    analysis = analyze_failures(failures, args.frames, args.output)
    print(f"\nFailure analysis complete:")
    print(f"  Total: {analysis['total_failures']}")
    print(f"  Categories: {analysis['by_category']}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
