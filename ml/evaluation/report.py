"""
ASTRA — Evaluation Report Generator

Produces human-readable Markdown reports from evaluation results.
Covers all 16 required sections.
"""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_report(
    results: Dict[str, Any],
    output_dir: str = "eval_results",
    report_name: str = "evaluation_report",
) -> str:
    """
    Generate a comprehensive Markdown evaluation report.

    Returns the path to the generated report.
    """
    os.makedirs(output_dir, exist_ok=True)
    lines = []

    def add(text=""):
        lines.append(text)

    def add_metric(name, value, indent=0):
        prefix = "  " * indent
        if isinstance(value, float):
            if math.isnan(value):
                lines.append(f"{prefix}- **{name}**: ⚠️ insufficient data")
            else:
                lines.append(f"{prefix}- **{name}**: `{value:.4f}`")
        elif value is None:
            lines.append(f"{prefix}- **{name}**: ⚠️ not available")
        else:
            lines.append(f"{prefix}- **{name}**: `{value}`")

    def add_table(headers, rows):
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            cells = [str(c) if c is not None else "—" for c in row]
            lines.append("| " + " | ".join(cells) + " |")

    # ============================================================
    # Header
    # ============================================================
    add("# ASTRA Evaluation Report")
    add()
    add(f"**Generated**: {datetime.now().isoformat()}")
    add()
    add("---")
    add()

    # ============================================================
    # 1. Dataset Summary
    # ============================================================
    add("## 1. Dataset Summary")
    add()
    ds = results.get("dataset_summary", {})
    add_metric("Name", ds.get("name", "?"))
    add_metric("Split", ds.get("split", "?"))
    add_metric("Samples", ds.get("n_samples", 0))
    add_metric("Sessions", ds.get("n_sessions", 0))
    add_metric("Videos", ds.get("n_videos", 0))
    add_metric("Object classes", ds.get("n_classes", 0))
    add_metric("Action classes", ds.get("n_actions", 0))
    add_metric("Steps", ds.get("n_steps", 0))
    add()

    data_avail = []
    for key in ["has_detections", "has_poses", "has_actions", "has_step_sequence", "has_anomalies"]:
        status = "✅" if ds.get(key) else "❌"
        data_avail.append([key.replace("has_", ""), status])
    add_table(["Annotation Type", "Available"], data_avail)
    add()

    # ============================================================
    # 2. Detection Performance
    # ============================================================
    add("## 2. Detection Performance")
    add()
    det = results.get("detection", {})
    if det.get("status") == "insufficient_data":
        add(f"> ⚠️ {det.get('reason', 'No ground truth')}")
    else:
        add_metric("Total GT boxes", det.get("n_gt_total"))
        add_metric("Total predictions", det.get("n_pred_total"))
        for key in ["mAP@50", "mAP@75", "mAP@50:95"]:
            val = det.get(key)
            if isinstance(val, dict):
                add_metric(key, val.get("mAP", float("nan")))
            else:
                add_metric(key, val)

        # Per-class table
        map50 = det.get("mAP@50", {})
        if isinstance(map50, dict) and "per_class" in map50:
            add()
            add("### Per-Class AP@50")
            rows = []
            for cls, data in sorted(map50["per_class"].items()):
                ap = data.get("ap", "—")
                n_gt = data.get("n_gt", 0)
                rows.append([cls, f"{ap:.4f}" if isinstance(ap, float) and not math.isnan(ap) else "—", n_gt])
            add_table(["Class", "AP@50", "GT Count"], rows)
    add()

    # ============================================================
    # 3. Pose Performance
    # ============================================================
    add("## 3. Pose Performance")
    add()
    pose = results.get("pose", {})
    if pose.get("status") == "insufficient_data":
        add(f"> ⚠️ {pose.get('reason', 'No ground truth')}")
    else:
        add_metric("PCK@0.2", pose.get("pck@0.2"))
        add_metric("MPJPE (px)", pose.get("mpjpe_px"))
        add_metric("Evaluated", pose.get("n_evaluated"))

        pj = pose.get("per_joint_pck", {})
        if pj:
            add()
            add("### Per-Joint PCK")
            rows = [[j, f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else "—"]
                     for j, v in pj.items()]
            add_table(["Joint", "PCK@0.2"], rows)
    add()

    # ============================================================
    # 4. Action Performance
    # ============================================================
    add("## 4. Action Performance")
    add()
    act = results.get("action", {})
    if act.get("status") == "insufficient_data":
        add("> ⚠️ Insufficient ground truth action data")
    else:
        add_metric("Frame accuracy", act.get("frame_accuracy"))
        add_metric("Top-3 accuracy", act.get("top3_accuracy"))

        prf = act.get("frame_prf", {})
        if prf:
            add_metric("Macro precision", prf.get("precision"))
            add_metric("Macro recall", prf.get("recall"))
            add_metric("Macro F1", prf.get("f1"))

        for key in ["segment_f1@25", "segment_f1@50", "segment_f1@75"]:
            sf = act.get(key)
            if sf:
                add_metric(key, sf.get("f1"))

        add_metric("Edit score", act.get("edit_score"))
    add()

    # ============================================================
    # 5. HOI Performance
    # ============================================================
    add("## 5. HOI Performance")
    add()
    hoi = results.get("hoi", {})
    if hoi.get("status") == "insufficient_data":
        add(f"> ⚠️ {hoi.get('reason', 'No ground truth')}")
    else:
        add_metric("Interaction type accuracy", hoi.get("interaction_type_accuracy"))
        add_metric("Hand assignment accuracy", hoi.get("hand_accuracy"))
        add_metric("Contact accuracy", hoi.get("contact_accuracy"))
        cprf = hoi.get("contact_prf", {})
        if cprf:
            add_metric("Contact precision", cprf.get("precision"))
            add_metric("Contact recall", cprf.get("recall"))
            add_metric("Contact F1", cprf.get("f1"))
    add()

    # ============================================================
    # 6. Object State Performance
    # ============================================================
    add("## 6. Object State Performance")
    add()
    os_r = results.get("object_state", {})
    if os_r.get("status") == "insufficient_data":
        add(f"> ⚠️ {os_r.get('reason', 'No ground truth')}")
    else:
        add_metric("State accuracy", os_r.get("state_accuracy"))
        tprf = os_r.get("transition_prf", {})
        if tprf:
            add_metric("Transition precision", tprf.get("precision"))
            add_metric("Transition recall", tprf.get("recall"))
    add()

    # ============================================================
    # 7. Temporal Performance
    # ============================================================
    add("## 7. Temporal Performance")
    add()
    tmp = results.get("temporal", {})
    add_metric("Action accuracy", tmp.get("action_accuracy"))
    add_metric("Next action accuracy", tmp.get("next_action_accuracy"))
    add_metric("Step accuracy", tmp.get("step_accuracy"))
    cal = tmp.get("action_calibration", {})
    if cal:
        add_metric("Action ECE", cal.get("ece"))
    add()

    # ============================================================
    # 8–11. PROCEDURE VALIDATION (PRIMARY METRIC)
    # ============================================================
    add("## 8–11. Procedure Validation Performance ⭐")
    add()
    add("> This is the **PRIMARY ASTRA METRIC**.")
    add()
    proc = results.get("procedure", {})

    # Composite
    add_metric("🏆 COMPOSITE SCORE", proc.get("composite_score"))
    add()

    # Sequence
    seq = proc.get("sequence", {})
    add("### 8. Sequence Accuracy")
    add_metric("Sequence accuracy", seq.get("sequence_accuracy"))
    add_metric("Exact match", seq.get("exact_match"))
    add_metric("Edit distance", seq.get("edit_distance"))
    add_metric("Order correct", seq.get("order_correct"))
    add_metric("Correct steps", seq.get("correct_steps"))
    add_metric("Missed steps", seq.get("missed_steps"))
    add_metric("Extra steps", seq.get("extra_steps"))
    add()

    # Step validation
    sv = proc.get("step_validation", {})
    add("### 9. Step Validation")
    add_metric("Step precision", sv.get("step_precision"))
    add_metric("Step recall", sv.get("step_recall"))
    add_metric("Step F1", sv.get("step_f1"))
    add_metric("False VALID rate", sv.get("false_valid_rate"))
    add_metric("UNCERTAIN rate", sv.get("uncertain_rate"))
    add()

    # Per-step breakdown
    ps = sv.get("per_step", {})
    if ps:
        add("### Per-Step Breakdown")
        rows = []
        for sid, data in sorted(ps.items()):
            rows.append([
                sid, data.get("gt", "—"), data.get("pred", "—"),
                "✅" if data.get("correct") else "❌",
                "⚠️" if data.get("false_valid") else "",
            ])
        add_table(["Step", "GT Status", "Pred Status", "Correct", "False Valid"], rows)
        add()

    # Anomaly detection
    add("### 10. Out-of-Order Detection")
    ooo = proc.get("out_of_order_detection", {})
    add_metric("Precision", ooo.get("precision"))
    add_metric("Recall", ooo.get("recall"))
    add_metric("F1", ooo.get("f1"))
    add()

    add("### Skipped-Step Detection")
    skip = proc.get("skipped_step_detection", {})
    add_metric("Precision", skip.get("precision"))
    add_metric("Recall", skip.get("recall"))
    add_metric("F1", skip.get("f1"))
    add()

    # Experiment completion
    add("### 11. Experiment Completion")
    ec = proc.get("experiment_completion", {})
    add_metric("GT complete", ec.get("gt_complete"))
    add_metric("Pred complete", ec.get("pred_complete"))
    add_metric("Correct", ec.get("correct"))
    add()

    # ============================================================
    # 12. Latency
    # ============================================================
    add("## 12. Latency")
    add()
    lat = results.get("latency", {})
    add_metric("Mean latency", f"{lat.get('mean_ms', '?')} ms")
    add_metric("P50 latency", f"{lat.get('p50_ms', '?')} ms")
    add_metric("P95 latency", f"{lat.get('p95_ms', '?')} ms")
    add_metric("FPS", lat.get("fps"))
    add_metric("Frames processed", lat.get("n_frames"))
    add()

    # ============================================================
    # 13. Memory
    # ============================================================
    add("## 13. Memory")
    add()
    mem = results.get("memory", {})
    if "rss_mb" in mem:
        add_metric("RSS", f"{mem['rss_mb']} MB")
        add_metric("VMS", f"{mem.get('vms_mb', '?')} MB")
    else:
        add(f"> {mem.get('note', 'Memory data not available')}")
    add()

    # ============================================================
    # 14. Failure Cases
    # ============================================================
    add("## 14. Failure Cases")
    add()
    failures = results.get("failure_cases", [])
    if not failures:
        add("> ✅ No failure cases detected.")
    else:
        add(f"**{len(failures)} failure case(s) found:**")
        add()
        rows = []
        for fc in failures:
            rows.append([
                fc.get("step_id", "?"),
                fc.get("expected_status", "?"),
                fc.get("predicted_status", "?"),
                f"{fc.get('confidence', '?')}",
                fc.get("reason", "?"),
            ])
        add_table(["Step", "Expected", "Predicted", "Confidence", "Reason"], rows)
    add()

    # ============================================================
    # 15. Confidence Calibration
    # ============================================================
    add("## 15. Confidence Calibration")
    add()
    step_cal = proc.get("step_calibration", {})
    if step_cal:
        add_metric("Step validation ECE", step_cal.get("ece"))
        bins = step_cal.get("bins", [])
        if bins:
            rows = [[b["bin"], b["n"], f"{b['avg_confidence']:.3f}", f"{b['avg_accuracy']:.3f}"]
                     for b in bins]
            add_table(["Bin", "Count", "Avg Confidence", "Avg Accuracy"], rows)
    else:
        add("> Calibration data not available.")
    add()

    # ============================================================
    # 16. Known Limitations
    # ============================================================
    add("## 16. Known Limitations")
    add()
    limits = results.get("known_limitations", [])
    if limits:
        for lim in limits:
            add(f"- ⚠️ {lim}")
    else:
        add("> No known limitations identified.")
    add()

    # ============================================================
    # Write report
    # ============================================================
    report_text = "\n".join(lines)
    md_path = os.path.join(output_dir, f"{report_name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Report saved to: {md_path}")
    return md_path
