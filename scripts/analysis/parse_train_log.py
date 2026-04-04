#!/usr/bin/env python3
"""Parse train.log files from SBF training runs into structured CSV files.

Supports three run types, auto-detected from the log content:
  - redesigned:   RedesignedSupportFocusLoss (SmoothL1+Tversky support, optional Lovasz focus)
  - focus:        SupportGuidedSemanticFocusLoss (Phase 7 active route)
  - support_only: SemanticBoundaryLoss (support-only baseline)

Usage:
    python scripts/analysis/parse_train_log.py <train_log_path> [--output-dir DIR]

Outputs:
    metrics_epoch.csv  - one row per eval epoch with train/val summary metrics
    per_class_iou.csv  - one row per eval epoch with per-class IoU values
"""

import argparse
import csv
import os
import re
import sys

# ---------------------------------------------------------------------------
# Column definitions per run type
# ---------------------------------------------------------------------------

FOCUS_COLUMNS = [
    "epoch",
    "val_mIoU",
    "val_mAcc",
    "val_allAcc",
    "val_boundary_mIoU",
    "val_boundary_mAcc",
    "boundary_point_ratio",
    "support_bce",
    "support_cover",
    "valid_ratio",
    "train_loss",
    "train_loss_semantic",
    "train_loss_support",
    "train_loss_focus",
]

REDESIGNED_COLUMNS = [
    "epoch",
    "val_mIoU",
    "val_mAcc",
    "val_allAcc",
    "val_boundary_mIoU",
    "val_boundary_mAcc",
    "boundary_point_ratio",
    "support_reg_error",
    "support_cover",
    "valid_ratio",
    "train_loss",
    "train_loss_semantic",
    "train_loss_support",
    "train_loss_support_reg",
    "train_loss_support_cover",
    "train_loss_focus",
]

SUPPORT_ONLY_COLUMNS = [
    "epoch",
    "val_mIoU",
    "val_mAcc",
    "val_allAcc",
    "train_loss",
    "train_loss_semantic",
    "train_loss_edge",
    "train_loss_support",
    "train_loss_support_reg",
    "train_loss_support_cover",
    "train_loss_dir",
    "train_loss_dist",
]

CLASS_NAMES = [
    "balustrade",
    "balcony",
    "advboard",
    "wall",
    "eave",
    "column",
    "window",
    "clutter",
]

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

RE_KV = re.compile(r"(\w+)=([\d.]+)")
RE_VAL_SUMMARY = re.compile(
    r"Val result: mIoU/mAcc/allAcc (\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)"
)
RE_PER_CLASS = re.compile(r"(\w+) Result: iou/accuracy ([\d.]+)/([\d.]+)")
RE_VAL_BATCH = re.compile(r"Val/Test: \[(\d+)/(\d+)\]")
RE_BOUNDARY_MIOU = re.compile(r"val_boundary_mIoU: ([\d.]+)")
RE_BOUNDARY_MACC = re.compile(r"val_boundary_mAcc: ([\d.]+)")
RE_BOUNDARY_RATIO = re.compile(r"boundary_point_ratio: ([\d.]+)")
RE_SUPPORT_BCE = re.compile(r"support_bce: ([\d.]+)")
RE_SUPPORT_REG_ERROR = re.compile(r"support_reg_error: ([\d.]+)")
RE_SUPPORT_COVER = re.compile(r"support_cover: ([\d.]+)")
RE_VALID_RATIO = re.compile(r"valid_ratio: ([\d.]+)")


# ---------------------------------------------------------------------------
# Detect run type
# ---------------------------------------------------------------------------

def detect_run_type(log_path: str) -> str:
    """Scan log for run-type indicator in Train result lines."""
    with open(log_path, "r") as f:
        for i, line in enumerate(f):
            if i >= 5000:
                break
            if "Train result:" in line:
                if "loss_support_reg=" in line and "loss_focus=" in line:
                    return "redesigned"
                if "loss_focus=" in line:
                    return "focus"
                if "loss_edge=" in line:
                    return "support_only"
    print("ERROR: Could not detect run type from train.log", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Parse the log
# ---------------------------------------------------------------------------

def parse_log(log_path: str, run_type: str):
    """Line-by-line parse producing epoch rows and per-class rows.

    Log order per epoch: Val/Test batches -> Val result -> per-class results -> Train result.
    Train result for epoch N comes AFTER Val result for epoch N, so we create the row
    at Val result time and backfill train metrics when the Train result line appears.
    """
    eval_epoch = 0
    current_per_class = {}

    epoch_rows = []
    per_class_rows = []

    # Track the last Val/Test batch line per validation block for boundary metrics
    last_val_batch_boundary = {}

    with open(log_path, "r") as f:
        for line in f:
            # --- Val/Test batch line (focus/redesigned type: collect boundary metrics) ---
            if run_type in ("focus", "redesigned") and "Val/Test:" in line:
                m_batch = RE_VAL_BATCH.search(line)
                if m_batch:
                    boundary_data = {}
                    for pat, key in [
                        (RE_BOUNDARY_MIOU, "val_boundary_mIoU"),
                        (RE_BOUNDARY_MACC, "val_boundary_mAcc"),
                        (RE_BOUNDARY_RATIO, "boundary_point_ratio"),
                        (RE_SUPPORT_BCE, "support_bce"),
                        (RE_SUPPORT_REG_ERROR, "support_reg_error"),
                        (RE_SUPPORT_COVER, "support_cover"),
                        (RE_VALID_RATIO, "valid_ratio"),
                    ]:
                        m = pat.search(line)
                        if m:
                            boundary_data[key] = m.group(1)
                    # Always overwrite: we want the last batch (running avg = epoch avg)
                    last_val_batch_boundary = boundary_data
                continue

            # --- Val result summary (comes BEFORE Train result for same epoch) ---
            m_val = RE_VAL_SUMMARY.search(line)
            if m_val:
                eval_epoch += 1
                row = {"epoch": eval_epoch}
                row["val_mIoU"] = m_val.group(1)
                row["val_mAcc"] = m_val.group(2)
                row["val_allAcc"] = m_val.group(3)

                # Merge boundary metrics (focus/redesigned type)
                if run_type in ("focus", "redesigned"):
                    row.update(last_val_batch_boundary)
                    last_val_batch_boundary = {}

                epoch_rows.append(row)
                continue

            # --- Train result (comes AFTER Val result for same epoch) ---
            if "Train result:" in line:
                kvs = dict(RE_KV.findall(line))
                # Backfill into the most recent epoch row
                if epoch_rows:
                    row = epoch_rows[-1]
                    if run_type == "redesigned":
                        row["train_loss"] = kvs.get("loss", "")
                        row["train_loss_semantic"] = kvs.get("loss_semantic", "")
                        row["train_loss_support"] = kvs.get("loss_support", "")
                        row["train_loss_support_reg"] = kvs.get("loss_support_reg", "")
                        row["train_loss_support_cover"] = kvs.get("loss_support_cover", "")
                        row["train_loss_focus"] = kvs.get("loss_focus", "")
                    elif run_type == "focus":
                        row["train_loss"] = kvs.get("loss", "")
                        row["train_loss_semantic"] = kvs.get("loss_semantic", "")
                        row["train_loss_support"] = kvs.get("loss_support", "")
                        row["train_loss_focus"] = kvs.get("loss_focus", "")
                    elif run_type == "support_only":
                        row["train_loss"] = kvs.get("loss", "")
                        row["train_loss_semantic"] = kvs.get("loss_semantic", "")
                        row["train_loss_edge"] = kvs.get("loss_edge", "")
                        row["train_loss_support"] = kvs.get("loss_support", "")
                        row["train_loss_support_reg"] = kvs.get("loss_support_reg", "")
                        row["train_loss_support_cover"] = kvs.get("loss_support_cover", "")
                        row["train_loss_dir"] = kvs.get("loss_dir", "")
                        row["train_loss_dist"] = kvs.get("loss_dist", "")
                continue

            # --- Per-class result ---
            m_cls = RE_PER_CLASS.search(line)
            if m_cls:
                cls_name = m_cls.group(1)
                iou_val = m_cls.group(2)
                if cls_name in CLASS_NAMES:
                    current_per_class[cls_name] = iou_val
                    # After all 8 classes, flush
                    if len(current_per_class) == len(CLASS_NAMES):
                        cls_row = {"epoch": eval_epoch}
                        for cn in CLASS_NAMES:
                            cls_row[cn] = current_per_class.get(cn, "")
                        per_class_rows.append(cls_row)
                        current_per_class = {}
                continue

    return epoch_rows, per_class_rows


# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

def write_csv(rows, columns, output_path, comment=None):
    """Write rows as CSV with the given column order."""
    with open(output_path, "w", newline="") as f:
        if comment:
            f.write(f"# {comment}\n")
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse SBF train.log into structured CSV files."
    )
    parser.add_argument("train_log_path", help="Path to train.log file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for CSV files (default: same as train.log)",
    )
    args = parser.parse_args()

    log_path = args.train_log_path
    if not os.path.isfile(log_path):
        print(f"ERROR: File not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(log_path))
    os.makedirs(output_dir, exist_ok=True)

    # Detect run type
    run_type = detect_run_type(log_path)
    print(f"Detected run type: {run_type}")

    # Parse
    epoch_rows, per_class_rows = parse_log(log_path, run_type)

    if not epoch_rows:
        print("ERROR: Zero eval epochs found in log.", file=sys.stderr)
        sys.exit(1)

    # Determine columns
    if run_type == "redesigned":
        columns = REDESIGNED_COLUMNS
    elif run_type == "focus":
        columns = FOCUS_COLUMNS
    else:
        columns = SUPPORT_ONLY_COLUMNS

    # Write metrics CSV
    metrics_path = os.path.join(output_dir, "metrics_epoch.csv")
    write_csv(epoch_rows, columns, metrics_path)
    print(f"Wrote {len(epoch_rows)} epoch rows to: {metrics_path}")

    # Write per-class IoU CSV
    per_class_path = os.path.join(output_dir, "per_class_iou.csv")
    per_class_columns = ["epoch"] + CLASS_NAMES
    write_csv(per_class_rows, per_class_columns, per_class_path)
    print(f"Wrote {len(per_class_rows)} epoch rows to: {per_class_path}")

    # Summary: best val_mIoU
    best_miou = 0.0
    best_epoch = 0
    for row in epoch_rows:
        miou = float(row.get("val_mIoU", 0))
        if miou > best_miou:
            best_miou = miou
            best_epoch = row["epoch"]

    print(f"Best val_mIoU: {best_miou:.4f} at epoch {best_epoch}")
    print(f"Total eval epochs: {len(epoch_rows)}")


if __name__ == "__main__":
    main()
