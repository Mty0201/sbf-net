#!/usr/bin/env python3
"""Summarize training logs into markdown and JSON artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TIMESTAMPED_LINE_RE = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<level>[A-Z]+)\] (?P<message>.*)$"
)
TRAIN_LINE_RE = re.compile(
    r"^Train:\s+\[(?P<epoch>\d+)/(?P<epoch_total>\d+)\]\[(?P<iter>\d+)/(?P<iter_total>\d+)\]\s*(?P<body>.*)$"
)
VAL_LINE_RE = re.compile(
    r"^Val/Test:\s+\[(?P<step>\d+)/(?P<step_total>\d+)\]\s*(?P<body>.*)$"
)
KV_COLON_RE = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z0-9_./-]*)\s*:\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)
KV_EQUAL_RE = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z0-9_./-]*)=(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)
CLASS_RESULT_RE = re.compile(
    r"^(?P<label>.+?) Result:\s+iou/accuracy\s+(?P<iou>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"/(?P<accuracy>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$"
)
BEST_METRIC_RE = re.compile(
    r"^Best validation (?P<metric>[A-Za-z0-9_./-]+) updated to:\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$"
)
CURRENT_METRIC_RE = re.compile(
    r"^Current (?P<metric>[A-Za-z0-9_./-]+):\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$"
)
TOTAL_EPOCH_RE = re.compile(
    r"^Total epoch:\s*(?P<total_epoch>\d+),\s*Eval epoch:\s*(?P<eval_epoch>\d+),\s*Train loop:\s*(?P<train_loop>\d+)$"
)
TOTAL_TRAIN_ITER_RE = re.compile(r"^Total train iterations:\s*(?P<value>\d+)$")
START_EPOCH_RE = re.compile(r"^Start epoch:\s*(?P<value>\d+)$")
SAVE_PATH_RE = re.compile(r"^Save path:\s*(?P<value>.+)$")
CHECKPOINT_RE = re.compile(r"^(?:Saving checkpoint to|Checkpoint (?:last|best)):\s*(?P<value>.+)$")
VALUE_LINE_RE = re.compile(r"^(?P<key>[^:]+):\s*(?P<value>.+)$")

LOWER_IS_BETTER_TOKENS = ("loss", "error")
HIGHER_IS_BETTER_TOKENS = ("miou", "macc", "allacc", "acc", "accuracy", "cosine", "cover")
NEUTRAL_METRIC_TOKENS = ("ratio", "mean", "lr", "data", "batch", "remain", "steps")
WARNING_TOKENS = ("warning", "warn", "error", "exception", "traceback", "nan", "inf", "oom", "killed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize one or more training logs into markdown and JSON artifacts."
    )
    parser.add_argument("logs", nargs="+", help="Path(s) to train log files.")
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir()),
        help="Directory for *.summary.md and *.summary.json outputs.",
    )
    parser.add_argument(
        "--recent-count",
        type=int,
        default=5,
        help="How many recent records to keep in the summary.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional output stem. Only valid when summarizing a single log.",
    )
    return parser.parse_args()


def default_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "reports" / "log_summaries"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_float(raw: str) -> Optional[float]:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value:
        return None
    return value


def safe_int(raw: str) -> Optional[int]:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def human_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    units = ["KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        size /= 1024.0
        if size < 1024.0:
            return f"{size:.2f} {unit}"
    return f"{size:.2f} PB"


def sanitize_output_stem(log_path: Path) -> str:
    base = f"{log_path.parent.name}__{log_path.stem}" if log_path.parent.name else log_path.stem
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    return re.sub(r"_+", "_", base).strip("_") or "train_log"


def parse_timestamp(raw: str) -> Optional[str]:
    try:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S").isoformat()
    except ValueError:
        return None


def parse_numeric_kv_pairs(text: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for regex in (KV_COLON_RE, KV_EQUAL_RE):
        for match in regex.finditer(text):
            value = safe_float(match.group("value"))
            if value is None:
                continue
            values[match.group("key")] = value
    return values


def infer_metric_direction(metric_name: str) -> str:
    name = metric_name.lower()
    if any(token in name for token in NEUTRAL_METRIC_TOKENS):
        return "neutral"
    if any(token in name for token in LOWER_IS_BETTER_TOKENS):
        return "lower_is_better"
    if any(token in name for token in HIGHER_IS_BETTER_TOKENS):
        return "higher_is_better"
    return "unknown"


def format_number(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1000 or (0 < abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.4f}"


def make_point_record(
    *,
    stage: str,
    metric: str,
    value: float,
    timestamp: Optional[str],
    epoch: Optional[int] = None,
    iter_idx: Optional[int] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "metric": metric,
        "value": value,
        "timestamp": timestamp,
        "epoch": epoch,
        "iter": iter_idx,
        "source": source,
    }


def new_session(start_line: int) -> Dict[str, Any]:
    return {
        "start_line": start_line,
        "end_line": start_line,
        "start_timestamp": None,
        "end_timestamp": None,
        "device": None,
        "save_path": None,
        "smoke_mode": None,
        "optimizer": None,
        "start_epoch": None,
        "total_epoch": None,
        "eval_epoch": None,
        "train_loop": None,
        "total_train_iterations": None,
        "train_records": 0,
        "val_records": 0,
        "train_result_records": 0,
        "checkpoints": [],
        "warnings": [],
        "errors": [],
    }


def append_issue(container: List[Dict[str, Any]], line_no: int, timestamp: Optional[str], message: str) -> None:
    container.append({"line": line_no, "timestamp": timestamp, "message": message})


def summarize_log(log_path: Path, recent_count: int) -> Dict[str, Any]:
    file_size = log_path.stat().st_size
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        raw_lines = handle.readlines()

    summary: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "input": {
            "path": str(log_path.resolve()),
            "name": log_path.name,
            "size_bytes": file_size,
            "size_human": human_size(file_size),
            "line_count": len(raw_lines),
        },
        "basic_info": {},
        "sessions": [],
        "last_values": {},
        "best_values": {},
        "recent_changes": {},
        "loss_trends": {},
        "warnings": [],
        "anomalies": [],
        "auto_questions": [],
        "notes": [],
    }

    sessions: List[Dict[str, Any]] = []
    current_session: Optional[Dict[str, Any]] = None

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    train_result_records: List[Dict[str, Any]] = []
    scalar_records: List[Dict[str, Any]] = []
    class_results: List[Dict[str, Any]] = []
    checkpoint_paths: List[str] = []
    metric_points: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    first_timestamp: Optional[str] = None
    last_timestamp: Optional[str] = None
    min_epoch: Optional[int] = None
    max_epoch: Optional[int] = None
    min_iter: Optional[int] = None
    max_iter: Optional[int] = None
    total_epoch_hint: Optional[int] = None
    iter_total_hint: Optional[int] = None

    for line_no, raw_line in enumerate(raw_lines, start=1):
        line = raw_line.rstrip("\n")
        matched = TIMESTAMPED_LINE_RE.match(line)
        if matched:
            timestamp_text = matched.group("timestamp")
            level = matched.group("level")
            message = matched.group("message").strip()
        else:
            timestamp_text = None
            level = "UNKNOWN"
            message = line.strip()

        timestamp_iso = parse_timestamp(timestamp_text) if timestamp_text else None
        if timestamp_iso and first_timestamp is None:
            first_timestamp = timestamp_iso
        if timestamp_iso:
            last_timestamp = timestamp_iso

        if current_session is None or message.startswith("=> Loading config"):
            current_session = new_session(line_no)
            current_session["start_timestamp"] = timestamp_iso
            sessions.append(current_session)
        current_session["end_line"] = line_no
        current_session["end_timestamp"] = timestamp_iso or current_session["end_timestamp"]

        warning_flag = level in {"WARN", "WARNING", "ERROR", "CRITICAL"} or any(
            token in message.lower() for token in WARNING_TOKENS
        )
        if warning_flag:
            append_issue(summary["warnings"], line_no, timestamp_iso, message)
            if level in {"ERROR", "CRITICAL"} or "traceback" in message.lower() or "exception" in message.lower():
                append_issue(summary["anomalies"], line_no, timestamp_iso, message)
                append_issue(current_session["errors"], line_no, timestamp_iso, message)
            else:
                append_issue(current_session["warnings"], line_no, timestamp_iso, message)

        if "traceback" in message.lower():
            append_issue(summary["anomalies"], line_no, timestamp_iso, message)

        save_match = SAVE_PATH_RE.match(message)
        if save_match:
            current_session["save_path"] = save_match.group("value").strip()
            continue

        value_match = VALUE_LINE_RE.match(message)
        if value_match:
            key = value_match.group("key").strip()
            value = value_match.group("value").strip()
            if key == "Device":
                current_session["device"] = value
            elif key == "Smoke mode":
                current_session["smoke_mode"] = value.lower() == "true"
            elif key == "Optimizer":
                current_session["optimizer"] = value

        total_epoch_match = TOTAL_EPOCH_RE.match(message)
        if total_epoch_match:
            current_session["total_epoch"] = safe_int(total_epoch_match.group("total_epoch"))
            current_session["eval_epoch"] = safe_int(total_epoch_match.group("eval_epoch"))
            current_session["train_loop"] = safe_int(total_epoch_match.group("train_loop"))
            total_epoch_hint = current_session["total_epoch"] or total_epoch_hint
            continue

        total_iter_match = TOTAL_TRAIN_ITER_RE.match(message)
        if total_iter_match:
            current_session["total_train_iterations"] = safe_int(total_iter_match.group("value"))
            continue

        start_epoch_match = START_EPOCH_RE.match(message)
        if start_epoch_match:
            current_session["start_epoch"] = safe_int(start_epoch_match.group("value"))
            continue

        checkpoint_match = CHECKPOINT_RE.match(message)
        if checkpoint_match:
            checkpoint_path = checkpoint_match.group("value").strip()
            checkpoint_paths.append(checkpoint_path)
            current_session["checkpoints"].append(checkpoint_path)
            continue

        class_result_match = CLASS_RESULT_RE.match(message)
        if class_result_match:
            class_results.append(
                {
                    "timestamp": timestamp_iso,
                    "label": class_result_match.group("label").strip(),
                    "iou": safe_float(class_result_match.group("iou")),
                    "accuracy": safe_float(class_result_match.group("accuracy")),
                }
            )
            continue

        train_match = TRAIN_LINE_RE.match(message)
        if train_match:
            epoch = safe_int(train_match.group("epoch"))
            epoch_total = safe_int(train_match.group("epoch_total"))
            iter_idx = safe_int(train_match.group("iter"))
            iter_total = safe_int(train_match.group("iter_total"))
            metrics = parse_numeric_kv_pairs(train_match.group("body"))
            record = {
                "line": line_no,
                "timestamp": timestamp_iso,
                "epoch": epoch,
                "epoch_total": epoch_total,
                "iter": iter_idx,
                "iter_total": iter_total,
                "metrics": metrics,
            }
            train_records.append(record)
            current_session["train_records"] += 1
            if epoch is not None:
                min_epoch = epoch if min_epoch is None else min(min_epoch, epoch)
                max_epoch = epoch if max_epoch is None else max(max_epoch, epoch)
            if iter_idx is not None:
                min_iter = iter_idx if min_iter is None else min(min_iter, iter_idx)
                max_iter = iter_idx if max_iter is None else max(max_iter, iter_idx)
            if epoch_total is not None:
                total_epoch_hint = epoch_total if total_epoch_hint is None else max(total_epoch_hint, epoch_total)
            if iter_total is not None:
                iter_total_hint = iter_total if iter_total_hint is None else max(iter_total_hint, iter_total)
            for metric_name, metric_value in metrics.items():
                metric_points["train"][metric_name].append(
                    make_point_record(
                        stage="train",
                        metric=metric_name,
                        value=metric_value,
                        timestamp=timestamp_iso,
                        epoch=epoch,
                        iter_idx=iter_idx,
                        source="train_line",
                    )
                )
            continue

        val_match = VAL_LINE_RE.match(message)
        if val_match:
            step = safe_int(val_match.group("step"))
            step_total = safe_int(val_match.group("step_total"))
            metrics = parse_numeric_kv_pairs(val_match.group("body"))
            record = {
                "line": line_no,
                "timestamp": timestamp_iso,
                "step": step,
                "step_total": step_total,
                "metrics": metrics,
            }
            val_records.append(record)
            current_session["val_records"] += 1
            for metric_name, metric_value in metrics.items():
                metric_points["val"][metric_name].append(
                    make_point_record(
                        stage="val",
                        metric=metric_name,
                        value=metric_value,
                        timestamp=timestamp_iso,
                        epoch=max_epoch,
                        iter_idx=step,
                        source="val_line",
                    )
                )
            continue

        if message.startswith("Train result:"):
            metrics = parse_numeric_kv_pairs(message[len("Train result:") :])
            record = {"line": line_no, "timestamp": timestamp_iso, "metrics": metrics}
            train_result_records.append(record)
            current_session["train_result_records"] += 1
            for metric_name, metric_value in metrics.items():
                metric_points["train_result"][metric_name].append(
                    make_point_record(
                        stage="train_result",
                        metric=metric_name,
                        value=metric_value,
                        timestamp=timestamp_iso,
                        epoch=max_epoch,
                        source="train_result",
                    )
                )
            continue

        best_metric_match = BEST_METRIC_RE.match(message)
        if best_metric_match:
            metric_name = best_metric_match.group("metric")
            metric_value = safe_float(best_metric_match.group("value"))
            if metric_value is not None:
                metric_points["best_line"][metric_name].append(
                    make_point_record(
                        stage="best_line",
                        metric=metric_name,
                        value=metric_value,
                        timestamp=timestamp_iso,
                        epoch=max_epoch,
                        source="best_line",
                    )
                )
            continue

        current_metric_match = CURRENT_METRIC_RE.match(message)
        if current_metric_match:
            metric_name = current_metric_match.group("metric")
            metric_value = safe_float(current_metric_match.group("value"))
            if metric_value is not None:
                point = make_point_record(
                    stage="scalar",
                    metric=metric_name,
                    value=metric_value,
                    timestamp=timestamp_iso,
                    epoch=max_epoch,
                    source="current_line",
                )
                scalar_records.append(point)
                metric_points["scalar"][metric_name].append(point)

    summary["basic_info"] = {
        "file_name": log_path.name,
        "file_size_bytes": file_size,
        "file_size_human": human_size(file_size),
        "line_count": len(raw_lines),
        "timestamp_range": {"first": first_timestamp, "last": last_timestamp},
        "epoch_range": {"min": min_epoch, "max": max_epoch, "hint_total": total_epoch_hint},
        "iter_range": {"min": min_iter, "max": max_iter, "hint_total": iter_total_hint},
        "runs_detected": len(sessions),
        "checkpoint_count": len(checkpoint_paths),
    }

    for session in sessions:
        if session["train_records"] == 0 and session["val_records"] == 0:
            status = "startup_only"
        elif session["train_records"] > 0 and session["val_records"] == 0:
            status = "train_only"
        elif session["checkpoints"]:
            status = "validated_with_checkpoints"
        else:
            status = "validated_no_checkpoint"
        session["status"] = status
    summary["sessions"] = sessions

    last_values: Dict[str, Dict[str, float]] = {}
    for stage_name, stage_series in metric_points.items():
        stage_last: Dict[str, float] = {}
        for metric_name, points in stage_series.items():
            if points:
                stage_last[metric_name] = points[-1]["value"]
        if stage_last:
            last_values[stage_name] = stage_last
    summary["last_values"] = last_values

    best_values: Dict[str, Dict[str, Any]] = {}
    for stage_name, stage_series in metric_points.items():
        stage_best: Dict[str, Any] = {}
        for metric_name, points in stage_series.items():
            direction = infer_metric_direction(metric_name)
            if direction in {"neutral", "unknown"}:
                continue
            chooser = min if direction == "lower_is_better" else max
            best_point = chooser(points, key=lambda item: item["value"])
            stage_best[metric_name] = {
                "value": best_point["value"],
                "timestamp": best_point["timestamp"],
                "epoch": best_point["epoch"],
                "iter": best_point["iter"],
                "direction": direction,
            }
        if stage_best:
            best_values[stage_name] = stage_best
    summary["best_values"] = best_values

    recent_changes: Dict[str, Any] = {}
    if train_records:
        recent_changes["train"] = train_records[-recent_count:]
    if val_records:
        recent_changes["val"] = val_records[-recent_count:]
    if train_result_records:
        recent_changes["train_result"] = train_result_records[-recent_count:]
    if scalar_records:
        recent_changes["scalar"] = scalar_records[-recent_count:]
    summary["recent_changes"] = recent_changes

    loss_trends: Dict[str, Dict[str, Any]] = {}
    for stage_name, stage_series in metric_points.items():
        stage_trends: Dict[str, Any] = {}
        for metric_name, points in stage_series.items():
            if not (metric_name.startswith("loss") or metric_name.startswith("val_loss")):
                continue
            values = [point["value"] for point in points]
            if not values:
                continue
            first_value = values[0]
            last_value = values[-1]
            delta = last_value - first_value
            threshold = max(1e-6, abs(first_value) * 0.02)
            if abs(delta) <= threshold:
                trend = "flat"
            elif delta < 0:
                trend = "down"
            else:
                trend = "up"
            stage_trends[metric_name] = {
                "count": len(values),
                "first": first_value,
                "last": last_value,
                "delta": delta,
                "min": min(values),
                "max": max(values),
                "trend": trend,
            }
        if stage_trends:
            loss_trends[stage_name] = stage_trends
    summary["loss_trends"] = loss_trends

    if sessions and len(sessions) > 1:
        summary["anomalies"].append(
            {
                "line": sessions[-1]["start_line"],
                "timestamp": sessions[-1]["start_timestamp"],
                "message": f"Detected {len(sessions)} training starts in one log file.",
            }
        )

    last_session = sessions[-1] if sessions else None
    if last_session and last_session["status"] == "startup_only":
        summary["anomalies"].append(
            {
                "line": last_session["start_line"],
                "timestamp": last_session["start_timestamp"],
                "message": "Latest session stopped at startup without train or validation records.",
            }
        )
    elif last_session and last_session["status"] == "train_only":
        summary["anomalies"].append(
            {
                "line": last_session["start_line"],
                "timestamp": last_session["start_timestamp"],
                "message": "Latest session has train records but no validation records.",
            }
        )

    if not val_records:
        summary["notes"].append("No validation records were detected in this log.")
    if not checkpoint_paths:
        summary["notes"].append("No checkpoint save lines were detected in this log.")
    if not train_records:
        summary["notes"].append("No per-iteration train records were detected in this log.")

    auto_questions: List[str] = []
    if sessions and len(sessions) > 1:
        auto_questions.append(
            f"Why were there {len(sessions)} startup attempts in the same log, and which session should be treated as authoritative?"
        )
    if last_session and last_session["status"] == "startup_only":
        auto_questions.append("What blocked the latest run before the first train iteration started?")
    if last_session and last_session["status"] == "train_only":
        auto_questions.append("Why did the latest run stop before validation completed?")
    if not checkpoint_paths:
        auto_questions.append("Why were no checkpoints recorded in the log output?")
    if val_records:
        val_miou_points = metric_points["val"].get("mIoU", []) + metric_points["scalar"].get("val_mIoU", [])
        if val_miou_points:
            last_miou = val_miou_points[-1]["value"]
            if last_miou < 0.1:
                auto_questions.append(
                    f"Is the latest val_mIoU {format_number(last_miou)} expected for this run, or does it point to a config / data / label mismatch?"
                )
    else:
        auto_questions.append("Should a summary artifact be generated only after validation appears, or is the current run intentionally startup-only?")
    if last_session and last_session.get("device") == "cpu":
        auto_questions.append("Is CPU execution expected here, or should this run have used a CUDA-enabled environment?")
    summary["auto_questions"] = auto_questions

    if class_results:
        summary["class_results_tail"] = class_results[-recent_count:]

    return summary


def trim_record_metrics(record: Dict[str, Any], limit: int = 8) -> Dict[str, Any]:
    metrics = record.get("metrics", {})
    important_order = [
        "loss",
        "loss_semantic",
        "loss_edge",
        "loss_support",
        "loss_axis",
        "loss_side",
        "loss_dir",
        "loss_dist",
        "mIoU",
        "mAcc",
        "allAcc",
        "support_cover",
        "support_error",
        "axis_cosine",
        "side_accuracy",
        "dir_cosine",
        "dist_error",
        "val_loss_edge",
        "val_loss_support",
        "val_loss_axis",
        "val_loss_side",
        "val_loss_dir",
        "val_loss_dist",
    ]
    picked: Dict[str, Any] = {}
    for key in important_order:
        if key in metrics and key not in picked:
            picked[key] = metrics[key]
        if len(picked) >= limit:
            break
    if len(picked) < limit:
        for key in sorted(metrics):
            if key not in picked:
                picked[key] = metrics[key]
            if len(picked) >= limit:
                break
    trimmed = {k: v for k, v in record.items() if k != "metrics"}
    trimmed["metrics"] = picked
    return trimmed


def markdown_for_summary(summary: Dict[str, Any], recent_count: int) -> str:
    basic = summary["basic_info"]
    lines: List[str] = [
        f"# Log Summary: {basic['file_name']}",
        "",
        "## Basic Info",
        "",
        f"- File: `{summary['input']['path']}`",
        f"- Size: `{basic['file_size_human']}` ({basic['file_size_bytes']} bytes)",
        f"- Lines: `{basic['line_count']}`",
        f"- Time range: `{basic['timestamp_range']['first']}` -> `{basic['timestamp_range']['last']}`",
        f"- Runs detected: `{basic['runs_detected']}`",
        f"- Epoch range: `{basic['epoch_range']['min']}` -> `{basic['epoch_range']['max']}` (hint total `{basic['epoch_range']['hint_total']}`)",
        f"- Iter range: `{basic['iter_range']['min']}` -> `{basic['iter_range']['max']}` (hint total `{basic['iter_range']['hint_total']}`)",
        f"- Checkpoint lines detected: `{basic['checkpoint_count']}`",
        "",
        "## Sessions",
        "",
    ]

    for index, session in enumerate(summary["sessions"], start=1):
        lines.extend(
            [
                f"- Session {index}: status `{session['status']}`, device `{session['device']}`, smoke `{session['smoke_mode']}`, "
                f"train records `{session['train_records']}`, val records `{session['val_records']}`, checkpoints `{len(session['checkpoints'])}`",
                f"  Start `{session['start_timestamp']}`, end `{session['end_timestamp']}`, save path `{session['save_path']}`",
            ]
        )

    lines.extend(["", "## Last Values", ""])
    stage_annotations = {
        "val": "val (step/batch-level, NOT epoch-aggregated)",
        "scalar": "scalar (epoch-aggregated, authoritative for val_mIoU)",
        "train": "train",
        "train_result": "train_result",
    }
    for stage_name in ("train_result", "val", "scalar", "train"):
        values = summary["last_values"].get(stage_name)
        if not values:
            continue
        lines.append(f"### {stage_annotations.get(stage_name, stage_name)}")
        lines.append("")
        for metric_name in sorted(values):
            lines.append(f"- `{metric_name}` = `{format_number(values[metric_name])}`")
        lines.append("")

    lines.extend(["## Best Values", ""])
    for stage_name in ("val", "train_result", "train", "scalar"):
        values = summary["best_values"].get(stage_name)
        if not values:
            continue
        lines.append(f"### {stage_annotations.get(stage_name, stage_name)}")
        lines.append("")
        for metric_name in sorted(values):
            item = values[metric_name]
            lines.append(
                f"- `{metric_name}` best `{format_number(item['value'])}` ({item['direction']}, ts `{item['timestamp']}`, epoch `{item['epoch']}`, iter `{item['iter']}`)"
            )
        lines.append("")

    lines.extend(["## Recent Changes", ""])
    for stage_name in ("train", "val", "train_result", "scalar"):
        records = summary["recent_changes"].get(stage_name)
        if not records:
            continue
        header = stage_annotations.get(stage_name, stage_name)
        lines.append(f"### {header} (last {min(recent_count, len(records))})")
        lines.append("")
        for record in records:
            if "metrics" in record:
                trimmed = trim_record_metrics(record)
                metric_text = ", ".join(
                    f"{name}={format_number(value)}" for name, value in trimmed["metrics"].items()
                )
                location = []
                if trimmed.get("epoch") is not None:
                    location.append(f"epoch {trimmed['epoch']}")
                if trimmed.get("iter") is not None:
                    location.append(f"iter {trimmed['iter']}")
                if trimmed.get("step") is not None:
                    location.append(f"step {trimmed['step']}")
                location_text = ", ".join(location) if location else "no step info"
                lines.append(f"- `{trimmed.get('timestamp')}` {location_text}: {metric_text}")
            else:
                lines.append(
                    f"- `{record.get('timestamp')}` `{record.get('metric')}` = `{format_number(record.get('value'))}`"
                )
        lines.append("")

    lines.extend(["## Loss Trends", ""])
    for stage_name in ("train", "val", "train_result"):
        trends = summary["loss_trends"].get(stage_name)
        if not trends:
            continue
        lines.append(f"### {stage_name}")
        lines.append("")
        for metric_name in sorted(trends):
            item = trends[metric_name]
            lines.append(
                f"- `{metric_name}`: first `{format_number(item['first'])}`, last `{format_number(item['last'])}`, "
                f"delta `{format_number(item['delta'])}`, min `{format_number(item['min'])}`, max `{format_number(item['max'])}`, trend `{item['trend']}`"
            )
        lines.append("")

    lines.extend(["## Warnings And Anomalies", ""])
    if not summary["warnings"] and not summary["anomalies"]:
        lines.append("- None detected.")
    else:
        for item in summary["warnings"][:20]:
            lines.append(f"- Warning line {item['line']}: `{item['message']}`")
        for item in summary["anomalies"][:20]:
            lines.append(f"- Anomaly line {item['line']}: `{item['message']}`")
    lines.append("")

    lines.extend(["## Auto Questions", ""])
    if not summary["auto_questions"]:
        lines.append("- None.")
    else:
        for question in summary["auto_questions"]:
            lines.append(f"- {question}")
    lines.append("")

    if summary.get("notes"):
        lines.extend(["## Notes", ""])
        for note in summary["notes"]:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_outputs(summary: Dict[str, Any], output_dir: Path, output_stem: str, recent_count: int) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{output_stem}.summary.md"
    json_path = output_dir / f"{output_stem}.summary.json"
    ensure_parent(md_path)
    ensure_parent(json_path)
    md_path.write_text(markdown_for_summary(summary, recent_count), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return md_path, json_path


def run_one(log_path: Path, output_dir: Path, recent_count: int, output_stem: Optional[str]) -> Tuple[Path, Path]:
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    summary = summarize_log(log_path, recent_count=recent_count)
    stem = output_stem or sanitize_output_stem(log_path)
    return write_outputs(summary, output_dir, stem, recent_count)


def main() -> int:
    args = parse_args()
    if args.name and len(args.logs) != 1:
        print("--name can only be used when summarizing a single log.", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).resolve()
    exit_code = 0
    for index, raw_log in enumerate(args.logs):
        try:
            md_path, json_path = run_one(
                Path(raw_log).resolve(),
                output_dir=output_dir,
                recent_count=max(args.recent_count, 1),
                output_stem=args.name if index == 0 else None,
            )
            print(f"Wrote {md_path}")
            print(f"Wrote {json_path}")
        except Exception as exc:  # pragma: no cover - last resort guard
            exit_code = 1
            print(f"Failed to summarize {raw_log}: {exc}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
