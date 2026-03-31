#!/usr/bin/env python3
"""Generate conservative round-end writeback drafts from current task checkpoint artifacts."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SECTION_RE = re.compile(r"^(##+)\s+(.*)$")
BACKTICK_RE = re.compile(r"`([^`]+)`")
SUMMARY_MD_SUFFIX = ".summary.md"
SUMMARY_JSON_SUFFIX = ".summary.json"

TARGET_NOTES = {
    "web_chat": [
        "Keep the draft compact enough to paste into a new web chat without replaying full history.",
        "Favor packet and summary paths over raw logs or long document excerpts.",
    ],
    "claude": [
        "Treat the draft as a writeback proposal layer. Canonical facts still live in AGENTS/current_state/task.",
        "Use the draft to update the smallest set of docs instead of replaying a long handoff.",
    ],
    "codex": [
        "Use the draft as the preferred closeout checklist before editing task/current_state/handoff.",
        "If task, summary, or packet changed during the round, regenerate the draft instead of hand-merging old notes.",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build conservative round-end writeback drafts from checkpoint artifacts for task/current_state/handoff."
    )
    parser.add_argument(
        "--mode",
        choices=("draft", "preview", "apply"),
        default="draft",
        help="Draft writes the proposal artifact. Preview shows fixed-scope diffs. Apply writes the fixed blocks.",
    )
    parser.add_argument(
        "--target",
        choices=("web_chat", "claude", "codex"),
        default="codex",
        help="Target audience for the round draft.",
    )
    parser.add_argument(
        "--current-state",
        default=str(default_current_state_path()),
        help="Path to project_memory/current_state.md.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Optional explicit current task path. Defaults to the task pointer inside current_state.md.",
    )
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        help="Optional explicit summary markdown/json path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--summary-dir",
        default=str(default_summary_dir()),
        help="Directory containing *.summary.md and *.summary.json files.",
    )
    parser.add_argument(
        "--packet",
        default=None,
        help="Optional explicit context packet path.",
    )
    parser.add_argument(
        "--packet-dir",
        default=str(default_packet_dir()),
        help="Directory containing generated context packets.",
    )
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="Optional short manual note to fold into the draft. Can be passed multiple times.",
    )
    parser.add_argument(
        "--notes-file",
        default=None,
        help="Optional text or markdown file with additional manual notes.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir()),
        help="Directory for generated round draft markdown files.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional output stem. Defaults to <TASK-STEM>.<target>.round_update.draft.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required with --mode apply to explicitly confirm writing canonical files.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_current_state_path() -> Path:
    return repo_root() / "project_memory" / "current_state.md"


def default_summary_dir() -> Path:
    return repo_root() / "reports" / "log_summaries"


def default_packet_dir() -> Path:
    return repo_root() / "reports" / "context_packets"


def default_output_dir() -> Path:
    return repo_root() / "reports" / "round_updates"


def default_latest_round_path() -> Path:
    return repo_root() / "handoff" / "latest_round.md"


def resolve_cli_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == repo_root().name:
        return (repo_root().parent / path).resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (repo_root() / path).resolve()


def read_text_safe(path: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        return path.read_text(encoding="utf-8"), None
    except FileNotFoundError:
        return None, f"Missing file: {path}"
    except OSError as exc:
        return None, f"Failed to read {path}: {exc}"


def read_json_safe(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, f"Missing file: {path}"
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON in {path}: {exc}"
    except OSError as exc:
        return None, f"Failed to read {path}: {exc}"


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root()))
    except ValueError:
        return str(path.resolve())


def sanitize_output_stem(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "round_update"


def shorten(text: str, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def format_number(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1000 or (0 < abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.4f}"


def split_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {"__root__": []}
    current = "__root__"
    for raw_line in text.splitlines():
        match = SECTION_RE.match(raw_line.strip())
        if match:
            current = match.group(2).strip()
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(raw_line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def list_items(text: str, max_items: int = 4) -> List[str]:
    items: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("- "):
            items.append(line[2:].strip())
        elif re.match(r"^\d+\.\s+", line):
            items.append(re.sub(r"^\d+\.\s+", "", line))
        if len(items) >= max_items:
            break
    return items


def extract_backticks(text: str) -> List[str]:
    return [match.group(1).strip() for match in BACKTICK_RE.finditer(text)]


def extract_task_path(current_state_text: str, current_state_path: Path) -> Path:
    for token in extract_backticks(current_state_text):
        if token.startswith("project_memory/tasks/") and token.endswith(".md"):
            return (repo_root() / token).resolve()
    return current_state_path.parent / "tasks" / "TASK-UNKNOWN.md"


def summary_group_key(path: Path) -> Optional[str]:
    name = path.name
    if name.endswith(SUMMARY_MD_SUFFIX):
        return name[: -len(SUMMARY_MD_SUFFIX)]
    if name.endswith(SUMMARY_JSON_SUFFIX):
        return name[: -len(SUMMARY_JSON_SUFFIX)]
    return None


def collect_summary_pairs(explicit_paths: Sequence[str], summary_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    issues: List[str] = []
    groups: Dict[str, Dict[str, Any]] = {}
    candidate_paths: List[Path] = []

    if explicit_paths:
        for raw in explicit_paths:
            path = resolve_cli_path(raw)
            if not path.exists():
                issues.append(f"Missing explicit summary input: {path}")
                continue
            candidate_paths.append(path)
    else:
        if summary_dir.exists():
            candidate_paths.extend(sorted(summary_dir.glob("*.summary.md")))
            candidate_paths.extend(sorted(summary_dir.glob("*.summary.json")))
        else:
            issues.append(f"Summary directory not found: {summary_dir}")

    for path in candidate_paths:
        key = summary_group_key(path)
        if key is None:
            continue
        bucket = groups.setdefault(
            key,
            {"name": key, "md_path": None, "json_path": None, "mtime": 0.0},
        )
        if path.name.endswith(SUMMARY_MD_SUFFIX):
            bucket["md_path"] = path
        elif path.name.endswith(SUMMARY_JSON_SUFFIX):
            bucket["json_path"] = path
        try:
            bucket["mtime"] = max(bucket["mtime"], path.stat().st_mtime)
        except OSError:
            pass

    ordered = sorted(groups.values(), key=lambda item: item["mtime"], reverse=True)
    return ordered, issues


def get_nested(data: Any, *keys: str) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def coerce_metric_value(raw: Any) -> Optional[float]:
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, dict):
        value = raw.get("value")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def first_metric_value(mapping: Any, metric_names: Sequence[str]) -> Optional[float]:
    if not isinstance(mapping, dict):
        return None
    for metric in metric_names:
        value = coerce_metric_value(mapping.get(metric))
        if value is not None:
            return value
    lowered = {str(key).lower(): key for key in mapping.keys()}
    for metric in metric_names:
        match = lowered.get(metric.lower())
        if match is None:
            continue
        value = coerce_metric_value(mapping.get(match))
        if value is not None:
            return value
    return None


def load_summary_snapshot(summary_pair: Dict[str, Any]) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "name": summary_pair["name"],
        "md_path": summary_pair.get("md_path"),
        "json_path": summary_pair.get("json_path"),
        "issues": [],
        "available": False,
        "status": None,
        "device": None,
        "runs_detected": None,
        "checkpoint_count": None,
        "source_log": None,
        "latest_val_miou": None,
        "best_val_miou": None,
        "checkpoint_paths": [],
        "warnings": [],
        "anomalies": [],
        "auto_questions": [],
        "notes": [],
    }

    data: Optional[Dict[str, Any]] = None
    json_path = summary_pair.get("json_path")
    if isinstance(json_path, Path):
        data, error = read_json_safe(json_path)
        if error:
            snapshot["issues"].append(error)

    if data:
        snapshot["available"] = True
        sessions = data.get("sessions") if isinstance(data.get("sessions"), list) else []
        latest_session = sessions[-1] if sessions else {}
        snapshot["status"] = latest_session.get("status")
        snapshot["device"] = latest_session.get("device")
        snapshot["checkpoint_count"] = len(latest_session.get("checkpoints") or [])
        snapshot["checkpoint_paths"] = [
            str(path)
            for path in dict.fromkeys(str(item) for item in (latest_session.get("checkpoints") or []) if item)
        ]
        snapshot["runs_detected"] = get_nested(data, "basic_info", "runs_detected")
        snapshot["source_log"] = get_nested(data, "input", "path")
        last_val = get_nested(data, "last_values", "val")
        best_val = get_nested(data, "best_values", "val")
        snapshot["latest_val_miou"] = first_metric_value(last_val, ("val_mIoU", "mIoU", "miou"))
        snapshot["best_val_miou"] = first_metric_value(best_val, ("val_mIoU", "mIoU", "miou"))
        snapshot["warnings"] = [shorten(item.get("message", "")) for item in data.get("warnings", [])[:4]]
        snapshot["anomalies"] = [shorten(item.get("message", "")) for item in data.get("anomalies", [])[:4]]
        snapshot["auto_questions"] = [shorten(item) for item in data.get("auto_questions", [])[:4]]
        snapshot["notes"] = [shorten(item) for item in data.get("notes", [])[:4]]

    md_path = summary_pair.get("md_path")
    if isinstance(md_path, Path):
        text, error = read_text_safe(md_path)
        if error:
            snapshot["issues"].append(error)
        elif text and not snapshot["available"]:
            snapshot["available"] = True
            sections = split_sections(text)
            basic_excerpt = list_items(sections.get("Basic Info", ""), max_items=3)
            warning_excerpt = list_items(sections.get("Warnings And Anomalies", ""), max_items=4)
            question_excerpt = list_items(sections.get("Auto Questions", ""), max_items=4)
            snapshot["notes"].extend(shorten(item) for item in basic_excerpt if item)
            snapshot["warnings"].extend(shorten(item) for item in warning_excerpt if item)
            snapshot["auto_questions"].extend(shorten(item) for item in question_excerpt if item)

    return snapshot


def find_packet_path(task_path: Path, target: str, explicit_packet: Optional[str], packet_dir: Path) -> Tuple[Optional[Path], List[str]]:
    issues: List[str] = []
    if explicit_packet:
        packet_path = resolve_cli_path(explicit_packet)
        if packet_path.exists():
            return packet_path, issues
        issues.append(f"Missing explicit packet input: {packet_path}")
        return None, issues

    packet_path = packet_dir / f"{task_path.stem}.{target}.context_packet.md"
    if packet_path.exists():
        return packet_path, issues
    issues.append(f"Packet not found for target `{target}`: {packet_path}")
    return None, issues


def load_manual_notes(raw_notes: Sequence[str], notes_file: Optional[str]) -> Tuple[List[str], List[str]]:
    notes = [shorten(note) for note in raw_notes if note and note.strip()]
    issues: List[str] = []
    if not notes_file:
        return notes, issues

    path = Path(notes_file)
    if not path.is_absolute():
        path = resolve_cli_path(notes_file)
    text, error = read_text_safe(path)
    if error:
        issues.append(error)
        return notes, issues
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s+", "", line)
        notes.append(shorten(line))
    return notes, issues


def classify_round_outcome(primary_summary: Optional[Dict[str, Any]]) -> str:
    if not primary_summary or not primary_summary.get("available"):
        return "pending"
    status = str(primary_summary.get("status") or "").lower()
    if status == "validated_with_checkpoints":
        return "validated"
    if "validated" in status:
        return "partially_validated"
    if "train" in status:
        return "train_only"
    if "startup" in status:
        return "startup_only"
    return "unknown"


def build_evidence_lines(
    primary_summary: Optional[Dict[str, Any]],
    summary_snapshots: Sequence[Dict[str, Any]],
    packet_path: Optional[Path],
    manual_notes: Sequence[str],
) -> List[str]:
    lines: List[str] = []
    if primary_summary and primary_summary.get("available"):
        evidence = (
            f"日志摘要 `{primary_summary['name']}` 显示 latest session "
            f"`{primary_summary.get('status') or 'unknown'}`"
        )
        if primary_summary.get("device"):
            evidence += f"，device `{primary_summary['device']}`"
        if primary_summary.get("checkpoint_count") is not None:
            evidence += f"，checkpoint `{primary_summary['checkpoint_count']}`"
        evidence += "。"
        lines.append(evidence)
        if primary_summary.get("source_log"):
            lines.append(f"原始训练日志入口仍是 `{primary_summary['source_log']}`，但当前回写草稿以摘要为主。")
        if primary_summary.get("runs_detected") is not None:
            lines.append(f"同一日志中检测到 `{primary_summary['runs_detected']}` 次启动 / 会话，应以最后一个有效 session 为当前 round 的主要证据。")
        latest = primary_summary.get("latest_val_miou")
        best = primary_summary.get("best_val_miou")
        if latest is not None or best is not None:
            lines.append(
                "指标快照："
                f"latest val `mIoU={format_number(latest)}`，"
                f"best val `mIoU={format_number(best)}`。"
            )
    else:
        lines.append("当前未找到可用日志摘要，因此不能把本轮写成训练结果已确认。")

    if packet_path:
        lines.append(f"当前 round 还可回看 `{relpath(packet_path)}` 获取模块图、相关文档和摘要路径指引。")
    else:
        lines.append("当前未找到对应 target 的 context packet；如需补足上下文，应先生成 packet 再收尾。")

    for snapshot in summary_snapshots[1:2]:
        if snapshot.get("available"):
            lines.append(f"另有摘要 `{snapshot['name']}` 可作为补充证据。")

    if manual_notes:
        lines.append(f"人工备注已并入本轮草稿：{'; '.join(manual_notes[:3])}")
    return lines


def build_task_result_draft(
    task_sections: Dict[str, str],
    primary_summary: Optional[Dict[str, Any]],
    summary_snapshots: Sequence[Dict[str, Any]],
    packet_path: Optional[Path],
    manual_notes: Sequence[str],
) -> str:
    outcome = classify_round_outcome(primary_summary)
    current_status = {
        "validated": "已核证。",
        "partially_validated": "部分核证，仍需人工确认。",
        "train_only": "仅核证到训练阶段，验证证据仍不足。",
        "startup_only": "仅定位到启动信息，未形成有效训练证据。",
        "unknown": "有摘要，但结论仍需人工确认。",
        "pending": "待人工确认。",
    }[outcome]
    lines = ["## Result", "", f"- 当前状态：{current_status}"]
    lines.extend(f"- {line}" for line in build_evidence_lines(primary_summary, summary_snapshots, packet_path, manual_notes))
    if primary_summary and primary_summary.get("auto_questions"):
        lines.extend(f"- 待继续分析：{question}" for question in primary_summary["auto_questions"][:2])
    elif outcome == "pending":
        lines.append("- 待继续分析：先补生成对应 `*.summary.md` / `*.summary.json`，再决定是否更新结果。")
    return "\n".join(lines)


def build_task_next_step_draft(primary_summary: Optional[Dict[str, Any]]) -> str:
    outcome = classify_round_outcome(primary_summary)
    lines = ["## Next step", ""]
    if outcome == "validated":
        lines.append("- 下一轮切到 `axis-side` full train 验证，但保留当前 smoke 作为唯一已核证的训练证据入口。")
        lines.append("- full train 前先解释当前 smoke 中的低 `val_mIoU` 是否属于预期、标签问题或配置问题。")
        lines.append("- 开新窗口前先重新生成 packet 和 round update draft，避免继续沿用旧的“待核证”表述。")
    elif outcome in {"partially_validated", "train_only"}:
        lines.append("- 先补齐缺失的验证证据，再决定是否允许切到 full train。")
        lines.append("- 仅在摘要能确认 val 与 checkpoint 后，再把 smoke 写成通过。")
    else:
        lines.append("- 第一优先级仍是补齐或重跑 `axis-side` smoke，并为日志生成摘要。")
        lines.append("- 在没有摘要证据前，不要把 `current_state.md` 或 task 里的 smoke 状态改成已通过。")
    return "\n".join(lines)


def build_current_state_workspace_block(
    current_state_sections: Dict[str, str],
    primary_summary: Optional[Dict[str, Any]],
    packet_path: Optional[Path],
) -> str:
    outcome = classify_round_outcome(primary_summary)
    lines = ["## 当前 workspace 状态", ""]
    lines.append("- `axis-side` 的 loss / evaluator / trainer / config 修改都已在当前 working tree 中落地。")
    if outcome == "validated" and primary_summary:
        lines.append(
            "- 当前 `axis-side` smoke 已有一份可直接复核的摘要证据：latest session "
            f"`{primary_summary.get('status') or 'unknown'}`，"
            f"device `{primary_summary.get('device') or 'unknown'}`，"
            f"checkpoint `{primary_summary.get('checkpoint_count')}`。"
        )
        if primary_summary.get("runs_detected") is not None:
            lines.append(
                f"- 同一日志中检测到 `{primary_summary['runs_detected']}` 次启动 / 会话；当前应以最后一个 "
                "`validated_with_checkpoints` session 作为有效 smoke 证据。"
            )
        if primary_summary.get("source_log"):
            lines.append(
                f"- 可直接复核入口：`{primary_summary['source_log']}` 与对应 "
                "`reports/log_summaries/*.summary.{md,json}`。"
            )
    else:
        original = list_items(current_state_sections.get("当前 workspace 状态", ""), max_items=3)
        if original:
            lines.extend(f"- {item}" for item in original)
        else:
            lines.append("- 当前尚无可直接确认的 round-end 训练摘要证据。")
    if packet_path:
        lines.append(f"- 当前 target 的 context packet 位于 `{relpath(packet_path)}`。")
    return "\n".join(lines)


def build_current_state_next_step_block(primary_summary: Optional[Dict[str, Any]]) -> str:
    outcome = classify_round_outcome(primary_summary)
    lines = ["## 下一步", ""]
    if outcome == "validated":
        lines.append("- 第一优先级：围绕 `axis-side` full train 建立下一轮 task，并保留当前 smoke 摘要作为准入证据。")
        lines.append("- 第二优先级：解释当前 smoke 的低 `val_mIoU` 与多次启动记录，避免把“可跑通”误写成“效果正常”。")
    else:
        lines.append("- 第一优先级：补齐 `axis-side` smoke 的摘要证据，必要时在 CUDA-enabled `ptv3` 环境中重跑。")
        lines.append("- 第二优先级：仅在摘要能确认 val 与 checkpoint 后，再决定是否运行 `axis-side` full train。")
    return "\n".join(lines)


def build_handoff_latest_round_block(
    task_path: Path,
    packet_path: Optional[Path],
    summary_snapshots: Sequence[Dict[str, Any]],
    manual_notes: Sequence[str],
) -> str:
    latest_round_path = default_latest_round_path()
    primary_summary = summary_snapshots[0] if summary_snapshots else None
    outcome = classify_round_outcome(primary_summary)
    lines = ["# Latest Round", "", "## Summary", ""]
    if outcome == "validated" and primary_summary:
        lines.append(f"- 当前 round 围绕 `{relpath(task_path)}` 收尾；`axis-side` smoke 已在摘要层形成一次有效核证。")
        lines.append("- 核心证据来自 `reports/log_summaries/` 中的最新摘要，而不是手工回放原始长日志。")
    else:
        lines.append(f"- 当前 round 围绕 `{relpath(task_path)}` 收尾；训练结果仍需人工确认或补齐摘要。")
    lines.extend(["", "## Read First", ""])
    lines.append("- `AGENTS.md`")
    lines.append("- `project_memory/current_state.md`")
    lines.append(f"- `{relpath(task_path)}`")
    if packet_path:
        lines.append(f"- `{relpath(packet_path)}`")
    if primary_summary and primary_summary.get("md_path"):
        lines.append(f"- `{relpath(primary_summary['md_path'])}`")
    lines.extend(["", "## Evidence", ""])
    if primary_summary and primary_summary.get("available"):
        status = primary_summary.get("status") or "unknown"
        lines.append(f"- Latest summary status: `{status}`.")
        if primary_summary.get("device"):
            lines.append(f"- Device: `{primary_summary['device']}`.")
        if primary_summary.get("checkpoint_count") is not None:
            lines.append(f"- Checkpoints observed: `{primary_summary['checkpoint_count']}`.")
        if primary_summary.get("latest_val_miou") is not None:
            lines.append(f"- Latest val `mIoU={format_number(primary_summary['latest_val_miou'])}`.")
        if primary_summary.get("auto_questions"):
            lines.append(f"- Follow-up question: {primary_summary['auto_questions'][0]}")
    else:
        lines.append("- No usable summary was found for this draft; do not mark the round as verified yet.")
    if manual_notes:
        lines.extend(["", "## Manual Notes", ""])
        lines.extend(f"- {note}" for note in manual_notes[:4])
    lines.extend(["", "## Next Window", ""])
    if outcome == "validated":
        lines.append("- Open the next task for `axis-side` full train verification.")
        lines.append("- Keep checking summaries first; only open raw logs if the summary is insufficient.")
    else:
        lines.append("- Finish or regenerate the smoke summary before updating canonical state docs.")
        lines.append("- Regenerate the context packet if task or evidence changed during the next round.")

    action = "replace-file draft" if latest_round_path.exists() else "add-file draft"
    block = "\n".join(lines)
    return f"Target file: `{relpath(latest_round_path)}` ({action})\n\n```md\n{block}\n```"


def normalize_text(text: str) -> str:
    stripped = text.strip("\n")
    return (stripped + "\n") if stripped else ""


def normalize_draft_artifact(text: str) -> str:
    kept_lines = [line for line in normalize_text(text).splitlines() if not line.startswith("- Generated at: `")]
    return "\n".join(kept_lines).rstrip() + "\n" if kept_lines else ""


def summary_ref_label(primary_summary: Optional[Dict[str, Any]]) -> str:
    if primary_summary and primary_summary.get("name"):
        return f"reports/log_summaries/{primary_summary['name']}.summary.{{md,json}}"
    if primary_summary and primary_summary.get("md_path"):
        return relpath(primary_summary["md_path"])
    if primary_summary and primary_summary.get("json_path"):
        return relpath(primary_summary["json_path"])
    return "reports/log_summaries/*.summary.{md,json}"


def checkpoint_dir_label(primary_summary: Optional[Dict[str, Any]]) -> Optional[str]:
    if not primary_summary:
        return None
    checkpoint_paths = primary_summary.get("checkpoint_paths") or []
    if checkpoint_paths:
        return relpath(Path(str(checkpoint_paths[0])).parent) + "/"
    source_log = primary_summary.get("source_log")
    if source_log:
        return relpath(Path(str(source_log)).parent / "model") + "/"
    return None


def find_section_bounds(text: str, title: str) -> Optional[Tuple[int, int]]:
    lines = text.splitlines()
    header = f"## {title}"
    start: Optional[int] = None
    for idx, line in enumerate(lines):
        if line.strip() == header:
            start = idx
            break
    if start is None:
        return None
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].strip().startswith("## "):
            end = idx
            break
    return start, end


def extract_section_text(text: str, title: str) -> Optional[str]:
    bounds = find_section_bounds(text, title)
    if bounds is None:
        return None
    start, end = bounds
    return normalize_text("\n".join(text.splitlines()[start:end]))


def replace_section_text(text: str, title: str, replacement: str) -> Tuple[Optional[str], Optional[str]]:
    bounds = find_section_bounds(text, title)
    if bounds is None:
        return None, f"Missing section `## {title}`"
    lines = text.splitlines()
    start, end = bounds
    new_lines = lines[:start] + normalize_text(replacement).splitlines() + lines[end:]
    return "\n".join(new_lines).rstrip() + "\n", None


def tail_from_first_marker(section_text: str, markers: Sequence[str]) -> str:
    lines = normalize_text(section_text).splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if any(marker in stripped for marker in markers):
            return "\n".join(lines[idx:]).strip()
    return ""


def build_task_result_apply_block(primary_summary: Optional[Dict[str, Any]]) -> str:
    outcome = classify_round_outcome(primary_summary)
    lines = ["## Result", ""]
    if outcome == "validated" and primary_summary:
        lines.append("- 当前状态：已核证。")
        lines.append(
            f"- 根据 `{summary_ref_label(primary_summary)}`，`axis-side` smoke 的最后一个有效 session 为 "
            f"`{primary_summary.get('status') or 'unknown'}`，device `{primary_summary.get('device') or 'unknown'}`，"
            f"并在 `{checkpoint_dir_label(primary_summary) or 'outputs/.../model/'}` 下生成 `model_best.pth` 与 `model_last.pth`。"
        )
        if primary_summary.get("runs_detected") is not None:
            lines.append(f"- 同一 `train.log` 中共检测到 `{primary_summary['runs_detected']}` 次启动 / 会话；当前以最后一个有效 session 作为本轮 smoke 结论。")
        latest = primary_summary.get("latest_val_miou")
        best = primary_summary.get("best_val_miou")
        if latest is not None or best is not None:
            lines.append(f"- 指标快照：latest / best val `mIoU={format_number(latest if latest is not None else best)}`。")
        if primary_summary.get("runs_detected") is not None and latest is not None:
            lines.append(
                f"- 待继续分析：为何同一日志中出现 `{primary_summary['runs_detected']}` 次启动，以及当前 `val_mIoU={format_number(latest)}` 是否指向 config / data / label mismatch。"
            )
        elif primary_summary.get("auto_questions"):
            lines.append(f"- 待继续分析：{primary_summary['auto_questions'][0]}")
    else:
        lines.append("- 当前状态：待人工确认。")
        lines.append("- 当前缺少可直接写回 canonical 文档的摘要证据；先补齐 `summary` 和 `packet`。")
    return "\n".join(lines)


def build_task_next_step_apply_block(primary_summary: Optional[Dict[str, Any]]) -> str:
    outcome = classify_round_outcome(primary_summary)
    lines = ["## Next step", ""]
    if outcome == "validated":
        lines.append("- 下一轮应新建 `axis-side` full train 验证 task，并保留当前 smoke 摘要作为准入证据。")
        lines.append("- 在 full train 前先解释当前 smoke 的低 `val_mIoU` 与 `4` 次启动记录。")
        lines.append("- 开新窗口前重新生成 context packet 与 round update draft，避免继续沿用旧状态。")
    else:
        lines.append("- 第一优先级仍是补齐 `axis-side` smoke 的摘要证据。")
        lines.append("- 只有先确认 smoke，下一步才有资格进入 `axis-side` full train。")
    return "\n".join(lines)


def build_current_state_workspace_apply_block(existing_section_text: str, primary_summary: Optional[Dict[str, Any]]) -> str:
    lines = ["## 当前 workspace 状态", ""]
    lines.append("- `axis-side` 的 loss / evaluator / trainer / config 修改都已在当前 working tree 中落地。")
    if classify_round_outcome(primary_summary) == "validated" and primary_summary:
        lines.append(
            "- 当前 `axis-side` smoke 已有一份可直接复核的摘要证据：最后一个有效 session 为 "
            f"`{primary_summary.get('status') or 'unknown'}`，device `{primary_summary.get('device') or 'unknown'}`，"
            f"并在 `{checkpoint_dir_label(primary_summary) or 'outputs/.../model/'}` 下生成 `model_best.pth` 与 `model_last.pth`。"
        )
        if primary_summary.get("runs_detected") is not None:
            lines.append(f"- 同一 `train.log` 中共检测到 `{primary_summary['runs_detected']}` 次启动 / 会话；当前以最后一个有效 session 作为 smoke 结论。")
    else:
        lines.append("- 当前 `axis-side` smoke 仍缺少可直接写回的摘要证据。")
    tail = tail_from_first_marker(existing_section_text, ("`Route A`", "`B′`"))
    if tail:
        lines.append(tail)
    return "\n".join(lines)


def build_current_state_next_step_apply_block(existing_section_text: str, primary_summary: Optional[Dict[str, Any]]) -> str:
    lines = ["## 下一步", ""]
    if classify_round_outcome(primary_summary) == "validated":
        value = format_number(primary_summary.get("latest_val_miou")) if primary_summary else "n/a"
        lines.append("- 第一优先级：围绕 `axis-side` full train 建立下一轮 task，并保留当前 smoke 摘要作为准入证据。")
        lines.append(f"- 第二优先级：解释当前 smoke 的低 `val_mIoU={value}` 与多次启动记录，避免把“可跑通”误写成“效果正常”。")
    else:
        lines.append("- 第一优先级：补齐 `axis-side` smoke 的摘要证据。")
        lines.append("- 第二优先级：只有在 smoke 核证后，再决定是否进入 `axis-side` full train。")
    tail = tail_from_first_marker(existing_section_text, ("若任务需要补充上下文",))
    if tail:
        lines.append(tail)
    return "\n".join(lines)


def build_latest_round_apply_text(task_path: Path, primary_summary: Optional[Dict[str, Any]]) -> str:
    task_ref = relpath(task_path)
    lines = ["# Latest Round", "", "## Summary", ""]
    if classify_round_outcome(primary_summary) == "validated" and primary_summary:
        lines.append(f"- 本轮围绕 `{task_ref}` 收尾，`axis-side` smoke 已在摘要层形成一次有效核证。")
        lines.append("- 当前 canonical 文档已同步到“smoke 已核证，但 `axis-side` full train 尚未开始”的状态。")
    else:
        lines.append(f"- 本轮围绕 `{task_ref}` 收尾，但训练结论仍需补齐摘要后再同步。")
    lines.extend(["", "## Read First", ""])
    lines.append("- `AGENTS.md`")
    lines.append("- `project_memory/current_state.md`")
    lines.append(f"- `{task_ref}`")
    for target_name in ("web_chat", "claude", "codex"):
        lines.append(f"- `reports/context_packets/{task_path.stem}.{target_name}.context_packet.md`")
    if primary_summary and primary_summary.get("md_path"):
        lines.append(f"- `{relpath(primary_summary['md_path'])}`")
    elif primary_summary and primary_summary.get("json_path"):
        lines.append(f"- `{relpath(primary_summary['json_path'])}`")
    lines.extend(["", "## Evidence", ""])
    if primary_summary:
        lines.append(f"- Latest summary status: `{primary_summary.get('status') or 'unknown'}`.")
        if primary_summary.get("device"):
            lines.append(f"- Device: `{primary_summary['device']}`.")
        if primary_summary.get("checkpoint_count") is not None:
            lines.append(f"- Checkpoints observed: `{primary_summary['checkpoint_count']}`.")
        latest = primary_summary.get("latest_val_miou")
        best = primary_summary.get("best_val_miou")
        if latest is not None or best is not None:
            lines.append(f"- Latest / best val `mIoU={format_number(latest if latest is not None else best)}`.")
        if primary_summary.get("runs_detected") is not None and latest is not None:
            lines.append(
                f"- Follow-up question: why were there `{primary_summary['runs_detected']}` startup attempts in the same log, and should the low `val_mIoU` be treated as expected smoke behavior or as a config / data / label mismatch?"
            )
    else:
        lines.append("- Latest summary is unavailable; do not treat the round as verified yet.")
    lines.extend(["", "## Next Window", ""])
    if classify_round_outcome(primary_summary) == "validated":
        lines.append("- 为 `axis-side` full train 新建下一轮 task。")
        lines.append("- 继续优先使用 `summary -> context packet -> round update draft`，原始长日志只按需下钻。")
        lines.append("- 在 full train 前先解释当前 smoke 的低 `val_mIoU` 与多次启动记录。")
    else:
        lines.append("- 先补齐 smoke 的摘要和 packet，再同步 canonical 文档。")
    return "\n".join(lines) + "\n"


def make_operation(*, path: Path, section: str, current_text: str, proposed_text: str) -> Dict[str, Any]:
    current_norm = normalize_text(current_text)
    proposed_norm = normalize_text(proposed_text)
    return {"path": path, "section": section, "current": current_norm, "proposed": proposed_norm, "changed": current_norm != proposed_norm}


def build_apply_operations(
    *,
    task_path: Path,
    task_text: str,
    current_state_path: Path,
    current_state_text: str,
    latest_round_path: Path,
    latest_round_text: Optional[str],
    primary_summary: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    issues: List[str] = []
    operations: List[Dict[str, Any]] = []

    task_result_current = extract_section_text(task_text, "Result")
    if task_result_current is None:
        issues.append(f"Missing section `## Result` in {task_path}")
    else:
        operations.append(make_operation(path=task_path, section="Result", current_text=task_result_current, proposed_text=build_task_result_apply_block(primary_summary)))

    task_next_current = extract_section_text(task_text, "Next step")
    if task_next_current is None:
        issues.append(f"Missing section `## Next step` in {task_path}")
    else:
        operations.append(make_operation(path=task_path, section="Next step", current_text=task_next_current, proposed_text=build_task_next_step_apply_block(primary_summary)))

    workspace_current = extract_section_text(current_state_text, "当前 workspace 状态")
    if workspace_current is None:
        issues.append(f"Missing section `## 当前 workspace 状态` in {current_state_path}")
    else:
        operations.append(make_operation(path=current_state_path, section="当前 workspace 状态", current_text=workspace_current, proposed_text=build_current_state_workspace_apply_block(workspace_current, primary_summary)))

    next_current = extract_section_text(current_state_text, "下一步")
    if next_current is None:
        issues.append(f"Missing section `## 下一步` in {current_state_path}")
    else:
        operations.append(make_operation(path=current_state_path, section="下一步", current_text=next_current, proposed_text=build_current_state_next_step_apply_block(next_current, primary_summary)))

    operations.append(make_operation(path=latest_round_path, section="whole_file", current_text=latest_round_text or "", proposed_text=build_latest_round_apply_text(task_path, primary_summary)))
    return operations, issues


def build_noop_markdown(
    *,
    target: str,
    current_state_path: Path,
    task_path: Path,
    packet_path: Optional[Path],
    primary_summary: Optional[Dict[str, Any]],
    issues: Sequence[str],
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    lines = [
        f"# Round Update Draft: {task_path.stem} / {target}",
        "",
        f"- Generated at: `{generated_at}`",
        "- Mode: `draft`",
        "- Status: `no-op`",
        f"- Current state source: `{relpath(current_state_path)}`",
        f"- Task source: `{relpath(task_path)}`",
        f"- Context packet: `{relpath(packet_path)}`" if packet_path else "- Context packet: `missing`",
        f"- Primary summary: `{relpath(primary_summary['md_path'])}`" if primary_summary and primary_summary.get("md_path") else f"- Primary summary: `{summary_ref_label(primary_summary)}`",
        "",
        "## No-op",
        "",
        "- Canonical fixed blocks already match the current validated proposal.",
        "- Checked scope:",
        "- `project_memory/tasks/TASK-*.md` -> `Result`, `Next step`",
        "- `project_memory/current_state.md` -> `当前 workspace 状态`, `下一步`",
        "- `handoff/latest_round.md` -> whole file",
        "",
        "## Notes",
        "",
        "- Non-fixed sections and unrelated docs still require manual review.",
        "- If summary or packet changes, regenerate the draft before preview/apply.",
    ]
    if issues:
        lines.extend(["", "## Input Gaps", ""])
        lines.extend(f"- {issue}" for issue in issues)
    return "\n".join(lines) + "\n"


def build_markdown(
    *,
    target: str,
    current_state_path: Path,
    task_path: Path,
    packet_path: Optional[Path],
    summary_snapshots: Sequence[Dict[str, Any]],
    task_sections: Dict[str, str],
    current_state_sections: Dict[str, str],
    manual_notes: Sequence[str],
    issues: Sequence[str],
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    primary_summary = summary_snapshots[0] if summary_snapshots else None
    metadata_lines = [
        f"# Round Update Draft: {task_path.stem} / {target}",
        "",
        f"- Generated at: `{generated_at}`",
        "- Mode: `draft`",
        f"- Target: `{target}`",
        f"- Current state source: `{relpath(current_state_path)}`",
        f"- Task source: `{relpath(task_path)}`",
        f"- Context packet: `{relpath(packet_path)}`" if packet_path else "- Context packet: `missing`",
    ]
    if primary_summary and primary_summary.get("md_path"):
        metadata_lines.append(f"- Primary summary: `{relpath(primary_summary['md_path'])}`")
    elif primary_summary and primary_summary.get("json_path"):
        metadata_lines.append(f"- Primary summary: `{relpath(primary_summary['json_path'])}`")
    else:
        metadata_lines.append("- Primary summary: `missing`")

    lines = metadata_lines
    lines.extend(["", "## Purpose", ""])
    lines.append("- Provide a short, reviewable writeback draft for `task`, `current_state`, and `handoff/latest_round.md`.")
    lines.append("- Keep the round-end summary anchored to packet and log-summary artifacts instead of replaying long history.")
    lines.extend(["", "## Notes For This Target", ""])
    lines.extend(f"- {note}" for note in TARGET_NOTES[target])
    if issues:
        lines.extend(["", "## Input Gaps", ""])
        lines.extend(f"- {issue}" for issue in issues)
    lines.extend(["", "## Task Result Draft", ""])
    lines.append(f"Suggested replacement for `## Result` in `{relpath(task_path)}`:")
    lines.extend(["", "```md", build_task_result_draft(task_sections, primary_summary, summary_snapshots, packet_path, manual_notes), "```"])
    lines.extend(["", "## Task Next Step Draft", ""])
    lines.append(f"Suggested replacement for `## Next step` in `{relpath(task_path)}`:")
    lines.extend(["", "```md", build_task_next_step_draft(primary_summary), "```"])
    lines.extend(["", "## Current State Patch Draft", ""])
    lines.append(f"Suggested replacements for `## 当前 workspace 状态` and `## 下一步` in `{relpath(current_state_path)}`:")
    lines.extend(["", "```md", build_current_state_workspace_block(current_state_sections, primary_summary, packet_path), "", build_current_state_next_step_block(primary_summary), "```"])
    lines.extend(["", "## Latest Round Handoff Patch Draft", ""])
    lines.append(build_handoff_latest_round_block(task_path, packet_path, summary_snapshots, manual_notes))
    return "\n".join(lines) + "\n"


def render_preview(operations: Sequence[Dict[str, Any]]) -> str:
    changed_ops = [op for op in operations if op["changed"]]
    changed_files: List[Path] = []
    for op in changed_ops:
        if op["path"] not in changed_files:
            changed_files.append(op["path"])
    lines = ["# Preview", "", f"- Files to modify: `{len(changed_files)}`"]
    for path in changed_files:
        block_names = [op["section"] for op in changed_ops if op["path"] == path]
        lines.append(f"- `{relpath(path)}` -> {', '.join(block_names)}")
    lines.extend(["", "## Block Diffs", ""])
    for op in changed_ops:
        label = f"{relpath(op['path'])} :: {op['section']}"
        lines.append(f"### {label}")
        lines.append("")
        lines.append("```diff")
        diff_lines = list(difflib.unified_diff(op["current"].splitlines(), op["proposed"].splitlines(), fromfile=f"{label} (current)", tofile=f"{label} (proposed)", lineterm=""))
        lines.extend(diff_lines or ["(no diff)"])
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def preview_prereq_errors(
    *,
    issues: Sequence[str],
    packet_path: Optional[Path],
    primary_summary: Optional[Dict[str, Any]],
    output_path: Path,
    expected_draft_text: str,
) -> List[str]:
    errors: List[str] = list(issues)
    if primary_summary is None or not primary_summary.get("available"):
        errors.append("Missing usable summary input. Refresh summary artifacts before preview/apply.")
    if packet_path is None or not packet_path.exists():
        errors.append("Missing context packet input. Refresh packet artifacts before preview/apply.")
    draft_text, draft_error = read_text_safe(output_path)
    if draft_error or draft_text is None:
        errors.append(f"Missing draft artifact: {output_path}. Run --mode draft first.")
    elif normalize_draft_artifact(draft_text) != normalize_draft_artifact(expected_draft_text):
        errors.append(f"Draft artifact is stale relative to current inputs: {output_path}. Run --mode draft first.")
    return errors


def apply_operations(
    *,
    operations: Sequence[Dict[str, Any]],
    task_path: Path,
    task_text: str,
    current_state_path: Path,
    current_state_text: str,
    latest_round_path: Path,
    latest_round_text: Optional[str],
) -> Tuple[List[Path], List[str]]:
    changed_ops = [op for op in operations if op["changed"]]
    if not changed_ops:
        return [], []

    issues: List[str] = []
    original_map = {
        task_path: normalize_text(task_text),
        current_state_path: normalize_text(current_state_text),
        latest_round_path: normalize_text(latest_round_text or ""),
    }
    new_contents = dict(original_map)

    for op in changed_ops:
        if op["path"] == latest_round_path and op["section"] == "whole_file":
            new_contents[latest_round_path] = normalize_text(op["proposed"])
            continue
        updated_text, error = replace_section_text(new_contents[op["path"]], op["section"], op["proposed"])
        if error or updated_text is None:
            issues.append(f"{relpath(op['path'])}: {error or 'Failed to replace section.'}")
            continue
        new_contents[op["path"]] = updated_text

    if issues:
        return [], issues

    written: List[Path] = []
    for path, content in new_contents.items():
        if normalize_text(content) == original_map[path]:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(normalize_text(content), encoding="utf-8")
        written.append(path)
    return written, []


def main() -> int:
    args = parse_args()

    current_state_path = resolve_cli_path(args.current_state)
    current_state_text, error = read_text_safe(current_state_path)
    if error or current_state_text is None:
        print(error or f"Unable to read current state: {current_state_path}", file=sys.stderr)
        return 1
    current_state_sections = split_sections(current_state_text)

    task_path = resolve_cli_path(args.task) if args.task else extract_task_path(current_state_text, current_state_path).resolve()
    task_text, task_error = read_text_safe(task_path)
    if task_text is None:
        print(task_error or f"Unable to read task file: {task_path}", file=sys.stderr)
        return 1
    task_sections = split_sections(task_text)

    issues: List[str] = []
    if task_error:
        issues.append(task_error)

    summary_dir = resolve_cli_path(args.summary_dir)
    summary_pairs, summary_issues = collect_summary_pairs(args.summary, summary_dir)
    issues.extend(summary_issues)
    summary_snapshots = [load_summary_snapshot(pair) for pair in summary_pairs[:3]]
    for snapshot in summary_snapshots:
        issues.extend(snapshot.get("issues", []))
    primary_summary = summary_snapshots[0] if summary_snapshots else None

    packet_dir = resolve_cli_path(args.packet_dir)
    packet_path, packet_issues = find_packet_path(task_path, args.target, args.packet, packet_dir)
    issues.extend(packet_issues)

    manual_notes, note_issues = load_manual_notes(args.note, args.notes_file)
    issues.extend(note_issues)

    latest_round_path = default_latest_round_path()
    latest_round_text, latest_round_error = read_text_safe(latest_round_path)
    if latest_round_error and not latest_round_error.startswith("Missing file:"):
        issues.append(latest_round_error)
        latest_round_text = None

    operations, op_issues = build_apply_operations(
        task_path=task_path,
        task_text=task_text,
        current_state_path=current_state_path,
        current_state_text=current_state_text,
        latest_round_path=latest_round_path,
        latest_round_text=latest_round_text,
        primary_summary=primary_summary,
    )
    issues.extend(op_issues)
    changed_ops = [op for op in operations if op["changed"]]

    output_dir = resolve_cli_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = sanitize_output_stem(args.name.strip()) if args.name and args.name.strip() else sanitize_output_stem(f"{task_path.stem}.{args.target}.round_update.draft")
    output_path = output_dir / f"{output_stem}.md"

    draft_markdown = build_markdown(
        target=args.target,
        current_state_path=current_state_path,
        task_path=task_path,
        packet_path=packet_path,
        summary_snapshots=summary_snapshots,
        task_sections=task_sections,
        current_state_sections=current_state_sections,
        manual_notes=manual_notes,
        issues=issues,
    ) if changed_ops else build_noop_markdown(
        target=args.target,
        current_state_path=current_state_path,
        task_path=task_path,
        packet_path=packet_path,
        primary_summary=primary_summary,
        issues=issues,
    )

    if args.mode == "draft":
        output_path.write_text(draft_markdown, encoding="utf-8")
        print(output_path.resolve())
        return 0

    prereq_errors = preview_prereq_errors(
        issues=issues,
        packet_path=packet_path,
        primary_summary=primary_summary,
        output_path=output_path,
        expected_draft_text=draft_markdown,
    )
    if prereq_errors:
        for item in prereq_errors:
            print(item, file=sys.stderr)
        return 1

    if not changed_ops:
        print("NO-OP: canonical files already match the fixed apply scope.")
        return 0

    preview_text = render_preview(operations)
    if args.mode == "preview":
        print(preview_text)
        return 0

    if not args.confirm:
        print("Apply mode requires --confirm after reviewing preview/diff.", file=sys.stderr)
        return 2

    written_files, apply_issues = apply_operations(
        operations=operations,
        task_path=task_path,
        task_text=task_text,
        current_state_path=current_state_path,
        current_state_text=current_state_text,
        latest_round_path=latest_round_path,
        latest_round_text=latest_round_text,
    )
    if apply_issues:
        for item in apply_issues:
            print(item, file=sys.stderr)
        return 1

    noop_markdown = build_noop_markdown(
        target=args.target,
        current_state_path=current_state_path,
        task_path=task_path,
        packet_path=packet_path,
        primary_summary=primary_summary,
        issues=[],
    )
    output_path.write_text(noop_markdown, encoding="utf-8")

    print(preview_text.rstrip())
    print("")
    print(f"Applied {len(written_files)} file(s).")
    print(f"Refreshed draft artifact to no-op: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
