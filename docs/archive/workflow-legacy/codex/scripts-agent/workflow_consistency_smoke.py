#!/usr/bin/env python3
"""Check whether the current task canonical and checkpoint chain is present, fresh, and aligned."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


BACKTICK_RE = re.compile(r"`([^`]+)`")
SECTION_RE = re.compile(r"^(##+)\s+(.*)$")
HEADER_VALUE_RE = re.compile(r"^- ([^:]+): `([^`]+)`$")
SUMMARY_MD_SUFFIX = ".summary.md"
SUMMARY_JSON_SUFFIX = ".summary.json"
PACKET_TARGETS = ("web_chat", "claude", "codex")
POSITIVE_STATUS_HINTS = ("已核证", "已通过", "已确认")
NEGATIVE_STATUS_HINTS = ("待核证", "尚未确认", "未确认通过", "未核证")


@dataclass
class LayerReport:
    name: str
    facts: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)

    def add_fact(self, text: str) -> None:
        self.facts.append(text)

    def add_issue(self, text: str) -> None:
        self.issues.append(text)


@dataclass
class GlobalIssues:
    missing: List[str] = field(default_factory=list)
    stale: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect whether the current task canonical and checkpoint chain is closed, fresh, and aligned."
    )
    parser.add_argument(
        "--target",
        choices=("web_chat", "claude", "codex", "all"),
        default="codex",
        help="Target packet/draft view to inspect. Use 'all' for cross-target inspection.",
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
        "--summary-dir",
        default=str(default_summary_dir()),
        help="Directory containing *.summary.md and *.summary.json artifacts.",
    )
    parser.add_argument(
        "--packet-dir",
        default=str(default_packet_dir()),
        help="Directory containing *.context_packet.md artifacts.",
    )
    parser.add_argument(
        "--round-output-dir",
        default=str(default_round_output_dir()),
        help="Directory containing *.round_update.draft.md artifacts.",
    )
    parser.add_argument(
        "--latest-round",
        default=str(default_latest_round_path()),
        help="Path to handoff/latest_round.md.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir()),
        help="Directory for generated workflow consistency markdown reports.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional output stem. Defaults to <TASK-STEM>.workflow_consistency_smoke or workflow_consistency_smoke.",
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


def default_round_output_dir() -> Path:
    return repo_root() / "reports" / "round_updates"


def default_latest_round_path() -> Path:
    return repo_root() / "handoff" / "latest_round.md"


def default_output_dir() -> Path:
    return repo_root() / "reports" / "workflow_smokes"


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
    return re.sub(r"_+", "_", text).strip("_") or "workflow_consistency_smoke"


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


def extract_backticks(text: str) -> List[str]:
    return [match.group(1).strip() for match in BACKTICK_RE.finditer(text)]


def extract_task_path(current_state_text: str, current_state_path: Path) -> Optional[Path]:
    for token in extract_backticks(current_state_text):
        if token.startswith("project_memory/tasks/") and token.endswith(".md"):
            return (repo_root() / token).resolve()
    return current_state_path.parent / "tasks" / "TASK-UNKNOWN.md"


def markdown_header_map(text: str) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for raw_line in text.splitlines():
        match = HEADER_VALUE_RE.match(raw_line.strip())
        if match:
            headers[match.group(1).strip()] = match.group(2).strip()
    return headers


def summary_group_key(path: Path) -> Optional[str]:
    name = path.name
    if name.endswith(SUMMARY_MD_SUFFIX):
        return name[: -len(SUMMARY_MD_SUFFIX)]
    if name.endswith(SUMMARY_JSON_SUFFIX):
        return name[: -len(SUMMARY_JSON_SUFFIX)]
    return None


def normalize_timestamp(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    candidate = raw.strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(candidate).astimezone(timezone.utc).isoformat()
    except ValueError:
        return None


def detect_status_hint(text: str) -> str:
    positive = any(token in text for token in POSITIVE_STATUS_HINTS)
    negative = any(token in text for token in NEGATIVE_STATUS_HINTS)
    if positive and not negative:
        return "positive"
    if negative and not positive:
        return "negative"
    if positive and negative:
        return "mixed"
    return "unknown"


def latest_mtime(paths: Sequence[Path]) -> Optional[float]:
    mtimes = [path.stat().st_mtime for path in paths if path.exists()]
    if not mtimes:
        return None
    return max(mtimes)


def is_older_than(path: Path, dependency_mtime: Optional[float], slack_seconds: float = 1.0) -> bool:
    if dependency_mtime is None or not path.exists():
        return False
    return path.stat().st_mtime + slack_seconds < dependency_mtime


def format_ts(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def collect_summary_groups(summary_dir: Path) -> Dict[str, Dict[str, Path]]:
    groups: Dict[str, Dict[str, Path]] = {}
    if not summary_dir.exists():
        return groups
    for path in sorted(summary_dir.glob("*.summary.md")):
        key = summary_group_key(path)
        if key:
            groups.setdefault(key, {})["md"] = path.resolve()
    for path in sorted(summary_dir.glob("*.summary.json")):
        key = summary_group_key(path)
        if key:
            groups.setdefault(key, {})["json"] = path.resolve()
    return groups


def referenced_summary_keys(texts: Sequence[str]) -> List[str]:
    keys: List[str] = []
    seen = set()
    for text in texts:
        for token in extract_backticks(text):
            if not token.startswith("reports/log_summaries/"):
                continue
            key = summary_group_key(Path(token))
            if key and key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def choose_primary_summary_key(
    groups: Dict[str, Dict[str, Path]],
    preferred_keys: Sequence[str],
) -> Optional[str]:
    for key in preferred_keys:
        if key in groups:
            return key
    latest_key = None
    latest_value = None
    for key, entries in groups.items():
        value = latest_mtime(list(entries.values()))
        if value is not None and (latest_value is None or value > latest_value):
            latest_key = key
            latest_value = value
    return latest_key


def packet_path(packet_dir: Path, task_path: Path, target: str) -> Path:
    return (packet_dir / f"{task_path.stem}.{target}.context_packet.md").resolve()


def round_draft_path(round_dir: Path, task_path: Path, target: str) -> Path:
    return (round_dir / f"{task_path.stem}.{target}.round_update.draft.md").resolve()


def add_missing(layer: LayerReport, global_issues: GlobalIssues, text: str) -> None:
    layer.add_issue(f"Missing: {text}")
    global_issues.missing.append(text)


def add_stale(layer: LayerReport, global_issues: GlobalIssues, text: str) -> None:
    layer.add_issue(f"Stale: {text}")
    global_issues.stale.append(text)


def add_conflict(layer: LayerReport, global_issues: GlobalIssues, text: str) -> None:
    layer.add_issue(f"Conflict: {text}")
    global_issues.conflicts.append(text)


def status_from_issues(global_issues: GlobalIssues, critical_failures: Sequence[str]) -> str:
    if critical_failures or global_issues.conflicts:
        return "FAIL"
    if global_issues.missing or global_issues.stale:
        return "WARN"
    return "PASS"


def render_layer(lines: List[str], layer: LayerReport) -> None:
    lines.append(f"## {layer.name}")
    lines.append("")
    if layer.facts:
        for fact in layer.facts:
            lines.append(f"- {fact}")
    else:
        lines.append("- No facts collected.")
    if layer.issues:
        lines.append("- Issues:")
        for issue in layer.issues:
            lines.append(f"  - {issue}")
    else:
        lines.append("- Issues: none.")
    lines.append("")


def build_report(
    *,
    task_path: Optional[Path],
    target_label: str,
    verdict: str,
    global_issues: GlobalIssues,
    canonical_layer: LayerReport,
    task_layer: LayerReport,
    summary_layer: LayerReport,
    packet_layer: LayerReport,
    round_layer: LayerReport,
) -> str:
    lines = [
        f"# Workflow Consistency Smoke: {task_path.stem if task_path else 'TASK-UNKNOWN'}",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`",
        f"- Target scope: `{target_label}`",
        f"- Task: `{relpath(task_path) if task_path else 'unknown'}`",
        "",
    ]
    render_layer(lines, canonical_layer)
    render_layer(lines, task_layer)
    render_layer(lines, summary_layer)
    render_layer(lines, packet_layer)
    render_layer(lines, round_layer)
    lines.append("## Overall Verdict")
    lines.append("")
    lines.append(f"- Verdict: `{verdict}`")
    lines.append(f"- Missing items: `{len(global_issues.missing)}`")
    lines.append(f"- Stale items: `{len(global_issues.stale)}`")
    lines.append(f"- Conflict items: `{len(global_issues.conflicts)}`")
    if global_issues.missing:
        lines.append("- Missing detail:")
        for item in global_issues.missing:
            lines.append(f"  - {item}")
    if global_issues.stale:
        lines.append("- Stale detail:")
        for item in global_issues.stale:
            lines.append(f"  - {item}")
    if global_issues.conflicts:
        lines.append("- Conflict detail:")
        for item in global_issues.conflicts:
            lines.append(f"  - {item}")
    if not (global_issues.missing or global_issues.stale or global_issues.conflicts):
        lines.append("- Detail: no missing, stale, or conflict items were detected for the inspected scope.")
    lines.append("")
    lines.append("## Suggested Fixes")
    lines.append("")
    if global_issues.suggestions:
        for item in global_issues.suggestions:
            lines.append(f"- {item}")
    else:
        lines.append("- No immediate fix is required. Re-run this smoke after the next evidence refresh or before the next agent handoff.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    targets = list(PACKET_TARGETS) if args.target == "all" else [args.target]
    target_label = ",".join(targets)
    output_dir = resolve_cli_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_layer = LayerReport("Canonical Layer")
    task_layer = LayerReport("Task Layer")
    summary_layer = LayerReport("Summary Layer")
    packet_layer = LayerReport("Packet Layer")
    round_layer = LayerReport("Round Update Layer")
    global_issues = GlobalIssues()
    critical_failures: List[str] = []

    current_state_path = resolve_cli_path(args.current_state)
    current_state_text, current_state_error = read_text_safe(current_state_path)
    current_state_sections: Dict[str, str] = {}
    current_state_rel = relpath(current_state_path)
    task_path: Optional[Path] = resolve_cli_path(args.task) if args.task else None

    if current_state_text is None:
        add_missing(canonical_layer, global_issues, f"`{current_state_path}`")
        critical_failures.append("current_state_missing")
    else:
        current_state_sections = split_sections(current_state_text)
        canonical_layer.add_fact(f"`{current_state_rel}` is present.")
        if args.task:
            canonical_layer.add_fact(f"Task is explicitly provided as `{relpath(task_path)}`.")
        else:
            task_path = extract_task_path(current_state_text, current_state_path)
            canonical_layer.add_fact(f"Current task pointer resolves to `{relpath(task_path)}`.")

    if task_path is not None:
        task_text, task_error = read_text_safe(task_path)
    else:
        task_text, task_error = None, "Missing task path."

    if task_text is None:
        if task_path is not None:
            add_missing(task_layer, global_issues, f"`{task_path}`")
        else:
            add_missing(task_layer, global_issues, "Current task path could not be resolved.")
        critical_failures.append("task_missing")
        task_sections: Dict[str, str] = {}
    else:
        task_sections = split_sections(task_text)
        task_layer.add_fact(f"`{relpath(task_path)}` is present.")

    latest_round_path = resolve_cli_path(args.latest_round)
    latest_round_text, latest_round_error = read_text_safe(latest_round_path)
    if latest_round_text is None:
        add_missing(canonical_layer, global_issues, f"`{latest_round_path}`")
    else:
        canonical_layer.add_fact(f"`{relpath(latest_round_path)}` is present.")

    if current_state_text and task_text:
        current_task_section = current_state_sections.get("当前 task", current_state_text)
        if relpath(task_path) not in current_task_section and relpath(task_path) not in current_state_text:
            add_conflict(
                canonical_layer,
                global_issues,
                f"`{current_state_rel}` does not clearly reference `{relpath(task_path)}`.",
            )

        workspace_status = detect_status_hint(current_state_sections.get("当前 workspace 状态", current_state_text))
        result_status = detect_status_hint(task_sections.get("Result", task_text))
        if workspace_status != "unknown":
            task_layer.add_fact(f"Task result status heuristic: `{result_status}`.")
            canonical_layer.add_fact(f"Current state status heuristic: `{workspace_status}`.")
        if workspace_status != "unknown" and result_status != "unknown" and workspace_status != result_status:
            add_conflict(
                canonical_layer,
                global_issues,
                f"`{current_state_rel}` and `{relpath(task_path)}` disagree on verification status (`{workspace_status}` vs `{result_status}`).",
            )

    summary_dir = resolve_cli_path(args.summary_dir)
    summary_groups = collect_summary_groups(summary_dir)
    summary_layer.add_fact(f"Summary directory: `{relpath(summary_dir)}`.")
    preferred_summary_keys = referenced_summary_keys(
        [text for text in (current_state_text, task_text, latest_round_text) if text]
    )
    primary_summary_key = choose_primary_summary_key(summary_groups, preferred_summary_keys)
    primary_summary_mtime: Optional[float] = None
    primary_summary_status: Optional[str] = None
    primary_summary_input: Optional[Path] = None

    if not summary_groups:
        add_missing(summary_layer, global_issues, f"No `*.summary.md/json` artifacts found in `{summary_dir}`.")
        critical_failures.append("summary_missing")
    elif primary_summary_key is None:
        add_missing(summary_layer, global_issues, "No summary group could be selected for the current task.")
        critical_failures.append("summary_unresolved")
    else:
        group = summary_groups[primary_summary_key]
        md_path = group.get("md")
        json_path = group.get("json")
        summary_layer.add_fact(f"Primary summary group: `{primary_summary_key}`.")
        if md_path:
            summary_layer.add_fact(f"Markdown summary: `{relpath(md_path)}`.")
        else:
            add_missing(summary_layer, global_issues, f"Markdown summary missing for `{primary_summary_key}`.")
        if json_path:
            summary_layer.add_fact(f"JSON summary: `{relpath(json_path)}`.")
        else:
            add_missing(summary_layer, global_issues, f"JSON summary missing for `{primary_summary_key}`.")

        primary_summary_mtime = latest_mtime([path for path in (md_path, json_path) if path])
        if json_path:
            summary_json, summary_json_error = read_json_safe(json_path)
            if summary_json is None:
                add_missing(summary_layer, global_issues, summary_json_error or f"Unreadable JSON: `{json_path}`")
            else:
                sessions = summary_json.get("sessions") or []
                if sessions:
                    primary_summary_status = str(sessions[-1].get("status") or "unknown")
                    summary_layer.add_fact(f"Latest summary session status: `{primary_summary_status}`.")
                input_path_raw = str((summary_json.get("input") or {}).get("path") or "").strip()
                if input_path_raw:
                    primary_summary_input = Path(input_path_raw).resolve()
                    summary_layer.add_fact(f"Summary input log: `{relpath(primary_summary_input)}`.")
                if primary_summary_input and primary_summary_input.exists() and md_path and json_path:
                    if is_older_than(md_path, primary_summary_input.stat().st_mtime) or is_older_than(
                        json_path, primary_summary_input.stat().st_mtime
                    ):
                        add_stale(
                            summary_layer,
                            global_issues,
                            f"Summary group `{primary_summary_key}` is older than `{relpath(primary_summary_input)}`.",
                        )

    if current_state_text and task_text and primary_summary_status:
        workspace_status = detect_status_hint(current_state_sections.get("当前 workspace 状态", current_state_text))
        result_status = detect_status_hint(task_sections.get("Result", task_text))
        if primary_summary_status.startswith("validated"):
            if workspace_status == "negative":
                add_stale(
                    canonical_layer,
                    global_issues,
                    f"`{current_state_rel}` still reads as unverified while summary status is `{primary_summary_status}`.",
                )
            if result_status == "negative":
                add_stale(
                    task_layer,
                    global_issues,
                    f"`{relpath(task_path)}` still reads as unverified while summary status is `{primary_summary_status}`.",
                )
            if latest_round_text and detect_status_hint(latest_round_text) == "negative":
                add_stale(
                    canonical_layer,
                    global_issues,
                    f"`{relpath(latest_round_path)}` still reads as unverified while summary status is `{primary_summary_status}`.",
                )

    packet_dir = resolve_cli_path(args.packet_dir)
    packet_layer.add_fact(f"Packet directory: `{relpath(packet_dir)}`.")
    source_packet_mtime = latest_mtime(
        [path for path in (current_state_path, task_path) if path is not None and path.exists()]
    )
    if primary_summary_mtime is not None:
        source_packet_mtime = max(filter(None, [source_packet_mtime, primary_summary_mtime]))

    for target in targets:
        if task_path is None:
            add_missing(packet_layer, global_issues, f"Cannot resolve packet for `{target}` without a task path.")
            continue
        path = packet_path(packet_dir, task_path, target)
        text, error = read_text_safe(path)
        if text is None:
            add_missing(packet_layer, global_issues, f"`{relpath(path)}`")
            continue
        packet_layer.add_fact(f"`{target}` packet present: `{relpath(path)}`.")
        headers = markdown_header_map(text)
        expected_task = relpath(task_path)
        if headers.get("Task source") and headers["Task source"] != expected_task:
            add_conflict(
                packet_layer,
                global_issues,
                f"`{relpath(path)}` points to task `{headers['Task source']}` instead of `{expected_task}`.",
            )
        if headers.get("Current state source") and headers["Current state source"] != current_state_rel:
            add_conflict(
                packet_layer,
                global_issues,
                f"`{relpath(path)}` points to current state `{headers['Current state source']}` instead of `{current_state_rel}`.",
            )
        if is_older_than(path, source_packet_mtime):
            add_stale(
                packet_layer,
                global_issues,
                f"`{relpath(path)}` is older than current state, task, or primary summary inputs.",
            )

    round_output_dir = resolve_cli_path(args.round_output_dir)
    round_layer.add_fact(f"Round update directory: `{relpath(round_output_dir)}`.")
    for target in targets:
        if task_path is None:
            add_missing(round_layer, global_issues, f"Cannot resolve round draft for `{target}` without a task path.")
            continue
        path = round_draft_path(round_output_dir, task_path, target)
        text, error = read_text_safe(path)
        if text is None:
            add_missing(round_layer, global_issues, f"`{relpath(path)}`")
            continue
        round_layer.add_fact(f"`{target}` round draft present: `{relpath(path)}`.")
        headers = markdown_header_map(text)
        expected_task = relpath(task_path)
        if headers.get("Task source") and headers["Task source"] != expected_task:
            add_conflict(
                round_layer,
                global_issues,
                f"`{relpath(path)}` points to task `{headers['Task source']}` instead of `{expected_task}`.",
            )
        expected_packet = relpath(packet_path(packet_dir, task_path, target))
        if headers.get("Context packet") and headers["Context packet"] != expected_packet:
            add_conflict(
                round_layer,
                global_issues,
                f"`{relpath(path)}` points to packet `{headers['Context packet']}` instead of `{expected_packet}`.",
            )
        expected_primary_summary = (
            f"reports/log_summaries/{primary_summary_key}.summary.md" if primary_summary_key else None
        )
        if expected_primary_summary and headers.get("Primary summary") and headers["Primary summary"] != expected_primary_summary:
            add_conflict(
                round_layer,
                global_issues,
                f"`{relpath(path)}` points to summary `{headers['Primary summary']}` instead of `{expected_primary_summary}`.",
            )
        draft_dependencies = [source_packet_mtime]
        target_packet = packet_path(packet_dir, task_path, target)
        if target_packet.exists():
            draft_dependencies.append(target_packet.stat().st_mtime)
        dependency_mtime = max(value for value in draft_dependencies if value is not None) if any(
            value is not None for value in draft_dependencies
        ) else None
        if is_older_than(path, dependency_mtime):
            add_stale(
                round_layer,
                global_issues,
                f"`{relpath(path)}` is older than the packet or upstream canonical inputs for `{target}`.",
            )
        if "- Status: `no-op`" in text:
            round_layer.add_fact(f"`{target}` draft status is `no-op`.")

    if latest_round_text and task_path is not None:
        latest_round_headers = markdown_header_map(latest_round_text)
        canonical_layer.add_fact(f"`handoff/latest_round.md` current task reference check uses `{relpath(task_path)}`.")
        if relpath(task_path) not in latest_round_text:
            add_stale(
                canonical_layer,
                global_issues,
                f"`{relpath(latest_round_path)}` does not mention the current task `{relpath(task_path)}`.",
            )

    if not global_issues.missing and not global_issues.stale and not global_issues.conflicts:
        global_issues.suggestions.append(
            "The inspected workflow chain is aligned. Re-run this smoke after the next evidence refresh or before the next handoff."
        )
    else:
        if primary_summary_key is None or any("summary" in item.lower() for item in global_issues.missing + global_issues.stale):
            global_issues.suggestions.append(
                "If summary artifacts are missing or stale, run `scripts/agent/refresh_round_artifacts.py --mode draft --target codex --log <train.log>` to rebuild the chain in the correct order."
            )
        if any("context_packets" in item or "packet" in item.lower() for item in global_issues.missing + global_issues.stale):
            global_issues.suggestions.append(
                "If packet artifacts are missing or stale, run `scripts/agent/refresh_round_artifacts.py --mode draft --target <target>` so summary -> packet -> round draft are refreshed together."
            )
        if any("round_update" in item or "round draft" in item.lower() for item in global_issues.missing + global_issues.stale):
            global_issues.suggestions.append(
                "If round draft artifacts are missing or stale, refresh them through `scripts/agent/refresh_round_artifacts.py` before preview/apply."
            )
        if global_issues.conflicts:
            global_issues.suggestions.append(
                "If canonical and derived layers disagree, resolve the canonical source first, then regenerate packet and round draft instead of hand-editing stale artifacts."
            )
        if latest_round_text is None:
            global_issues.suggestions.append(
                "Create or refresh `handoff/latest_round.md` after the next successful round closeout so agent switching does not depend on memory."
            )

    verdict = status_from_issues(global_issues, critical_failures)
    output_stem = args.name.strip() if args.name else (
        f"{task_path.stem}.{args.target}.workflow_consistency_smoke"
        if task_path is not None
        else f"{args.target}.workflow_consistency_smoke"
    )
    output_path = (output_dir / f"{sanitize_output_stem(output_stem)}.md").resolve()
    report = build_report(
        task_path=task_path,
        target_label=target_label,
        verdict=verdict,
        global_issues=global_issues,
        canonical_layer=canonical_layer,
        task_layer=task_layer,
        summary_layer=summary_layer,
        packet_layer=packet_layer,
        round_layer=round_layer,
    )
    output_path.write_text(report, encoding="utf-8")
    print(output_path)
    return 1 if verdict == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
