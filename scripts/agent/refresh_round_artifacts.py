#!/usr/bin/env python3
"""Refresh summary, context packet, and round update artifacts in one ordered chain."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


BACKTICK = "`"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh log summaries, context packets, and round updates in one ordered chain."
    )
    parser.add_argument(
        "--mode",
        choices=("draft", "preview", "apply"),
        default="draft",
        help="Draft refreshes summary/packet/draft. Preview refreshes first, then prints the fixed-scope diff. Apply refreshes first, then applies with confirmation.",
    )
    parser.add_argument(
        "--target",
        choices=("web_chat", "claude", "codex"),
        default="codex",
        help="Round update target passed to update_round_artifacts.py.",
    )
    parser.add_argument(
        "--packet-target",
        choices=("web_chat", "claude", "codex", "all"),
        default="all",
        help="Context packet target(s) to refresh before round update.",
    )
    parser.add_argument(
        "--current-state",
        default=str(default_current_state_path()),
        help="Path to project_memory/current_state.md.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Optional explicit task markdown path. Defaults to the task pointer in current_state.md.",
    )
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        help="Optional train log path to summarize. Can be passed multiple times.",
    )
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        help="Optional existing summary markdown/json path. Can be passed multiple times when no new log is provided.",
    )
    parser.add_argument(
        "--summary-dir",
        default=str(default_summary_dir()),
        help="Directory for summary artifacts.",
    )
    parser.add_argument(
        "--summary-name",
        default=None,
        help="Optional summary output stem. Only valid when a single --log is provided.",
    )
    parser.add_argument(
        "--recent-count",
        type=int,
        default=5,
        help="How many recent records summarize_train_log.py should keep.",
    )
    parser.add_argument(
        "--packet-dir",
        default=str(default_packet_dir()),
        help="Directory for context packet artifacts.",
    )
    parser.add_argument(
        "--max-summaries",
        type=int,
        default=3,
        help="Maximum number of summaries build_context_packet.py should include.",
    )
    parser.add_argument(
        "--round-output-dir",
        default=str(default_round_output_dir()),
        help="Directory for round update artifacts.",
    )
    parser.add_argument(
        "--round-name",
        default=None,
        help="Optional round update output stem. Passed through to update_round_artifacts.py.",
    )
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="Optional short manual note forwarded to update_round_artifacts.py.",
    )
    parser.add_argument(
        "--notes-file",
        default=None,
        help="Optional notes file forwarded to update_round_artifacts.py.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required with --mode apply. Passed through to update_round_artifacts.py.",
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


def summarize_script_path() -> Path:
    return repo_root() / "scripts" / "agent" / "summarize_train_log.py"


def packet_script_path() -> Path:
    return repo_root() / "scripts" / "agent" / "build_context_packet.py"


def round_update_script_path() -> Path:
    return repo_root() / "scripts" / "agent" / "update_round_artifacts.py"


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


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root()))
    except ValueError:
        return str(path.resolve())


def read_text_safe(path: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        return path.read_text(encoding="utf-8"), None
    except FileNotFoundError:
        return None, f"Missing file: {path}"
    except OSError as exc:
        return None, f"Failed to read {path}: {exc}"


def extract_task_path(current_state_text: str, current_state_path: Path) -> Path:
    for chunk in current_state_text.split(BACKTICK):
        token = chunk.strip()
        if token.startswith("project_memory/tasks/") and token.endswith(".md"):
            return (repo_root() / token).resolve()
    return current_state_path.parent / "tasks" / "TASK-UNKNOWN.md"


def discover_summary_artifacts(summary_dir: Path) -> List[Path]:
    if not summary_dir.exists():
        return []
    return sorted(summary_dir.glob("*.summary.md")) + sorted(summary_dir.glob("*.summary.json"))


def parse_written_paths(stdout: str) -> List[Path]:
    paths: List[Path] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("Wrote "):
            paths.append(Path(line[len("Wrote ") :].strip()).resolve())
    return paths


def parse_single_output_path(stdout: str) -> Optional[Path]:
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("# ") or line.startswith("NO-OP:"):
            continue
        if line.endswith(".md") and not line.startswith("- "):
            return Path(line).resolve()
    return None


def packet_path_for_target(task_path: Path, packet_dir: Path, target: str, generated_paths: Sequence[Path]) -> Path:
    for path in generated_paths:
        if path.name == f"{task_path.stem}.{target}.context_packet.md":
            return path.resolve()
    return (packet_dir / f"{task_path.stem}.{target}.context_packet.md").resolve()


def indent_block(text: str, prefix: str = "    ") -> List[str]:
    return [prefix + line if line else prefix.rstrip() for line in text.rstrip().splitlines()]


def make_step(name: str, status: str, details: Sequence[str], output: Optional[str] = None) -> dict:
    return {"name": name, "status": status, "details": list(details), "output": output}


def render_report(
    *,
    mode: str,
    target: str,
    packet_target: str,
    task_path: Path,
    steps: Sequence[dict],
) -> str:
    lines = [
        "# Refresh Round Artifacts",
        "",
        f"- Mode: {BACKTICK}{mode}{BACKTICK}",
        f"- Target: {BACKTICK}{target}{BACKTICK}",
        f"- Packet target: {BACKTICK}{packet_target}{BACKTICK}",
        f"- Task: {BACKTICK}{relpath(task_path)}{BACKTICK}",
        "",
        "## Steps",
        "",
    ]
    for idx, step in enumerate(steps, start=1):
        lines.append(f"{idx}. {BACKTICK}{step['name']}{BACKTICK} -> {BACKTICK}{step['status']}{BACKTICK}")
        for detail in step["details"]:
            lines.append(f"   - {detail}")
        if step.get("output"):
            lines.append("   - Output:")
            lines.extend(indent_block(step["output"]))
    return "\n".join(lines) + "\n"


def run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=str(repo_root()),
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> int:
    args = parse_args()

    if args.mode == "apply" and not args.confirm:
        print("Apply mode requires --confirm.", file=sys.stderr)
        return 2
    if args.summary_name and len(args.log) != 1:
        print("--summary-name is only valid when exactly one --log is provided.", file=sys.stderr)
        return 2

    current_state_path = resolve_cli_path(args.current_state)
    current_state_text, current_state_error = read_text_safe(current_state_path)
    if current_state_error or current_state_text is None:
        print(current_state_error or f"Unable to read current state: {current_state_path}", file=sys.stderr)
        return 1

    task_path = resolve_cli_path(args.task) if args.task else extract_task_path(current_state_text, current_state_path).resolve()
    task_text, task_error = read_text_safe(task_path)
    if task_error or task_text is None:
        print(task_error or f"Unable to read task file: {task_path}", file=sys.stderr)
        return 1

    summary_dir = resolve_cli_path(args.summary_dir)
    packet_dir = resolve_cli_path(args.packet_dir)
    round_output_dir = resolve_cli_path(args.round_output_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    packet_dir.mkdir(parents=True, exist_ok=True)
    round_output_dir.mkdir(parents=True, exist_ok=True)

    logs = [resolve_cli_path(item) for item in args.log]
    missing_logs = [path for path in logs if not path.exists()]
    if missing_logs:
        for path in missing_logs:
            print(f"Missing log input: {path}", file=sys.stderr)
        return 1

    explicit_summaries = [resolve_cli_path(item) for item in args.summary]
    missing_summaries = [path for path in explicit_summaries if not path.exists()]
    if missing_summaries:
        for path in missing_summaries:
            print(f"Missing summary input: {path}", file=sys.stderr)
        return 1

    steps: List[dict] = []
    summary_refs: List[Path] = []

    if logs:
        summary_cmd: List[str] = [
            sys.executable,
            str(summarize_script_path()),
            *[str(path) for path in logs],
            "--output-dir",
            str(summary_dir),
            "--recent-count",
            str(args.recent_count),
        ]
        if args.summary_name:
            summary_cmd.extend(["--name", args.summary_name])
        summary_run = run_command(summary_cmd)
        if summary_run.returncode != 0:
            steps.append(
                make_step(
                    "summarize_train_log.py",
                    "FAILED",
                    [f"Command exited with {summary_run.returncode}."],
                    summary_run.stderr or summary_run.stdout,
                )
            )
            print(
                render_report(
                    mode=args.mode,
                    target=args.target,
                    packet_target=args.packet_target,
                    task_path=task_path,
                    steps=steps,
                ),
                end="",
            )
            return summary_run.returncode
        summary_refs = parse_written_paths(summary_run.stdout)
        summary_details = [f"Processed {len(logs)} log file(s)."]
        if summary_refs:
            summary_details.extend(f"Wrote {BACKTICK}{path}{BACKTICK}" for path in summary_refs)
        steps.append(make_step("summarize_train_log.py", "EXECUTED", summary_details))
    else:
        if explicit_summaries:
            summary_refs = explicit_summaries
            steps.append(
                make_step(
                    "summarize_train_log.py",
                    "SKIPPED",
                    [f"Using {len(explicit_summaries)} explicit summary artifact(s)."],
                )
            )
        else:
            existing_summaries = discover_summary_artifacts(summary_dir)
            if not existing_summaries:
                print(
                    "No --log was provided and no existing summary artifacts were found. "
                    "Refresh cannot continue safely.",
                    file=sys.stderr,
                )
                return 1
            steps.append(
                make_step(
                    "summarize_train_log.py",
                    "SKIPPED",
                    [f"Using existing summaries from {BACKTICK}{relpath(summary_dir)}{BACKTICK}."],
                )
            )

    packet_cmd: List[str] = [
        sys.executable,
        str(packet_script_path()),
        "--target",
        args.packet_target,
        "--current-state",
        str(current_state_path),
        "--task",
        str(task_path),
        "--summary-dir",
        str(summary_dir),
        "--output-dir",
        str(packet_dir),
        "--max-summaries",
        str(args.max_summaries),
    ]
    for path in summary_refs:
        packet_cmd.extend(["--summary", str(path)])
    packet_run = run_command(packet_cmd)
    if packet_run.returncode != 0:
        steps.append(
            make_step(
                "build_context_packet.py",
                "FAILED",
                [f"Command exited with {packet_run.returncode}."],
                packet_run.stderr or packet_run.stdout,
            )
        )
        print(
            render_report(
                mode=args.mode,
                target=args.target,
                packet_target=args.packet_target,
                task_path=task_path,
                steps=steps,
            ),
            end="",
        )
        return packet_run.returncode
    packet_paths = parse_written_paths(packet_run.stdout)
    target_packet_path = packet_path_for_target(task_path, packet_dir, args.target, packet_paths)
    packet_details = [f"Refreshed packet target {BACKTICK}{args.packet_target}{BACKTICK}."]
    if packet_paths:
        packet_details.extend(f"Wrote {BACKTICK}{path}{BACKTICK}" for path in packet_paths)
    else:
        packet_details.append(f"Using packet path {BACKTICK}{target_packet_path}{BACKTICK}.")
    steps.append(make_step("build_context_packet.py", "EXECUTED", packet_details))

    draft_cmd: List[str] = [
        sys.executable,
        str(round_update_script_path()),
        "--mode",
        "draft",
        "--target",
        args.target,
        "--current-state",
        str(current_state_path),
        "--task",
        str(task_path),
        "--summary-dir",
        str(summary_dir),
        "--packet-dir",
        str(packet_dir),
        "--packet",
        str(target_packet_path),
        "--output-dir",
        str(round_output_dir),
    ]
    if args.round_name:
        draft_cmd.extend(["--name", args.round_name])
    for path in summary_refs:
        draft_cmd.extend(["--summary", str(path)])
    for note in args.note:
        draft_cmd.extend(["--note", note])
    if args.notes_file:
        draft_cmd.extend(["--notes-file", str(resolve_cli_path(args.notes_file))])

    draft_run = run_command(draft_cmd)
    if draft_run.returncode != 0:
        steps.append(
            make_step(
                "update_round_artifacts.py --mode draft",
                "FAILED",
                [f"Command exited with {draft_run.returncode}."],
                draft_run.stderr or draft_run.stdout,
            )
        )
        print(
            render_report(
                mode=args.mode,
                target=args.target,
                packet_target=args.packet_target,
                task_path=task_path,
                steps=steps,
            ),
            end="",
        )
        return draft_run.returncode

    draft_output_path = parse_single_output_path(draft_run.stdout)
    if draft_output_path is None:
        draft_output_path = (
            round_output_dir
            / (
                (args.round_name.strip() if args.round_name else f"{task_path.stem}.{args.target}.round_update.draft")
                + ".md"
            )
        ).resolve()
    draft_text, _ = read_text_safe(draft_output_path)
    draft_status = "NO-OP" if draft_text and "- Status: `no-op`" in draft_text else "EXECUTED"
    draft_details = [f"Refreshed draft artifact {BACKTICK}{draft_output_path}{BACKTICK}."]
    if draft_status == "NO-OP":
        draft_details.append("Canonical fixed blocks already match the current validated proposal.")
    steps.append(make_step("update_round_artifacts.py --mode draft", draft_status, draft_details))

    if args.mode == "draft":
        print(
            render_report(
                mode=args.mode,
                target=args.target,
                packet_target=args.packet_target,
                task_path=task_path,
                steps=steps,
            ),
            end="",
        )
        return 0

    final_cmd: List[str] = [
        sys.executable,
        str(round_update_script_path()),
        "--mode",
        args.mode,
        "--target",
        args.target,
        "--current-state",
        str(current_state_path),
        "--task",
        str(task_path),
        "--summary-dir",
        str(summary_dir),
        "--packet-dir",
        str(packet_dir),
        "--packet",
        str(target_packet_path),
        "--output-dir",
        str(round_output_dir),
    ]
    if args.round_name:
        final_cmd.extend(["--name", args.round_name])
    for path in summary_refs:
        final_cmd.extend(["--summary", str(path)])
    for note in args.note:
        final_cmd.extend(["--note", note])
    if args.notes_file:
        final_cmd.extend(["--notes-file", str(resolve_cli_path(args.notes_file))])
    if args.mode == "apply":
        final_cmd.append("--confirm")

    final_run = run_command(final_cmd)
    if final_run.returncode != 0:
        steps.append(
            make_step(
                f"update_round_artifacts.py --mode {args.mode}",
                "FAILED",
                [f"Command exited with {final_run.returncode}."],
                final_run.stderr or final_run.stdout,
            )
        )
        print(
            render_report(
                mode=args.mode,
                target=args.target,
                packet_target=args.packet_target,
                task_path=task_path,
                steps=steps,
            ),
            end="",
        )
        return final_run.returncode

    final_stdout = final_run.stdout.strip()
    final_status = "NO-OP" if "NO-OP:" in final_stdout else "EXECUTED"
    final_details = [f"Ran final mode {BACKTICK}{args.mode}{BACKTICK} against refreshed draft and packet."]
    if final_status == "NO-OP":
        final_details.append("Canonical files were already aligned within the fixed apply scope.")
    elif args.mode == "apply":
        final_details.append("Apply respected update_round_artifacts.py fixed-section safety boundaries.")
    steps.append(
        make_step(
            f"update_round_artifacts.py --mode {args.mode}",
            final_status,
            final_details,
            final_stdout if final_stdout else None,
        )
    )

    print(
        render_report(
            mode=args.mode,
            target=args.target,
            packet_target=args.packet_target,
            task_path=task_path,
            steps=steps,
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
