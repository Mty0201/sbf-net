#!/usr/bin/env python3
"""Build a target-specific markdown context packet from current project state."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


TARGET_NOTES = {
    "web_chat": {
        "purpose": "Provide a compact packet that can be pasted or attached to a new web chat without manually restating long context.",
        "read_order": [
            "This packet",
            "AGENTS.md for global rules",
            "project_memory/current_state.md for current facts",
            "Current task file for this round",
            "Only the linked docs or log summaries that remain necessary",
        ],
        "notes": [
            "Favor short excerpts and explicit file paths that can be copied into a web chat if needed.",
            "Do not dump raw logs; use linked summaries first and only quote small log fragments if the summary is insufficient.",
        ],
    },
    "claude": {
        "purpose": "Provide Claude with a startup packet that is thinner than full memory replay but still grounded in the current task, rules, and evidence.",
        "read_order": [
            "This packet",
            "AGENTS.md for canonical workflow rules",
            "project_memory/current_state.md for current facts and task pointer",
            "Current task file for execution intent",
            "Linked summaries and module files only when the task needs them",
        ],
        "notes": [
            "Treat this packet as the preferred entry layer, but keep AGENTS/current_state/task as the source of truth if anything conflicts.",
            "Use the linked module files and summaries rather than reopening full handoff or raw logs by default.",
        ],
    },
    "codex": {
        "purpose": "Provide Codex with an execution-oriented packet that points directly to the current task, relevant modules, and training evidence.",
        "read_order": [
            "This packet",
            "AGENTS.md for boundaries and workflow",
            "project_memory/current_state.md for the active state snapshot",
            "Current task file for the round objective and validation bar",
            "Linked source files and log summaries before any raw long log",
        ],
        "notes": [
            "Use this packet to reduce repeated discovery work before opening code or evidence files.",
            "If the packet becomes stale after task, state, or summary changes, regenerate it rather than hand-editing old context.",
        ],
    },
}

BACKTICK_RE = re.compile(r"`([^`]+)`")
SECTION_RE = re.compile(r"^(##+)\s+(.*)$")
IDENTIFIER_RE = re.compile(r"^[A-Z][A-Za-z0-9_]+$")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
SUMMARY_MD_SUFFIX = ".summary.md"
SUMMARY_JSON_SUFFIX = ".summary.json"

MODULE_CATEGORY_HINTS = (
    ("Model", ("主模型", "model")),
    ("Loss", ("loss 模块", "loss")),
    ("Evaluator", ("evaluator 模块", "evaluator")),
    ("Runtime", ("runtime 模块", "trainer")),
    ("Dataset", ("数据模块", "dataset")),
    ("Transform", ("变换模块", "transform")),
    ("Config", ("config", "配置")),
)

SEARCH_ROOTS = ("project", "configs", "scripts")
SEARCH_SUFFIXES = {".py", ".md", ".toml"}
CATEGORY_PREFERRED_SUBPATHS = {
    "Model": ("project/models",),
    "Loss": ("project/losses",),
    "Evaluator": ("project/evaluator",),
    "Runtime": ("project/trainer",),
    "Dataset": ("project/datasets",),
    "Transform": ("project/transforms",),
    "Config": ("configs",),
    "Script": ("scripts",),
}
KEYWORD_STOPWORDS = {
    "project",
    "memory",
    "current",
    "state",
    "task",
    "train",
    "training",
    "stage",
    "phase",
    "route",
    "result",
    "results",
    "support",
    "semantic",
    "boundary",
    "field",
    "smoke",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build target-specific context packet markdown files.")
    parser.add_argument(
        "--target",
        choices=("web_chat", "claude", "codex", "all"),
        default="all",
        help="Packet target to generate. Use 'all' to build every supported packet.",
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
        "--summary-dir",
        default=str(default_summary_dir()),
        help="Directory that stores *.summary.md and *.summary.json log summaries.",
    )
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        help="Optional explicit summary markdown/json file to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir()),
        help="Directory for generated context packet markdown files.",
    )
    parser.add_argument(
        "--max-summaries",
        type=int,
        default=3,
        help="Maximum number of summary artifacts to include.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional output stem. Only valid when generating a single target packet.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_current_state_path() -> Path:
    return repo_root() / "project_memory" / "current_state.md"


def default_summary_dir() -> Path:
    return repo_root() / "reports" / "log_summaries"


def default_output_dir() -> Path:
    return repo_root() / "reports" / "context_packets"


def read_text_safe(path: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        return path.read_text(encoding="utf-8"), None
    except FileNotFoundError:
        return None, f"Missing file: {path}"
    except OSError as exc:
        return None, f"Failed to read {path}: {exc}"


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root()))
    except ValueError:
        return str(path.resolve())


def shorten(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def sanitize_output_stem(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "context_packet"


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


def excerpt_text(text: str, max_items: int = 3) -> List[str]:
    items = list_items(text, max_items=max_items)
    if items:
        return [shorten(item) for item in items]
    excerpt: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        excerpt.append(shorten(line))
        if len(excerpt) >= max_items:
            break
    return excerpt


def extract_backticks(text: str) -> List[str]:
    return [match.group(1).strip() for match in BACKTICK_RE.finditer(text)]


def extract_task_path(current_state_text: str, current_state_path: Path) -> Optional[Path]:
    for token in extract_backticks(current_state_text):
        if token.startswith("project_memory/tasks/") and token.endswith(".md"):
            return (repo_root() / token).resolve()
    return current_state_path.parent / "tasks" / "TASK-UNKNOWN.md"


def resolve_repo_hint(token: str) -> Optional[Path]:
    if not ("/" in token or token.endswith((".py", ".md", ".json", ".toml", ".log", ".txt"))):
        return None
    token = token.strip()
    candidate = (repo_root() / token).resolve()
    if candidate.exists():
        return candidate
    return None


def find_matches_for_identifier(
    identifier: str, *, category: str = "General", max_matches: int = 4
) -> List[Path]:
    root = repo_root()
    matches: List[Path] = []
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", identifier).lower()
    for folder in SEARCH_ROOTS:
        search_root = root / folder
        if not search_root.exists():
            continue
        for path in search_root.rglob("*"):
            if not path.is_file() or path.suffix not in SEARCH_SUFFIXES:
                continue
            try:
                if identifier in path.read_text(encoding="utf-8", errors="ignore"):
                    matches.append(path.resolve())
            except OSError:
                continue
    unique_matches = list(dict.fromkeys(matches))

    preferred_subpaths = CATEGORY_PREFERRED_SUBPATHS.get(category, ())

    def score(path: Path) -> Tuple[int, int, str]:
        bonus = 0
        if path.stem == snake:
            bonus += 30
        elif snake in path.name.lower():
            bonus += 20
        path_text = relpath(path)
        if any(path_text.startswith(prefix) for prefix in preferred_subpaths):
            bonus += 25
        if path.name != "__init__.py":
            bonus += 10
        if path.suffix == ".py":
            bonus += 5
        return (bonus, -len(path.parts), path.name)

    unique_matches.sort(key=score, reverse=True)
    return unique_matches[:max_matches]


def infer_category(line: str, fallback: str = "General") -> str:
    lower = line.lower()
    for category, hints in MODULE_CATEGORY_HINTS:
        if any(hint.lower() in lower for hint in hints):
            return category
    return fallback


def build_module_map(task_text: str, architecture_text: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()

    def add_entry(category: str, symbol: str, path: Optional[Path], reason: str) -> None:
        path_text = relpath(path) if path else "not found"
        if category == "General" and any(
            existing["symbol"] == symbol and existing["path_text"] == path_text for existing in entries
        ):
            return
        key = (category, symbol, path_text)
        if key in seen:
            return
        seen.add(key)
        entries.append(
            {
                "category": category,
                "symbol": symbol,
                "path": path,
                "path_text": path_text,
                "reason": reason,
            }
        )

    for raw_line in architecture_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        category = infer_category(line)
        for token in extract_backticks(line):
            resolved_path = resolve_repo_hint(token)
            if resolved_path is not None:
                add_entry(category, token, resolved_path, "Referenced as a path in architecture notes.")
                continue
            if IDENTIFIER_RE.match(token):
                matches = find_matches_for_identifier(token, category=category)
                if matches:
                    add_entry(category, token, matches[0], "Resolved from architecture symbol search.")
                else:
                    add_entry(category, token, None, "Mentioned in architecture notes but no local file match was found.")

    for token in extract_backticks(task_text):
        resolved_path = resolve_repo_hint(token)
        if resolved_path is not None and resolved_path.suffix == ".py":
            add_entry("Config" if "configs/" in token else "Script", token, resolved_path, "Referenced by the current task.")

    return entries


def parse_read_first_paths(task_sections: Dict[str, str]) -> List[Path]:
    read_first = task_sections.get("Read first", "")
    paths: List[Path] = []
    for item in list_items(read_first, max_items=20):
        tokens = extract_backticks(item) or [item]
        for token in tokens:
            path = resolve_repo_hint(token)
            if path is not None and path not in paths:
                paths.append(path)
    return paths


def build_doc_entries(task_sections: Dict[str, str]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    doc_paths = parse_read_first_paths(task_sections)
    skip_paths = {
        (repo_root() / "AGENTS.md").resolve(),
        (repo_root() / "project_memory" / "current_state.md").resolve(),
    }
    doc_paths = [path for path in doc_paths if path.resolve() not in skip_paths]
    for extra in (
        repo_root() / "project_memory" / "05_active_stage.md",
        repo_root() / "project_memory" / "90_archived_decisions.md",
    ):
        if extra.exists() and extra not in doc_paths:
            doc_paths.append(extra)

    preferred_sections = {
        "01_current_architecture.md": ("Current Active Mainline: Axis-Side", "Shared Base", "Supporting Modules"),
        "02_loss_design.md": ("Current Active Mainline: Axis-Side", "Shared Semantic Term"),
        "04_training_rules.md": ("Training Rules",),
        "05_active_stage.md": ("Active Stage", "当前优先级"),
        "90_archived_decisions.md": ("Archived Decisions",),
    }

    for path in doc_paths:
        text, error = read_text_safe(path)
        excerpt: List[str]
        if text is None:
            excerpt = [error or f"Missing file: {path}"]
        else:
            sections = split_sections(text)
            excerpt = []
            for section_name in preferred_sections.get(path.name, ()):
                if sections.get(section_name):
                    excerpt = excerpt_text(sections[section_name], max_items=2)
                    break
            if not excerpt:
                excerpt = excerpt_text(text, max_items=2)
        entries.append({"path": path, "path_text": relpath(path), "excerpt": excerpt})
    return entries


def target_keywords(texts: Sequence[str]) -> List[str]:
    keywords: List[str] = []
    seen = set()
    for text in texts:
        for token in TOKEN_RE.findall(text):
            lower = token.lower()
            if lower in KEYWORD_STOPWORDS:
                continue
            if len(lower) < 4:
                continue
            if lower not in seen:
                seen.add(lower)
                keywords.append(lower)
    return keywords


def summary_candidates(summary_dir: Path) -> Dict[str, Dict[str, Path]]:
    pairs: Dict[str, Dict[str, Path]] = {}
    if not summary_dir.exists():
        return pairs
    for path in summary_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if name.endswith(SUMMARY_MD_SUFFIX):
            stem = name[: -len(SUMMARY_MD_SUFFIX)]
            pairs.setdefault(stem, {})["md"] = path.resolve()
        elif name.endswith(SUMMARY_JSON_SUFFIX):
            stem = name[: -len(SUMMARY_JSON_SUFFIX)]
            pairs.setdefault(stem, {})["json"] = path.resolve()
    return pairs


def pick_summary_pairs(
    *,
    task_text: str,
    current_state_text: str,
    summary_dir: Path,
    explicit_summary_paths: Sequence[Path],
    max_summaries: int,
) -> List[Dict[str, Optional[Path]]]:
    discovered = summary_candidates(summary_dir)
    for path in explicit_summary_paths:
        name = path.name
        if name.endswith(SUMMARY_MD_SUFFIX):
            stem = name[: -len(SUMMARY_MD_SUFFIX)]
            discovered.setdefault(stem, {})["md"] = path
        elif name.endswith(SUMMARY_JSON_SUFFIX):
            stem = name[: -len(SUMMARY_JSON_SUFFIX)]
            discovered.setdefault(stem, {})["json"] = path

    if not discovered:
        return []

    keywords = target_keywords([task_text, current_state_text])
    scored: List[Tuple[int, float, str]] = []
    for stem, files in discovered.items():
        score = sum(1 for keyword in keywords if keyword in stem.lower())
        newest_mtime = 0.0
        for candidate in files.values():
            if candidate is not None:
                try:
                    newest_mtime = max(newest_mtime, candidate.stat().st_mtime)
                except OSError:
                    pass
        scored.append((score, newest_mtime, stem))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    selected: List[Dict[str, Optional[Path]]] = []
    for _, _, stem in scored[: max_summaries]:
        files = discovered[stem]
        selected.append({"stem": stem, "md": files.get("md"), "json": files.get("json")})
    return selected


def read_summary_snapshot(summary_pair: Dict[str, Optional[Path]]) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "stem": summary_pair["stem"],
        "md": summary_pair.get("md"),
        "json": summary_pair.get("json"),
        "excerpt": [],
        "auto_questions": [],
        "notes": [],
    }
    json_path = summary_pair.get("json")
    if json_path is not None:
        text, error = read_text_safe(json_path)
        if text is None:
            snapshot["notes"].append(error)
            return snapshot
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            snapshot["notes"].append(f"Failed to parse summary json: {exc}")
            return snapshot
        basic = data.get("basic_info", {})
        sessions = data.get("sessions") or []
        last_session = sessions[-1] if sessions else {}
        last_val = (data.get("last_values") or {}).get("val", {})
        best_val = (data.get("best_values") or {}).get("val", {})
        if basic:
            source_path = data.get("input", {}).get("path", "unknown")
            source_rel = source_path
            try:
                source_rel = relpath(Path(source_path))
            except (TypeError, ValueError, OSError):
                source_rel = source_path
            snapshot["excerpt"].append(
                f"Source `{source_rel}`; runs `{basic.get('runs_detected')}`, "
                f"epoch range `{basic.get('epoch_range', {}).get('min')}` -> `{basic.get('epoch_range', {}).get('max')}`."
            )
        if last_session:
            snapshot["excerpt"].append(
                f"Latest session status `{last_session.get('status')}`, device `{last_session.get('device')}`, checkpoints `{len(last_session.get('checkpoints') or [])}`."
            )
        if "mIoU" in last_val:
            snapshot["excerpt"].append(f"Latest val `mIoU={last_val['mIoU']:.4f}`.")
        elif "val_mIoU" in (data.get("last_values") or {}).get("scalar", {}):
            snapshot["excerpt"].append(
                f"Latest scalar `val_mIoU={(data.get('last_values') or {}).get('scalar', {}).get('val_mIoU'):.4f}`."
            )
        if "mIoU" in best_val:
            snapshot["excerpt"].append(f"Best val `mIoU={best_val['mIoU']['value']:.4f}`.")
        elif "best_val_mIoU" in (data.get("best_values") or {}).get("scalar", {}):
            snapshot["excerpt"].append(
                f"Best scalar `best_val_mIoU={(data.get('best_values') or {}).get('scalar', {}).get('best_val_mIoU', {}).get('value'):.4f}`."
            )
        snapshot["auto_questions"] = (data.get("auto_questions") or [])[:2]
        return snapshot

    md_path = summary_pair.get("md")
    if md_path is not None:
        text, error = read_text_safe(md_path)
        if text is None:
            snapshot["notes"].append(error)
        else:
            snapshot["excerpt"] = excerpt_text(text, max_items=4)
    return snapshot


def snapshot_current_state(current_state_text: str) -> List[str]:
    sections = split_sections(current_state_text)
    snapshot: List[str] = []
    snapshot.extend(f"Current task: {item}" for item in excerpt_text(sections.get("当前 task", ""), max_items=1))
    snapshot.extend(excerpt_text(sections.get("当前有效事实", ""), max_items=4))
    snapshot.extend(excerpt_text(sections.get("已确认结果", ""), max_items=3))
    snapshot.extend(excerpt_text(sections.get("当前 workspace 状态", ""), max_items=2))
    snapshot.extend(excerpt_text(sections.get("下一步", ""), max_items=2))
    return snapshot


def snapshot_task(task_text: str) -> List[str]:
    sections = split_sections(task_text)
    snapshot: List[str] = []
    for section_name, limit in (
        ("Goal", 1),
        ("Why now", 2),
        ("In scope", 2),
        ("Constraints", 2),
        ("Validation", 2),
        ("Next step", 2),
    ):
        if sections.get(section_name):
            excerpt = excerpt_text(sections[section_name], max_items=limit)
            snapshot.extend(f"{section_name}: {item}" for item in excerpt)
    return snapshot


def build_packet_markdown(
    *,
    target: str,
    current_state_path: Path,
    current_state_text: str,
    task_path: Path,
    task_text: str,
    doc_entries: Sequence[Dict[str, Any]],
    module_entries: Sequence[Dict[str, Any]],
    evidence_entries: Sequence[Dict[str, Any]],
) -> str:
    target_meta = TARGET_NOTES[target]
    task_name = task_path.stem if task_path else "unknown_task"

    lines: List[str] = [
        f"# Context Packet: {task_name} / {target}",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')}`",
        f"- Target: `{target}`",
        f"- Current state source: `{relpath(current_state_path)}`",
        f"- Task source: `{relpath(task_path)}`",
        "",
        "## Purpose",
        "",
        f"- {target_meta['purpose']}",
        "- This packet is excerpt-oriented. Canonical facts still live in `AGENTS.md`, `project_memory/current_state.md`, the current task file, and linked summaries.",
        "",
        "## Read Order",
        "",
    ]
    for index, item in enumerate(target_meta["read_order"], start=1):
        lines.append(f"{index}. {item}")

    lines.extend(["", "## Current State Snapshot", ""])
    for item in snapshot_current_state(current_state_text):
        lines.append(f"- {item}")

    lines.extend(["", "## Current Task Snapshot", ""])
    for item in snapshot_task(task_text):
        lines.append(f"- {item}")

    lines.extend(["", "## Related Files", "", "### Core Docs", ""])
    lines.append(f"- `AGENTS.md` -> `AGENTS.md`")
    lines.append(f"- `Current state` -> `{relpath(current_state_path)}`")
    lines.append(f"- `Current task` -> `{relpath(task_path)}`")
    for entry in doc_entries:
        excerpt = " | ".join(entry["excerpt"]) if entry["excerpt"] else "No excerpt available."
        lines.append(f"- `{entry['path_text']}` -> {excerpt}")

    if module_entries:
        lines.extend(["", "### Module Map", ""])
        for entry in module_entries:
            if entry["path"] is not None:
                lines.append(
                    f"- `{entry['category']}`: `{entry['symbol']}` -> `{entry['path_text']}` ({shorten(entry['reason'], 120)})"
                )
            else:
                lines.append(
                    f"- `{entry['category']}`: `{entry['symbol']}` -> not found ({shorten(entry['reason'], 120)})"
                )

    lines.extend(["", "## Training Evidence", ""])
    if evidence_entries:
        for evidence in evidence_entries:
            md_path = evidence.get("md")
            json_path = evidence.get("json")
            evidence_paths = []
            if md_path is not None:
                evidence_paths.append(f"`{relpath(md_path)}`")
            if json_path is not None:
                evidence_paths.append(f"`{relpath(json_path)}`")
            path_hint = ", ".join(evidence_paths) if evidence_paths else "`missing summary files`"
            lines.append(f"- Summary `{evidence['stem']}` -> {path_hint}")
            for excerpt in evidence.get("excerpt", [])[:4]:
                lines.append(f"  - {excerpt}")
            for question in evidence.get("auto_questions", [])[:2]:
                lines.append(f"  - Auto question: {question}")
            for note in evidence.get("notes", [])[:2]:
                lines.append(f"  - Note: {note}")
    else:
        lines.append("- No log summary artifacts were found under `reports/log_summaries/`.")
        lines.append(
            "- Generate them first with `scripts/agent/summarize_train_log.py`, then rebuild this packet so Training Evidence can point to summaries instead of raw logs."
        )

    lines.extend(["", "## Notes For This Target", ""])
    for note in target_meta["notes"]:
        lines.append(f"- {note}")

    return "\n".join(lines).rstrip() + "\n"


def generate_packet(
    *,
    target: str,
    current_state_path: Path,
    task_path: Path,
    summary_dir: Path,
    explicit_summary_paths: Sequence[Path],
    output_dir: Path,
    max_summaries: int,
    output_stem: Optional[str],
) -> Path:
    current_state_text, current_state_error = read_text_safe(current_state_path)
    if current_state_text is None:
        raise FileNotFoundError(current_state_error or f"Missing current state file: {current_state_path}")

    task_text, task_error = read_text_safe(task_path)
    if task_text is None:
        task_text = f"# Missing Task\n\n- {task_error or f'Missing task file: {task_path}'}\n"

    architecture_path = repo_root() / "project_memory" / "01_current_architecture.md"
    architecture_text, _ = read_text_safe(architecture_path)
    architecture_text = architecture_text or ""

    task_sections = split_sections(task_text)
    doc_entries = build_doc_entries(task_sections)
    module_entries = build_module_map(task_text, architecture_text)

    evidence_pairs = pick_summary_pairs(
        task_text=task_text,
        current_state_text=current_state_text,
        summary_dir=summary_dir,
        explicit_summary_paths=explicit_summary_paths,
        max_summaries=max_summaries,
    )
    evidence_entries = [read_summary_snapshot(pair) for pair in evidence_pairs]

    markdown = build_packet_markdown(
        target=target,
        current_state_path=current_state_path,
        current_state_text=current_state_text,
        task_path=task_path,
        task_text=task_text,
        doc_entries=doc_entries,
        module_entries=module_entries,
        evidence_entries=evidence_entries,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem or sanitize_output_stem(f"{task_path.stem}.{target}.context_packet")
    output_path = output_dir / f"{stem}.md"
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def main() -> int:
    args = parse_args()
    current_state_path = Path(args.current_state).resolve()
    current_state_text, current_state_error = read_text_safe(current_state_path)
    if current_state_text is None:
        print(current_state_error, file=sys.stderr)
        return 1

    if args.task:
        task_path = Path(args.task).resolve()
    else:
        inferred = extract_task_path(current_state_text, current_state_path)
        task_path = inferred.resolve() if inferred is not None else (current_state_path.parent / "tasks" / "TASK-UNKNOWN.md")

    explicit_summary_paths = [Path(item).resolve() for item in args.summary]
    output_dir = Path(args.output_dir).resolve()
    summary_dir = Path(args.summary_dir).resolve()

    targets = ["web_chat", "claude", "codex"] if args.target == "all" else [args.target]
    if args.name and len(targets) != 1:
        print("--name can only be used when generating a single target packet.", file=sys.stderr)
        return 2

    exit_code = 0
    for target in targets:
        try:
            output_path = generate_packet(
                target=target,
                current_state_path=current_state_path,
                task_path=task_path,
                summary_dir=summary_dir,
                explicit_summary_paths=explicit_summary_paths,
                output_dir=output_dir,
                max_summaries=max(args.max_summaries, 1),
                output_stem=args.name if len(targets) == 1 else None,
            )
            print(f"Wrote {output_path}")
        except Exception as exc:  # pragma: no cover - last-resort guard
            exit_code = 1
            print(f"Failed to build packet for {target}: {exc}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
