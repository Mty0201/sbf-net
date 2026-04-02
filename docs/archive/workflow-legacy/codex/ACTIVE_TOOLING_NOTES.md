# Active Tooling Notes

Phase 3 archived the repo-local manual orchestration layer and kept only the minimum Codex-side assets still required for the active GSD workflow.

## Retained Active Paths

- `.codex/get-shit-done/` - GSD runtime implementation and workflow assets
- `.codex/config.toml` - local GSD runtime configuration
- `.codex/gsd-file-manifest.json` - GSD-managed file manifest
- `.codex/agents/gsd-*.md` and `.codex/agents/gsd-*.toml` - active GSD agent definitions
- `.codex/skills/gsd-*/` - active GSD skill set used by the GSD-first workflow
- `.codex/agents/README.md` and `.codex/skills/README.md` - runtime-only labels for the retained active directories

## Archived Legacy Paths

- `docs/archive/workflow-legacy/codex/agents/architect.toml`
- `docs/archive/workflow-legacy/codex/agents/worker.toml`
- `docs/archive/workflow-legacy/codex/agents/maintainer.toml`
- `docs/archive/workflow-legacy/codex/skills/prepare-task-brief/`
- `docs/archive/workflow-legacy/codex/skills/refresh-round-artifacts/`
- `docs/archive/workflow-legacy/codex/skills/update-handoff-memory/`
- `docs/archive/workflow-legacy/codex/skills/workflow-consistency-smoke/`
- `docs/archive/workflow-legacy/codex/scripts-agent/`
- `docs/archive/workflow-legacy/codex/claude/`

No other `.codex/agents/` or `.codex/skills/` paths are intended to act as a repo-specific manual workflow layer.
