> Thin wrapper only. Default repository entry is now GSD plus local `.planning/` artifacts. Use this file only when a new window or a web handoff needs a compact redirect surface.

# Chat Entry

Start in this order:

1. `GSD`
2. `.planning/PROJECT.md`
3. `.planning/ROADMAP.md`
4. `.planning/STATE.md`
5. `docs/canonical/README.md`

Only after that, use compatibility surfaces narrowly if the task explicitly needs them:

- `handoff/web_to_agent_contract.md` for structured web-to-local handoff
- `project_memory/current_state.md` and the current `TASK-*.md` only when resuming a pre-migration continuity thread
- `handoff/handoff_for_chat.md` only when an old thread cites it directly
