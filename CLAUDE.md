# CLAUDE.md

This file is a compatibility-only thin wrapper for Claude-facing entry points. Do not maintain a second workflow here.

## Default Entry

For active repository work, start with:

1. `GSD` commands such as `$gsd-progress`, `$gsd-plan-phase N`, and `$gsd-execute-phase N`
2. `.planning/PROJECT.md`
3. `.planning/ROADMAP.md`
4. `.planning/STATE.md`
5. `docs/canonical/README.md`

## Compatibility Notes

- Use `handoff/chat_entry.md` only when a new window or web handoff needs a thin redirect surface
- Use `handoff/web_to_agent_contract.md` only when exchanging a structured web-to-local handoff
- Use `CLAUDE_AGENTS.md` only if an old thread explicitly references historical Claude role terminology

Everything else should follow the GSD-first path above.
