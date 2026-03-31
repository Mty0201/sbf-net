# DEVELOPMENT_WORKFLOW

> Legacy reference only. 当前项目的正式 workflow 规则以 `docs/workflows/sbf_net_workflow_v1.md` 为准；全局启动边界以 `AGENTS.md` 为准。本文件保留给旧链接和旧讨论窗口做兼容跳转，不再维护第二套完整流程说明。

## Current Mapping

- 全局规则与最小启动链：`AGENTS.md`
- 正式 task lifecycle / 角色分工 / closeout 规则：`docs/workflows/sbf_net_workflow_v1.md`
- 当前有效事实与 active task：`project_memory/current_state.md` + 当前 `TASK-*.md`
- 最近一轮已同步 closeout：`handoff/latest_round.md`

## Layer Map

- Canonical：`AGENTS.md`、`docs/workflows/sbf_net_workflow_v1.md`、`project_memory/current_state.md`、当前 `TASK-*.md`、`handoff/latest_round.md`
- Checkpoint / generated：`reports/log_summaries/`、`reports/context_packets/`、`reports/round_updates/`、`reports/workflow_smokes/`
- Thin wrappers：`CLAUDE.md`、`handoff/chat_entry.md`
- Compatibility / legacy reference：`docs/workflow.md`（本文件）、`CLAUDE_AGENTS.md`、`handoff/handoff_for_chat.md`

## Quick Migration

- 一个 task 默认覆盖一轮完整闭环，而不是只覆盖一次 artifact 动作。
- `summary / packet / workflow smoke / round update draft` 都只是 task 内 checkpoint，不默认代表 task 完成。
- 只有当前 task 的 `Done condition` 达成，或核心问题本质变化，才默认新建下一 task。
- 新的长期规则优先写入 `AGENTS.md` 或 `docs/workflows/sbf_net_workflow_v1.md`，不要再回写到本文件。
