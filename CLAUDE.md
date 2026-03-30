# CLAUDE.md

本文件只做 Claude 兼容入口，规范以 `AGENTS.md` 为准。

默认启动只读：
1. `AGENTS.md`
2. `project_memory/current_state.md`
3. `project_memory/current_state.md` 中指向的当前 task 文件
4. 仅在新窗口接手时，再读 `handoff/chat_entry.md`

新一轮开始前，优先生成或读取 `reports/context_packets/*.md` 中对应 `claude` target 的 packet；若 packet 过期，重新生成，不手工拼长上下文。

默认不要读：
- 完整 `handoff/`
- 完整 `project_memory/`
- 未先看摘要的 `outputs/` 原始长日志
- 全部 `.codex/skills/*/SKILL.md`

如需查看训练日志，先读 `reports/log_summaries/*.summary.md` 或 `*.summary.json`，只有摘要不足时再下钻原始日志。

一轮结束时或一轮证据刷新后，优先运行 `scripts/agent/refresh_round_artifacts.py`；它会顺序刷新 summary、context packet 和 round update，避免手工分步执行造成 stale packet / stale draft / stale preview。

重要轮次收尾、canonical 回写前，或 Claude / web / Codex 切换前，可先运行 `scripts/agent/workflow_consistency_smoke.py`，先看当前 task 链的 missing / stale / conflict，再决定是否需要先 refresh 或 apply。

如确需 Claude 侧旧角色说明，再按需查看 `CLAUDE_AGENTS.md`；它不是默认启动集合。
