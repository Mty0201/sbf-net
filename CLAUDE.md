# CLAUDE.md

本文件是 Claude 的 compatibility-only 薄入口。默认规则以 `AGENTS.md` 和 `docs/workflows/sbf_net_workflow_v1.md` 为准；不要在这里维护第二套完整 workflow 说明。

默认启动只读：
1. `AGENTS.md`
2. `project_memory/current_state.md`
3. `project_memory/current_state.md` 中指向的当前 task 文件
4. 仅在新窗口接手时，再读 `handoff/chat_entry.md`
5. 仅在产出或消费 web ChatGPT 结构化交付物时，再读 `handoff/web_to_agent_contract.md`

默认优先：
- 读取或生成对应 `claude` target 的 `reports/context_packets/*.context_packet.md`
- 先读 `reports/log_summaries/*.summary.md` 或 `*.summary.json`，再决定是否下钻原始长日志
- 用 `scripts/agent/refresh_round_artifacts.py` 刷新 checkpoint 产物
- 在 handoff / apply / 重要切换前，用 `scripts/agent/workflow_consistency_smoke.py` 做一致性巡检

默认不要读：
- 完整 `handoff/`
- 完整 `project_memory/`
- 未先看摘要的 `outputs/` 原始长日志
- 全部 `.codex/skills/*/SKILL.md` 或全部 `claude/skills/*.md`
- `CLAUDE_AGENTS.md`，除非明确需要查旧角色术语；它是 legacy reference only
