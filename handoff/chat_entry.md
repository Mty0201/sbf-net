> 本文件是网页端 ChatGPT / 新窗口的薄入口。正式 workflow 规则以 `AGENTS.md` 和 `docs/workflows/sbf_net_workflow_v1.md` 为准；当前事实以 `project_memory/current_state.md` 和当前 task 文件为准。

# Chat Entry

- 新窗口接手时，优先读取对应 target 的 `reports/context_packets/*.context_packet.md`；若 packet 缺失，再回到 `AGENTS.md`、`project_memory/current_state.md` 和当前 task 文件。
- 若需要训练证据，先读 `reports/log_summaries/*.summary.md` 或 `*.summary.json`；原始长日志只按需下钻。
- 若网页端需要把结果交给本地 Codex / Claude，优先按 `handoff/web_to_agent_contract.md` 输出单一结构化交付物，而不是长篇自由文本。
- `summary`、`packet`、`workflow smoke`、`round update draft` 以及 `refresh / preview / apply` 都只是 task 内 checkpoint，不默认代表 task 结束。
- `handoff/handoff_for_chat.md` 仅作 legacy reference，不是默认启动入口。
