> 新窗口 / 网页端接手时，优先读取对应 target 的 `reports/context_packets/*.md`。若 packet 不存在，再回到 `AGENTS.md`、`project_memory/current_state.md` 和当前 task 文件。若需要训练证据，先读 `reports/log_summaries/*.summary.md` 或 `*.summary.json`。不要默认展开完整 `handoff/`、完整 `project_memory/`、原始长日志或全部 `.codex/skills/`。

# Chat Entry

- 项目主体是 `semantic-boundary-field`；`Pointcept` 只是宿主依赖，默认只读。
- 当前阶段是 `Stage-2 architecture rollout / verification phase`。
- 当前 active 主线是 `axis + side + support`；作者口头中的 `magnitude` 在当前同步中等价于 `support`。
- 当前验证中心是 `semseg-pt-v3m1-0-base-bf-edge-axis-side-train` 及其 smoke config。
- 当前已确认实验事实：`semantic-only=73.8`，`support-only(reg=1, cover=0.2)=74.6`，`support + dir + dist = 71`。
- 当前 workspace 状态：`axis-side` 实现已落地，但 smoke 尚未确认通过。
- 下一步优先级：先在 CUDA-enabled `ptv3` 环境确认 `axis-side` smoke。
