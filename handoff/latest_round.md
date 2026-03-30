# Latest Round

## Summary

- 本轮围绕 `project_memory/tasks/TASK-2026-03-30-001.md` 收尾，`axis-side` smoke 已在摘要层形成一次有效核证。
- 当前 canonical 文档已同步到“smoke 已核证，但 `axis-side` full train 尚未开始”的状态。

## Read First

- `AGENTS.md`
- `project_memory/current_state.md`
- `project_memory/tasks/TASK-2026-03-30-001.md`
- `reports/context_packets/TASK-2026-03-30-001.web_chat.context_packet.md`
- `reports/context_packets/TASK-2026-03-30-001.claude.context_packet.md`
- `reports/context_packets/TASK-2026-03-30-001.codex.context_packet.md`
- `reports/log_summaries/semantic_boundary_axis_side_train_smoke_train.summary.md`

## Evidence

- Latest summary status: `validated_with_checkpoints`.
- Device: `cuda`.
- Checkpoints observed: `4`.
- Latest / best val `mIoU=0.0378`.
- Follow-up question: why were there `4` startup attempts in the same log, and should the low `val_mIoU` be treated as expected smoke behavior or as a config / data / label mismatch?

## Next Window

- 为 `axis-side` full train 新建下一轮 task。
- 继续优先使用 `summary -> context packet -> round update draft`，原始长日志只按需下钻。
- 在 full train 前先解释当前 smoke 的低 `val_mIoU` 与多次启动记录。
