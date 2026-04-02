# Latest Round

## Summary

- 本轮执行中心：`project_memory/tasks/TASK-2026-03-31-003.md`（已完成）。
- 修正了验证口径与产物绑定：`build_context_packet.py` 和 `summarize_train_log.py` 现在明确区分 step-level `val.mIoU` 和 epoch-aggregated scalar `val_mIoU`。
- summary / packet / smoke 链路已对齐，workflow smoke 达到 `PASS`。

## Read First

- `AGENTS.md`
- `project_memory/current_state.md`
- `project_memory/tasks/TASK-2026-03-31-003.md`
- `reports/context_packets/TASK-2026-03-31-003.claude.context_packet.md`
- `reports/log_summaries/semseg-pt-v3m1-0-base-bf-edge-support-shape-train_train.summary.md`

## Evidence

- Latest summary status: `validated_with_checkpoints`.
- Device: `cuda`.
- Checkpoints observed: `314`.
- 权威指标（epoch-aggregated scalar）：latest `val_mIoU = 0.7085`，best `val_mIoU = 0.7316`（epoch 65）。
- step-level val `mIoU` 最高 0.9423（单 batch），不作为权威口径。
- workflow smoke verdict: `PASS`（0 missing, 0 stale, 0 conflict）。

## Next Window

- 新建 `TASK-004`，在干净口径下分析 full-train `val_mIoU = 0.7316` 低于 baseline（74.6）的原因。
- 继续优先使用 `summary -> context packet -> round update draft`，原始长日志只按需下钻。
