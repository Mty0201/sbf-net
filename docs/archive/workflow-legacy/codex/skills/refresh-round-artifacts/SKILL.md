---
name: refresh-round-artifacts
description: Refresh log summaries, context packets, and round update drafts for semantic-boundary-field through the shared single-entry chain. Use when evidence changes, a round is ending, or before preview/apply to avoid stale artifacts.
---

# Refresh Round Artifacts

## When To Use

- 一轮证据刚更新，需要顺序刷新 `summary -> packet -> round draft`。
- 准备执行 `preview` 或 `apply`，但不想手工分步调用多个脚本。
- 发现 packet、round draft 或 preview 可能 stale，需要用单入口重新对齐派生产物。

## Read First

1. `AGENTS.md`
2. `project_memory/current_state.md`
3. 当前 task 文件
4. 只有在本轮有新日志时，再确认对应 `train.log` 路径

## How To Use

1. 默认从当前 task 出发，优先调用：
   `python scripts/agent/refresh_round_artifacts.py --mode draft --target codex`
2. 如果本轮有新的训练日志，在同一条命令上补：
   `--log <path/to/train.log>`
3. 需要检查 fixed-scope diff 时，使用：
   `--mode preview`
4. 只有在 preview 已看过、并且用户明确要落盘时，才使用：
   `--mode apply --confirm`

## Outputs

- `reports/log_summaries/*.summary.md`
- `reports/log_summaries/*.summary.json`
- `reports/context_packets/*.context_packet.md`
- `reports/round_updates/*.round_update.draft.md`
- 一份终端摘要，标明每一步是 `EXECUTED`、`SKIPPED`、`FAILED` 还是 `NO-OP`

## Do Not

- 不手工跳过顺序直接调 `preview/apply`，除非是在专门调试 `update_round_artifacts.py`
- 不把它当成业务代码修改入口
- 不在缺少 `--confirm` 的情况下执行 `apply`
- 不在摘要已经足够时回退到先读原始长日志
