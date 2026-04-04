---
name: workflow-consistency-smoke
description: Inspect the current sbf-net task chain for missing, stale, or conflicting canonical and derived artifacts. Use before handoff, before apply, or when packet/draft alignment is uncertain.
---

# Workflow Consistency Smoke

## When To Use

- 重要轮次收尾前，想确认当前 task 链是否已经闭环。
- 准备 `apply` 前，先检查 canonical、summary、packet、round draft 是否一致。
- agent / target 切换前，需要知道当前缺什么、哪些 stale、哪里冲突。

## Read First

1. `AGENTS.md`
2. `project_memory/current_state.md`
3. 当前 task 文件

## How To Use

1. 检查当前 Codex 视角链路：
   `python scripts/agent/workflow_consistency_smoke.py --target codex`
2. 若要一次检查 `web_chat / claude / codex` 三侧派生产物，使用：
   `--target all`
3. 打开生成的 `reports/workflow_smokes/*.md`，先看 `Overall Verdict`，再看各层问题和 `Suggested Fixes`

## Outputs

- `reports/workflow_smokes/*.workflow_consistency_smoke.md`
- 结构化层级结果：
  - `Canonical Layer`
  - `Task Layer`
  - `Summary Layer`
  - `Packet Layer`
  - `Round Update Layer`
  - `Overall Verdict`
  - `Suggested Fixes`
- verdict 语义：
  - `PASS`：当前检查范围内未发现 missing / stale / conflict
  - `WARN`：存在缺失或 stale，但未出现关键冲突
  - `FAIL`：关键 canonical 缺失、当前 task 无法解析、summary 层不可用，或出现明确 conflict

## Do Not

- 不把它当成修复脚本；它只负责检查，不负责刷新或回写
- 不把单个 target 的 `PASS` 误当成全部 target 都已齐全
- 不把历史多 task 审计扩展进来；当前只检查当前 task 主链
