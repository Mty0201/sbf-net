---
name: prepare-task-brief
description: Generate a minimal executable task brief for semantic-boundary-field before implementation. Use when an architect needs to convert current facts into a bounded worker brief without reopening the research mainline.
---

# Prepare Task Brief

## When To Use

- 需要在实现前先把任务压缩成单一、可执行、可验证的 brief。
- 需要先对齐“`2.5` 已结束、当前准备进入 `Stage-2`”的阶段边界，再交给 worker。
- 需要把用户要求映射到现有文件名、现有目录和最小影响文件集合。

## Required Inputs

1. `handoff/handoff_for_chat.md`
2. 与任务直接相关的 `project_memory/*`
3. 当前用户请求

## Workflow

1. 先确认 `2.5` 阶段已结束，当前处于 `Stage-2 entry preparation phase`。
2. 先确认唯一允许主动维护的项目根是 `semantic-boundary-field`；`Pointcept` 仅作为宿主依赖存在，默认只读。
3. 若问题疑似来自 `Pointcept`、宿主接口、依赖异常或边界语义冲突，停止继续下发实现任务，并在 brief 中写清停止原因与待分析问题。
4. 提炼当前有效事实、禁改项和允许改动项。
5. 若用户给出的文件名与仓库现状不一致，先做路径映射并写进 brief。
6. 把任务限制为一个最小可交付目标；若工作较大，也要拆成当前轮最小落地点。
7. 只输出 worker 需要的执行信息，不展开新的研究方案。

## Brief Template

- `任务名`
- `目标`
- `当前事实`
- `允许改动`
- `禁止改动`
- `最小影响文件`
- `验证方式`
- `停止条件`
- `需人工确认项`

## Guardrails

- 不把 brief 写成架构大设计。
- 不把 `handoff` 当成事实源替代 `project_memory`。
- 不因为“准备进入 Stage-2”而放松对主训练入口、主配置、主模型职责的保护。
- 不把 `Pointcept` 或 `semantic-boundary-field` 目录之外的路径写成常规可维护范围。
- 当前原型阶段禁止为了跑通、通过率或兼容性擅自加入 fallback、默认兼容层、吞错或自动绕过。
