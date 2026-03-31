# Web to Agent Contract

## 1. Purpose

- 规范网页端 ChatGPT 如何把分析结果交给本地 Codex / Claude。
- 避免只输出长篇自然语言建议，优先落成结构化交付物。
- 让本地 agent 能直接继续讨论、压缩 task，或按最小启动链开工。
- 本文件只定义 web -> agent 的交付格式；task lifecycle、checkpoint 和 closeout 规则以 `AGENTS.md` 与 `docs/workflows/sbf_net_workflow_v1.md` 为准。
- 一次分析默认只交付一个主输出类型；若确需补充，只追加最小说明，不复制整包 handoff / memory。
- 这些交付物默认是当前 task 内的讨论 / brief / 实现输入 checkpoint，不自动代表 task 完成，也不默认触发新 task。

## 2. Output Types

### A. Discussion Handoff

适用于：问题仍在讨论、事实与边界尚未完全收敛、本轮不应直接开工。

必须包含以下字段：

- `Confirmed facts`
- `Open questions`
- `Recommended direction`
- `Do not do`
- `Suggested next action`

要求：

- `Confirmed facts` 只写已被当前仓库事实、已有 artifact 或作者明确裁定支持的内容。
- `Open questions` 只写真正阻止收口的问题，不混入实现细节。
- `Recommended direction` 只给下一步判断，不伪装成可立即执行的实现 prompt。

### B. Task Brief Draft

适用于：已经准备收口成新 task，但仍需要本地 agent 检查边界、压缩范围或映射到仓库模板。

必须包含以下字段：

- `Task type`
- `Goal`
- `Why now`
- `In scope`
- `Out of scope`
- `Constraints`
- `Read first`
- `Checkpoints`
- `Done condition`
- `Review / acceptance plan`
- `Validation`
- `Deliverables`
- `Closeout expectation`

要求：

- 内容应能自然映射到 `project_memory/tasks/TASK_TEMPLATE.md`。
- `Read first` 只列最小文件集合，不要求重读整仓。
- `Checkpoints` 只写本轮内部检查点，不把 artifact 刷新写成 task 完成。
- `Done condition` 必须描述 task 什么时候允许 closeout，而不是一次 checkpoint 何时完成。
- `Deliverables` 只写本轮应产出的文件、结论或验证结果，不预填尚未发生的完成状态。
- `Closeout expectation` 只写达成 done condition 后需要同步什么，不把 summary / packet / smoke 本身写成 task 完成。

### C. Agent Prompt

适用于：边界已明确、依赖已知、可以直接发给本地 Codex / Claude 开工。

必须包含以下字段：

- `Read only`
- `Goal`
- `Boundary`
- `Do not do`
- `Validation`
- `Refresh artifacts`

要求：

- `Read only` 只列本轮启动所需最小文件，不把整仓扫描当默认前置。
- `Boundary` 需要明确本轮只做什么、不扩展到哪里。
- `Refresh artifacts` 只列本轮结束后需要刷新的 artifact；如果无需刷新，明确写 `None`。

## 3. Routing Rules

- 还没到实现阶段时，优先输出 `Discussion Handoff` 或 `Task Brief Draft`。
- 已经有明确边界、最小读取集合和验证方式时，才输出 `Agent Prompt`。
- 分析未收敛时，不允许直接跳到实现 prompt。
- `summary`、`packet`、`workflow smoke`、`round update draft` 的刷新默认仍属于当前 task 内部推进，不因为 artifact 已刷新就自动切到新 task。
- 若结论只是“建议继续收集信息”或“需要先确认边界”，默认输出 `Discussion Handoff`。
- 若结论已经接近 task 级边界，但本地 agent 还需要压缩范围或映射到 `TASK-*.md`，默认输出 `Task Brief Draft`。

## 4. How Codex / Claude Should Consume It

- 默认仍以 `AGENTS.md`、`docs/workflows/sbf_net_workflow_v1.md`、`project_memory/current_state.md` 和当前 task 为准；本合同不单独改写 task lifecycle 或 closeout 规则。
- `Discussion Handoff`：只做仓库内一致性审阅、补证据或继续讨论，不直接改代码。
- `Task Brief Draft`：先检查是否符合 `AGENTS.md`、当前 stage 边界和仓库禁改项；必要时压缩边界，再落地成 `TASK-*.md`。
- `Agent Prompt`：按最小启动链执行，只读取列出的文件与其直接依赖，不重新整仓接手。
- 若当前 task 的 done condition 未达成，默认继续留在同一 task 内推进 discussion / implementation / review / closeout，不因为单次 artifact 动作而封盘。
- 若结构化产物里的字段与仓库事实冲突，以作者裁定、`AGENTS.md`、`project_memory/current_state.md`、当前 task 和本地实现为准，再最小回写修正。

## 5. Guardrails

- 不要要求网页端 ChatGPT 重新读取整仓。
- 不要复制完整 handoff、完整 memory 或完整原始长日志。
- 不要在没有 task 边界时直接输出实现 prompt。
- 不要把未证实假设写成 `Confirmed facts`。
- 不要把“建议方向”写成“已经确认的实现决议”。
- 不要在 `Read first` 或 `Read only` 中塞入与本轮无关的大量文件。
- 不要让本地 agent 先做整仓总结，再开始处理当前交付物。
- 不要把本合同扩写成第二份 workflow 文档；新的长期规则应写回 `AGENTS.md` 或 `docs/workflows/sbf_net_workflow_v1.md`。

## 6. Reusable Templates

### Discussion Handoff Template

```md
# Discussion Handoff

## Confirmed facts
- ...

## Open questions
- ...

## Recommended direction
- ...

## Do not do
- ...

## Suggested next action
- ...
```

### Task Brief Draft Template

```md
# Task Brief Draft

## Goal
- ...

## Task type
- ...

## Why now
- ...

## In scope
- ...

## Out of scope
- ...

## Constraints
- ...

## Read first
- ...

## Checkpoints
- ...

## Done condition
- ...

## Review / acceptance plan
- ...

## Validation
- ...

## Deliverables
- ...

## Closeout expectation
- ...
```

### Agent Prompt Template

```md
# Agent Prompt

## Read only
- ...

## Goal
- ...

## Boundary
- ...

## Do not do
- ...

## Validation
- ...

## Refresh artifacts
- ...
```
