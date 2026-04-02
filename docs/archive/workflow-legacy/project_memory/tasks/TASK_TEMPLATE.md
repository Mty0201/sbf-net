# TASK_TEMPLATE

> 只写本轮 task 特有的目标、边界、验证和 closeout。不要把 `AGENTS.md` 或 `docs/workflows/sbf_net_workflow_v1.md` 的通用流程规则整段复制进每个 task。

## Task type

- `analysis` / `implementation` / `hybrid`
- 说明为什么这一轮属于该类型。

## Goal

- 这一轮要达成的单一目标。

## Why now

- 为什么这个任务是当前阶段的直接优先级。

## In scope

- 本轮明确要做的内容。

## Out of scope

- 本轮明确不做的内容。

## Constraints

- 必须遵守的边界、禁改项和环境约束。

## Read first

- 开始执行前只需要先读的最小文件集合。
- 优先列 canonical 文件和必要 artifact 路径，不要把完整 handoff / memory 当默认前置。

## Implementation plan

1. 列出最小执行步骤。
2. 只写本轮真的要做的步骤。

## Checkpoints

- 这里只写 task 内的中间检查点。
- 例如：`summary`、`packet`、`workflow smoke`、`round update draft`、`refresh / preview / apply`。
- 这些动作默认只代表当前 task 有推进，不代表 task 结束。
- 尽量用路径指针引用 checkpoint 产物，不要把大段 artifact 内容重新抄回 task。

## Validation

- 本轮如何判断完成，证据应来自什么。

## Done condition

- 只有满足这些条件，当前 task 才允许 closeout。

## Review / acceptance plan

- 本轮由谁 review、按什么标准验收、需要什么证据。

## Deliverables

- 本轮结束后应该产出的文件、日志或结论。

## Closeout expectation

- 达成 done condition 后，需要同步哪些 canonical / handoff / artifact。
- 若只完成 checkpoint 而未完成 closeout，也要明确写出。

## Result

- 填写当前结果、状态和关键证据。

## Next step

- 本轮结束后下一步应该衔接什么。
- 只有当前 task 完成，或核心问题本质变化时，才默认新建下一 task。
