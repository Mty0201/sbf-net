---
name: prepare-task-brief
description: Draft the next minimal task file for sbf-net from TASK_TEMPLATE.md. Use when a new round needs a bounded task proposal instead of a long handoff replay.
---

# Prepare Task Brief

## When To Use

- 需要为下一轮工作新建一个最小 task 草案。
- 当前 task 已收尾，想把下一步压缩成单轮、可执行、可验证的任务书。
- 用户需求还比较宽，需要先映射到 `TASK_TEMPLATE.md` 的固定结构，再决定是否真正落盘。

## Required Inputs

1. `AGENTS.md`
2. `project_memory/current_state.md`
3. `project_memory/tasks/TASK_TEMPLATE.md`
4. 若是承接上一轮，再读当前 task 文件
5. 仅当任务直接依赖证据时，再补读一个最相关的 summary / packet / 专题 memory

## Workflow

1. 从 `AGENTS.md` 和 `current_state.md` 确认当前阶段边界、当前 task 指针和禁改项。
2. 判断这是“新 task 草案”还是“当前 task 仍应继续”；若当前 task 还没收尾且用户没要求拆新轮，先停下来说明。
3. 只提炼当前轮真正需要的稳定事实，避免复制完整 handoff 或完整 memory。
4. 基于日期和目录现状，建议新的任务文件名：`project_memory/tasks/TASK-YYYY-MM-DD-XXX.md`。
5. 按 `TASK_TEMPLATE.md` 逐节生成草案，至少填好：
   - `Task type`
   - `Goal`
   - `Why now`
   - `In scope`
   - `Out of scope`
   - `Constraints`
   - `Read first`
   - `Implementation plan`
   - `Checkpoints`
   - `Done condition`
   - `Review / acceptance plan`
   - `Validation`
   - `Deliverables`
   - `Closeout expectation`
   - `Result`
   - `Next step`
6. `Result` 只写当前已知状态，例如“待开始”或“等待验证”，不要凭空补结果。
7. 输出草案时同时给出最小读取清单和建议文件名。

## Outputs

- 一份符合 `project_memory/tasks/TASK_TEMPLATE.md` 的 task 草案
- 一个建议文件名
- 一组最小 `Read first` 文件
- 如有必要，一小组需人工确认的边界问题

## Guardrails

- 不把 task 草案写成长 handoff 或研究设计文档
- 不复制完整 `project_memory`、完整 `handoff` 或完整原始日志
- 不在这个 skill 里顺手回写 `current_state.md`、`handoff/latest_round.md` 或旧 task
- 不虚构验证结论、训练结果或尚未出现的产物
- 不把 `AGENTS.md` 或 workflow 文档的通用规则整段复制进 task 草案
