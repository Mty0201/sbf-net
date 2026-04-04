# Skill: prepare-task-brief（SBF 项目专用）

围绕 `project_memory/tasks/TASK_TEMPLATE.md` 生成下一轮最小 task 草案。
对应 Codex 的 `.codex/skills/prepare-task-brief/SKILL.md`。

---

## 触发场景

- 当前 task 已收尾，准备新开一轮 task
- 用户需求还比较宽，需要先压缩成单轮、可执行、可验证的任务书
- 需要把当前状态、下一步和边界映射到 `TASK_TEMPLATE.md`
- 需要为 web ChatGPT、Claude、Codex 共享同一份 task 草案，而不是继续写长 handoff

---

## 必要输入

1. `AGENTS.md`
2. `project_memory/current_state.md`
3. `project_memory/tasks/TASK_TEMPLATE.md`
4. 若是承接上一轮，再读当前 task 文件
5. 若任务直接依赖证据，再补一个最相关的 summary / packet / 专题 memory

---

## 生成流程

**Step 1：边界确认**
- 从 `AGENTS.md` 和 `current_state.md` 确认当前阶段边界、当前 task 指针和禁改项
- 确认唯一允许主动维护的项目根是 `sbf-net/`；`Pointcept` 默认只读
- 若问题疑似来自宿主接口或边界语义冲突，停止继续生成 task 草案，输出停止声明

**Step 2：事实提炼**
- 只提炼当前轮真正需要的稳定事实
- 避免复制完整 handoff、完整 memory 或完整原始日志
- 若需要证据，只补读一个最相关的 summary / packet / 专题文档

**Step 3：任务文件名**
- 基于日期和目录现状，建议新的任务文件名：
  `project_memory/tasks/TASK-YYYY-MM-DD-XXX.md`
- 若用户给出的文件名与仓库现状不一致，先写清映射，不静默猜路径

**Step 4：按模板生成**
- 按 `TASK_TEMPLATE.md` 逐节生成 task 草案，至少填好：
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

**Step 5：结果约束**
- `Result` 只写当前已知状态，例如“待开始”或“等待验证”
- 不得凭空补训练结果、验证结论或已完成状态
- 同时给出最小 `Read first` 清单和建议文件名

---

## 产出

- 一份符合 `TASK_TEMPLATE.md` 的 task 草案
- 一个建议文件名
- 一组最小 `Read first` 文件
- 如有必要，一小组需人工确认的边界问题

---

## Guardrails

- 不把 task 草案写成长 handoff 或架构大设计
- 不把 `handoff` 当作事实源替代 `project_memory`
- 不顺手回写 `current_state.md`、`handoff/latest_round.md` 或旧 task
- 不虚构验证证据、训练结果或尚未出现的产物
- 当前处于原型 / 研究验证阶段，task 草案必须帮助暴露问题，而不是掩盖问题
- 不把 `AGENTS.md` 或 workflow 文档里的通用规则整段复制进 task 草案
