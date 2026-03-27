# AGENTS.md

## Scope

- IDE 工作区根可能是 `Pointcept`，但这不代表 `Pointcept` 是项目主体。
- 唯一允许主动维护的项目根是 `semantic-boundary-field`。
- `Pointcept` 只是宿主依赖与接口边界，默认只读。
- 未经明确授权，禁止修改 `semantic-boundary-field` 目录之外的代码。
- 若问题疑似来自 `Pointcept` 或宿主接口，必须停止并汇报，不得自行兜底修补。
- 本文件只保留全局协议；角色职责交给 `.codex/agents/`，高频流程交给 `.codex/skills/`。

## Current Stage Boundary

- `2.5` 阶段探索性实验已完成并已收束。
- 当前阶段状态是 `Stage-2 entry preparation phase`，尚未把 `Stage-2` 的架构改进实现真正展开。
- 当前进入 `Stage-2` 的已确认实验事实是：`semantic-only=73.8`，`support-only(reg=1, cover=0.2)=74.5` 为最佳，`support + dir + dist = 71` 在当前结构下失败。
- `Stage-2` 的正式核心是从架构改进角度重新接入 direction 项，而不是继续扫 support 参数。

## Canonical Sources

1. `project_memory/` 是唯一执行级事实源。
2. `handoff/handoff_for_chat.md` 是外部 web chat 接手入口，不承担长期记忆职责。
3. `docs/research_plan*.md` 只作为背景或历史参考，不参与当前执行判定。

## Collaboration Stack

- `AGENTS.md`: 全局协议与边界。
- `.codex/agents/architect.toml`: 只读分析、影响面扫描、任务书生成。
- `.codex/agents/worker.toml`: 接收 brief 后实现、验证、汇报。
- `.codex/agents/maintainer.toml`: 更新 `handoff`、`project_memory` 与执行总结。
- `.codex/skills/prepare-task-brief/`: 生成可执行 brief。
- `.codex/skills/update-handoff-memory/`: 同步长期记忆与交接文档。

## Required Workflow

1. 开始前先读 `handoff/handoff_for_chat.md`，再读相关 `project_memory`。
2. 先确认当前处于 `Stage-2 entry preparation phase`，并明确 `Stage-2` 的核心目标是架构改进接入 direction，再决定最小影响文件集合。
3. 若冲突只属于事实层的小型、明确、无歧义同步，可先在 `project_memory` 中做最小必要同步，再继续执行。
4. 若冲突涉及边界不清、语义冲突、宿主/依赖疑点，或需要 fallback 才能继续，必须停止并输出结构化分析，不继续打补丁。
5. `project_memory` 只写已确认、当前有效的事实；`handoff` 只写新 chat 快速接手所需的最小背景。
6. 角色分工由 subagents 与 skills 承担，不再在 `AGENTS.md` 中模拟 Developer / Reviewer / Planner 多角色轮换。

## Guardrails

- 不修改 `semantic-boundary-field` 目录之外的代码，除非用户明确授权。
- 不修改 Pointcept 主体，不重写 Pointcept registry / trainer / dataset 协议。
- 不改当前训练入口 `scripts/train/train.py`。
- 不改当前主配置 `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py`。
- 不改当前主模型职责划分，除非未来正式阶段切换并先同步 `project_memory`。
- 不把“准备进入 `Stage-2`”误写成“`Stage-2` 已经完成方向项接入”。
- 不在本轮顺手引入与当前 dual-task 主线无关的大改。
- 当前处于原型/研究验证阶段，优先目标是暴露问题，而不是掩盖问题。
- 禁止为了“跑通 / 通过率 / 兼容性”擅自加入 fallback、默认兼容层、自动绕过、吞错或保守修补。
- “最小侵入”指少改代码、不扩大影响面，不代表允许通过兼容层隐藏问题。
