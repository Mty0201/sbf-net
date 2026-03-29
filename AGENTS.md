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
- 当前阶段状态是 `Stage-2 architecture rollout / verification phase`；架构更新已经开始并已落地到 `axis-side` 主线实现，当前处于新主线核证阶段。
- 当前 active 主线表达统一为 `axis + side + support`；作者口头中的 `magnitude` 在当前代码 / 文档同步中等价于 `support`，不要误写成独立 magnitude 分支已落地。
- 当前验证中心是 `semseg-pt-v3m1-0-base-bf-edge-axis-side-train` 及其 smoke config。
- 当前已确认实验事实是：`semantic-only=73.8`，`support-only(reg=1, cover=0.2)=74.6` 为最佳，`support + dir + dist = 71` 在旧 signed-direction 结构下失败。
- `Stage-2` 的当前正式核心是通过 `axis + side + support` 重写旧的显式 signed direction 表达，而不是继续扫 support 参数。

## Canonical Sources

1. `project_memory/` 是唯一执行级事实源。
2. `handoff/handoff_for_chat.md` 是外部 web chat 接手入口，不承担长期记忆职责。
3. `docs/research_plan*.md` 只作为背景或历史参考，不参与当前执行判定。
4. 若作者明确裁定与旧文档冲突，以作者裁定 + 本地仓库实现为优先，再同步回 `project_memory`。

## Collaboration Stack

- `AGENTS.md`: 全局协议与边界。
- `.codex/agents/architect.toml`: 只读分析、影响面扫描、任务书生成。
- `.codex/agents/worker.toml`: 接收 brief 后实现、验证、汇报。
- `.codex/agents/maintainer.toml`: 更新 `handoff`、`project_memory` 与执行总结。
- `.codex/skills/prepare-task-brief/`: 生成可执行 brief。
- `.codex/skills/update-handoff-memory/`: 同步长期记忆与交接文档。

## Required Workflow

1. 开始前先读 `handoff/handoff_for_chat.md`，再读相关 `project_memory`。
2. 先确认当前处于 `Stage-2 architecture rollout / verification phase`，并明确当前 active 主线已切到 `axis + side + support`、验证中心是 `axis-side` train / smoke，再决定最小影响文件集合。
3. 若作者明确裁定与旧文档冲突，先以作者裁定 + 本地仓库实现做最小必要同步，再继续执行。
4. 若冲突涉及边界不清、语义冲突、宿主 / 依赖疑点，或需要 fallback 才能继续，必须停止并输出结构化分析，不继续打补丁。
5. `project_memory` 只写已确认、当前有效的事实；`handoff` 只写新 chat 快速接手所需的最小背景，并明确区分“作者已确认事实 / 当前 workspace 可直接复核产物 / 仅代码已落地状态”。
6. 角色分工由 subagents 与 skills 承担，不再在 `AGENTS.md` 中模拟 Developer / Reviewer / Planner 多角色轮换。

## Guardrails

- 不修改 `semantic-boundary-field` 目录之外的代码，除非用户明确授权。
- 不修改 Pointcept 主体，不重写 Pointcept registry / trainer / dataset 协议。
- 不改当前训练入口 `scripts/train/train.py`。
- 不改当前主配置 `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py`。
- 不改当前主模型职责划分，除非未来正式阶段切换并先同步 `project_memory`。
- 不把当前 `axis-side` 主线已落地误写成已经完成 smoke / full train 验证。
- 不在本轮顺手引入与当前 dual-task 主线无关的大改。
- 当前处于原型 / 研究验证阶段，优先目标是暴露问题，而不是掩盖问题。
- 禁止为了“跑通 / 通过率 / 兼容性”擅自加入 fallback、默认兼容层、自动绕过、吞错或保守修补。
- 不把作者已确认实验事实、当前 workspace 可直接复核产物与仅代码已落地状态混写成同一证据层。
- “最小侵入”指少改代码、不扩大影响面，不代表允许通过兼容层隐藏问题。
