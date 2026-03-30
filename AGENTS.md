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

1. `AGENTS.md` 定义全局边界与启动规则。
2. `project_memory/current_state.md` 是默认执行入口，也是当前有效事实的最小集合。
3. `project_memory/tasks/` 存放单轮执行任务书；当前 task 文件由 `project_memory/current_state.md` 指向。
4. `handoff/chat_entry.md` 是网页端 ChatGPT / 新窗口接手入口，只提供最小上下文。
5. `handoff/handoff_for_chat.md` 是扩展 handoff，默认不读取。
6. 其余 `project_memory/*.md` 是专题参考，按需精确读取，不批量读取。
7. `reports/context_packets/*.md` 是共享上下文包层；新一轮讨论 / 执行开始前，优先生成或读取对应 target 的 packet。packet 是派生层，不替代事实源。
8. `reports/log_summaries/*.summary.md` 与 `*.summary.json` 是日志阅读首选层；默认先读摘要，再决定是否下钻原始日志。
9. `reports/round_updates/*.md` 是共享回写草稿层；一轮结束时优先通过单入口刷新链生成或刷新 round draft，再人工确认回写 `task` / `current_state` / `handoff`。
10. `docs/research_plan*.md` 只作为背景或历史参考，不参与默认执行判定。
11. `outputs/` 下原始日志 / checkpoint 只在摘要不足或证据核对时按需读取，不作为默认启动材料。
12. `.codex/skills/` 只在明确要使用某个 skill 时再读对应 `SKILL.md`，禁止默认通读全部 skill 文档。
13. 若作者明确裁定与旧文档冲突，以作者裁定 + 本地仓库实现为优先，再最小同步回 `project_memory`。

## Default Startup Set

- 若已有对应 target 的 context packet，优先读取；若缺失或过期，先用 `scripts/agent/build_context_packet.py` 生成。
- 默认只读取以下最小上下文集合：
  1. `AGENTS.md`
  2. `project_memory/current_state.md`
  3. `project_memory/current_state.md` 中指向的当前 task 文件
  4. 仅当当前会话是网页端 ChatGPT / Claude 新窗口 / 无本地上下文接手时，再读 `handoff/chat_entry.md`
- 默认禁止：
  - 读取完整 `handoff/`
  - 读取完整 `project_memory/`
  - 在未先查看日志摘要时直接读取完整原始长日志
  - 读取完整训练输出或完整历史记录
  - 读取全部 `.codex/skills/*/SKILL.md`
- 若任务明确落在某个专题，再只补读一个最相关的专题文件，例如：
  - 架构：`project_memory/01_current_architecture.md`
  - loss：`project_memory/02_loss_design.md`
  - 训练 / 运行：`project_memory/04_training_rules.md`

## Collaboration Stack

- `AGENTS.md`: 全局协议与边界。
- `CLAUDE.md`: Claude 兼容薄入口，回指本文件。
- `handoff/chat_entry.md`: 网页端 / 新 chat 薄入口。
- `project_memory/current_state.md`: 当前有效事实摘要。
- `project_memory/tasks/`: 单轮执行任务书。
- `.codex/agents/architect.toml`: 只读分析、影响面扫描、任务书生成。
- `.codex/agents/worker.toml`: 接收 brief 后实现、验证、汇报。
- `.codex/agents/maintainer.toml`: 更新 `handoff`、`project_memory` 与执行总结。
- `.codex/skills/prepare-task-brief/`: 基于 `TASK_TEMPLATE.md` 生成下一轮 task 草案。
- `.codex/skills/refresh-round-artifacts/`: 顺序刷新 summary、context packet 和 round draft。
- `.codex/skills/workflow-consistency-smoke/`: 检查当前 task 链的 missing / stale / conflict。
- `.codex/skills/update-handoff-memory/`: 同步长期记忆与交接文档。

## Required Workflow

1. 新一轮讨论或执行开始前，优先生成或读取对应 target 的 context packet，再按 `Default Startup Set` 补齐其背后的最小事实源。
2. 先确认当前处于 `Stage-2 architecture rollout / verification phase`，并明确当前 active 主线已切到 `axis + side + support`、验证中心是 `axis-side` train / smoke，再决定最小影响文件集合。
3. 单轮执行默认以当前 task 文件为中心；若 task 与专题 memory 冲突，以作者裁定 + 本地实现 + 当前有效事实为准，再最小同步任务书或 memory。
4. context packet 以“摘录 + 路径指引”为主，不替代 `AGENTS.md`、`project_memory/current_state.md`、当前 task 与日志摘要；当这些源发生变化时，应重新生成 packet，而不是手工修补旧 packet。
5. 若当前任务需要查看训练证据，先读 `reports/log_summaries/*.summary.md` 或 `*.summary.json`；只有摘要不足时，才按需下钻对应原始日志片段。
6. 一轮执行结束前或一轮证据刷新后，优先运行 `scripts/agent/refresh_round_artifacts.py`；它会按顺序串联 `summarize_train_log.py -> build_context_packet.py -> update_round_artifacts.py`，避免手工分步执行导致 stale summary / stale packet / stale draft / stale preview。
7. `scripts/agent/update_round_artifacts.py` 仍是回写层核心脚本，但默认应由单入口刷新链驱动；只有在明确需要单独调试 fixed-scope preview/apply 时，再直接调用它。
8. 重要轮次收尾、canonical 回写前，或 agent / target 切换前，可先运行 `scripts/agent/workflow_consistency_smoke.py`，显式检查当前 task 链的 missing / stale / conflict 项，再决定是否先 refresh 或 apply。
9. 若当前任务直接依赖某个专题，再按需只读一个最相关的 `project_memory/*.md` 或一小段必要日志，不做整包扩展。
10. 若作者明确裁定与旧文档冲突，先以作者裁定 + 本地仓库实现做最小必要同步，再继续执行。
11. 若冲突涉及边界不清、语义冲突、宿主 / 依赖疑点，或需要 fallback 才能继续，必须停止并输出结构化分析，不继续打补丁。
12. `project_memory/current_state.md` 只写当前有效事实与当前 task 指针；其余 memory 文件只保留专题稳定知识，不承担默认启动入口职责。
13. `handoff/chat_entry.md` 只服务新窗口快速接手；`handoff/handoff_for_chat.md` 仅在需要扩展背景时按需读取。
14. 角色分工由 subagents 与 skills 承担，不再在 `AGENTS.md` 中模拟 Developer / Reviewer / Planner 多角色轮换。

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
- 禁止把历史大段 handoff、整包 memory、未先摘要的完整原始长日志或全部 skill 文档当作默认启动上下文。
- “最小侵入”指少改代码、不扩大影响面，不代表允许通过兼容层隐藏问题。
