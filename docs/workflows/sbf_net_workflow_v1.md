# SBF Net Workflow v1

## Purpose

- 把项目正式切换为“一个 task = 一轮假设闭环”的开发模型。
- 统一 discussion、brief、implementation、validation、review、closeout 之间的边界。
- 明确 summary、packet、workflow smoke、round draft、refresh / preview / apply 都只是 task 内 checkpoint，不默认代表新 task 或 closeout。

## Repository Layers

### Canonical / source of truth

- `AGENTS.md`：全局边界、最小启动链和 guardrails。
- `docs/workflows/sbf_net_workflow_v1.md`：正式 task lifecycle、角色分工与 closeout 规则。
- `project_memory/current_state.md`：当前有效事实和 active task 指针。
- 当前 `project_memory/tasks/TASK-*.md`：当前 task 的目标、边界、验证和 closeout 条件。
- `handoff/latest_round.md`：最近一轮已完成或已同步 closeout 的最小结果摘要；用于收尾，不是默认启动入口。
- `handoff/web_to_agent_contract.md`：web ChatGPT 到本地 agent 的结构化交付协议；只在产出或消费 web handoff 时生效，不替代 workflow。

### Checkpoint / generated

- `reports/log_summaries/*.summary.md|json`
- `reports/context_packets/*.context_packet.md`
- `reports/round_updates/*.round_update.draft.md`
- `reports/workflow_smokes/*.workflow_consistency_smoke.md`

这些都属于 task 内 checkpoint 产物。它们服务于分析、接手、回写和巡检，但不替代 canonical 事实源。

### Thin wrappers

- `CLAUDE.md`
- `handoff/chat_entry.md`

这些文件只负责薄入口和例外说明，不再维护第二套完整 workflow 规则。

### Compatibility / legacy reference

- `docs/workflow.md`
- `CLAUDE_AGENTS.md`
- `handoff/handoff_for_chat.md`

这些文件保留给旧链接、旧窗口或旧术语做兼容引用，不属于默认启动集合。

## Roles

### webchat

- 负责讨论、假设收敛、方向筛选。
- 负责输出 `Discussion Handoff`、`Task Brief Draft` 或 `Agent Prompt` 之一。
- 不直接根据模糊想法推动本地代码实现。

### codex-discuss

- 负责仓库内分析、边界校正、task 落地。
- 负责把讨论结果压缩成当前 task 或新的 `TASK-*.md`。
- 负责判断当前问题是否仍应留在同一 task 内。

### claude-implement

- 负责在已经收敛的 task brief 边界内实现。
- 默认只消费当前 task、最小 `Read first` 集合和必要 artifact。
- 不直接根据“想法”扩 scope，不自行升级为新 task。

### codex-review

- 负责验收 Claude 改动、复核验证证据、判断是否达到 done condition。
- 负责决定当前 task 应继续、closeout，还是因核心问题本质变化而切到下一 task。
- 负责最终封盘与 canonical 同步判断。

## Task Lifecycle

### Task starts when

- 当前有一个新的核心假设、决策问题或实现边界，需要单独闭环。
- 当前 task 的核心问题已经本质变化，继续沿用旧 task 会混淆目标、证据和 done condition。

### What may happen inside one task

- 讨论与假设收敛。
- brief / task 定义和微调。
- 实现。
- 验证与复核。
- review / 验收。
- closeout / 收尾。
- refresh summary / packet / workflow smoke / round draft。

### Task closes out when

- 该 task 的 `Done condition` 达成。
- `Review / acceptance plan` 已完成。
- `Closeout expectation` 已满足，或未完成项已被显式记录为后续 task 输入。

## Checkpoints vs Closeout

### Task-internal checkpoints

- `summary`
- `packet`
- `workflow smoke`
- `round update draft`
- `refresh`
- `preview`
- `apply`
- 中间日志分析
- 阶段性 review

这些动作默认只说明“task 内部进度发生了推进”，不代表 task 已完成，不自动触发新 task。

### What counts as closeout

- 当前 task 的核心问题已经被回答。
- 本轮要求的验证证据已经齐备。
- review 已明确通过，或剩余问题已被压缩成新的核心问题。
- canonical 层知道当前 task 已结束，或明确切换到下一 task。

## Recommended Threading Model

- `webchat`：适合长期复用，围绕同一个问题域持续讨论与收敛。
- `codex-discuss`：适合按问题域或连续 task 复用；只要分析边界没本质变化，可以持续使用。
- `claude-implement`：更适合按实现 task 或按同一 task 的稳定实现阶段复用；当任务目标发生本质变化时更适合新开。
- `codex-review`：更适合按 review / closeout 阶段开独立窗口；同一 task 的验收可复用，跨 task 不建议默认继承。

## Handoff Rules

### web -> codex

- 优先按 `handoff/web_to_agent_contract.md` 交付单一结构化产物。
- 若讨论尚未收敛，输出 `Discussion Handoff`。
- 若已接近新 task，输出 `Task Brief Draft`。
- 只有边界清楚、可直接开工时，才输出 `Agent Prompt`。

### codex -> claude

- 只在当前 task 已收敛到可实现边界时交给 Claude。
- 交付内容默认是当前 task、最小 `Read first`、必要 artifact 和明确 validation。
- 不允许把“想法列表”直接当实现输入。

### claude -> codex review

- 必须回交：
  - 实现摘要
  - 验证证据
  - 风险 / 未决项
  - 当前是否达到 done condition 的自评
- Codex review 负责最终判断是否 closeout，而不是由 Claude 单方面封盘。

## Closeout Criteria

- 允许封盘：
  - 当前 task 的 done condition 已达成。
  - review / acceptance 已完成。
  - 当前 task 不再需要继续留在同一假设闭环内。

- 必须继续留在当前 task：
  - 只是刷新了 summary / packet / smoke / round draft。
  - 只是完成了局部实现或局部验证，但还没回答核心问题。
  - 只是发现了新的中间现象，但还没有形成新的核心问题。

- 必须新建下一 task：
  - 当前 task 已经完成，且下一步是新的核心问题。
  - 或当前核心问题发生本质变化，继续沿用旧 task 会混淆目标、边界和验收标准。

## Anti-patterns

- 不能把分析一次日志就当成一个 task。
- 不能把刷新 summary / packet / workflow smoke / round draft 当成 task 完成。
- 不能让 Claude 直接根据“想法”写代码，而没有收敛过的 task brief。
- 不能因为一次 preview / apply 或一次 workflow smoke = PASS，就默认开启下一 task。
- 不能把 discussion、implementation、review、closeout 拆成多个默认 task，除非核心问题已经本质变化。

## Migration

### Why the old flow drifted toward action-level closeout

- artifact 刷新动作很清晰、很可见，容易被误当成“这一轮已经结束”。
- 一次日志分析、一次 smoke、一次 packet 刷新都很容易被误记成单独阶段。
- 角色边界不够统一时，discussion、implementation、review、closeout 会被拆成多个“动作封盘”。

### What changes in the new flow

- artifact refresh 被降级为 task 内 checkpoint。
- `Done condition`、`Review / acceptance plan`、`Closeout expectation` 成为 task closeout 的正式判定依据。
- 只有 task 完成，或核心问题本质变化，才默认新建下一 task。

### Recommended usage in this repo

- `webchat` 长期讨论窗口：持续收敛方向、比较路线、产出结构化交付物。
- `codex-discuss` 讨论窗口：把 web 结果映射到仓库、校正边界、落 task。
- `claude-implement` 实现窗口：只在 task 边界已收敛时进入。
- `codex-review` review 窗口：验收 Claude 改动、判断 done condition、负责封盘。

如果当前只是刷新了 summary、packet、workflow smoke 或 round draft，那么默认仍留在当前 task 内继续推进，而不是开新 task。

## Documentation Placement

- 新的长期 workflow 规则，优先写入 `AGENTS.md` 或本文件。
- 当前轮次的事实、结果和 closeout 状态，优先写入 `project_memory/current_state.md`、当前 `TASK-*.md` 和 `handoff/latest_round.md`。
- summary / packet / round draft / workflow smoke 继续只放在 `reports/`，视为 checkpoint 产物。
- `CLAUDE.md`、`handoff/chat_entry.md` 等薄入口只保留启动指针和必要例外，不再复制完整 lifecycle 规则。
- 不要把新的持久规则再写进 `docs/workflow.md`、`CLAUDE_AGENTS.md` 或 `handoff/handoff_for_chat.md`。
