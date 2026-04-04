---
name: update-handoff-memory
description: Sync confirmed changes into project_memory and handoff for sbf-net. Use when implementation or process changes need a minimal, factual memory update without mixing in speculation.
---

# Update Handoff Memory

## When To Use

- 本轮结果已经落地，需要把稳定事实写回 `project_memory`。
- 需要刷新 `handoff/handoff_for_chat.md`，便于新的 web chat 快速接手。
- 需要把阶段边界、当前任务边界和协作层变更保持一致。

## Decision Rules

- `project_memory/`: 只放当前有效、已确认、可复用的事实。
- `handoff/handoff_for_chat.md`: 只放下一窗口最需要的最小背景。
- `project_memory/90_archived_decisions.md`: 只放被替代但仍有历史价值的结论。

## Workflow

1. 先判断本轮变化是否已经稳定，不稳定就不要写入长期记忆。
2. 优先同步 `project_memory`，保证它仍是唯一事实源。
3. 若需要新 chat 接手，再同步 `handoff`，并保持内容短、小、可立即继续。
4. 若任务中止，必须如实记录：中止原因、影响范围、待分析问题；不要把中止写成“已解决”或“已兼容”。
5. 明确写出：
   - `2.5` 阶段已经完成
   - 当前处于 `Stage-2 architecture rollout / verification phase`
   - 当前 active 主线是 `axis + side + support`
   - 当前验证中心是 `axis-side` train / smoke
   - `Stage-2` 的核心仍是从架构改进角度重写旧的 signed direction 表达

## Guardrails

- 不写讨论过程、猜测或未决方案。
- 不把 handoff 扩写成 project_memory 的镜像副本。
- 不遗漏阶段边界同步，避免文档内继续出现“还在准备进入 Stage-2”的过时表述。
- 不得把 fallback、兼容层、宿主补丁或临时绕过美化为正式解决。
- 不得把超出 `sbf-net` 的修改写成常规维护行为。
- handoff / memory 的更新必须保持“问题显性化”，不能用摘要语言掩盖真实问题。
