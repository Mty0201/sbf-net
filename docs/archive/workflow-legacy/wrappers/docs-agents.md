# Agent Collaboration Notes

## Current Canonical Structure

- 全局协议: `AGENTS.md`
- 角色配置: `.codex/agents/architect.toml`, `.codex/agents/worker.toml`, `.codex/agents/maintainer.toml`
- 高频流程: `.codex/skills/prepare-task-brief/`, `.codex/skills/update-handoff-memory/`
- 长期事实源: `project_memory/`
- 新 chat 接手入口: `handoff/handoff_for_chat.md`

## Stage Boundary

- `2.5` 阶段已结束并已收束
- 当前阶段状态为 `Stage-2 entry preparation phase`
- `Stage-2` 即将正式进入 architecture improvement 阶段
- `Stage-2` 的核心是从架构改进角度重新接入 direction 项，而不是继续扫 support 参数

## Hard Constraints

- IDE 工作区根可能是 `Pointcept`，但唯一允许主动维护的项目根是 `sbf-net`
- `Pointcept` 仅作为宿主依赖存在，默认只读
- 未经明确授权，禁止修改 `sbf-net` 目录之外的代码
- 若问题疑似来自 `Pointcept` 或宿主接口，必须停止并汇报，不得自行兜底修补
- 当前处于原型/研究验证阶段，优先目标是暴露问题，而不是掩盖问题
- 禁止为了“跑通 / 通过率 / 兼容性”擅自加入 fallback、默认兼容层、自动绕过、吞错、保守修补
- “最小侵入”指少改代码、不扩大影响面，不代表允许通过兼容层隐藏问题

## Migration Note

旧的“在单一文档中模拟多角色轮换”的方案，已经在前一轮协作层重构中被拆分为三层：

1. `AGENTS.md` 只保留全局协议与禁改边界。
2. `.codex/agents/*.toml` 承载 `architect / worker / maintainer` 的角色职责。
3. `.codex/skills/*` 只封装高频流程，不再和角色职责混写。

## Documentation Rules

- `project_memory` 继续是唯一执行级事实源。
- `handoff` 继续只服务新 chat 快速接手。
- 本文件只做协作结构说明，不再作为唯一执行协议来源。
