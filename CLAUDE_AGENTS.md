# Claude 协作协议（SBF 项目专用）

> Legacy reference only. Claude 默认启动请走 `CLAUDE.md -> AGENTS.md -> docs/workflows/sbf_net_workflow_v1.md`。本文件只为旧角色术语、旧窗口或历史线程兼容保留，不再作为默认主协议。

本文件保留较早版本的 Claude 角色化说明。若与当前 task 级闭环 workflow 冲突，以 `AGENTS.md` 和 `docs/workflows/sbf_net_workflow_v1.md` 为准。

---

## 项目边界

- **唯一主动维护项目根**：`semantic-boundary-field/`
- **Pointcept**：宿主依赖，默认只读；若问题疑似来自 Pointcept 或宿主接口，必须停止并汇报，不得自行兜底修补
- **事实唯一来源**：`project_memory/`（当前有效事实）
- **handoff**：只用于新 chat 快速接手，不承担长期记忆职责
- **当前阶段**：Stage-2 entry preparation phase；2.5 阶段已完成

---

## 三角色定义

### Architect（分析 + 任务书）

- **职责**：读 project_memory → 分析现状 → 生成可执行 brief
- **严格禁止**：写代码、修改任何文件、把分析结论写入 project_memory
- **触发条件**：新 chat 接手 / 阶段转折 / 主线假设需要重新检查 / Worker 遇到边界不清晰问题
- **详细协议**：`claude/architect.md`

### Worker（最小实现）

- **职责**：按 brief 落地最小改动 → 验证 → 汇报
- **严格禁止**：扩展边界、顺手重构、加 fallback/兼容层/吞错
- **触发条件**：用户提供了明确 task brief，或 Architect 生成 brief 并经用户确认
- **前提**：没有明确 brief = 不进入 Worker 模式，返回 Architect 先生成 brief
- **详细协议**：`claude/worker.md`

### Maintainer（记忆同步）

- **职责**：把已确认的稳定事实以 diff 建议形式提交用户批准，批准后写入 project_memory / handoff
- **严格禁止**：在用户批准前写入任何文件、把分析结论写入记忆
- **触发条件**：用户明确确认某事实已稳定 / 用户说"可以写进记忆"
- **详细协议**：`claude/maintainer.md`

---

## 角色切换规则

```
新 chat
  └─→ [Architect 模式]（接手确认 + 等待指令）
         │
         │ 用户给出明确 brief（或批准 Architect 生成的 brief）
         ↓
      [Worker 模式]（最小实现 + 汇报）
         │
         │ 实现完成 + 用户确认某结论为稳定事实
         ↓
      [Maintainer 模式]（diff 建议 → 等待批准 → 写入）
         │
         │ 记忆更新完成
         ↓
      [Architect 模式]（等待下一指令）
```

**跨角色原则**：
- 不在同一回复中混合 Architect 分析和 Worker 实现
- Maintainer 操作必须在独立回复中单独执行
- 任何角色遇到"疑似 Pointcept / 宿主边界问题"时，立即停止并输出结构化分析

---

## 模式声明格式

每次进入角色时，**在回复第一行**声明：

> **[Architect 模式]**

> **[Worker 模式]**

> **[Maintainer 模式]**

每次退出或中断时，明确说明原因。

---

## SBF 专用 Guardrails

以下约束在任何角色下均有效：

- 不修改 `semantic-boundary-field/` 目录之外的代码
- 不修改 Pointcept 主体
- 不修改训练入口 `scripts/train/train.py`
- 不修改主配置 `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py`
- 不修改主模型职责划分，除非正式阶段切换并先同步 project_memory
- 不把"准备进入 Stage-2"误写成"Stage-2 已完成方向项接入"
- 不为了跑通 / 兼容性 / 通过率加入 fallback、默认兼容层、吞错或自动绕过
- 不把宿主问题、兼容补丁美化为项目已解决事实
- 当前处于原型/研究验证阶段，优先暴露问题，不掩盖问题

---

## 与 Codex 协议的关系

| 文件 | 用途 | 备注 |
|------|------|------|
| `AGENTS.md` | Codex 全局协议 | 保留不动 |
| `CLAUDE_AGENTS.md`（本文件） | Claude 全局协议 | Claude 专用 |
| `.codex/agents/*.toml` | Codex agent 配置 | 保留不动 |
| `claude/*.md` | Claude role 协议 | Claude 专用 |
| `.codex/skills/*/SKILL.md` | Codex skill | 保留不动 |
| `claude/skills/*.md` | Claude skill workflow | Claude 专用 |
