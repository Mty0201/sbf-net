# Skill: prepare-task-brief（SBF 项目专用）

生成一份最小可执行 task brief，供 Worker 按 brief 落地改动。
对应 Codex 的 `.codex/skills/prepare-task-brief/SKILL.md`。

---

## 触发场景

- 需要在实现前把任务压缩成单一、可执行、可验证的 brief
- 用户提出一个改动需求，需要先确认边界再交给 Worker
- 需要把用户的模糊需求映射到具体文件和最小影响集合
- 需要在对齐"2.5 已结束、当前准备进入 Stage-2"的阶段边界后，再下发实现任务

---

## 必要输入（Architect 执行本 skill 前必须确认已读）

1. `handoff/handoff_for_chat.md`
2. 与任务直接相关的 `project_memory/` 文件
3. 用户当前请求

---

## 生成流程

**Step 1：阶段状态确认**
- 确认 2.5 阶段已结束，当前处于 `Stage-2 entry preparation phase`
- 确认唯一允许主动维护的项目根是 `semantic-boundary-field/`；Pointcept 只读
- 若问题疑似来自 Pointcept 或宿主接口，停止继续生成 brief，输出停止声明

**Step 2：事实提炼**
- 从 `project_memory/` 提炼与本任务相关的已确认事实
- 明确哪些是稳定事实，哪些是待验证假设（后者放"需人工确认项"）
- 验证数字：semantic-only = 73.8 / support-only best = 74.6 / Stage-2 v1 best = 71.34

**Step 3：文件名映射**
- 若用户给出的文件名与仓库现状不一致，先做路径映射
- 把映射结果写进 brief，不静默使用"猜测路径"
- 当前已落地的 Stage-2 路径：
  - `stage2-support-dir` config
  - `stage2-v2` model/train/train-smoke config
  - `stage2-support-dir-train-smoke.py`（sample smoke）
  - `stage2-v2-train-smoke.py`（v2 sample smoke）

**Step 4：边界划定**
- 固定禁改项（任何 brief 都不得包含）：
  - `scripts/train/train.py`
  - `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py`
  - `project_memory/` 下的任何文件
  - `handoff/handoff_for_chat.md`
- 确认本任务允许改动的最小文件集合

**Step 5：任务压缩**
- 把任务限制为一个最小可交付目标
- 若工作量较大，拆成当前轮最小落地点，而不是一次性大改
- 不在 brief 中展开新的研究方案

**Step 6：输出 brief**（使用下方模板）

---

## Brief 模板

```markdown
## Task Brief：[任务名]

**目标**：[一句话，可验证]

**当前事实**（来自 project_memory）：
- [事实1，来源：project_memory/XX.md]
- [事实2，来源：...]

**允许改动**：
- [文件或模块，精确到文件名]

**禁止改动**：
- `scripts/train/train.py`
- `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py`
- [本任务其他禁止项]

**最小影响文件**：
- [预计需要改动的文件列表]

**验证方式**：
- [具体可执行步骤，如：python scripts/train/train.py --config ... 并确认进入 training loop]

**停止条件**：
- [Worker 遇到什么情况应停止并汇报]

**需人工确认项**：
- [模糊或有歧义的部分，需用户决策后再实现]

[待验证设计选择，写入 brief 而非 project_memory]
```

---

## Guardrails

- 不把 brief 写成架构大设计，只写最小可落地目标
- 不把 `handoff` 当作事实源替代 `project_memory`
- 不因为"准备进入 Stage-2"就放松对训练入口 / 主配置 / 主模型职责的保护
- 不把 Pointcept 目录路径写成常规可维护范围
- 禁止为了跑通 / 通过率 / 兼容性在 brief 中预设 fallback 或兼容层
- 当前处于原型/研究验证阶段，brief 中的验证方式必须能真实暴露问题，不掩盖问题
