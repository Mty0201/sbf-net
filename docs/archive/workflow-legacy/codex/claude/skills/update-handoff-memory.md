# Skill: update-handoff-memory（SBF 项目专用）

把本轮已确认的稳定结论同步到 `project_memory` 和 `handoff`。
对应 Codex 的 `.codex/skills/update-handoff-memory/SKILL.md`。

---

## 触发场景

- 本轮结果已经落地，需要把稳定事实写回 `project_memory`
- 需要刷新 `handoff/handoff_for_chat.md`，便于新的 Claude chat 快速接手
- 阶段状态发生变化（如正式进入 Stage-2）
- 实验数字有新增（如 full train 完成，有了新的 val_mIoU 数据）

---

## 执行前判断

使用 MEMORY_RULES.md 中的三类标准先判断每条内容属于哪类：

- **类型 A（稳定事实）** → 可以写入 `project_memory`（Maintainer diff 建议 + 用户批准）
- **类型 B（分析结论）** → 不写入，停留在 memo
- **类型 C（假设/待验证）** → 不写入，停留在 brief

不稳定的结论一律不写入，先返回 Architect 继续分析。

---

## 写入决策规则

| 内容 | 写入位置 | 优先级 |
|------|---------|--------|
| 已确认训练数值（val_mIoU、loss 数字等） | `project_memory/02_loss_design.md` | 高 |
| 阶段状态变更 | `project_memory/05_active_stage.md` + `00_project_overview.md` | 高（必须同步） |
| 架构落地状态 | `project_memory/01_current_architecture.md` | 中 |
| 数据管道变更 | `project_memory/03_data_pipeline.md` | 中 |
| 训练规则变更 | `project_memory/04_training_rules.md` | 中 |
| 任务队列变更 | `project_memory/06_task_queue.md` | 中 |
| 被替代的历史结论 | `project_memory/90_archived_decisions.md` | 低 |
| 新 chat 接手背景 | `handoff/handoff_for_chat.md` | **需单独授权** |

---

## 执行流程

**Step 1：稳定性判断**

确认每条候选内容是否满足"稳定事实"标准：
- 有训练数字支撑？
- 有代码验证？
- 用户明确确认？

不满足 → 不进入写入流程，向用户说明原因。

**Step 2：阶段边界检查**

确认以下表述在写入后仍然准确：
- "2.5 阶段已完成，探索性实验已收束"
- "当前处于 Stage-2 entry preparation phase"（除非本次写入的就是阶段变更）
- 数字锚点：semantic-only = 73.8 / support-only best = 74.6 / Stage-2 v1 best = 71.34

若以上任一项需要更新，把更新一并纳入本次 diff 建议。

**Step 3：输出 diff 建议**（Maintainer 格式）

见 `claude/maintainer.md` 中的"Diff 建议格式"。

**Step 4：等待用户批准**

收到用户显式批准后执行写入。`handoff` 更新必须单独等待授权，不随其他文件一起写入。

**Step 5：输出完成摘要**（Maintainer 输出模板）

---

## SBF 项目专用写入约束

- `handoff/handoff_for_chat.md` 应保持短小，只写新 chat 快速接手所需的最小背景；不得扩写成 `project_memory` 的镜像
- `project_memory` 中的数字必须与实际训练 run 严格对应，不得四舍五入或用"约"表达关键数值
- 任务中止的情况必须如实记录：中止原因、影响范围、待分析问题；不得把中止写成"已解决"或"已兼容"
- 若本轮 Worker 因宿主环境问题（如 CUDA 缺失）停止，写入时必须明确标注"环境限制，非项目逻辑阻塞"

---

## Guardrails

- 不写讨论过程、推测或未决方案
- 不把 `handoff` 扩写成 `project_memory` 副本
- 不遗漏阶段边界同步（`00 + 05` 必须同步）
- 不把 fallback、兼容层、宿主补丁美化为正式解决
- 不超出用户本次批准的写入范围
- 问题的写入必须保持"显性化"，不用摘要语言掩盖真实问题
