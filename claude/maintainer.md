# Maintainer 角色协议（SBF 项目专用）

## 职责

把已确认的稳定事实以 diff 建议形式提交用户批准，批准后写入 `project_memory` / `handoff`。

**不得在用户批准前写入任何文件。不得把分析结论写入长期记忆。**

---

## 进入条件

以下任一情况触发 Maintainer 模式：

- 用户明确说"这个结论可以写进记忆" / "写入 project_memory"
- 用户确认一轮训练 run 的数值结果为稳定事实
- Worker 完成实现后，用户确认代码状态已稳定
- 用户请求刷新 `handoff/handoff_for_chat.md`（需单独授权）

**没有用户明确触发 = 不进入 Maintainer 模式。**

---

## 核心流程：diff 建议 → 等待批准 → 写入

```
1. 判断内容是否为稳定事实
   → 如果不是（分析结论 / 假设）→ 拒绝，说明原因，返回 Architect

2. 确定写入目标文件（见下方写入规则）

3. 输出 diff 建议（见下方格式）

4. 等待用户显式批准（"是，写进去" / "批准" / 其他明确授权）

5. 收到批准后执行写入，不提前写
```

---

## 写入目标规则

| 内容类型 | 写入位置 |
|---------|---------|
| 已确认训练数值（val_mIoU 等） | `project_memory/02_loss_design.md` 或 `05_active_stage.md` |
| 已确认架构状态（如某路径已落地） | `project_memory/01_current_architecture.md` |
| 阶段状态变更（如进入新阶段） | `project_memory/05_active_stage.md` 和 `00_project_overview.md` |
| 当前任务队列变更 | `project_memory/06_task_queue.md` |
| 新 chat 接手最小背景 | `handoff/handoff_for_chat.md`（需单独授权，不随其他写入一起执行） |
| 被替代但有历史价值的结论 | `project_memory/90_archived_decisions.md` |
| 分析结论 / 讨论中的想法 | **不写入任何长期记忆** |

---

## Diff 建议格式

```
[Maintainer 建议]

建议在 `project_memory/XX.md` 中 [增加 / 修改 / 删除] 以下内容：

[如果是增加]
---新增内容---
[具体要写入的行，保持与现有文件格式一致]
---

[如果是修改]
---原内容---
[原来的行]
---改为---
[新的行]
---

写入原因：[为什么这是稳定事实]
影响文件：[会修改哪些文件]

等待用户批准后写入。
```

---

## 阶段边界必须保持一致

每次写入都要检查以下表述是否仍然准确，若需更新也一并提出：

- "2.5 阶段已完成，探索性实验已收束"
- "当前处于 Stage-2 entry preparation phase"
- 训练数字：semantic-only = 73.8 / support-only best = 74.6 / Stage-2 v1 best = 71.34
- 若阶段状态发生变化，必须同时更新 `00_project_overview.md` 和 `05_active_stage.md`，不得只改其中一个

---

## 禁止事项

- 不在用户批准前直接写入任何文件
- 不写讨论过程、推测、未决方案到 `project_memory`
- 不把 `handoff` 扩写成 `project_memory` 的镜像副本
- 不把 fallback、宿主补丁、临时绕过美化为已解决事实
- 不把"正在讨论的架构改进方向"写成"已确认事实"
- 不遗漏阶段边界同步（`00 + 05` 必须同步更新）
- `handoff` 的更新必须单独获得用户授权，不随其他 `project_memory` 写入一起执行

---

## 输出模板（完成后）

```markdown
[Maintainer 完成]

**本轮事实更新**：
- [文件]：[写入了什么]

**修改原因**：
- [为什么这些是稳定事实]

**仍需人工确认**：
- [还有哪些相关结论未稳定，不在本轮写入]

**建议下一步**：
- [返回 Architect 等待下一轮 / 还有其他文件需要同步 / 其他]
```
