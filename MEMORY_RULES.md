# 记忆写入规则（SBF 项目专用）

本文件规定哪些内容允许写入 `project_memory`，哪些只能停留在分析 memo 中。
这是 Claude 协作中防止"分析污染事实记忆"的核心防护机制。

---

## 三种内容类型

### 类型 A：稳定事实（可写入 project_memory）

**判断标准**（至少满足一条）：
- 有实际训练 run 的数值支撑（如 val_mIoU 数字）
- 有明确代码验证的实现状态（如 smoke 进入 training loop）
- 有 diff-level 的代码改动记录
- 用户明确说"这是稳定事实"或"可以写进记忆"

**归宿**：→ `project_memory/` 对应文件，经 Maintainer 以 diff 建议形式提交、用户**显式批准**后写入

---

### 类型 B：分析结论（不得写入 project_memory）

**判断标准**：
- 基于推理 / 类比 / 理论，未经实验验证
- 无训练数据支撑
- 属于"我认为 X 是原因"类表述
- 属于对 GT 或架构的解读分析

**归宿**：→ 只停留在当前 chat 的 memo 中；结尾必须标注 `[分析结论，未经实验验证，不写入 project_memory]`

---

### 类型 C：假设 / 待验证设计方案（不得写入 project_memory）

**判断标准**：
- 提出了一个设计选择，但尚未实现或实验
- 属于"如果 X 则 Y"类表述
- 属于架构改进方向建议

**归宿**：→ 写入 task brief 的"待验证"字段；不写入 project_memory；标注 `[待验证设计选择，写入 brief 而非 project_memory]`

---

## SBF 项目具体案例

| 内容 | 类型 | 允许写入 project_memory？ |
|------|------|--------------------------|
| `support-only(reg=1, cover=0.2) val_mIoU = 74.6` | A | **是** |
| `Stage-2 v1` 最佳 `val_mIoU = 71.34`（epoch 36） | A | **是** |
| `Stage-2 v2` smoke 已进入 training loop | A | **是** |
| direction 失败是因为 support domain partition 被压扁 | B | **否** |
| support 应该作为 direction field 的 organizer | B | **否** |
| 应该在 `edge.npy` 中加入 `support_id` 字段 | C | **否**（写 brief 里） |
| boundary_adapter 能解决特征竞争问题 | C | **否**（待实验验证） |
| `dist` 在一个 epoch 内快速降到约 `0.0002` | A | **是** |
| train `dir_cosine` 后期约 `0.65 ~ 0.75` | A | **是** |

---

## Claude 操作约定

### Architect 输出分析 memo 时

结尾必须标注：
> **[分析结论，未经实验验证，不写入 project_memory]**

### Architect 输出 task brief 中包含设计假设时

在相关字段标注：
> **[待验证，不写入 project_memory]**

### Maintainer 提出写入建议时

必须用以下格式，等待用户批准后再执行：
> **[Maintainer 建议]** 建议在 `project_memory/XX.md` 中增加以下内容：
> ```
> [具体 diff 内容]
> ```
> 等待用户批准后写入。

---

## 绝对禁止

- 在用户批准前，Maintainer 不得直接写入任何 `project_memory` 文件
- 不把讨论过程、推测、未决方案写入 `project_memory`
- 不把 `handoff/handoff_for_chat.md` 扩写成 `project_memory` 的镜像副本
- 不把 fallback、兼容层、宿主补丁、临时绕过美化为已解决事实
- 不把"正在讨论的架构改进方向"写成"已确认下一步"
- 不遗漏阶段边界同步（避免 `project_memory` 中出现"Stage-2 已完成"的误表述）
