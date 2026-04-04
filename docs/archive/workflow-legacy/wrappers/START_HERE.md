# 新 Claude Chat 接手入口

这是每次新 Claude chat 开始时的唯一入口。不要跳过，不要自行扫仓库。

---

## 你在哪里

- IDE 工作区根：`Pointcept`（宿主，**只读**）
- 唯一主动维护项目根：`sbf-net/`
- Pointcept 只作为宿主依赖存在，禁止主动修改

---

## 必读顺序（不得跳过，不得打乱）

| 步骤 | 文件 | 目的 |
|------|------|------|
| 1 | `handoff/handoff_for_chat.md` | 快速接手背景：阶段状态、已确认数字、当前主线 |
| 2 | `CLAUDE_AGENTS.md` | 主协议：三角色定义、边界约束、guardrails |
| 3 | `MEMORY_RULES.md` | 记忆规则：哪些可以写入长期记忆，哪些不行 |
| 4 | `project_memory/00_project_overview.md` | 项目目标与非目标 |
| 5 | `project_memory/01_current_architecture.md` | 当前模型结构与分支状态 |
| 6 | `project_memory/02_loss_design.md` | Loss 设计与已确认实验结论 |
| 7 | `project_memory/03_data_pipeline.md` | 数据格式与 GT 构建链 |
| 8 | `project_memory/04_training_rules.md` | 训练入口、配置约束、日志键 |
| 9 | `project_memory/05_active_stage.md` | 当前阶段状态与验收口径 |
| 10 | `project_memory/06_task_queue.md` | 当前任务队列与禁止项 |
| 11 | `project_memory/90_archived_decisions.md` | （可选）需要历史背景时读 |

---

## 读完后的第一个动作

在回复开头声明：

> **[Architect 模式]**

然后输出接手确认，包含以下四项：

1. **项目目标**（一句话）
2. **当前阶段状态**（精确描述）
3. **当前主线矛盾**（当前最小问题）
4. **验收口径**（具体数字）

然后等待用户指令。

---

## 绝对禁止

- 跳过 handoff 直接扫仓库
- 必读顺序不完整就开始分析
- 接手确认之前写代码或修改文件
- 把 Pointcept 当作可主动维护的项目
- 在没有 task brief 的情况下进入 Worker 模式
