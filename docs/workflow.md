# DEVELOPMENT_WORKFLOW

## 1. 目标

本项目采用 **task + artifact 驱动开发**，而不是长会话驱动开发。

目标有两个：

1. 持续推进 `sbf-net` 的真实研发
2. 让 Claude / Codex / 网页端 ChatGPT 在同一套最小上下文和共享产物上协同工作

核心原则：

- **canonical 文档是事实源**
- **task 文件是单轮执行中心**
- **summary / packet / smoke / round draft 是派生产物**
- **Claude 和 Codex 共享同一套脚本与流程**
- **网页端 ChatGPT 只做讨论与任务边界收口，不重新整仓接手**

---

## 2. 角色分工

### 2.1 网页端 ChatGPT
职责：

- 分析与讨论
- 收口任务边界
- 判断下一轮是分析轮还是实现轮
- 生成给 Claude / Codex 的提示词

不负责：

- 重新整仓接手
- 默认读取完整 handoff / 完整 memory / 原始长日志
- 直接落地本地代码修改

### 2.2 Claude
职责：

- 本地分析
- 本地实现
- 刷新 summary / packet / round draft
- 执行 closeout
- 运行 consistency smoke

适合：

- 实现线程
- 分析线程
- 收尾线程
- 局部验证线程

### 2.3 Codex
职责与 Claude 基本对等：

- Claude 额度不足时接手
- 文档和 task 维护
- 共享流程脚本落地
- 本地分析 / 实现 / 收尾

原则：

- Claude 和 Codex 不是两套体系
- 两者只是同一工作流下的两个执行端

---

## 3. 信息分层

### 3.1 canonical 层（事实源）
这些文件是权威事实源：

- `AGENTS.md`
- `CLAUDE.md`
- `project_memory/current_state.md`
- `project_memory/tasks/TASK-*.md`
- `handoff/latest_round.md`

规则：

- 只写当前有效事实
- 不把完整历史堆成必读长文
- 不把派生产物当事实源

### 3.2 artifact 层（派生产物）
这些文件是根据 canonical 和日志生成的派生产物：

- `reports/log_summaries/*.summary.md`
- `reports/log_summaries/*.summary.json`
- `reports/context_packets/*.context_packet.md`
- `reports/round_updates/*.round_update.draft.md`
- `reports/workflow_smokes/*.workflow_consistency_smoke.md`

规则：

- artifact 不是长期事实源
- artifact 用于分析、交接、验证、收尾
- canonical 更新后，应重新生成相关 artifact

### 3.3 skill 层（动作入口）
当前共享 skill：

- `prepare-task-brief`
- `refresh-round-artifacts`
- `workflow-consistency-smoke`

规则：

- skill 只做薄包装
- skill 不复制大段项目背景
- skill 调用共享脚本，不重复发明流程

### 3.4 subagent 层（隔离执行）
subagent 的角色：

- 处理局部验证
- 处理局部分析
- 避免污染主线程上下文

适合：

- 跑 smoke
- 检查 summary / packet / draft 是否 stale
- 单独分析某一条日志结论
- 单独执行 closeout 验证

不适合：

- 承担整个项目长期记忆
- 替代 canonical 文档体系

---

## 4. 单轮开发的基本单位

本项目以 **task round** 为基本单位，而不是以长对话为单位。

每轮任务必须有一个 `TASK-*.md` 文件。

### 4.1 任务类型

#### A. 分析决策轮
目标：

- 解释现象
- 区分已确认事实与待解释问题
- 判断下一轮是否需要代码改动

例如：

- `TASK-2026-03-30-002`
- `TASK-2026-03-31-004`

#### B. 实现轮
目标：

- 在明确边界内改代码
- 用最小验证或现有日志验证改动

例如：

- `TASK-2026-03-31-003`

#### C. 收尾
不单独编号，但每轮结束必须执行：

- refresh
- workflow smoke
- preview / apply
- canonical 同步

---

## 5. 启动规则

任何新一轮开始时，默认只读最小集合：

1. `AGENTS.md`
2. `project_memory/current_state.md`
3. 当前 active `TASK-*.md`

然后按 task 的 `Read first` 精确补读：

- 相关源码文件
- 相关 summary
- 相关 context packet
- 相关 workflow smoke

默认禁止：

- 完整 handoff 历史
- 完整 project_memory 历史
- 原始长日志
- 全部 skill 文档

---

## 6. 标准轮次流程

### Step 1：确认 active task
先确认：

- 当前 active task 是什么
- 这一轮属于分析轮还是实现轮
- 当前 canonical 是否已同步

### Step 2：生成或刷新 artifact
如果本轮涉及日志或结果分析，优先运行：

```bash
python scripts/agent/refresh_round_artifacts.py --mode draft --target <target> --log <train.log>