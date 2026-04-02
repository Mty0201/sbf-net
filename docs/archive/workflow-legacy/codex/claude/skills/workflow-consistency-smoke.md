# Skill: workflow-consistency-smoke（SBF 项目专用）

检查当前 task 主链是否闭环、哪些文档 stale、哪些产物缺失。
对应 Codex 的 `.codex/skills/workflow-consistency-smoke/SKILL.md`。

---

## 触发场景

- 重要轮次收尾前，想确认当前链路是否已经闭环
- canonical 回写前，先看是否还有 missing / stale / conflict
- Claude / web ChatGPT / Codex 切换前，需要一份一致性体检报告

---

## 必要输入

1. `AGENTS.md`
2. `project_memory/current_state.md`
3. 当前 task 文件

---

## 调用方式

- 检查 Claude 当前链路：
  `python scripts/agent/workflow_consistency_smoke.py --target claude`
- 一次检查三侧：
  `python scripts/agent/workflow_consistency_smoke.py --target all`
- 结果写到 `reports/workflow_smokes/*.md`，优先看 `Overall Verdict` 和 `Suggested Fixes`

---

## 产出

- 一份结构化 markdown 报告，至少包含：
  - `Canonical Layer`
  - `Task Layer`
  - `Summary Layer`
  - `Packet Layer`
  - `Round Update Layer`
  - `Overall Verdict`
  - `Suggested Fixes`
- verdict 含义：
  - `PASS`：当前检查范围内无 missing / stale / conflict
  - `WARN`：存在缺失或 stale，但没有关键冲突
  - `FAIL`：关键 canonical 缺失、当前 task 无法解析、summary 层不可用，或出现明确冲突

---

## 不要做什么

- 不把它当成修复脚本
- 不把单 target 的 `PASS` 误当成全部 target 都已齐全
- 不扩展成跨多个历史 task 的大审计；当前只看当前 task 主链
