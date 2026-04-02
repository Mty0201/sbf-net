# Skill: refresh-round-artifacts（SBF 项目专用）

顺序刷新 `summary -> context packet -> round update draft`。
对应 Codex 的 `.codex/skills/refresh-round-artifacts/SKILL.md`。

---

## 触发场景

- 本轮日志、摘要或任务状态刚更新，需要统一刷新派生产物
- 准备做 `preview` 或 `apply`，但不想手工分步调用多个脚本
- 怀疑 packet / round draft 已经 stale，需要走单入口重建

---

## 必要输入

1. `AGENTS.md`
2. `project_memory/current_state.md`
3. 当前 task 文件
4. 若本轮有新日志，再提供对应 `train.log` 路径

---

## 调用方式

- Claude 默认目标用：
  `python scripts/agent/refresh_round_artifacts.py --mode draft --target claude`
- 若本轮有新日志，同一条命令补：
  `--log <path/to/train.log>`
- 需要看 fixed-scope diff 时，用：
  `--mode preview`
- 只有用户明确要求落盘、且已经过 preview 时，才用：
  `--mode apply --confirm`

---

## 产出

- `reports/log_summaries/*.summary.md`
- `reports/log_summaries/*.summary.json`
- `reports/context_packets/*.context_packet.md`
- `reports/round_updates/*.round_update.draft.md`
- 一份终端步骤摘要：`EXECUTED / SKIPPED / FAILED / NO-OP`

---

## 不要做什么

- 不把它当成业务代码修改入口
- 不绕开顺序手工先调 `preview/apply`
- 不在没有 `--confirm` 的情况下执行 `apply`
- 不在摘要足够时回到先读原始长日志
