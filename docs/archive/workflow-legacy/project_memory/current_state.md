# Current State

## 当前 task

- 当前单轮执行中心：`project_memory/tasks/TASK-2026-03-31-005.md`
- `TASK-2026-03-31-004` 已完成分析决策收束，并已把 `TASK-2026-03-31-005` 从 candidate brief 升格为当前 active formal analysis task；本轮仍不进入实现。

## 当前有效事实

- 唯一允许主动维护的项目根是 `semantic-boundary-field`；`Pointcept` 只是宿主依赖，默认只读。
- 当前阶段状态是 `Stage-2 architecture rollout / verification phase`。
- 当前 active 主线统一为 `axis + side + support`；作者口头中的 `magnitude` 在当前代码 / 文档同步中等价于 `support`。
- 当前分析中心保留 `semseg-pt-v3m1-0-base-bf-edge-support-shape-train` 的 full train 结果作为失败旁证；当前 active rollout 仍是 `axis + side + support`，`axis-side` smoke 作为前置准入证据保留。
- 当前阶段主评判指标仍是 semantic `val_mIoU`。
- 当前不做：不重新扫 support 参数，不重开 `dist` 主线，不扩展到 Pointcept 改写、test/export/visualization。
- 旧失败 run 中 `dist` 诊断值曾改善，但它不是该轮主训练目标，不能与当前 active rollout 的监督结构混写。

## 已确认结果

- `semantic-only baseline = 73.8`。
- `support-only(reg=1, cover=0.2) = 74.6`，当前最佳。
- `support + dir + dist = 71`，是旧 signed-direction 结构下的失败事实。
- `Stage-2 v1` full train: best `71.34`，final `68.31`。
- `Stage-2 v2` full train: best `72.38`。

## 当前 workspace 状态

- `axis-side` 的 loss / evaluator / trainer / config 修改都已在当前 working tree 中落地。
- 当前 `axis-side` smoke 已有一份可直接复核的摘要证据：最后一个有效 session 为 `validated_with_checkpoints`，device `cuda`，并在 `/home/mty/Python_Proj/for_build_seg/Pointcept/sbf-net/outputs/semantic_boundary_support_shape_train/model/` 下生成 `model_best.pth` 与 `model_last.pth`。
- 同一 `train.log` 中共检测到 `1` 次启动 / 会话；当前以最后一个有效 session 作为 smoke 结论。
- `Route A` smoke 已通过，且当前 workspace 中可直接复核 `train.log + model_best.pth + model_last.pth`。
- `B′` smoke 已通过，且当前 workspace 中可直接复核 `train.log + model_best.pth + model_last.pth`。
- `outputs/semseg-pt-v3m1-0-base-bf-edge-support-shape-train/train.log` 已存在；对应权威摘要已进入当前 canonical 分析链，当前剩余问题是根因分析与路线选择。
- `TASK-2026-03-31-003` 已完成验证口径修正：summary / packet / smoke 链路已对齐到 scalar `val_mIoU` 为权威口径。
- 当前 full-train 权威指标：latest scalar `val_mIoU = 0.7085`，best scalar `val_mIoU = 0.7316`（epoch 65）。
- 当前 `TASK-2026-03-31-004` 对应 workflow smoke 为 `WARN`，且仅 `1` 个 stale（`handoff/latest_round.md` 未同步），不影响本轮对 full-train 的性能判断。
- `TASK-2026-03-31-005` 已正式落盘为 active analysis task；其来源 candidate brief 保留在 `reports/round_updates/TASK-2026-03-31-005.codex.candidate_brief.draft.md` 作为 checkpoint artifact。
- `TASK-2026-03-31-005` 当前已锁定的唯一优先结构方向是：保留 support 对 boundary branch 的直接监督，但 axis / side 只能在 support-conditioned 私有分支上学习，其损失不再直接回传 shared backbone。

## 下一步

- 第一优先级：在 `TASK-2026-03-31-005` 内把单一路线正式收敛为后续实现轮的唯一入口，写清 why / boundary / validation / risks。
- 第二优先级：保持本轮为 formal analysis，不进入实现、不写代码、不跑训练。
- 若任务需要补充上下文，只按需读取一个专题文件：
  - 架构：`project_memory/01_current_architecture.md`
  - loss：`project_memory/02_loss_design.md`
  - 训练 / 运行：`project_memory/04_training_rules.md`
