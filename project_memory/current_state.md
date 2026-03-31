# Current State

## 当前 task

- 当前单轮执行中心：`project_memory/tasks/TASK-2026-03-31-004.md`
- `TASK-2026-03-31-003` 已完成验证口径修正并把 workflow smoke 收敛到 `PASS`；当前下一轮聚焦 full-train 低于 baseline 的根因分析与决策。

## 当前有效事实

- 唯一允许主动维护的项目根是 `semantic-boundary-field`；`Pointcept` 只是宿主依赖，默认只读。
- 当前阶段状态是 `Stage-2 architecture rollout / verification phase`。
- 当前 active 主线统一为 `axis + side + support`；作者口头中的 `magnitude` 在当前代码 / 文档同步中等价于 `support`。
- 当前分析中心已切到 `semseg-pt-v3m1-0-base-bf-edge-support-shape-train` 的 full train 结果；`axis-side` smoke 作为前置准入证据保留。
- 当前阶段主评判指标仍是 semantic `val_mIoU`。
- 当前不做：不重新扫 support 参数，不重开 `dist` 主线，不扩展到 Pointcept 改写、test/export/visualization。

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
- workflow smoke 达到 `PASS`。

## 下一步

- 第一优先级：在干净口径下分析为什么当前 full-train best scalar `val_mIoU = 0.7316` 低于 `support-only baseline = 74.6`。
- 第二优先级：在调参重训、结构改动和补额外验证之间选定下一轮唯一优先路线。
- 若任务需要补充上下文，只按需读取一个专题文件：
  - 架构：`project_memory/01_current_architecture.md`
  - loss：`project_memory/02_loss_design.md`
  - 训练 / 运行：`project_memory/04_training_rules.md`
