# 0. 项目一句话定义

`semantic-boundary-field` 的目标是在 Pointcept PTv3 语义分割主干上加入 boundary field 监督，但当前阶段的主评判仍是 semantic segmentation 的 `val_mIoU`。

## 1. 当前阶段定位

- 当前阶段: `Stage-3 Architecture Design Phase`
- 已结束阶段: `Stage-2.5 loss design / support 收束阶段`
- 当前不做: 不继续做 loss 细碎扫参，不扩展到 test/export/visualization

## 2. 当前已确认结论

- support 分支确实能带来收益。
- 当前 support loss 已基本收束到 `support_reg_weight=1.0`、`support_cover_weight≈0.25`。
- 当前对 support 的有效解释是: `reg` 为主，`cover` 为弱辅助；`cover` 不能去掉，但也不能过强。
- `dist` 以及历史 `vec` 相关项的损失会很快降到较小量级，目前没有证据表明它们是当前主矛盾。
- direction 不是学不会；不训练时 `dir_cosine` 约在 0 附近，训练后可提升到约 0.6。
- 但在现有 `shared backbone + thin edge head` 架构下，一旦真实接入 direction supervision，semantic `val_mIoU` 会显著下降到约 71。
- 当前最合理的工作假设是: 现有结构的分支耦合过强，direction 优化压力会破坏 semantic 表征。

## 3. 下一阶段唯一核心问题

在不低于 `semantic-only` 基线 `val_mIoU=73.8` 的前提下，成功接入 direction supervision。

补充约束:

- 当前 boundary 是 semantic boundary，因此 semantic `val_mIoU` 仍是主评判指标。
- 下一阶段优先考虑最小侵入式架构改动，而不是重新发明整个系统。
- 当前需要先解决 shared backbone 下的分支耦合问题。

## 4. 建议下一窗口先读

- `project_memory/01_current_architecture.md`
- `project_memory/02_loss_design.md`
- `project_memory/05_active_stage.md`
- `project_memory/06_task_queue.md`
- `project_memory/90_archived_decisions.md`
