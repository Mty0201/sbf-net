# Project Overview

- 项目目标: 在 Pointcept PTv3 backbone 上训练带 semantic boundary field 监督的点云语义分割模型。
- 当前阶段状态: `B′ vs A experiment comparison phase`。
- `2.5` 阶段已完成，support 参数探索性实验已收束。
- 当前阶段主线任务: 保持 `semantic val_mIoU 73.8` 作为安全线，并以超越 `support-only best 74.6` 作为当前主目标。
- 当前最佳已确认路径: `support only, reg=1, cover=0.2 -> mIoU 74.6`；`reg=1, cover=0.25 -> mIoU 74.4` 可作为次优且稳定参考。
- 当前已确认失败路径: `support + dir + dist -> mIoU 71`，明显劣于 `semantic-only baseline 73.8`。
- `Stage-2 v1` full train 结果: 最佳 `val_mIoU = 71.34`（epoch 36），最终 `68.31`（epoch 100），未过安全线。
- `Stage-2 v2` full train 结果: 有效 best 约 `72.5`，未过 `73.8` 安全线；v2 相比 v1 有改善。
- 当前已落地的实验路径:
  - `Stage-2 v1`: `SupportConditionedEdgeHead` + 独立 config。
  - `Stage-2 v2`: post-backbone branch split + 独立 config。
  - `Route A`: `RouteASemanticBoundaryLoss` + sidecar `edge_support_id.npy` + basin coherence + 独立 config。Smoke 已通过，尚未 full train。
  - `B′`: `SemanticBoundaryLoss(support_weighted_edge=True)` + 独立 config。Smoke 已通过，具备 full train 条件。
- 非目标: 不重写 Pointcept 框架。
- 非目标: 不把单独的 semantic-only calibration 作为主任务；但 semantic `val_mIoU` 仍是当前阶段主评判指标与门槛。
- 非目标: 不扩展 test pipeline、result export、visualization export、distributed training。
