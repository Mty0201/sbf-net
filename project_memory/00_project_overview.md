# Project Overview

- 项目目标: 在 Pointcept PTv3 backbone 上训练带 semantic boundary field 监督的点云语义分割模型。
- 当前阶段状态: `Stage-2 entry preparation phase`。
- `2.5` 阶段已完成，support 参数探索性实验已收束。
- 当前阶段主线任务: 保持 `semantic val_mIoU 73.8` 作为安全线，并以超越 `support-only best 74.6` 作为当前 `Stage-2` 架构实验主目标。
- 当前最佳已确认路径: `support only, reg=1, cover=0.2 -> mIoU 74.6`；`reg=1, cover=0.25 -> mIoU 74.4` 可作为次优且稳定参考。
- 当前已确认失败路径: `support + dir + dist -> mIoU 71`，明显劣于 `semantic-only baseline 73.8`。
- `Stage-2` 的正式目标: 从架构改进角度重新接入 direction 项，而不是继续扫 support 参数。
- 当前已落地一条隔离的 `Stage-2` 最小实验路径: `SupportConditionedEdgeHead` + 独立 `stage2-support-dir` config；主线 config 与旧 `EdgeHead` 默认路径保持不变。
- 非目标: 不重写 Pointcept 框架。
- 非目标: 不把单独的 semantic-only calibration 作为主任务；但 semantic `val_mIoU` 仍是当前阶段主评判指标与门槛。
- 非目标: 不扩展 test pipeline、result export、visualization export、distributed training。
