# Project Overview

- 项目目标: 在 Pointcept PTv3 backbone 上训练带 semantic boundary field 监督的点云语义分割模型。
- 当前阶段主线任务: `Stage-3 Architecture Design Phase`，在不低于 `semantic-only` 基线 `val_mIoU=73.8` 的前提下，安全接入 direction supervision。
- 已结束阶段: `Stage-2.5 loss design / support 收束阶段`，当前不再继续围绕 support / dist 做细碎 loss 扫参。
- 非目标: 不重写 Pointcept 框架。
- 非目标: 不把单独的 semantic-only calibration 作为主任务；但 semantic `val_mIoU` 仍是当前阶段主评判指标与门槛。
- 非目标: 不扩展 test pipeline、result export、visualization export、distributed training。
