# Project Overview

- 项目目标: 在 Pointcept PTv3 backbone 上训练带 semantic boundary field 监督的点云语义分割模型。
- 当前阶段状态: `Stage-2 architecture rollout / verification phase`。
- `2.5` 阶段已完成，support 参数探索性实验已收束。
- 当前 active 主表达统一为 `axis + side + support`；作者口头中的 `magnitude` 在当前代码 / 文档同步中等价于 `support`。
- 当前正式主线与验证中心: `semseg-pt-v3m1-0-base-bf-edge-axis-side-train` 及其 smoke config；当前任务是把已落地的 `axis-side` 主线核证成可确认的 smoke / full-train 路线。
- 当前目标线: `semantic val_mIoU 73.8` 是安全线，`support-only best 74.6` 是主目标线。
- 作者已确认的实验事实: `support only, reg=1, cover=0.2 -> 74.6`；`reg=1, cover=0.25 -> 74.4`；`support + dir + dist -> 71`；`Stage-2 v1 best = 71.34, final = 68.31`；`Stage-2 v2 best = 72.38`。
- 当前路线状态:
  - `Stage-2 v1`: 历史 signed-direction full-train 路线，已确认失败。
  - `Stage-2 v2`: 历史 signed-direction full-train 路线，较 v1 改善但仍未过安全线。
  - `Route A`: 平行 signed-direction 路线；smoke 已通过，尚未确认 full train。
  - `B′`: 平行 signed-direction 路线；smoke 已通过，当前 workspace 未定位到 full-train 产物。
  - `axis-side`: 当前 active 主线；loss / evaluator / trainer / config 已落地，当前 workspace 仅定位到一次 smoke 启动日志，尚未确认 smoke 通过。
- 长期 memory 中的 baseline / full-train 数字可来自作者已确认结果；当前 workspace 是否保留完整 artifact 需单独标注。
- 非目标: 不重写 Pointcept 框架。
- 非目标: 不把单独的 semantic-only calibration 作为主任务；但 semantic `val_mIoU` 仍是当前阶段主评判指标与门槛。
- 非目标: 不扩展 test pipeline、result export、visualization export、distributed training。
