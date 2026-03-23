# Task Queue

1. 运行正式双任务配置，确认真实 BF 数据上首个 train/val 周期完整结束。
2. 校准 `dist_scale=0.08`，对照 `loss_dist`、`dist_error`、`dist_error_scaled` 是否同步稳定。
3. 检查 support 分支是否满足覆盖优先目标，重点看 `support_cover` 与 `support_error`。
4. 对比 dual-task 与 semantic-only 基线的 `val_mIoU` 和 per-class IoU，确认边界分支未破坏语义主线。
5. 补项目内 test pipeline 与 result export，保持不改 Pointcept 主框架。
