# Active Stage

- 阶段名: `Stage-1 Dual-Task Training Phase`。
- 唯一核心任务: 在当前 `direction + distance + support` GT 定义下跑通并稳定化正式双任务训练闭环。
- 验收标准: `scripts/train/train.py` 能用 full config 完成至少一个 displayed epoch 的 train + val。
- 验收标准: 验证日志同时出现 `val_mIoU`, `support_cover`, `dir_cosine`, `dist_error`。
- 验收标准: 输出 `outputs/semantic_boundary_train/model/model_last.pth`。
- 验收标准: 若 `val_mIoU` 刷新最佳值，同时输出 `model_best.pth`。
