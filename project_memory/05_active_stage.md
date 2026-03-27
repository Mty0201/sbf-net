# Active Stage

- 当前阶段状态: `Stage-2 entry preparation phase`。
- `2.5` 阶段已完成，探索性实验已收束。
- 当前唯一核心任务: 完成 `2.5` 阶段结论固化，并切换到准备正式进入 `Stage-2` 的状态。
- 当前阶段主评判指标: semantic `val_mIoU`。
- 当前基线: `semantic-only val_mIoU = 73.8`。
- 当前 `2.5` 阶段最佳结果: `support-only(reg=1, cover=0.2) -> 74.5`；`reg=1, cover=0.25 -> 74.4` 为次优且稳定参考。
- 当前已确认失败结果: `support + dir + dist -> 71`，明显劣于基线。
- 当前阶段结论: `support` 参数设计已收束；`dist` 不是主要矛盾；`dir` 可学习但会伤害 semantic 主任务。
- `Stage-2` 即将开始，其核心目标是从架构改进角度重新接入 direction 项，而不是继续扫 support 参数。
- `Stage-2` 成功门槛: direction 项重新接入后，semantic `val_mIoU` 不能继续跌破 `semantic-only baseline 73.8`。
