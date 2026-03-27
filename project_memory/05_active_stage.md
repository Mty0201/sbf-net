# Active Stage

- 当前阶段状态: `Stage-2 entry preparation phase`。
- `2.5` 阶段已完成，探索性实验已收束。
- 当前核心任务: 使用仓库内已落地的 `Stage-2` 最小实验路径，在真实开发环境中继续做 direction 重接入验证。
- 当前阶段主评判指标: semantic `val_mIoU`。
- 当前基线: `semantic-only val_mIoU = 73.8`。
- 当前 `2.5` 阶段最佳结果: `support-only(reg=1, cover=0.2) -> 74.6`；`reg=1, cover=0.25 -> 74.4` 为次优且稳定参考。
- 当前已确认失败结果: `support + dir + dist -> 71`，明显劣于基线。
- 当前阶段结论: `support` 参数设计已收束；`dist` 不是主要矛盾；`dir` 可学习但会伤害 semantic 主任务。
- `Stage-2` 即将开始，其核心目标是从架构改进角度重新接入 direction 项，而不是继续扫 support 参数。
- 当前仓库已落地一条隔离的 `Stage-2` 最小实验路径: `SupportConditionedEdgeHead` + 独立 `stage2-support-dir` config；主线 train config 未被污染。
- `Stage-2` 验收口径: `<73.8` 失败；`73.8 ~ 74.6` 仅说明 direction 不再明显伤害 semantic；只有 `val_mIoU > 74.6` 才说明 architecture improvement 让 direction 成为净增益项。
