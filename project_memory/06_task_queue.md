# Task Queue

1. 把 `2.5` 阶段结论视为已完成事实，不再继续做 support 参数 sweep；当前主参考点固定为 `support-only(reg=1, cover=0.2)=74.5`，`reg=1, cover=0.25)=74.4` 作为次优稳定参考。
2. 正式准备进入 `Stage-2 architecture improvement phase`，目标是在不破坏 semantic 主任务的前提下重新接入 direction 项。
3. `Stage-2` 的首要工作是围绕当前 shared backbone + thin multi-head 结构的特征竞争问题提出架构改进方案，而不是继续调 support 参数。
4. 当前不做: 不把当前问题重新定义成 loss sweep，不重开 `dist` 主线，不顺手扩展到 Pointcept 改写、test/export/visualization。
