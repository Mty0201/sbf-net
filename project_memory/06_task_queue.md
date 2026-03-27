# Task Queue

1. 把 `2.5` 阶段结论视为已完成事实，不再继续做 support 参数 sweep；当前主参考点固定为 `support-only(reg=1, cover=0.2)=74.6`，`reg=1, cover=0.25)=74.4` 作为次优稳定参考。
2. 基于已落地的 `Stage-2` 最小实验路径，在真实开发环境 `ptv3` 中继续验证 direction 重接入效果。
3. 当前 `Stage-2` 的主目标不是只回到 `73.8`，而是超越 `support-only best 74.6`；`73.8` 仅作为安全线。
4. 当前不做: 不把当前问题重新定义成 loss sweep，不重开 `dist` 主线，不顺手扩展到 Pointcept 改写、test/export/visualization。
