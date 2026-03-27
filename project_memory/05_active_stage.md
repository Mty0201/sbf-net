# Active Stage

- 阶段名: `Stage-3 Architecture Design Phase`。
- 2.5 阶段已收束: support loss 已基本收束，direction 可学习性已确认，loss 设计不再是当前主线问题。
- 当前唯一核心问题: 如何在不破坏 semantic 主任务的前提下安全接入 direction supervision。
- 当前阶段主评判指标: semantic `val_mIoU`。
- 当前基线: `semantic-only val_mIoU = 73.8`。
- 当前阶段门槛: 架构改进必须先满足“接入 direction supervision 且不低于 semantic-only baseline”。
- 当前工作假设: 现有 `shared backbone + thin edge head` 结构导致 semantic 与 boundary 特征分化不足，direction 的优化压力会破坏 semantic 表征。
- 注意: 上述结构性解释目前是工作假设，不是已经完全证明的最终定论。
- 当前阶段策略: 优先最小侵入式架构改动，先解决分支耦合问题，不继续做 loss 细碎扫参。
