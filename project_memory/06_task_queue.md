# Task Queue

1. 围绕 `shared backbone + thin edge head` 的当前结构，明确 direction 接入后伤害 semantic 的最可能耦合位置与结构性原因。
2. 提出最小侵入式架构改进方向，只允许围绕分支解耦、特征缓冲或边界分支能力补强，不重写整体系统。
3. 为下一轮设计/实现先固定首个成功门槛: 接入 direction supervision 后，semantic `val_mIoU` 不低于 `semantic-only` 基线 `73.8`。
4. 仅在 architecture 尝试之后仍有证据指向 loss 主矛盾时，才重新打开 support / dist 相关 sweep；当前不继续 2.5 阶段式扫参。
