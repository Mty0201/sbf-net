# Project

- 项目: `semantic-boundary-field`
- 目标: 在 Pointcept PTv3 backbone 上训练带 semantic boundary field 监督的点云语义分割模型。
- 当前主线: stage-1 双任务训练，联合学习 `semantic + direction + distance + support`。
- Pointcept 角色: 只作为上游依赖，不是主项目。

# Just Finished

- 当前阶段已收束为 `Stage-1 Dual-Task Training Phase`。
- 当前正式训练入口已固定为 `scripts/train/train.py`。
- 当前正式训练配置已固定为 `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py`。
- 当前系统快照与约束已固化，可直接讨论下一主题。

# Confirmed Decisions

- 主模型固定为 shared backbone 双分支结构。
- edge 分支固定预测 `direction + distance + support`。
- `valid` 只作为 GT 有效监督域，不是预测目标。
- support 固定为 coarse boundary proposal，不做精确边界重建。
- direction 固定为 cosine 监督。
- distance 固定为显式物理步长监督。
- distance 训练固定使用 `dist_scale=0.08` 的线性重标定。
- 训练入口固定使用项目内 trainer，不使用 Pointcept trainer。

# Current System Snapshot

- backbone: `PT-v3m1`
- semantic head: 输出 `seg_logits`
- edge head: 输出 `dir_pred(3) + dist_pred(1) + support_pred(1)`
- 总损失: `loss = loss_semantic + loss_edge`
- edge 损失: `loss_edge = loss_support + loss_dir + loss_dist`
- semantic loss: `CrossEntropy + Lovasz`
- support loss: coverage-first + regression auxiliary
- direction loss: cosine consistency
- distance loss: `SmoothL1(dist_pred / dist_scale, dist_gt / dist_scale)`
- 关键验收日志: `val_mIoU`, `support_cover`, `dir_cosine`, `dist_error`

# Next Discussion Topic

- 优先讨论 `distance` 是否真正学到。
- 核心切口: `loss_dist`、`dist_error`、`dist_error_scaled` 是否同步。
- 次级切口: support 的 coverage-first 语义是否足够转化为 semantic 收益。

# Core Questions

- 当前 `distance` 是真实收敛，还是只在重标定空间里看起来收敛。
- 当前 support 作为 coarse proposal，是否弱于需要的边界先验。
- 当前双任务分支是否已经开始影响 semantic 主任务。
- 当前 dual-task 与 semantic-only 的差距，主要来自 support、direction 还是 distance。

# Hard Constraints

- 不把 Pointcept 当作主项目讨论。
- 不重写 Pointcept trainer 或 Pointcept 源码。
- 不把 semantic-only 当作主任务替代。
- 不扩展到 test pipeline、result export、visualization、distributed training。
- 讨论必须基于当前已确认系统，不回到已废弃方案。
