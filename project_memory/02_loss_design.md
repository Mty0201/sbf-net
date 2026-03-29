# Loss Design

## Shared Semantic Term

- 总损失: `loss = loss_semantic + loss_edge`。
- semantic loss: `CrossEntropy(ignore_index=-1) + LovaszLoss`。

## Current Active Mainline: Axis-Side

- 当前 active edge loss 主表达: `loss_edge = loss_support + loss_axis + loss_side`。
- 当前 active edge 权重: `support_weight=1.0`, `axis_weight=1.0`, `side_weight=1.0`。
- 当前 active 主表达统一为 `axis + side + support`；作者口头中的 `magnitude` 在当前代码 / 文档同步中等价于 `support`。
- support 语义: 预测边界邻域覆盖，不预测精确外轮廓。
- support 预测: `support_pred = sigmoid(support_logit)`。
- support target: `support_gt * valid_gt`。
- support cover loss: 对 `valid_gt` 区域做 Tversky loss。
- Tversky 参数: `alpha=0.3`, `beta=0.7`。
- support reg loss: 对 `support_pred` 与 `support_target` 做加权 `SmoothL1`。
- support 合成: `loss_support = 1.0 * loss_support_reg + 0.25 * loss_support_cover`。
- 当前 support sweep 已收束；已确认结果是 `semantic-only=73.8`, `support-only(reg=1, cover=0.2)=74.6`, `support-only(reg=1, cover=0.25)=74.4`, `support-only(reg=1, cover=0.3)=73.7`，其它 cover 参数均不优于 `73.8`。
- 当前对 support loss 的有效解释: `reg` 仍是主项，`cover` 只能做弱辅助；当前问题已不再是 support 参数设计。
- axis 语义: 预测 unsigned direction axis，与 GT 方向做 sign-invariant cosine。
- axis 有效域: `valid_gt > 0.5` 且 `dist_gt > tau_dir`。
- axis 阈值: `tau_dir=1e-3`。
- axis loss: `1 - |cos(axis_pred_unit, dir_gt_unit)|`。
- side 语义: 预测 support 两侧的二值 hemisphere 标签。
- side loss: 对运行时由 `dir_gt` 导出的 `side GT` 做 BCE。
- `side_support_threshold` 默认 `0.0`。
- `dir_cosine` 当前保留为 signed comparison metric，用于与历史 signed-direction 路线做对照；它不再是当前 active 主线的主损失项。
- 当前 axis-side full / smoke config 固定为: `support_reg_weight=1.0`, `support_cover_weight=0.2`, `axis_weight=1.0`, `side_weight=1.0`。
- 当前 `Stage-2` 的验收口径已更新为: `73.8` 只是安全线，只有 `val_mIoU > 74.6` 才能说明 architecture improvement 让 direction 成为净增益项。
- 阶段结论: `2.5` 阶段已完成；当前问题已从 support 参数设计转向架构问题，当前阶段是 `Stage-2 architecture rollout / verification phase`，active 主线是 `axis + side + support`。

## Historical Signed-Direction Routes

- 历史 signed-direction route 的 edge loss 是 `loss_edge = loss_support + loss_dir + loss_dist`。
- 历史 signed-direction 权重: `support_weight=1.0`, `dir_weight=1.0`, `dist_weight=1.0`；`Stage-2` 独立 config 中固定 `support_reg_weight=1.0`, `support_cover_weight=0.2`, `dir_weight=1.0`, `dist_weight=0.0`。
- direction 语义: 预测单位方向，与 GT 方向做余弦一致性。
- direction loss: `1 - cosine(dir_pred_unit, dir_gt_unit)`。
- 当前已确认: direction 是可以学习的；不训练时 `dir_cosine` 约在 0 附近，训练后可提升到约 `0.6`。
- 当前已确认: 在现有 shared-backbone + thin-head 架构下，`support + dir + dist` 的 `val_mIoU` 约为 `71`，明显劣于 `semantic-only baseline 73.8`。
- 当前有效结论: 旧 signed-direction 路线中的 `dir` 学习会以 semantic 主任务性能为代价。
- distance 语义: 预测点到最近 support 的物理距离。
- distance loss: 对 `dist_pred / dist_scale` 与 `dist_gt / dist_scale` 做加权 `SmoothL1`。
- distance 重标定: `dist_scale=0.08`。
- distance 指标: `dist_error` 保持原始物理单位，不跟随缩放。
- 当前判断: 旧 signed-direction 路线中的 `dist` 项会在不到一个 epoch 的训练中快速下降到极小值（约 `0.0002`），当前不是主要矛盾。
- 当前已确认 `Stage-2 v1` 真实 full train 结果: 在上述 signed-direction 设定下，最佳 `val_mIoU = 71.34`（epoch 36），最终 `68.31`（epoch 100）。
- 当前已确认 `Stage-2 v1` 训练 / 验证观察: train `dir_cosine` 后期可到约 `0.65 ~ 0.75`，但验证期按样本均值统计的 `dir_cosine` 在最佳点约 `0.27`、末期约 `0.31`；同一 run 中验证 `support_cover` 从首轮约 `0.72` 下降到最佳点约 `0.53`、末期约 `0.51`，`support_error` 从首轮约 `0.077` 上升到最佳点约 `0.142`、末期约 `0.140`。
- `Stage-2 v2` full train 结果: best `72.38`，未过 `73.8` 安全线；v2 相比 v1 有改善。
- `B′` 路线: `SemanticBoundaryLoss` 新增 `support_weighted_edge` 开关（默认 `False`）。开启时 `loss_dir` 改为 `weighted_mean(dir_error, support_gt * valid_gt * (dist_gt > tau_dir))`，`loss_dist` 改为 `weighted_mean(dist_error, support_gt * valid_gt)`。不引入 `support_id`，不改模型结构。
- `Route A` 路线: `RouteASemanticBoundaryLoss` 继承 `SemanticBoundaryLoss`，增加 `coherence_weight * loss_coherence`。需要 sidecar `edge_support_id.npy`。参数: `coherence_weight=0.1`, `local_radius=0.30`, `max_points_per_basin=100`。
