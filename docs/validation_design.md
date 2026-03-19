# Validation Design (Stage 1)

## 1. 设计目标

当前阶段的 validation 只服务于“最小训练闭环验证”。

本阶段不追求完整评估体系，也不追求一次性覆盖所有潜在分析维度。当前优先目标是：

- 与训练 loss 保持一致
- 指标具备可解释性
- 实现简单、稳定、便于后续最小接入

因此，当前 validation 设计强调“够用、清晰、可落地”，而不是复杂化。

## 2. Semantic Validation Metrics

当前 semantic 分支的 validation 指标完全沿用 Pointcept 默认语义分割指标，不做任何改动。

最小指标集合如下：

- `val_mIoU`
- `val_mAcc`
- `val_allAcc`

说明：

- `val_mIoU` 反映类别层面的平均交并比，是当前 semantic 分支的核心指标
- `val_mAcc` 反映类别平均准确率
- `val_allAcc` 反映整体点级准确率

当前阶段明确：

- 主指标：`val_mIoU`
- 用于 best checkpoint 判定：`val_mIoU`

## 3. Edge Validation Metrics

Edge prediction tensor definition:

edge_pred = [vec_x, vec_y, vec_z, strength, mask_logit]

对应：

- edge_pred[:, 0:3] → vector
- edge_pred[:, 3]   → strength
- edge_pred[:, 4]   → mask (logit)

### 3.1 Loss-based Metrics

当前 edge 分支最小 loss 型 validation 指标如下：

- `val_loss_mask`
- `val_loss_vec`
- `val_loss_strength`

说明：

- 这些指标与当前训练 loss 完全一致
- 它们首先服务于观察训练是否收敛
- 第一阶段优先保证 validation 与训练目标一致，而不是额外设计复杂评价函数

### 3.2 Mask Metrics

当前 mask 分支最小解释性指标如下：

- `mask_precision`
- `mask_recall`
- `mask_f1`

当前阶段的定义方式为：

- 对 `mask_pred_logit` 先做 `sigmoid`
- 使用固定阈值 `threshold = 0.5`
- 将结果转成二值预测
- 与 `mask_gt` 计算 `precision / recall / f1`

当前必须明确：

- `threshold = 0.5` 是当前阶段的固定策略
- 该策略优先服务于第一阶段的简单、稳定观察
- 后续如有必要，可以再讨论 soft mask 评估或阈值策略细化

### 3.3 Vector & Strength Metrics

当前 vec / strength 分支最小解释性指标如下：

- `vec_error_masked`
- `strength_error_masked`

定义方式：

- 使用 `MSE`
- 只在 `mask_gt == 1` 内统计

说明：

- 这两个指标与当前训练监督完全一致
- 第一阶段不引入额外误差定义，如 `L1`、`RMSE` 或更复杂的方向指标
- 当前优先保证定义简单、稳定、与训练目标一致

当当前样本中不存在 mask_gt == 1 的点时：

- vec_error_masked = 0
- strength_error_masked = 0

该规则用于避免 NaN，并保持指标统计稳定。

## 4. Mask Gating Rule in Validation

当前 validation 的 gating 规则必须与 training 保持完全一致。

具体规则如下：

- `mask loss`：全点计算
- `vec / strength`：只在 `mask_gt == 1` 内计算

这里需要强调：

- validation 不引入额外 gating 逻辑
- validation 不改变当前训练监督的参与范围
- 训练与验证在 gating 规则上必须保持一致，以避免解释偏差

## 5. Best Checkpoint Policy

当前阶段的 best checkpoint 仅按 semantic validation 的 mIoU 决定。

edge 分支相关指标在 validation 中全部记录，包括 mask、vec、strength 的损失与解释性指标，但这些 edge 指标在第一阶段不参与 best checkpoint 判定。

待 edge 指标定义与训练行为稳定后，再讨论是否引入联合指标或新的主指标。

## 6. Relationship to Pointcept

当前 validation 机制的组织方式参考 Pointcept 的现有设计，重点参考以下两层：

- `evaluator`
- `hook (after_epoch)`

当前阶段的扩展重点仅在“指标内容”上，而不在“核心机制”上。

换句话说：

- validation 的触发方式继续参考 Pointcept 训练后 `after_epoch` 的 evaluator 思路
- 当前仅扩展我们自己的双任务指标集合
- 不修改 Pointcept 的核心 validation 机制

## 7. Out of Scope (当前不做)

当前阶段明确不做以下内容：

- 不设计复杂联合指标
- 不做 edge 主指标排序
- 不做 soft mask 评估
- 不做 test 输出格式设计
- 不做可视化导出
- 不做多数据集适配
