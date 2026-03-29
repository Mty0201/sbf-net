# Active Stage

- 当前阶段状态: `axis-side implementation verification phase`。
- `2.5` 阶段已完成，support 参数探索已收束。
- `Stage-2 v1` 已完成 full train: 最佳 `val_mIoU = 71.34`（epoch 36），最终 `68.31`（epoch 100），未过安全线。
- `Stage-2 v2` 已完成 full train: best `72.38`，未过 `73.8` 安全线；v2 相比 v1 有改善。
- 当前阶段主评判指标: semantic `val_mIoU`。
- 当前基线: `semantic-only val_mIoU = 73.8`（安全线），`support-only best = 74.6`（当前主目标）。
- 验收口径: `<73.8` 失败；`73.8 ~ 74.6` 说明 direction 不再伤害 semantic；`>74.6` 说明 direction 成为净增益项。

## Route A 当前状态

- Route A 第一版最小实现已落地，包括:
  - `RouteASemanticBoundaryLoss`: 在 `SemanticBoundaryLoss` 基础上加 local within-basin direction coherence
  - `BFDataset` 支持可选 sidecar `edge_support_id.npy`
  - trainer 支持向 loss 传递 `coord / support_id / offset`
  - 独立 config: `route-a-train.py` / `route-a-train-smoke.py`
  - 数据脚本: `add_support_id_to_edge_dataset.py`
- Route A smoke 已通过（`loss_coherence` 非零）。
- Route A 尚未完成正式 full train 验证。
- Route A 第一轮实验建议 `mix_prob=0.0`（避免 support_id 跨场景冲突）。

## B′ 当前状态

- `SemanticBoundaryLoss` 新增 `support_weighted_edge` 开关（默认 `False`，不影响旧路径）。
- B′ 开启时: `loss_dir` 和 `loss_dist` 改为 `support_gt * valid_gt` 归一化加权平均。
- B′ 不引入 `support_id`，不加 basin coherence，不改模型结构。
- 独立 config: `stage2-bprime-train.py` / `stage2-bprime-train-smoke.py`。
- B′ smoke 已通过。
- B′ 具备进入 full train 的条件。
- 当前 workspace 中尚未定位到可确认的 `B′` full train 输出；不要把 `B′ full train ≈72.8` 写成已证实事实。

## Axis-Side 当前状态

- 当前 working tree 已落地 `AxisSideSemanticBoundaryLoss`、`AxisSideEvaluator`、`axis-side-train.py` 和 `axis-side-train-smoke.py`。
- `axis-side` 复用 `Stage-2 v2` 模型与现有五通道 edge 输出，不改 `edge.npy` 六列格式，也不引入新的 sidecar。
- 当前 workspace 尚未确认成功的 `axis-side` smoke 输出。
- 本轮在 CPU-only `ptv3` 环境复跑 `axis-side` smoke 时，PTv3 / spconv 在首个训练 step 前因缺少 NVIDIA driver 失败；因此 `axis-side smoke passed` 当前不能写成已确认事实。

## 当前优先级

- 先在 CUDA-enabled `ptv3` 环境确认 `axis-side` smoke。
- 仅在 `axis-side` smoke 确认通过后，再安排 `axis-side` full train。
- 若需要把 `B′ full train` 写成已完成，先定位对应 log / output 证据。
