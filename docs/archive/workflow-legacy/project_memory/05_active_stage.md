# Active Stage

- 当前阶段状态: `Stage-2 architecture rollout / verification phase`。
- `2.5` 阶段已完成，support 参数探索已收束。
- 当前正式主线: `axis + side + support`；当前验证中心是 `axis-side` full / smoke config，不再继续扫 support 参数。
- 当前阶段主评判指标: semantic `val_mIoU`。
- 当前基线: `semantic-only val_mIoU = 73.8`（安全线），`support-only best = 74.6`（当前主目标）。
- 验收口径: `<73.8` 失败；`73.8 ~ 74.6` 说明 direction 不再伤害 semantic；`>74.6` 说明 direction 成为净增益项。
- 当前不做: 不重新扫 support 参数，不重开 `dist` 主线，不扩展到 Pointcept 改写、test/export/visualization。

## 作者已确认的实验事实

- `semantic-only baseline = 73.8`。
- `support-only(reg=1, cover=0.2) = 74.6`，当前最佳。
- `support + dir + dist = 71`，是旧 signed-direction 结构下的失败事实。
- `Stage-2 v1` 已完成 full train: 最佳 `val_mIoU = 71.34`（epoch 36），最终 `68.31`（epoch 100），未过安全线。
- `Stage-2 v2` 已完成 full train: best `72.38`，未过 `73.8` 安全线；v2 相比 v1 有改善。

## 当前 workspace 可直接复核的本地产物

- `Route A` smoke 输出包含 `train.log + model_best.pth + model_last.pth`，且 `loss_coherence` 非零。
- `B′` smoke 输出包含 `train.log + model_best.pth + model_last.pth`。
- `axis-side` 的 loss / evaluator / trainer / config 修改都已在当前 working tree 中落地。
- 当前 `axis-side` smoke 输出仅包含一次启动日志（`Device: cpu`），未见 train / val 行或 checkpoint。

## Route A 当前状态

- `Route A` 是平行 signed-direction 路线，不是当前 active 主线。
- `RouteASemanticBoundaryLoss`、可选 `edge_support_id.npy` sidecar、trainer `support_id` 透传和独立 config 都已落地。
- Route A smoke 已通过（`loss_coherence` 非零）。
- Route A 尚未完成正式 full train 验证。
- Route A 第一轮实验建议 `mix_prob=0.0`（避免 support_id 跨场景冲突）。

## B′ 当前状态

- `B′` 是平行 signed-direction 路线，不是当前 active 主线。
- `SemanticBoundaryLoss` 新增 `support_weighted_edge` 开关（默认 `False`，不影响旧路径）。
- B′ 开启时: `loss_dir` 和 `loss_dist` 改为 `support_gt * valid_gt` 归一化加权平均。
- B′ 不引入 `support_id`，不加 basin coherence，不改模型结构。
- 独立 config: `stage2-bprime-train.py` / `stage2-bprime-train-smoke.py`。
- B′ smoke 已通过。
- B′ 具备进入 full train 的条件。
- 当前 workspace 中尚未定位到可确认的 `B′` full train 输出；若要写入“workspace 可直接复核”，必须先定位对应证据。

## Axis-Side 当前状态

- `axis-side` 是当前 active 主线，不是旁路线。
- 当前 working tree 已落地 `AxisSideSemanticBoundaryLoss`、`AxisSideEvaluator`、trainer 的 axis-side train / val summary 分支，以及 `axis-side-train.py` / `axis-side-train-smoke.py`。
- `axis-side` 复用 `Stage-2 v2` 模型与现有五通道 edge 输出，不改 `edge.npy` 六列格式，也不引入新的 sidecar。
- 当前 workspace 仅定位到一次 `axis-side` smoke 启动日志（`Device: cpu`），未见 train / val 行或 checkpoint。
- 因此 `axis-side smoke passed` 当前不能写成已确认事实；当前状态应写成“已实现、待核证”。

## 当前优先级

- 先在 CUDA-enabled `ptv3` 环境确认 `axis-side` smoke；当前目标是把已落地主线核证成可确认 smoke 路线。
- 仅在 `axis-side` smoke 确认通过后，再安排 `axis-side` full train。
- 若需要把 `B′` 或 `Route A` 的 full-train 结果写成“workspace 可直接复核”，先定位对应 log / output 证据。
