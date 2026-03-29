# Current Architecture

## Shared Base

- 主模型: `SharedBackboneSemanticBoundaryModel`。
- 当前 active 主线复用 `Stage-2 v2` 的 shared-backbone split 结构: `PT-v3m1 shared backbone -> semantic_adapter -> semantic_head`，以及 `backbone -> boundary_adapter -> edge_head`。
- boundary path 内部仍沿用 `SupportConditionedEdgeHead`；`axis-side` 不新增独立 model / head class，只改 loss / evaluator / trainer log semantics 与 config 选择。
- semantic 分支输出 `seg_logits`，负责 8 类语义分割。
- 模型统一输出接口仍保持 `seg_logits / edge_pred`；`edge_pred` 固定为 5 通道张量。
- 当前结构观察: semantic 与 boundary 分支仍强共享 backbone 特征，edge 分支本身较薄，对 backbone 特征的直接依赖较强。

## Current Active Mainline: Axis-Side

- 当前 active 主表达统一为 `axis + side + support`。
- 当前 active `edge_pred` 解释为 `[axis(3), side_logit(1), support_logit(1)]`。
- `axis` 表示 unsigned direction axis；`side` 表示 support 两侧的二值符号；`support` 保持原有边界邻域覆盖语义。作者口头中的 `magnitude` 在当前代码 / 文档同步中等价于 `support`。
- `axis-side` 复用现有六列 GT `edge.npy = [dir_x, dir_y, dir_z, dist, support, valid]`；`side GT` 由现有 `dir_gt` 在 loss / evaluator 内运行时导出，不新增 sidecar。
- `dist_gt` 仍保留在 GT 中，用于 `tau_dir` 有效域判定与历史对照指标；当前 active 主线不再把 `dist` 作为独立预测通道。
- trainer / evaluator 已新增 axis-side 专用指标与日志分支: `loss_axis / loss_side / axis_cosine / side_accuracy / dir_cosine`。

## Historical Signed-Direction Routes

- 历史 signed-direction 主表达是 `support + dir + dist`；在这些路线下，同一 5 通道 `edge_pred` 被解释为 `[dir(3), dist(1), support(1)]`。
- `Stage-2 v1` 通过 `edge_head_cfg` 显式选择 `SupportConditionedEdgeHead`；当前已确认 full train 最佳 `71.34`，最终 `68.31`。
- `Stage-2 v2` 把第一次分流前移到 backbone 输出后，减轻 semantic / boundary 特征竞争；当前已确认 full train best `72.38`，仍未过 `73.8`。
- `B′` 和 `Route A` 都建立在上述 signed-direction 语义之上；它们是当前 `axis-side` 主线的前序路线或平行对照路线，不是当前 active 主表达。
- 当前已确认: 在现有 shared-backbone + thin-head 条件下，直接使用 signed direction supervision 会把 semantic `val_mIoU` 拉低到约 `71`；这正是当前主线改写为 `axis + side + support` 的背景。

## Supporting Modules

- 数据模块: `BFDataset` 负责加载样本目录中的正式六列 `edge.npy`；Route A 专用 `support_id` sidecar 仍是可选加载。
- 变换模块: `InjectIndexValidKeys` 负责把 `edge` 纳入 Pointcept 的索引同步链。
- loss 模块: 当前 active 主线使用 `AxisSideSemanticBoundaryLoss`；signed-direction 对照路线仍保留 `SemanticBoundaryLoss` 与 `RouteASemanticBoundaryLoss`。
- evaluator 模块: 当前 active 主线使用 `AxisSideEvaluator`；signed-direction 对照路线仍保留 `SemanticBoundaryEvaluator`。
- runtime 模块: `SemanticBoundaryTrainer` 负责训练、验证、scheduler、checkpoint，并已为 axis-side 增加 train / val summary 分支。
- trainer: 当 batch 中包含 `support_id` 时，自动向 loss 传递 `coord / support_id / offset`。
