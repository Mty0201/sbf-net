> **Claude 新 chat 接手入口**：请先阅读 `START_HERE.md`（项目根目录），按其指定顺序阅读后再继续本文件。

# 0. 项目一句话定义

`semantic-boundary-field` 的目标是在 Pointcept PTv3 语义分割主干上加入 boundary field 监督，但当前阶段的主评判仍是 semantic segmentation 的 `val_mIoU`。

## 1. 当前阶段定位

- `2.5` 阶段已完成
- `Stage-2 v1` full train 已完成: best `71.34`，失败
- `Stage-2 v2` full train 已完成: 有效 best 约 `72.5`，未过 `73.8`
- 当前阶段状态: `B′ vs A experiment comparison phase`
- B′ 和 Route A 均已实现并通过 smoke，正在进入 full train 对照阶段
- 当前不做: 不再继续扫 support 参数；不扩展到 test/export/visualization

## 1.5 执行硬约束

- IDE 工作区根可能是 `Pointcept`，但唯一允许主动维护的项目根是 `semantic-boundary-field`
- `Pointcept` 仅作为宿主依赖存在，默认只读
- 未经明确授权，禁止修改 `semantic-boundary-field` 目录之外的代码
- 若问题疑似来自 `Pointcept` 或宿主接口，必须停止并汇报，不得自行兜底修补
- 当前处于原型/研究验证阶段，优先目标是暴露问题，而不是掩盖问题
- 禁止为了“跑通 / 通过率 / 兼容性”擅自加入 fallback、默认兼容层、自动绕过、吞错、保守修补
- “最小侵入”指少改代码、不扩大影响面，不代表允许通过兼容层隐藏问题

## 2. 当前已确认结论

- `semantic-only baseline = 73.8`
- `support-only(reg=1, cover=0.2) = 74.6`，当前最佳
- `support-only(reg=1, cover=0.25) = 74.4`，次优且稳定，可作参考
- `support + dir + dist = 71`，在当前架构下失败
- `Stage-2 v1` full train: best `71.34`（epoch 36），最终 `68.31`（epoch 100），未过安全线
- `Stage-2 v2` full train: 有效 best 约 `72.5`，未过 `73.8` 安全线；v2 相比 v1 有改善
- `dir` 可学习但会以 semantic 主任务性能为代价
- `dist` 不是主要矛盾
- 仓库内 `samples` 已对齐到六列 `edge.npy` 格式，samples 中也包含 `edge_support_id.npy`
- 仓库内已落地的实验路径:
  - `Stage-2 v1`: `SupportConditionedEdgeHead` + 独立 config（full train 已完成，失败）
  - `Stage-2 v2`: post-backbone branch split + 独立 config（full train 已完成，未过线）
  - `Route A`: `RouteASemanticBoundaryLoss` + sidecar `edge_support_id.npy` + basin coherence + 独立 config（smoke 已通过）
  - `B′`: `SemanticBoundaryLoss(support_weighted_edge=True)` + 独立 config（smoke 已通过）
- smoke 验证需使用 `ptv3` conda 环境（含 `flash_attn`）

## 2.5 当前最小技术摘要

- 当前结构: `PTv3 shared backbone -> semantic_adapter -> semantic_head` + `backbone -> boundary_adapter -> edge_head`（Stage-2 v2 架构）。
- edge 分支输出: `support / dir / dist`。
- B′ 修改: `SemanticBoundaryLoss` 新增 `support_weighted_edge` 开关，开启时 `loss_dir` / `loss_dist` 使用 `support_gt * valid_gt` 归一化加权平均。
- Route A 修改: `RouteASemanticBoundaryLoss` 在 `SemanticBoundaryLoss` 基础上加 `loss_coherence`（local within-basin direction consistency）。需要 `edge_support_id.npy` 提供 basin 标识。

## 3. 当前正式主线

当前正式主线是 B′ vs Route A 的实验对照，以确定 direction 监督改善的最优路径。

- B′（D1-O0）: 只修正 direction/distance 监督域为 support-weighted，不加显式 basin 组织。优先 full train。
- Route A（D1-O1）: 在 B′ 基础上加 basin coherence（需要 `edge_support_id.npy`）。B′ 之后 full train。
- 验收口径: `<73.8` 失败；`73.8 ~ 74.6` direction 不再伤害 semantic；`>74.6` direction 成为净增益项。
- 约束: 不修改 Pointcept 主体，不更改训练入口。

## 4. 建议下一窗口先读

- `AGENTS.md`
- `project_memory/01_current_architecture.md`
- `project_memory/02_loss_design.md`
- `project_memory/05_active_stage.md`
- `project_memory/06_task_queue.md`
- `project_memory/90_archived_decisions.md`
