# 0. 项目一句话定义

`semantic-boundary-field` 的目标是在 Pointcept PTv3 语义分割主干上加入 boundary field 监督，但当前阶段的主评判仍是 semantic segmentation 的 `val_mIoU`。

## 1. 当前阶段定位

- `2.5` 阶段已完成
- 当前阶段状态: `Stage-2 entry preparation phase`
- `Stage-2` 即将正式进入 `architecture improvement phase`
- 当前不做: 不再继续扫 support 参数；不扩展到 test/export/visualization；不夸大为“方向项已解决”

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
- `support-only(reg=1, cover=0.3) = 73.7`，不优于基线；其它 cover 参数也均不优于 `73.8`
- `support + dir + dist = 71`，在当前架构下失败
- `Stage-2 v1` 真实 full train: `SupportConditionedEdgeHead + support(1.0/0.2) + dir(1.0) + dist(0.0)` 最佳 `val_mIoU = 71.34`（epoch 36），最终 `68.31`（epoch 100），仍低于 `73.8`
- `dist` 项在不到一个 epoch 内快速降到极小值（约 `0.0002`），当前不是主要矛盾
- `dir` 项可学习；训练时 `dir_cosine` 可到约 `0.6`，不训练时在 `0` 附近
- 但 `dir` 的学习会以 semantic 主任务性能为代价
- `Stage-2 v1` 真实 run 中，train `dir_cosine` 后期可到约 `0.65 ~ 0.75`，但验证期按样本均值统计仅约 `0.27`（best epoch）到 `0.31`（final epoch）；同一 run 中验证 `support_cover` 由约 `0.72` 下滑到约 `0.53 / 0.51`
- 当前简单线性层堆叠的多头结构表达能力不足，本质上仍在争抢 backbone 特征空间
- 仓库内已落地一条隔离的 `Stage-2` 最小实验路径：`SupportConditionedEdgeHead` + 独立 `stage2-support-dir` config；主线 train config 未被改写
- 仓库内已落地一条隔离的 `Stage-2 v2` 最小实现路径：post-backbone `semantic_adapter / boundary_adapter` branch split + 独立 `stage2-v2` model/train/train-smoke config；主线 train config 未被改写
- 仓库内已补齐 `Stage-2` 的 sample smoke 启动 config：`stage2-support-dir-train-smoke.py`
- `Stage-2 v2` smoke 已在当前会话环境完成 config / data / model / optimizer 初始化并进入 trainer training loop；随后在 PTv3/spconv forward 处因缺少 NVIDIA driver 停止，应视为宿主环境限制而非 v2 装配失败
- 训练入口中的 `smoke_mode` 启动期属性缺失问题已修复；当前由 trainer 内部按 `max_train_batches / max_val_batches` 自动初始化
- 仓库内 `samples` 已对齐到当前正式六列 `edge.npy` 格式，可优先用于最小 smoke / runtime 验证；本轮已修复旧五列 sample 资产导致的 `edge[:, 5]` 越界问题
- 在当前受限会话环境中，`Stage-2` smoke 已可进入第一轮训练迭代；后续若在 PTv3 forward 处再被 NVIDIA driver / CUDA 宿主条件卡住，应视为环境限制而非当前项目逻辑阻塞

## 2.5 当前最小技术摘要

- 当前最小结构是 `PTv3 shared backbone -> semantic head + boundary heads`；semantic head 负责分割，boundary 侧当前直接输出 `support / dir / dist`。
- 所谓“简单线性层堆叠多头”，指的是多个较薄的预测头直接共享同一份 backbone 特征，再各自做线性式映射或轻量回归。
- 这意味着 semantic 与 boundary 分支没有足够的中间解耦层，direction 一旦开始占用特征表达，就会直接挤压 semantic 主任务所需空间。
- `support-only` 有效，是因为它仍在做较粗的边界提醒，能辅助 semantic；`dir` 可学习但当前有害，是因为它需要更强几何表达，现结构承接不住；`dist` 很快降到极小值，说明它不是当前主矛盾。
- 因此 `2.5` 的结果不是把问题继续推回 loss 参数，而是把问题明确推向架构层。

## 3. 当前正式主线

当前正式主线已经从 `2.5` 阶段的 support 参数探索切换为：准备正式进入 `Stage-2`，从架构改进角度重新接入 direction 项，并以 `semantic-only baseline 73.8` 作为安全线、以超越 `support-only best 74.6` 作为当前主目标。

第一性问题:

- `Stage-2` 不是继续扫 support 参数，也不是把 `dir` 临时弱化或绕开。
- `Stage-2` 真正要解决的是：在不伤害 semantic `val_mIoU` 的前提下，如何通过架构改进给 direction 分支足够的表达与解耦空间。
- 第一问可以直接写成: 当前 shared-backbone + thin-head 结构下，怎样接入 direction supervision，才能不再用 semantic 主任务性能做代价？

补充约束:

- 当前 boundary 是 semantic boundary，因此 semantic `val_mIoU` 仍是主评判指标。
- `Stage-2` 的核心不是继续扫 support 参数，而是围绕当前结构的表达能力与分支竞争问题做架构改进。
- 当前最佳进入 `Stage-2` 的实验参考点是 `support-only(reg=1, cover=0.2)`；`0.25` 可作稳定次优参考。
- 当前 `Stage-2` 的验收口径是：`<73.8` 失败；`73.8 ~ 74.6` 仅说明 direction 不再明显伤害 semantic；只有 `val_mIoU > 74.6` 才能说明 architecture improvement 让 direction 成为净增益项。
- 不修改 Pointcept 主体，不更改当前训练入口、主 config、主模型职责划分。

## 4. 建议下一窗口先读

- `AGENTS.md`
- `project_memory/01_current_architecture.md`
- `project_memory/02_loss_design.md`
- `project_memory/05_active_stage.md`
- `project_memory/06_task_queue.md`
- `project_memory/90_archived_decisions.md`
