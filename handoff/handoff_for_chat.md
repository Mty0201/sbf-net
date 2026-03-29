> **接手入口**：本文件只提供新 chat 的最小背景。读完后请回到项目根目录先读 `AGENTS.md`，再按其指定顺序继续。

# 0. 项目一句话定义

`semantic-boundary-field` 的目标是在 Pointcept PTv3 语义分割主干上加入 boundary field 监督，但当前阶段的主评判仍是 semantic segmentation 的 `val_mIoU`。

## 1. 当前阶段与主线

- `2.5` 阶段已完成。
- 当前阶段状态: `Stage-2 architecture rollout / verification phase`。
- 当前 active 主表达: `axis + side + support`。
- 作者口头中的 `magnitude` 在当前代码 / 文档同步中等价于 `support`。
- 当前验证中心: `semseg-pt-v3m1-0-base-bf-edge-axis-side-train` 与其 smoke config。
- 当前不做: 不重新扫 support 参数；不重开 `dist` 主线；不扩展到 Pointcept 改写、test/export/visualization。

## 1.5 执行硬约束

- IDE 工作区根可能是 `Pointcept`，但唯一允许主动维护的项目根是 `semantic-boundary-field`。
- `Pointcept` 仅作为宿主依赖存在，默认只读。
- 未经明确授权，禁止修改 `semantic-boundary-field` 目录之外的代码。
- 若问题疑似来自 `Pointcept` 或宿主接口，必须停止并汇报，不得自行兜底修补。
- 当前处于原型 / 研究验证阶段，优先目标是暴露问题，而不是掩盖问题。
- 禁止为了“跑通 / 通过率 / 兼容性”擅自加入 fallback、默认兼容层、自动绕过、吞错、保守修补。

## 2. 当前已确认事实

- 作者已确认的实验事实:
  - `semantic-only baseline = 73.8`
  - `support-only(reg=1, cover=0.2) = 74.6`，当前最佳
  - `support-only(reg=1, cover=0.25) = 74.4`，次优且稳定，可作参考
  - `support + dir + dist = 71`，是旧 signed-direction 结构下的失败事实
  - `Stage-2 v1` full train: best `71.34`（epoch 36），最终 `68.31`（epoch 100）
  - `Stage-2 v2` full train: best `72.38`，未过 `73.8`
- 当前路线状态:
  - `Stage-2 v1` / `Stage-2 v2`: 历史 signed-direction full-train 路线
  - `Route A`: 平行 signed-direction 路线；smoke 已通过，full train 尚未确认
  - `B′`: 平行 signed-direction 路线；smoke 已通过，当前 workspace 未定位到 full-train 产物
  - `axis-side`: 当前 active 主线；loss / evaluator / trainer / config 已落地，smoke 尚未确认
- 仓库内 `samples` 已对齐到六列 `edge.npy` 格式，samples 中也包含 `edge_support_id.npy`。
- smoke 验证需使用 `ptv3` conda 环境（含 `flash_attn`）。

## 2.5 当前可直接复核的本地产物

- `Route A` smoke 输出包含 `train.log + model_best.pth + model_last.pth`，且 `loss_coherence` 非零。
- `B′` smoke 输出包含 `train.log + model_best.pth + model_last.pth`。
- `axis-side` 相关实现文件已在当前 working tree 中落地。
- 当前 `axis-side` smoke 输出仅有一次启动日志，未见 val 行或 checkpoint。
- 历史 full-train 数字仍按“作者已确认实验事实”处理，即使当前 workspace 未保留完整 artifact。

## 3. 当前正式主线

- 当前 active 主线是 `axis + side + support`，不是旧的 `support + dir + dist`。
- 当前 `axis-side` 路线复用 `Stage-2 v2` 模型和现有五通道 edge 输出；`edge_pred[:, 0:3]` 解释为 axis，`edge_pred[:, 3]` 解释为 `side_logit`，`edge_pred[:, 4]` 解释为 `support_logit`。
- `side GT` 从现有六列 `edge.npy` 的 `dir_gt` 运行时导出；不新增 sidecar。
- axis 用 sign-invariant cosine；side 用 BCE；support 保持不变。
- 当前 workspace 仅定位到一次 `axis-side` smoke 启动日志（`Device: cpu`），未见 train / val 行或 checkpoint；因此当前不能把 `axis-side smoke passed` 写成已确认事实。

## 4. 下一窗口先做什么

- 先读 `AGENTS.md`、`project_memory/01_current_architecture.md`、`project_memory/02_loss_design.md`、`project_memory/05_active_stage.md`、`project_memory/06_task_queue.md`。
- 第一优先级: 在 CUDA-enabled `ptv3` 环境中确认 `axis-side` smoke。
- 第二优先级: 仅在 smoke 确认通过后，再决定是否运行 `axis-side` full train。
- 若需要把 `B′` 或 `Route A` 的 full-train 结果写成“workspace 可直接复核”，先定位对应 output / log。
