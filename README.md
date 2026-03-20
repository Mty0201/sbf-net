# SBF-Net
## Semantic Boundary Field Network for Edge-Aware LiDAR Semantic Segmentation

## 1. Introduction / 项目简介

SBF-Net is a research-oriented project for LiDAR point cloud semantic segmentation with semantic boundary field supervision.  
SBF-Net 是一个面向研究的 LiDAR 点云语义分割项目，核心是引入语义边界场监督。

It extends a Pointcept PTv3 semantic segmentation baseline with an additional boundary-aware branch while keeping backbone reuse non-intrusive and maintainable.  
它在 Pointcept 的 PTv3 语义分割基线上增加了边界感知分支，同时保持对 backbone 复用方式的非侵入性与可维护性。

The project is designed as an independent repository on top of Pointcept rather than a fork that copies and rewrites the upstream training stack.  
本项目被设计为构建在 Pointcept 之上的独立仓库，而不是复制并重写上游训练体系的私有分叉。

## 2. Key Features / 项目特点

- Dual-task learning with semantic segmentation and boundary field prediction.  
  双任务学习，同时进行语义分割与边界场预测。

- Edge-aware supervision driven by semantic boundary field labels.  
  使用语义边界场标签进行边界感知监督。

- Built on top of Pointcept and PTv3 without modifying Pointcept source code.  
  构建在 Pointcept 和 PTv3 之上，且不修改 Pointcept 源码。

- Minimal intrusive integration for maintainable research iteration.  
  采用最小侵入式集成，便于后续持续迭代和维护。

## 3. Framework Overview / 方法概述

The current framework uses a shared PTv3 backbone for point feature extraction.  
当前框架使用共享的 PTv3 backbone 进行点特征提取。

On top of the shared backbone, SBF-Net keeps a semantic segmentation head and adds a boundary field head.  
在共享 backbone 之上，SBF-Net 保留语义分割 head，并新增边界场 head。

The current boundary field head predicts boundary offset vectors and boundary support scores from a lightweight shared stem.  
当前边界场 head 通过轻量共享 stem 预测边界偏移向量与边界支撑分数。

The current compact training label `edge.npy` uses the fixed layout `[vec_x, vec_y, vec_z, edge_support, edge_valid]`.  
当前紧凑训练标签 `edge.npy` 使用固定列语义 `[vec_x, vec_y, vec_z, edge_support, edge_valid]`。

`edge_valid` is only a numeric validity domain for supervision, not a predicted mask target. Legacy names `edge_strength` and `edge_mask` are compatibility aliases for `edge_support` and `edge_valid`.  
`edge_valid` 只是监督数值有效域，不是要预测的 mask 目标。旧名称 `edge_strength` 和 `edge_mask` 仅分别作为 `edge_support` 与 `edge_valid` 的兼容别名。

The current public release focuses on a trainable and reproducible first-stage system rather than a fully expanded task formulation.  
当前公开版本聚焦于一个可训练、可复现的第一阶段系统，而不是一次性展开完整任务设计。

## 4. Installation / 安装

SBF-Net depends on Pointcept and does not include PTv3 or Pointcept internals.  
SBF-Net 依赖 Pointcept，本仓库不包含 PTv3 或 Pointcept 内部实现。

You should prepare Pointcept first, then run SBF-Net in the same environment.  
你应先准备好 Pointcept，再在同一环境中运行 SBF-Net。

Recommended environment name: `ptv3`.  
推荐使用的环境名称为 `ptv3`。

Recommended quick-start workflow:  
推荐的快速开始流程如下：

```bash
git clone <your-pointcept-repo> Pointcept
git clone <your-sbf-net-repo> SBF-Net
conda activate ptv3
export POINTCEPT_ROOT=/path/to/Pointcept
export SBF_DATA_ROOT=/path/to/BF_edge_chunk_npy
```

Required environment variables:  
需要设置的环境变量如下：

```bash
export POINTCEPT_ROOT=/path/to/Pointcept
export SBF_DATA_ROOT=/path/to/BF_edge_chunk_npy
```

Please refer to the detailed installation guide for environment checks and layout suggestions.  
有关环境检查和仓库摆放建议，请参考详细安装文档。

- [install.md](install.md)

## 5. Training / 训练

SBF-Net currently provides both dual-task training configs and semantic-only calibration configs.  
SBF-Net 当前同时提供双任务训练配置和 semantic-only 校准配置。

Current edge-task semantics:
当前 edge 任务语义：

- model output: `vec + support`
- training target: `edge.npy = [vec_x, vec_y, vec_z, edge_support, edge_valid]`
- `edge_valid` is only used as a supervision validity domain
- trainer log keys may still show legacy names such as `loss_mask` / `loss_strength` for compatibility, but they no longer mean a predicted mask task

Quick Start for full dual-task training:  
正式双任务训练的最小快速开始命令如下：

```bash
conda run --no-capture-output -n ptv3 python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py \
  --pointcept-root "$POINTCEPT_ROOT"
```

Smoke test command:  
Smoke 测试命令：

```bash
conda run --no-capture-output -n ptv3 python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train-smoke.py \
  --pointcept-root "$POINTCEPT_ROOT"
```

Full training command:  
正式训练命令：

```bash
conda run --no-capture-output -n ptv3 python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py \
  --pointcept-root "$POINTCEPT_ROOT"
```

The current trainer supports scheduler, gradient accumulation, total_epoch/eval_epoch style organization, resume, and weight loading.  
当前 trainer 已支持 scheduler、梯度累积、`total_epoch/eval_epoch` 风格训练组织、resume 与 weight 加载。

The current full training path uses `OneCycleLR`, `grad_accum_steps`, and Pointcept-style displayed epoch organization based on `total_epoch / eval_epoch / train_loop`.  
当前正式训练路径使用 `OneCycleLR`、`grad_accum_steps`，以及基于 `total_epoch / eval_epoch / train_loop` 的 Pointcept 风格显示 epoch 组织。

The effective batch size is computed as `batch_size * grad_accum_steps`.  
当前有效 batch size 按 `batch_size * grad_accum_steps` 计算。

The best checkpoint is still selected strictly by validation semantic mIoU.  
当前 best checkpoint 仍然严格按验证集语义分支的 mIoU 进行选择。

Checkpoint outputs are stored under `outputs/.../model/`.  
Checkpoint 输出位于 `outputs/.../model/` 目录下。

The full training config currently uses scheduler, `grad_accum_steps`, `total_epoch`, and `eval_epoch` as the main runtime knobs.  
当前正式训练配置使用 scheduler、`grad_accum_steps`、`total_epoch` 和 `eval_epoch` 作为主要运行控制项。

SBF-Net also provides a semantic-only baseline for trainer calibration against the original Pointcept PTv3 semantic segmentation path.  
SBF-Net 还提供了 semantic-only 基线，用于和 Pointcept 原版 PTv3 语义分割训练路径做训练框架校准。

Semantic-only smoke command:  
semantic-only smoke 命令：

```bash
conda run --no-capture-output -n ptv3 python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-semantic-train-smoke.py \
  --pointcept-root "$POINTCEPT_ROOT"
```

Semantic-only full command:  
semantic-only 正式训练命令：

```bash
conda run --no-capture-output -n ptv3 python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-semantic-train.py \
  --pointcept-root "$POINTCEPT_ROOT"
```

Please refer to the training guide for runtime details.  
运行细节请参考训练文档。

- [train.md](train.md)

## 6. Project Structure / 项目结构

The repository keeps all project-specific code inside the SBF-Net workspace and treats Pointcept as an external upstream dependency.  
仓库将所有项目特有代码保留在 SBF-Net 工作区内部，并将 Pointcept 视为外部上游依赖。

- `project/`: project-local datasets, transforms, models, losses, evaluator, and trainer.  
  `project/`：项目内的数据集、变换、模型、损失、评估器与训练器实现。

- `configs/`: BF dataset configs, smoke configs, and full training configs.  
  `configs/`：BF 数据配置、smoke 配置与正式训练配置。

- `scripts/`: runnable training and smoke-check scripts.  
  `scripts/`：可直接运行的训练脚本与 smoke 检查脚本。

- `docs/`: internal design notes, boundary documents, and collaboration rules.  
  `docs/`：内部设计说明、边界文档与协作规则。

Detailed directory notes are available here.  
更详细的目录说明见：

- [docs/project_structure.md](docs/project_structure.md)

## 7. Current Status / 当前状态

Current status: the first public trainable release is available, and the project-local runtime has already moved beyond a one-off smoke-only stage.  
当前状态：第一版可公开训练的版本已经具备，项目内运行时系统也已经超出仅限 smoke 的最小原型阶段。

Implemented: training, validation, project-local trainer, checkpointing, scheduler, gradient accumulation, and Pointcept-style runtime organization in single-card mode.  
已完成：训练、验证、项目内 trainer、checkpoint、scheduler、梯度累积，以及单卡场景下更接近 Pointcept 的运行组织。

Implemented baselines now include both the main dual-task path and a semantic-only calibration path under the same project-local trainer.  
当前已实现的基线同时包括主双任务路径，以及在同一项目内 trainer 下运行的 semantic-only 校准路径。

Not implemented yet: test pipeline, result export, visualization export, and distributed training support.  
尚未完成：test pipeline、结果导出、可视化导出，以及分布式训练支持。

## 8. Roadmap / 未来计划

- Introduce refined support-field supervision variants.  
  引入更细化的 support-field 监督变体。

- Improve edge-side loss design and evaluation.  
  继续完善边界分支的 loss 设计与评估方式。

- Build a project-local test pipeline.  
  构建项目内的 test pipeline。

- Add result export and visualization utilities.  
  增加结果导出与可视化工具。

## 9. Citation / 引用

Coming soon.  
引用信息准备中。

If you use SBF-Net before the citation entry is finalized, please cite the upstream Pointcept project and mention this repository in your implementation details.  
如果你在正式引用条目发布前使用了 SBF-Net，请先引用 Pointcept，并在实现细节中说明本仓库。

## 10. License / 许可证

SBF-Net is released under the MIT License.  
SBF-Net 采用 MIT License 开源。

See the license file for details.  
详情请参见许可证文件。

- [LICENSE](LICENSE)
