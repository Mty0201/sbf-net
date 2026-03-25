# 0. 项目一句话定义

`semantic-boundary-field` 的目标是在 Pointcept PTv3 语义分割主干上加入 boundary field 监督，联合学习 semantic segmentation 和边界相关信号。  
当前最终优化目标仍是 semantic segmentation 的 `val_mIoU`，edge 分支只服务于提升边界区域的语义判别。

## 1. 当前阶段定位

- 当前阶段: `Stage-1 Dual-Task Training Phase`
- 当前在做: 验证当前 `semantic + direction + distance + support` 主线是否能稳定提升 semantic 边界表现
- 当前不做: 不改 Pointcept 主体，不扩展 test/export/visualization/distributed training
- 当前唯一主问题: support 是否提供了足够强的 boundary prior 来改善 semantic 边界

## 2. 阅读路径

### Step 1：先读这些文档

- `project_memory/00_project_overview.md`
- `project_memory/01_current_architecture.md`
- `project_memory/02_loss_design.md`
- `project_memory/03_data_pipeline.md`
- `project_memory/04_training_rules.md`
- `project_memory/05_active_stage.md`
- `project_memory/06_task_queue.md`
- `project_memory/07_pointcept_interface.md`
- `project_memory/90_archived_decisions.md`

### Step 2：再读这些模块

- `project/models/semantic_boundary_model.py`
- `project/models/heads.py`
- `project/losses/semantic_boundary_loss.py`
- `project/evaluator/semantic_boundary_evaluator.py`
- `project/datasets/bf.py`
- `project/trainer/trainer.py`
- `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py`
- `data_pre/bf_edge_v3/scripts/convert_edge_vec_to_dir_dist.py`

### Step 3：不要读这些内容

- Pointcept 上游主体实现
- 与当前阶段无关的上游通用代码
- 已废弃的旧语义命名所对应的历史路径
- 与当前主问题无关的仓库内容

## 3. 当前系统结构

- backbone: `PT-v3m1` shared backbone
- semantic 分支: 输出 `seg_logits`，负责 8 类语义分割
- edge 分支: 输出 `direction + distance + support`
- trainer/loss 入口: 统一走 `edge_pred = [dir, dist, support]`

## 4. 当前有效语义定义

- support: 当前主边界监督，表示点处于边界邻域中的连续强度
- vec: 预处理中的几何位移向量，定义为 `q - p`
- `q`: 点在最近 support 上的投影点
- `p`: 原始点坐标
- direction: `normalize(vec)`，表示朝最近边界 support 的单位方向
- distance: `||vec||`，表示到最近边界 support 的物理距离
- valid: 只表示监督有效域，不是预测目标
- 当前训练 GT 格式: `edge.npy = [dir_x, dir_y, dir_z, edge_dist, edge_support, edge_valid]`
- 旧命名 `mask/strength` 只作为兼容名保留，不代表当前正式语义

## 5. 当前 loss / evaluator 的作用划分

- semantic loss: `CrossEntropy + Lovasz`
- support loss: 当前关注重点
- support_reg: support 主项，负责连续 support 场回归
- support_cover: support 弱辅助项，负责 valid 区域覆盖约束
- direction loss: 当前保留的结构性几何约束
- distance loss: 当前保留的物理步长约束
- 当前训练路径没有单独的 `vec_loss`
- vec 当前只作为中间几何表示，用来生成 direction 和 distance GT
- 当前不是重新设计 loss，而是在验证 support 本身是否足够有效

## 6. 当前主问题

support 是否提供了足够强的 boundary prior 以改善 semantic 边界。

## 7. 下一步任务

在当前 `support_reg` 主导、`support_cover` 弱辅助的 support loss 下，回到 full dual-task 验证 semantic 是否恢复。

## 8. Web Chat 使用说明

- 你不能访问本地文件，只能读 GitHub 仓库
- 请先按本 handoff 的 Step 1 -> Step 2 顺序阅读
- 不要扩展到 Pointcept 主体或无关上游代码
- 不要把旧兼容命名误当成当前正式语义
- 分析必须围绕当前唯一主问题展开
- 当前阶段重点是 support 的有效性，不是整个系统 redesign
